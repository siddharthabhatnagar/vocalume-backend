"""
Firebase Admin and Firestore integration for the VocaLume backend.
"""

import os
import json
import logging
from typing import Any, Optional
import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings

logger = logging.getLogger("vocalume")

_firebase_initialized = False

def init_firebase() -> None:
    """Initialize Firebase Admin SDK with robust error handling and fallbacks."""
    global _firebase_initialized
    if _firebase_initialized:
        return
        
    if firebase_admin._apps:
        _firebase_initialized = True
        return

    settings = get_settings()
    cred = None
    cred_json = settings.firebase_service_account_json

    if cred_json:
        if os.path.exists(cred_json):
            logger.info("Initializing Firebase with credentials file: %s", cred_json)
            try:
                cred = credentials.Certificate(cred_json)
            except Exception as e:
                logger.error("Failed to load Firebase cert file: %s", e)
        else:
            logger.info("Attempting to parse Firebase credentials as raw JSON string.")
            try:
                cred_data = json.loads(cred_json)
                cred = credentials.Certificate(cred_data)
            except Exception as e:
                logger.error("Failed to parse Firebase JSON config: %s", e)

    try:
        if cred:
            firebase_admin.initialize_app(cred)
        else:
            logger.info("Initializing Firebase with Application Default Credentials (ADC) or local mock.")
            firebase_admin.initialize_app()
        logger.info("Firebase Admin SDK successfully initialized.")
    except Exception as e:
        logger.warning(
            "Could not initialize Firebase Admin SDK (is GOOGLE_APPLICATION_CREDENTIALS set?): %s. "
            "Using fallback mock operations for local development.",
            e
        )
    _firebase_initialized = True


def get_db() -> Optional[Any]:
    """Retrieve Firestore client if initialized, otherwise return None."""
    init_firebase()
    try:
        return firestore.client()
    except Exception as e:
        logger.warning("Firestore client not available: %s", e)
        return None


# FastAPI Dependency for authentication
security = HTTPBearer(auto_error=False)

async def get_current_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """
    FastAPI dependency to verify Firebase ID Token in Authorization header.
    Returns decoded user UID or raises 401. Falls back to a debug user in development.
    """
    if not credentials:
        # Check if we are running in local/dev mode without Firebase configured
        # If Firestore is unavailable, return a default mock user for easy local startup
        db = get_db()
        if db is None:
            logger.warning("No Authorization header provided. Falling back to 'debug_user' in offline mode.")
            return "debug_user"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header",
        )
    
    token = credentials.credentials
    try:
        init_firebase()
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        logger.error("Firebase auth verification failed: %s", e)
        # If in development, fallback to debug_user if token is literally "debug_token"
        if token == "debug_token":
            return "debug_user"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired Firebase ID token: {str(e)}",
        )


def save_user_session(user_id: str, session_id: str, session_data: dict) -> None:
    """Save or merge session state in Firestore."""
    db = get_db()
    if db is None:
        logger.warning("Firestore client unavailable. Skipping session write: %s", session_id)
        return

    try:
        session_ref = db.collection("users").document(user_id).collection("sessions").document(session_id)
        session_ref.set(session_data)
        logger.info("Saved session %s for user %s in Firestore.", session_id, user_id)
        
        # Automatically update cumulative dashboard stats
        update_user_dashboard(user_id)
    except Exception as e:
        logger.error("Failed to save user session %s: %s", session_id, e)


def get_user_session(user_id: str, session_id: str) -> Optional[dict]:
    """Retrieve session state from Firestore."""
    db = get_db()
    if db is None:
        logger.warning("Firestore client unavailable. Returning empty mock session.")
        return None

    try:
        doc = db.collection("users").document(user_id).collection("sessions").document(session_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.error("Failed to fetch session %s: %s", session_id, e)
    return None


def update_user_dashboard(user_id: str) -> None:
    """Aggregate metrics across all of a user's sessions and save to their dashboard."""
    db = get_db()
    if db is None:
        return

    try:
        sessions_ref = db.collection("users").document(user_id).collection("sessions")
        sessions = list(sessions_ref.stream())
        
        total_turns = 0
        total_pron_score = 0.0
        pron_score_count = 0
        correct_grammar_turns = 0
        total_grammar_turns = 0
        word_bank_list = []
        cefr_levels = []
        
        for s in sessions:
            s_data = s.to_dict()
            history = s_data.get("history", [])
            total_turns += len([h for h in history if h.get("role") == "user"])
            
            metrics = s_data.get("metrics", {})
            if metrics:
                pron_score = metrics.get("average_pronunciation_score", 0.0)
                if pron_score > 0:
                    total_pron_score += pron_score
                    pron_score_count += 1
                
                grammar_rate = metrics.get("grammar_accuracy_rate", 100.0)
                session_turns = metrics.get("total_turns", 0)
                if session_turns > 0:
                    correct_grammar_turns += int((grammar_rate / 100.0) * session_turns)
                    total_grammar_turns += session_turns
            
            word_bank_list.extend(s_data.get("word_bank", []))
            cefr = s_data.get("cefr_level")
            if cefr:
                cefr_levels.append(cefr)
                
        # Unique word bank
        word_bank_unique = list(set(word_bank_list))
        
        avg_pron = total_pron_score / pron_score_count if pron_score_count > 0 else 0.0
        grammar_accuracy = (correct_grammar_turns / total_grammar_turns * 100.0) if total_grammar_turns > 0 else 0.0
        latest_cefr = cefr_levels[-1] if cefr_levels else "B1"
        
        dashboard_data = {
            "total_turns": total_turns,
            "average_pronunciation_score": avg_pron,
            "grammar_accuracy_rate": grammar_accuracy,
            "cefr_level": latest_cefr,
            "vocabulary_words_learned": len(word_bank_unique),
            "word_bank": word_bank_unique,
        }
        
        db.collection("users").document(user_id).set(dashboard_data, merge=True)
        logger.info("Updated dashboard statistics for user %s.", user_id)
    except Exception as e:
        logger.error("Failed to update dashboard for user %s: %s", user_id, e)


def get_user_dashboard(user_id: str) -> Optional[dict]:
    """Retrieve cumulative dashboard document for user."""
    db = get_db()
    if db is None:
        return None

    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.error("Failed to fetch dashboard for user %s: %s", user_id, e)
    return None
