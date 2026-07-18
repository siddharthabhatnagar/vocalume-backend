"""
VocaLume backend application package.

This package contains the FastAPI application, LangGraph conversation
pipeline, Cerebras-backed LLM services, and analysis tools (grammar,
pronunciation, vocabulary, CEFR classification) that power the
VocaLume Android English-speaking-coach app. The package version is
exposed here so routers (e.g. /health, /info) can report it without
importing the full app module tree.
"""

__version__ = "0.1.0"
