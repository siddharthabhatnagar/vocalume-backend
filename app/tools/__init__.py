"""
Tools package for the VocaLume backend.

Each module here implements one analysis capability applied to a
learner's spoken-then-transcribed utterance: grammar correction,
pronunciation scoring, vocabulary extraction, and CEFR-level
classification. Grammar, vocabulary, and CEFR tools are LLM-backed
(via the Cerebras analyzer chat) and demand strict JSON output with
tolerant parsing and fail-safe fallbacks so a malformed LLM response
never crashes a request. Pronunciation scoring is a transparent
heuristic (not LLM-backed) designed to be swapped for a real ASR
confidence-based scorer later.
"""
