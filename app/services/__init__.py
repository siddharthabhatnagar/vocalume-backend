"""
Services package for the VocaLume backend.

Houses thin wrappers around external providers. Currently this contains
only `cerebras_llm.py`, which wraps `langchain_openai.ChatOpenAI`
pointed at the Cerebras Inference API's OpenAI-compatible endpoint.
There is intentionally no ASR/TTS or Riva service module here: Cerebras
is a text-only inference provider, so speech recognition and synthesis
are delegated to the Android client (SpeechRecognizer / TextToSpeech).
"""
