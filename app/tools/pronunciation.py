"""
Pronunciation scoring tool.

This is deliberately NOT LLM-based -- pronunciation quality can't be
judged from text alone, only from audio. Cerebras has no ASR/TTS
service, so real pronunciation confidence must come from the Android
client's on-device `SpeechRecognizer`, which can report a per-utterance
(and on some devices per-word) confidence value. When the client
supplies a `confidence_map` (word -> 0.0-1.0 confidence) inside
`user_profile`, we scale those confidences directly into 0-100 scores.

When no `confidence_map` is available (e.g. during early integration,
testing, or on devices whose ASR doesn't expose confidences), we fall
back to a transparent, deterministic PLACEHOLDER heuristic based on
word length and vowel count, clamped into [70, 95] so it never looks
alarmingly bad or suspiciously perfect. This heuristic carries no
linguistic meaning about actual pronunciation quality -- it exists
purely so the UI has *something* stable to render in a demo before
real ASR confidences are wired up. It is designed to be a drop-in
swap point for a real scorer (e.g. NVIDIA Riva pronunciation
assessment) in the future.
"""

from app.schemas import PronunciationFeedback, PronunciationWordScore


def _color_for_score(score: float) -> str:
    if score >= 85:
        return "green"
    if score >= 65:
        return "yellow"
    return "red"


def _placeholder_word_score(word: str) -> float:
    """Deterministic pseudo-score based on word shape. NOT a real pronunciation
    measure -- see module docstring. Clamped to [70, 95]."""
    letters = [c for c in word.lower() if c.isalpha()]
    if not letters:
        return 80.0
    vowels = sum(1 for c in letters if c in "aeiou")
    length_factor = min(len(letters), 12) / 12.0
    vowel_ratio = vowels / max(len(letters), 1)
    raw = 70 + (length_factor * 15) + (vowel_ratio * 10)
    return max(70.0, min(95.0, round(raw, 1)))


async def score_pronunciation(
    transcript: str,
    confidence_map: dict | None = None,
) -> PronunciationFeedback:
    """Score each word in the transcript, using real ASR confidences if
    available, otherwise a transparent placeholder heuristic. Never raises."""
    try:
        words = transcript.split()
        if not words:
            return PronunciationFeedback(overall_score=0.0, words=[])

        scored: list[PronunciationWordScore] = []
        for word in words:
            if confidence_map and word in confidence_map:
                score = max(0.0, min(1.0, float(confidence_map[word]))) * 100
            else:
                score = _placeholder_word_score(word)
            scored.append(
                PronunciationWordScore(
                    word=word,
                    score=round(score, 1),
                    color=_color_for_score(score),
                )
            )

        overall = round(sum(w.score for w in scored) / len(scored), 1)
        return PronunciationFeedback(overall_score=overall, words=scored)
    except Exception:
        return PronunciationFeedback(overall_score=75.0, words=[])
