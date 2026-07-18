# VocaLume Backend (Cerebras Edition)

Real-time AI backend for **VocaLume**, an Android English-speaking-coach app.
This backend analyzes a learner's spoken (then transcribed) English, gives
structured feedback on grammar, pronunciation, and vocabulary, tracks the
learner's CEFR level, and drives an in-character role-play conversation --
all powered by the **Cerebras Inference API** and deployable as a Vercel
serverless function.

## Tech stack

| Layer            | Choice                                              |
|-------------------|------------------------------------------------------|
| Language          | Python 3.12                                         |
| Web framework      | FastAPI + uvicorn                                    |
| AI orchestration   | LangChain 0.3 + LangGraph 0.2                        |
| LLM provider       | Cerebras Inference API (`inference.cerebras.ai/v1`) |
| LLM model          | `llama3.3-70b` (default)                             |
| Schema             | Pydantic v2                                          |
| Streaming          | Server-Sent Events (SSE) via `sse-starlette`         |
| HTTP client        | `httpx` (async)                                      |
| Deployment target  | Vercel serverless (`@vercel/python`)                 |

Cerebras Inference exposes an **OpenAI-compatible** `/chat/completions`
endpoint, so the LLM integration reuses `langchain_openai.ChatOpenAI`
unmodified -- we simply point `base_url` at Cerebras and use the Cerebras
API key in place of an OpenAI key.

## Directory structure

```
vocalume-backend/
├── api/
│   └── index.py              # Vercel entry point (imports FastAPI app)
├── app/
│   ├── __init__.py           # __version__ = "0.1.0"
│   ├── config.py             # Pydantic Settings (env vars)
│   ├── schemas.py             # Full wire contract
│   ├── main.py                # FastAPI app factory + CORS + router wiring
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py           # ConversationState TypedDict
│   │   ├── nodes.py           # analyze / classify / respond nodes
│   │   └── builder.py         # StateGraph assembly (compiled at import)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── grammar.py         # LLM grammar corrector
│   │   ├── pronunciation.py   # Heuristic pronunciation scorer
│   │   ├── vocab.py           # LLM vocabulary extractor
│   │   └── cefr.py            # LLM CEFR classifier with smoothing
│   ├── services/
│   │   ├── __init__.py
│   │   └── cerebras_llm.py    # ChatOpenAI wrapper pointed at Cerebras
│   └── routers/
│       ├── __init__.py
│       ├── health.py          # /health, /info
│       ├── chat.py            # /chat, /chat/stream (SSE)
│       └── session.py         # /session/* placeholder contract
├── requirements.txt
├── runtime.txt
├── vercel.json
├── .env.example
├── .gitignore
└── README.md
```

## Getting a Cerebras API key

1. Go to https://cloud.cerebras.ai/ and sign up (free tier available).
2. Open **API Keys** in the dashboard.
3. Click **Create API Key**, copy the value -- it starts with `csk-`.
4. Put it in your `.env` file (local dev) or your Vercel project's
   environment variables (production) as `CEREBRAS_API_KEY`.

Do **not** confuse this with an NVIDIA NIM key (`nvapi-...`) -- this backend
talks only to Cerebras.

## LangGraph pipeline

```
START --> analyze --> classify --> respond --> END
```

- **analyze**: runs grammar correction, pronunciation scoring, and vocabulary
  extraction **concurrently** via `asyncio.gather`. This is the main latency
  win -- three independent LLM/heuristic calls run in parallel instead of
  sequentially, so the analysis phase takes roughly as long as the slowest
  single call rather than the sum of all three.
- **classify**: runs a few-shot CEFR (A1-C2) classifier on the transcript,
  smoothed against the learner's prior level so one short utterance can't
  cause a wild jump in tracked level.
- **respond**: builds a role-play-appropriate system prompt calibrated to the
  learner's CEFR level, and calls the Cerebras conversational chat model to
  produce the in-character reply that will be spoken back via the Android
  client's TTS engine.

## API reference

### `GET /health`

```bash
curl https://your-deployment.vercel.app/health
```
```json
{"status": "ok", "service": "vocalume-backend", "version": "0.1.0"}
```

### `GET /info`

```bash
curl https://your-deployment.vercel.app/info
```
```json
{
  "service": "vocalume-backend",
  "version": "0.1.0",
  "cerebras_configured": true,
  "cerebras_model": "llama3.3-70b",
  "max_history_turns": 10
}
```

### `POST /chat`

Non-streaming: runs the full pipeline and returns reply + feedback together.

```bash
curl -X POST https://your-deployment.vercel.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "transcript": "I go to the store yesterday to buy some milk.",
    "history": [],
    "cefr_level": "B1",
    "role_play_mode": "free_talk"
  }'
```

### `POST /chat/stream`

Streaming via SSE: emits a `feedback` event first, then `token` events as the
reply is generated, then a final `done` event.

```bash
curl -N -X POST https://your-deployment.vercel.app/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "transcript": "I go to the store yesterday to buy some milk.",
    "history": [],
    "cefr_level": "B1",
    "role_play_mode": "restaurant"
  }'
```

Example event stream:
```
event: feedback
data: {"grammar": {...}, "pronunciation": {...}, "vocabulary": {...}, "cefr_level": "B1", "cefr_confidence": 0.72}

event: token
data: {"text": "Welcome"}

event: token
data: {"text": " in!"}

event: done
data: {"session_id": "abc123", "cefr_level": "B1", "reply": "Welcome in! What can I get you today?"}
```

### `GET /session/{id}`, `POST /session`, `GET /session/{id}/dashboard`

These are placeholder endpoints -- see `app/routers/session.py` docstring.
Vercel serverless functions have no persistent local storage, so real session
history, word bank, and progress metrics should be kept in the Android
client's local Room database, with these endpoints becoming real
database-backed sync points in a future iteration (e.g. Postgres or
Firestore).

```bash
curl https://your-deployment.vercel.app/session/abc123
curl -X POST https://your-deployment.vercel.app/session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "cefr_level": "B1"}'
curl https://your-deployment.vercel.app/session/abc123/dashboard
```

## Local development

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env and set CEREBRAS_API_KEY=csk-...

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000

# 5. Smoke test
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "dev", "transcript": "Hello, how are you?"}'
```

## Deploying to Vercel

### Path 1: Vercel CLI

```bash
npm install -g vercel   # if not already installed
cd vocalume-backend
vercel                  # first deploy, creates the project
vercel env add CEREBRAS_API_KEY
# paste your csk-... key when prompted, select all environments
vercel --prod
```

### Path 2: GitHub + Vercel dashboard

1. Push this project to a GitHub repository.
2. Go to https://vercel.com/new and import the repository.
3. In **Environment Variables**, add `CEREBRAS_API_KEY` (and optionally
   `CEREBRAS_MODEL`, `ALLOWED_ORIGINS`, etc. from `.env.example`).
4. Click **Deploy**.

### About `vercel.json`

- This project uses the modern `functions` + `rewrites` config rather than
  the legacy `builds`/`routes` config -- Vercel does not allow `builds` and
  `functions` to be specified in the same `vercel.json` (you'll get a
  "The `functions` property cannot be used in conjunction with the `builds`
  property" error if you mix them). `rewrites` sends all incoming paths to
  the `api/index.py` serverless function, and `functions` configures that
  function's `maxDuration` and `includeFiles`.
- `includeFiles: "**/*.py"` is used deliberately instead of `"app/**"` --
  the latter glob pattern has been unreliable in practice for bundling
  nested Python packages with `@vercel/python`, sometimes silently dropping
  files. `**/*.py` reliably includes every Python module in the project.
- There is no `env` block referencing `@secrets` (e.g.
  `"CEREBRAS_API_KEY": "@cerebras-api-key"`). Vercel Secrets are a separate,
  legacy mechanism from regular project environment variables, and
  referencing a `@secret` that hasn't been separately created causes a
  `Secret does not exist` deployment error. Just set plain environment
  variables via the dashboard or `vercel env add` as shown above.

### Vercel limits to be aware of

| Limit                          | Value                                   |
|---------------------------------|------------------------------------------|
| Max function duration           | 60s (Hobby plan) / 300s (Pro plan)       |
| Max unzipped deployment size    | 250 MB                                   |
| Cold start                      | ~1-3s typical                            |
| WebSocket support                | Not supported (use SSE instead)          |
| Filesystem                      | Read-only, except `/tmp` (ephemeral)     |

This is why the backend uses SSE (`sse-starlette`) instead of WebSocket for
streaming, and why session persistence is deferred to the Android client's
local Room database rather than the serverless filesystem.

## ASR/TTS: what Cerebras does and doesn't do

Cerebras Inference is a **text-only** LLM API -- it has no speech
recognition (ASR) or speech synthesis (TTS) service, unlike some other
platforms that bundle Riva-style speech services alongside their LLMs.

For VocaLume:

- **ASR** (speech-to-text) should happen **on-device** in the Android app
  using `android.speech.SpeechRecognizer`. Send the resulting transcript
  text to this backend's `/chat` or `/chat/stream` endpoints.
- **TTS** (text-to-speech) should also happen **on-device**, using
  `android.speech.tts.TextToSpeech`, to speak the `reply` field back to the
  learner.
- **Pronunciation scoring** in this backend (`app/tools/pronunciation.py`) is
  a transparent placeholder heuristic based on word length/vowel count when
  no real ASR confidence is available. If your Android client's
  `SpeechRecognizer` exposes per-word confidence scores, pass them as
  `user_profile.confidence_map` (a `{word: 0.0-1.0}` dict) in the
  `ChatRequest`, and the scorer will use those real confidences instead of
  the placeholder heuristic.

## Android integration (SSE via OkHttp)

A minimal example of consuming `/chat/stream` from Kotlin using OkHttp's
`EventSource` support (`com.squareup.okhttp3:okhttp-sse`):

```kotlin
val client = OkHttpClient()
val requestBody = """
    {"session_id":"abc123","transcript":"$transcript","cefr_level":"B1","role_play_mode":"free_talk"}
""".trimIndent().toRequestBody("application/json".toMediaType())

val request = Request.Builder()
    .url("https://your-deployment.vercel.app/chat/stream")
    .post(requestBody)
    .build()

val listener = object : EventSourceListener() {
    override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
        when (type) {
            "feedback" -> renderFeedbackPanel(data)   // parse JSON -> FeedbackEvent
            "token" -> appendReplyToken(data)          // parse {"text": "..."}
            "done" -> finalizeTurn(data)
            "error" -> showErrorToast(data)
        }
    }
}

EventSources.createFactory(client).newEventSource(request, listener)
```

## Quality notes

- All async: every graph node, tool function, and route handler is `async`.
- Strict JSON contracts: every LLM prompt in `app/tools/` demands JSON-only
  output with no markdown fences or commentary, and every parser tolerates
  fenced/malformed responses via regex extraction as a fallback.
- Fail-safe tools: grammar, vocabulary, and CEFR tools never raise -- on any
  error they return a safe no-op/default result so a single tool failure
  never breaks the whole conversation turn.
- Pydantic v2 throughout: `SettingsConfigDict`, `model_dump()`,
  `model_dump_json()`, no legacy `Config` classes or `.dict()` calls.

## License

MIT License. Use, modify, and deploy freely for VocaLume or any other
project built on this backend.
