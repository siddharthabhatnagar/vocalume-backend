"""
Routers package for the VocaLume backend.

Contains the three FastAPI `APIRouter` modules mounted by
`app/main.py`: `health` (liveness/info endpoints), `chat` (the core
conversation endpoints, both non-streaming and SSE-streaming), and
`session` (a placeholder REST contract for session/dashboard data,
since Vercel serverless functions have no persistent local storage --
production deployments should back this with a real database such as
Postgres).
"""
