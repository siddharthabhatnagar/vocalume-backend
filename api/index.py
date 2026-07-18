"""
Vercel serverless entry point for the VocaLume backend.

Vercel's @vercel/python builder looks for a module-level ASGI/WSGI
application object in the file referenced by vercel.json's "src" build
entry. We simply re-export the FastAPI app instance built in
`app/main.py` so Vercel can wrap it for serverless invocation.
"""

from app.main import app

__all__ = ["app"]
