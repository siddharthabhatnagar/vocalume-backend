"""Vercel serverless entry point.

Vercel's Python runtime auto-detects FastAPI apps: importing ``app`` here is
enough. The ``vercel.json`` config routes every path to this file.
"""
from app.main import app