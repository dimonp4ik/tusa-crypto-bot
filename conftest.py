"""Pytest process guard: importing main must never start background jobs."""
import os


os.environ["BOT_STARTUP_ENABLED"] = "0"
