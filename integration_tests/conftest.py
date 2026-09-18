import sys
from pathlib import Path

# This directory is intentionally NOT under tests/, so tests/conftest.py's
# stubbing of torch/transformers/keybert/openai never applies here - these
# tests need the real packages to talk to a real Ollama server.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
