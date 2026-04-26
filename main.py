"""
Fashion Recommendation PoC — entry point.

Usage:
    uv run python main.py

Prerequisites:
    Build the catalogue index first:
        uv run python -m backend.scripts.build_index
"""
from frontend.ui import _check_index, build_ui

if __name__ == "__main__":
    _check_index()
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
