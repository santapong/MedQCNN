"""Launch the MedQCNN MCP server for AI agent integration.

Usage:
    uv run python scripts/mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medqcnn.mcp.server import run_server

if __name__ == "__main__":
    run_server()
