"""Launch the MedQCNN REST API server.

Usage:
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8000 --checkpoint checkpoints/model_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="MedQCNN API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()

    import uvicorn

    from medqcnn.api.server import create_app

    app = create_app(checkpoint_path=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
