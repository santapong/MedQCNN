"""
MedQCNN Live Camera Detection

Captures frames from a webcam, runs them through the HybridQCNN model,
and displays predictions overlaid on the live video feed.

Requires a display server (X11/Wayland) — intended for local machines,
not headless servers.

Usage:
    uv run python scripts/camera.py
    uv run python scripts/camera.py --checkpoint checkpoints/best_model.pt
    uv run python scripts/camera.py --camera-id 1 --n-qubits 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed
from medqcnn.utils.logging import console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedQCNN Live Camera Detection",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=DEMO_QUBITS,
        help=f"Number of qubits (default: {DEMO_QUBITS})",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=NUM_ANSATZ_LAYERS,
        help=f"Number of ansatz layers (default: {NUM_ANSATZ_LAYERS})",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help='Class label names (default: "Benign" "Malignant")',
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Run inference every N frames for performance (default: 1)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU even if GPU is available",
    )
    return parser.parse_args()


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[HybridQCNN, list[str]]:
    """Load HybridQCNN model and return (model, labels)."""
    labels = args.labels or ["Benign", "Malignant"]
    n_classes = args.n_classes

    model = HybridQCNN(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        console.print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "labels" in checkpoint and checkpoint["labels"] is not None:
            labels = checkpoint["labels"]
            if n_classes != len(labels):
                n_classes = len(labels)
                model = HybridQCNN(
                    n_qubits=args.n_qubits,
                    n_layers=args.n_layers,
                    n_classes=n_classes,
                    pretrained=True,
                ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print("  [green]Checkpoint loaded successfully[/green]")
    elif args.checkpoint:
        console.print(
            f"  [yellow]Warning:[/yellow] Checkpoint not found: {args.checkpoint}"
        )
        console.print("  Using random weights")
    else:
        console.print("  No checkpoint specified — using random weights")

    model.eval()
    return model, labels


def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    probs: list[float],
    labels: list[str],
    fps: float,
) -> None:
    """Draw prediction overlay on the video frame."""
    h, w = frame.shape[:2]

    # Color based on confidence: green > 0.8, yellow > 0.5, red otherwise
    if confidence > 0.8:
        color = (0, 200, 0)
    elif confidence > 0.5:
        color = (0, 200, 200)
    else:
        color = (0, 0, 200)

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Prediction label + confidence
    text = f"{label}: {confidence:.1%}"
    cv2.putText(
        frame, text, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )

    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
    )
    cv2.putText(
        frame, fps_text, (w - tw - 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Class probabilities
    y_offset = 65
    for i, (lbl, prob) in enumerate(zip(labels, probs)):
        prob_text = f"{lbl}: {prob:.1%}"
        cv2.putText(
            frame, prob_text, (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )
        y_offset += 20
        if i >= 4:
            break


def run_camera(args: argparse.Namespace) -> None:
    """Main camera detection loop."""
    console.rule("[bold cyan]MedQCNN Live Camera Detection[/bold cyan]")

    set_seed()
    device = get_device(prefer_gpu=not args.no_gpu)

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Qubits:       {args.n_qubits}")
    console.print(f"  Ansatz layers: {args.n_layers}")
    console.print(f"  Device:       {device}")
    console.print(f"  Skip frames:  {args.skip_frames}")

    # Load model
    console.print("\n[bold yellow]Loading model...[/bold yellow]")
    model, labels = load_model(args, device)
    console.print(f"  Labels: {labels}")

    # Transform pipeline (matches model_service.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Open camera
    console.print(f"\n[bold yellow]Opening camera {args.camera_id}...[/bold yellow]")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        console.print(
            f"[bold red]Error:[/bold red] Cannot open camera {args.camera_id}"
        )
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    console.print(f"  Resolution: {cam_w}x{cam_h}")
    console.print("  Press [bold]'q'[/bold] to quit\n")

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    last_label = labels[0]
    last_confidence = 0.0
    last_probs = [1.0 / len(labels)] * len(labels)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[yellow]Warning:[/yellow] Failed to read frame")
                break

            # Run inference every N frames
            if frame_count % args.skip_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pil_image = Image.fromarray(gray)
                tensor = transform(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = logits.argmax(dim=1).item()
                    last_confidence = probs[0, pred_idx].item()
                    last_probs = probs[0].cpu().tolist()
                    last_label = (
                        labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
                    )

            frame_count += 1

            # FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = curr_time

            # Draw and display
            draw_overlay(frame, last_label, last_confidence, last_probs, labels, fps)
            cv2.imshow("MedQCNN Live Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    console.rule("[bold green]Camera Session Ended[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    run_camera(args)
