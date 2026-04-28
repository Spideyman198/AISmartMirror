#!/usr/bin/env python3
"""CLI wrapper for guided webcam enrollment."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging
from vision.guided_enrollment import default_config, run_guided_enrollment


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Guided auto-enrollment for face recognition")
    parser.add_argument("--name", "-n", required=True, help="Display name for the user")
    parser.add_argument("--user-id", default=None, help="Optional explicit user id")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Override target samples")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Save CNN crops only (skip embedding compatibility files)",
    )
    args = parser.parse_args()

    cfg = default_config()
    if args.samples:
        cfg.total_samples = max(10, args.samples)
    cfg.save_embeddings = not args.no_embeddings
    result = run_guided_enrollment(
        user_name=args.name,
        user_id=args.user_id,
        config=cfg,
        project_root=project_root,
    )

    if result.cancelled:
        print("Enrollment cancelled by user.")
        return

    print("\nEnrollment complete.")
    print(f"- User: {result.user_name} ({result.user_id})")
    print(f"- CNN samples saved: {result.total_samples} -> {result.saved_cnn_dir}")
    if result.saved_embedding_file:
        print(f"- Embedding profile saved: {result.saved_embedding_file}")
    print(f"- Pose counts: {result.pose_counts}")
    print("\nNote: CNN retraining is required before this user is recognized by the CNN model.")


if __name__ == "__main__":
    main()
