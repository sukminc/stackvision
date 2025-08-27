# Simple evaluation: loads weights and runs model.val()
# Extend later with count MAE & BB error.

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        default="runs/segment/train",
        help="path to last training run dir",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="path to best.pt if not using run dir",
    )
    args = ap.parse_args()

    weights = args.weights or f"{args.run}/weights/best.pt"
    model = YOLO(weights)
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()