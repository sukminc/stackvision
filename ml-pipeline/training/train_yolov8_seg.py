
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml-pipeline/datasets/pokerplace_v1/chips.yaml", help="YOLO dataset yaml")
    ap.add_argument("--weights", default="yolov8n-seg.pt", help="base weights (n/s/m/l)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--outdir", default="ml-pipeline/models", help="export dir")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, cos_lr=True, device=args.device)
    metrics = model.val()

    outdir = Path(args.outdir)
    (outdir / "coreml").mkdir(parents=True, exist_ok=True)
    (outdir / "onnx").mkdir(parents=True, exist_ok=True)

    # Export Core ML
    coreml_path = model.export(format="coreml", nms=True)
    onnx_path = model.export(format="onnx", opset=13)

    print(f"Exported CoreML: {coreml_path}")
    print(f"Exported ONNX:   {onnx_path}")
    print("\nMetrics:", metrics)

if __name__ == "__main__":
    main()
