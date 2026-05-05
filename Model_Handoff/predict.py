"""
Single-image inference for food waste estimation.

Usage:
    python predict.py path/to/your/image.jpg --model M5
    python predict.py path/to/your/image.jpg --model M6 --before path/to/before.jpg
"""
import sys
import argparse
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from model import build_model
from dataset import EXCEL_PATH

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BINS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
BAR_WIDTH = 40

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(model_id: str, ckpt_path: str) -> torch.nn.Module:
    if model_id == "M6":
        model = build_model("M6", 1).to(DEVICE)
    else:
        le = LabelEncoder()
        le.fit(pd.read_excel(EXCEL_PATH)["Name of the food"].dropna())
        model = build_model(model_id, len(le.classes_)).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, model_id: str, img_path: str, before_path: str = None):
    img = Image.open(img_path).convert("RGB")
    x = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    if model_id == "M6":
        x_before = VAL_TRANSFORM(Image.open(before_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        pred_r = float(model(x_before, x)[0].item())
        bin_idx = min(int(pred_r * 5), 4)
        return {
            "pred_r": pred_r,
            "bin": BINS[bin_idx],
            "method": "paired MSE regression",
            "top3_food": [("N/A (no classifier)", 1.0)],
            "attn_w": None,
            "threshold_str": None,
        }

    cls_logits, reg_logits, ord_logits, attn_w = model(x)

    le = LabelEncoder()
    le.fit(pd.read_excel(EXCEL_PATH)["Name of the food"].dropna())
    top3_idx = cls_logits.squeeze(0).softmax(0).topk(3).indices.cpu().tolist()
    top3_food = list(zip(
        [le.classes_[i] for i in top3_idx],
        cls_logits.squeeze(0).softmax(0).topk(3).values.cpu().tolist(),
    ))

    if model_id == "M5":
        probs = torch.sigmoid(ord_logits.squeeze(0))
        predicted_rank = int((probs > 0.5).sum().item())
        pred_r = [0.1, 0.3, 0.5, 0.7, 0.9][predicted_rank]
        method = "CORAL ordinal"
        threshold_str = "  ".join(
            f"t{k+1}={p:.2f}" for k, p in enumerate(probs.cpu().tolist())
        )
    else:
        pred_r = float(reg_logits.squeeze().clamp(0, 1).item())
        method = "MSE regression"
        threshold_str = None

    bin_idx = min(int(pred_r * 5), 4)

    return {
        "pred_r": pred_r,
        "bin": BINS[bin_idx],
        "method": method,
        "top3_food": top3_food,
        "attn_w": attn_w,
        "threshold_str": threshold_str,
    }


def render(result: dict, img_path: str, model_id: str):
    r = result["pred_r"]
    filled = int(r * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)

    print(f"\n{'─'*52}")
    print(f"  Image   : {os.path.basename(img_path)}")
    print(f"  Model   : {model_id}  ({result['method']})")
    print(f"{'─'*52}")
    print(f"\n  Consumption ratio  :  {r:.3f}  ({r*100:.1f}%)")
    print(f"  Ordinal bin        :  {result['bin']}")
    print(f"\n  [{bar}]")
    print(f"   0%{' '*34}100%")
    print(f"\n  Top-3 food predictions:")
    for food, prob in result["top3_food"]:
        print(f"    {food:<28} {prob*100:5.1f}%")
    if result["threshold_str"]:
        print(f"\n  CORAL thresholds : {result['threshold_str']}")
    print(f"{'─'*52}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the after-meal food image")
    parser.add_argument("--model", default="M5",
                        choices=["M0", "M1", "M2", "M3", "M4", "M5", "M6"])
    parser.add_argument("--before", default=None,
                        help="Path to before-meal image (required for M6)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path (default: ckpt_<MODEL>.pt in script dir)")
    args = parser.parse_args()

    if args.model == "M6" and args.before is None:
        print("M6 requires a before-meal image. Use: --before path/to/before.jpg")
        sys.exit(1)

    ckpt = args.ckpt or os.path.join(os.path.dirname(__file__), f"ckpt_{args.model.lower()}.pt")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading {args.model} from {ckpt} on {DEVICE}...")
    model = load_model(args.model, ckpt)
    result = predict(model, args.model, args.image, before_path=args.before)
    render(result, args.image, args.model)


if __name__ == "__main__":
    main()
