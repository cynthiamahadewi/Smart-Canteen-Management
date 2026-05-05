import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dataset import get_dataloaders, EXCEL_PATH
from model import build_model, coral_loss, ordinal_to_ratio

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    le.fit(pd.read_excel(EXCEL_PATH)["Name of the food"].dropna())
    return le


def get_cls_labels(food_names, le, device):
    known = set(le.classes_)
    labels = [le.transform([n])[0] if n in known else 0 for n in food_names]
    return torch.tensor(labels, dtype=torch.long).to(device)


# ── forward + loss for any model ──────────────────────────────────────────────
def forward_and_loss(model, batch, le, device, model_id):
    imgs = batch["image"].to(device)
    true_r = batch["r"].float().to(device)
    ordinal_labels = batch["ordinal"].to(device)
    cls_labels = get_cls_labels(batch["food_name"], le, device)

    output = model(imgs)

    if model_id == "M0" or model_id == "M1":
        # output: (cls_logits, reg_logits, None)
        cls_logits, reg_logits, _ = output
        loss_cls = nn.CrossEntropyLoss()(cls_logits, cls_labels)
        loss_reg = nn.MSELoss()(reg_logits.squeeze(1), true_r)
        loss = loss_cls + loss_reg
        pred_r = reg_logits.squeeze(1).detach().clamp(0, 1)

    elif model_id in ("M2", "M3", "M4"):
        # output: (cls_logits, reg_logits, ord_logits, attn_w) — use MSE
        cls_logits, reg_logits, ord_logits, _ = output
        loss_cls = nn.CrossEntropyLoss()(cls_logits, cls_labels)
        loss_reg = nn.MSELoss()(reg_logits.squeeze(1), true_r)
        loss = loss_cls + loss_reg
        pred_r = reg_logits.squeeze(1).detach().clamp(0, 1)

    elif model_id == "M5":  # CORAL loss
        cls_logits, reg_logits, ord_logits, _ = output
        loss_cls = nn.CrossEntropyLoss()(cls_logits, cls_labels)
        loss_ord = coral_loss(ord_logits, ordinal_labels)
        loss = loss_cls + loss_ord
        pred_r = ordinal_to_ratio(ord_logits).detach()

    else:  # M6 — paired regression, no classification
        imgs_before = batch["before"].to(device)
        pred_r_raw, _ = model(imgs_before, imgs)
        loss = nn.MSELoss()(pred_r_raw, true_r)
        pred_r = pred_r_raw.detach().clamp(0, 1)
        # return dummy cls_logits so evaluate() signature stays the same
        cls_logits = torch.zeros(imgs.size(0), 1, device=device)

    return loss, cls_logits, pred_r, true_r


# ── eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, le, device, model_id):
    model.eval()
    total_loss, pred_rs, true_rs, correct, total = 0.0, [], [], 0, 0
    for batch in loader:
        loss, cls_logits, pred_r, true_r = forward_and_loss(
            model, batch, le, device, model_id)
        total_loss += loss.item()
        pred_rs.extend(pred_r.cpu().tolist())
        true_rs.extend(true_r.cpu().tolist())
        cls_labels = get_cls_labels(batch["food_name"], le, device)
        correct += (cls_logits.argmax(1) == cls_labels).sum().item()
        total += len(cls_labels)

    mae = float(np.mean(np.abs(np.array(pred_rs) - np.array(true_rs))))
    acc = correct / total if total > 0 and model_id != "M6" else float("nan")
    return total_loss / len(loader), mae, acc


# ── train ─────────────────────────────────────────────────────────────────────
def train(
    model_id: str,
    num_epochs: int = 30,
    batch_size: int = 16,
    patience: int = 8,
):
    print(f"\n{'='*50}")
    print(f"Training {model_id}  |  device: {DEVICE}")
    print(f"{'='*50}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    le = build_label_encoder()
    num_food_classes = len(le.classes_)

    model = build_model(model_id, num_food_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # layerwise LR for ViT-based models, flat LR for CNN
    if model_id in ("M0", "M1"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    elif model_id == "M6":
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.visual.named_parameters()
                        if "lora" not in n and p.requires_grad], "lr": 5e-6},
            {"params": [p for n, p in model.visual.named_parameters()
                        if "lora" in n],                          "lr": 1e-4},
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("visual")],           "lr": 1e-3},
        ], weight_decay=0.05)
    else:
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.visual.named_parameters()
                        if "lora" not in n and p.requires_grad], "lr": 5e-6},
            {"params": [p for n, p in model.visual.named_parameters()
                        if "lora" in n],                          "lr": 1e-4},
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("visual")],           "lr": 1e-3},
        ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_acc": []}
    best_val_mae = float("inf")
    epochs_no_improve = 0
    ckpt_path = os.path.join(os.path.dirname(__file__), f"ckpt_{model_id}.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, *_ = forward_and_loss(model, batch, le, DEVICE, model_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_loss, val_mae, val_acc = evaluate(model, val_loader, le, DEVICE, model_id)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_MAE={val_mae:.4f} | val_acc={val_acc:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_mae, test_acc = evaluate(model, test_loader, le, DEVICE, model_id)
    print(f"\n[{model_id}] Test MAE={test_mae:.4f} | Test acc={test_acc:.3f}")

    return {"model_id": model_id, "test_mae": test_mae, "test_acc": test_acc,
            "history": history}


# ── run all models ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_ids = sys.argv[1:] if len(sys.argv) > 1 else ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    results = []
    for mid in model_ids:
        r = train(mid)
        results.append(r)

    print("\n" + "="*50)
    print("ABLATION RESULTS")
    print("="*50)
    print(f"{'Model':<6} {'Test MAE':>10} {'Test Acc':>10}")
    for r in results:
        print(f"{r['model_id']:<6} {r['test_mae']:>10.4f} {r['test_acc']:>10.3f}")
