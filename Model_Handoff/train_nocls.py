import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from dataset import get_dataloaders
from model_nocls import build_model_nocls, coral_loss, ordinal_to_ratio

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── forward + loss ────────────────────────────────────────────────────────────
def forward_and_loss(model, batch, device, model_id):
    imgs = batch["image"].to(device)
    true_r = batch["r"].float().to(device)
    ordinal_labels = batch["ordinal"].to(device)

    output, _ = model(imgs)   # (logits, attn_w)

    if model_id == "M5":
        loss = coral_loss(output, ordinal_labels)
        pred_r = ordinal_to_ratio(output).detach()
    else:
        loss = nn.MSELoss()(output.squeeze(1), true_r)
        pred_r = output.squeeze(1).detach().clamp(0, 1)

    return loss, pred_r, true_r


# ── eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, model_id):
    model.eval()
    total_loss, pred_rs, true_rs = 0.0, [], []
    for batch in loader:
        loss, pred_r, true_r = forward_and_loss(model, batch, device, model_id)
        total_loss += loss.item()
        pred_rs.extend(pred_r.cpu().tolist())
        true_rs.extend(true_r.cpu().tolist())
    mae = float(np.mean(np.abs(np.array(pred_rs) - np.array(true_rs))))
    return total_loss / len(loader), mae


# ── train ─────────────────────────────────────────────────────────────────────
def train(
    model_id: str,
    num_epochs: int = 40,
    batch_size: int = 16,
    patience: int = 12,      # +2 vs original: weaker gradient signal without cls
):
    print(f"\n{'='*50}")
    print(f"Training {model_id}-NoCls  |  device: {DEVICE}")
    print(f"{'='*50}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = build_model_nocls(model_id).to(DEVICE)
    print(f"Trainable parameters: {model.trainable_parameters():,}")

    if model_id == "M1":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
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

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    epochs_no_improve = 0
    ckpt_path = os.path.join(os.path.dirname(__file__), f"ckpt_{model_id}_nocls.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, *_ = forward_and_loss(model, batch, DEVICE, model_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_loss, val_mae = evaluate(model, val_loader, DEVICE, model_id)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train={train_loss:.4f} | val_loss={val_loss:.4f} | val_MAE={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_mae = evaluate(model, test_loader, DEVICE, model_id)
    print(f"\n[{model_id}-NoCls] Test MAE={test_mae:.4f}")

    return {"model_id": f"{model_id}-NoCls", "test_mae": test_mae, "history": history}


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_ids = sys.argv[1:] if len(sys.argv) > 1 else ["M1", "M2", "M3", "M4", "M5"]
    results = []
    for mid in model_ids:
        r = train(mid)
        results.append(r)

    print("\n" + "="*50)
    print("NoCls ABLATION RESULTS")
    print("="*50)
    print(f"{'Model':<12} {'Test MAE':>10}")
    for r in results:
        print(f"{r['model_id']:<12} {r['test_mae']:>10.4f}")
