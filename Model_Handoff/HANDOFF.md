# Model Handoff

## Model

After testing, M5 performs the best, so we use **M5** as the final model. M5 takes a single **after-meal image** as input and predicts the consumption ratio of the dish. No before-meal image is needed for inference.

## What's in this folder

| File | Description |
|------|-------------|
| `ckpt_m5.pt` | Trained M5 model checkpoint |
| `predict.py` | Inference script |
| `model.py` | Model architecture |
| `dataset.py` | Dataset utilities (needed for label encoder) |
| `data_original.xlsx` | Food label list (must stay in the same folder as `dataset.py`) |
| `presets/` | Curated menu images for demo |

### About the preset images

- `menu{N}.jpg` — after-meal images; these are what the model takes as input
- `menu{N}_before.jpg` — before-meal reference images; not used by the model, included so the dashboard can display the full dish for each menu item as a reference for the professor

---

## Setup

```bash
pip install torch torchvision transformers peft scikit-learn pandas openpyxl pillow
```

---

## How to call the model

```python
from predict import load_model, predict

# Load once at startup
model = load_model("M5", "ckpt_m5.pt")

# Run inference on one after-meal image
result = predict(model, "M5", "path/to/after_meal_image.jpg")
```

### Output fields

```python
result["pred_r"]      # Consumption ratio (float): 0.1 / 0.3 / 0.5 / 0.7 / 0.9
result["bin"]         # Ordinal bin (str): "0–20%", "20–40%", "40–60%", "60–80%", "80–100%"
result["top3_food"]   # Top-3 predicted food names (list of (name, prob) tuples)
```

> **Note:** M5 uses CORAL ordinal regression, so `pred_r` is always one of five fixed values (0.1, 0.3, 0.5, 0.7, 0.9) — not a continuous number.

---

## Quick test

After downloading, place the `presets/` folder inside a `test_images/` folder so the path looks like `test_images/presets/`. Then run:

```bash
python predict.py test_images/presets/menu1.jpg --model M5
```
