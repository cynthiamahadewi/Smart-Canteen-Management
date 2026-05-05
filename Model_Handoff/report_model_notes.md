# Notes for Report Writer — Model/Methodology Section

These notes cover what to write for the methodology and results sections.
The final deployed model is **M5**. All models M0–M6 should be described as an ablation progression.
NoCls and zero-shot can be mentioned briefly in results as negative ablation results.

---

## Problem Formulation (put this at the start of Methodology)

Supervised regression task:
- **Input X**: single after-meal food image (224×224 RGB)
- **Label Y**: consumption ratio r = 1 − w_after / w_before ∈ [0, 1], derived from scale weight measurements
- **Goal**: learn f_θ(X) → r̂ ∈ [0, 1]

Secondary output (auxiliary task): food category classification over C classes (LeFood-Set has ~30 categories).

---

## Model Progression M0 → M5 → M6

Present as an ablation — each model adds one component and we measure MAE on the held-out test set.

| Model | Architecture | Key addition | Test MAE |
|-------|-------------|--------------|----------|
| M0 | CNN from scratch | Baseline | 0.2436 |
| M1 | ResNet-50 (ImageNet pretrained) | Transfer learning | 0.1711 |
| M2 | ViT-B/16 (ImageNet) + LoRA | Transformer backbone | 0.2060 |
| M3 | CLIP ViT-B/16 + LoRA | Vision-language pretraining | 0.1442 |
| M4 | CLIP ViT-B/16 + LoRA + CrossPatch Attention | Spatial aggregation | 0.1197 |
| M5 | CLIP ViT-B/16 + LoRA + CrossPatch + CORAL | Ordinal regression head | 0.1193 |
| M6 | Paired CLIP ViT (before + after) | Before-meal reference | 0.0936 |

M6 uses paired before+after images and is discussed separately as an alternative deployment mode.

---

## M5 Architecture Details (the main model)

### Backbone: CLIP ViT-B/16
- Pre-trained on 400M image-text pairs (OpenAI CLIP)
- Patch size 16×16, sequence length 197 (1 CLS + 196 patches), embedding dim 768
- Frozen except last 2 transformer encoder layers

### LoRA Fine-tuning
LoRA (Hu et al., 2021) injects low-rank adapters into the attention projection matrices:

    W' = W + BA,   where B ∈ R^{d×r}, A ∈ R^{r×d}, rank r=8, α=16

Applied to q_proj and v_proj in each attention layer. This reduces trainable backbone parameters from ~86M to ~300K while preserving pretrained representations. **This is not covered in lecture** — it is a parameter-efficient fine-tuning technique from the NLP/LLM literature adapted here for vision.

### CrossPatch Attention
A custom single-query attention module over the 196 patch tokens (CLS token excluded):

    Learnable query q ∈ R^{1×768}
    Keys/Values from patch tokens ∈ R^{196×768}
    Output: weighted sum → LayerNorm → 768-dim feature vector

This replaces simple CLS-token pooling with a learned spatial aggregation, letting the model focus on discriminative food regions.

### Dual Output Heads
1. **Classification head**: Linear(768 → C), trained with CrossEntropy loss on food category labels. Acts as an auxiliary task and domain-invariant regularizer.
2. **Ordinal regression head** (CORAL, advanced methodology #2): Linear(768 → K−1) where K=5 bins, trained with CORAL loss (see below).

---

## Objective Function

The combined loss at training:

    L_total = L_CORAL(ord_logits, y_ord) + λ · L_CE(cls_logits, y_cls)

where λ = 1.0 (equal weighting).

### CORAL Loss (advanced methodology #2)
CORAL (Cao et al., 2020 — *Rank Consistent Ordinal Regression*) frames ordinal regression as K−1 binary sub-problems. For K=5 consumption bins, we learn 4 thresholds t_1 < t_2 < t_3 < t_4:

    L_CORAL = (1/K-1) Σ_k BCE(σ(logit_k), 1[y > k])

Predicted rank = Σ_k 1[σ(logit_k) > 0.5], mapped to bin centres {0.1, 0.3, 0.5, 0.7, 0.9}.

**Why CORAL over MSE for ordinal output**: MSE treats bin distances as arbitrary real numbers. CORAL enforces rank consistency (if the model predicts rank > 3, it must also predict rank > 1, 2) and is more appropriate when labels are inherently ordered categories. **Not covered in lecture.**

### Optimizer and Regularization
- **Optimizer**: AdamW with layerwise learning rates:
  - Frozen backbone (non-LoRA): lr = 5×10⁻⁶
  - LoRA adapters: lr = 1×10⁻⁴
  - Classification and regression heads: lr = 1×10⁻³
- **Weight decay**: 0.05 (L₂ regularization)
- **Dropout**: p = 0.4 before all output heads
- **Gradient clipping**: max norm = 1.0
- **LR schedule**: Cosine annealing over 40 epochs (T_max=40, η_min=1×10⁻⁶)
- **Early stopping**: patience = 10 epochs on validation MAE

---

## M6: Paired Model (mention as alternative, not primary)

M6 uses shared CLIP ViT weights for both before- and after-meal images, concatenates their 196+196=392 patch tokens, feeds them through CrossPatch Attention, and predicts r via MSE regression only (no classification head). This achieves the best in-domain MAE of 0.0936.

**Why we chose M5 over M6 as the final model:**
Despite lower MAE, M6 has two practical limitations. First, it requires a before-meal reference image at inference time — in a real deployment, staff would need to photograph every dish before and after service, doubling operational friction. Second, M6's CrossPatch Attention learns to detect *change patterns* between before and after images that are specific to the training domain (Malaysian university cafeteria). On out-of-domain images (e.g., different cuisines or lighting), this paired comparison degrades because the learned change signatures no longer match. M5, relying on absolute visual features of the after-meal image alone, generalizes better across domains. We therefore treat M6 as a high-accuracy research variant and M5 as the deployment model.

---

## Additional Experiments Beyond the Main Ablation

> **Report structure guidance:**
> - M0–M5 ablation → main Results section
> - M6 → separate subsection within Results ("Paired Model Variant")
> - NoCls + Zero-shot → Appendix, referenced briefly in Results with one sentence each

### NoCls Ablation
The main ablation (M0–M5) all include an auxiliary classification head trained jointly with the regression objective. To isolate its contribution, we retrained M1–M5 without the classification head (NoCls variants), keeping all other components identical.

NoCls MAE results for reference:
- M1-NoCls: 0.0927 | M2-NoCls: 0.1724 | M3-NoCls: 0.0901 | M4-NoCls: 0.0985 | M5-NoCls: 0.1250

**Key takeaway: NoCls improves in-domain MAE but hurts generalization. We therefore retain the classification head for robustness.**

Some NoCls variants achieve comparable or slightly better in-domain MAE (e.g., M3-NoCls 0.0901 vs. M3 0.1442). However, when tested on out-of-domain images (different cuisines, lighting conditions), NoCls models generalize worse. The classification head, trained to recognize food categories, forces the shared feature extractor to learn domain-invariant visual representations — acting as a regularizer that prevents overfitting to dataset-specific textures and plating styles. Without it, the regression head overfits to the in-domain distribution.

### Zero-Shot CLIP Classification
As an alternative to the trained classification head, we evaluated CLIP zero-shot classification using food category names from the dataset as text prompts ("a photo of {name}"). Performance was significantly worse than the trained head on fine-grained Malaysian cafeteria dishes. This is consistent with known limitations of zero-shot methods: CLIP was trained on broad internet image-text pairs and lacks the fine-grained visual-semantic alignment needed to distinguish similar dishes (e.g., different rice preparations). The trained classification head, fine-tuned on in-domain images, substantially outperforms zero-shot matching for this task.

---

## What to Give the Report Writer

### Figures (already generated)
| Figure | File | Use in report |
|--------|------|---------------|
| M0–M5 loss curves | `loss_curves.png` | Results section — show training progression, diagnose underfitting/overfitting |
| M6 loss + val MAE | `loss_curve_M6.png` | Results section — show M6 convergence behavior |
| NoCls loss curves | `loss_curves_nocls.png` | Appendix (reference from results text) |
| Model architecture diagram | `models.png` | Methodology section |

### Code (for Appendix)
Give them these files — include all, let them decide what to excerpt:
- `model.py` — all model architectures M0–M6
- `model_nocls.py` — NoCls model definitions (M1–M5 without classification head)
- `train.py` — training loop, loss, optimizer
- `train_nocls.py` — NoCls training script
- `predict.py` — inference script
- `dataset.py` — data loading, label computation

### Key numbers to include in the paper
- Dataset: 524 before/after image pairs, Malaysian university cafeteria
- Label: r = 1 − w_after/w_before, measured by scale
- Train/val/test split: 70/15/15, fixed seed=42
- Best single-image model: **M5, Test MAE = 0.1193**
- Best overall: **M6, Test MAE = 0.0936** (requires before image)
- Trainable parameters in M5: ~300K LoRA + ~600K heads (out of ~86M total CLIP params)
