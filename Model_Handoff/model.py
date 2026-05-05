import torch
import torch.nn as nn
from transformers import CLIPModel, ViTModel
from peft import LoraConfig, get_peft_model
import torchvision.models as tv_models

NUM_ORDINAL = 4  # K-1 binary classifiers for 5 bins


# ── Losses ────────────────────────────────────────────────────────────────────
def coral_loss(logits: torch.Tensor, ordinal_labels: torch.Tensor) -> torch.Tensor:
    """Cao et al., CORAL loss (IEEE SPL 2020)."""
    K = logits.shape[1] + 1
    targets = torch.zeros_like(logits)
    for k in range(K - 1):
        targets[:, k] = (ordinal_labels > k).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def mse_loss_from_ordinal(logits: torch.Tensor, true_r: torch.Tensor) -> torch.Tensor:
    """MSE on predicted ratio vs true continuous ratio."""
    pred_r = torch.sigmoid(logits).squeeze(1)  # (B,)
    return nn.functional.mse_loss(pred_r, true_r.float())


def ordinal_to_ratio(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    predicted_rank = (probs > 0.5).sum(dim=1).float()
    bin_centres = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=logits.device)
    return bin_centres[predicted_rank.long()]


# ── Cross-Patch Attention ─────────────────────────────────────────────────────
class CrossPatchAttention(nn.Module):
    """
    Single learnable query attends over patch tokens.
    Manual implementation to avoid MPS/batch_first compatibility issues.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_tokens: torch.Tensor):
        B, N, D = patch_tokens.shape
        H, HD = self.num_heads, self.head_dim

        q = self.W_q(self.query.expand(B, -1)).view(B, 1, H, HD).transpose(1, 2)
        k = self.W_k(patch_tokens).view(B, N, H, HD).transpose(1, 2)
        v = self.W_v(patch_tokens).view(B, N, H, HD).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, D)
        out = self.norm(self.W_o(out))
        return out, attn_weights.mean(dim=1).squeeze(1)  # (B,D), (B,N)


# ── M0: Custom CNN from scratch ───────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    M0 — 5-block CNN built from scratch, no pretrained weights.
    CNNs are preferred over scratch ViT on small datasets due to
    locality/translation-invariance inductive biases (see DeiT, ICCV 2021).
    5 conv blocks → 512-dim feature, parameter count comparable to ResNet-50.
    """
    def __init__(self, num_food_classes: int, dropout: float = 0.4):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.backbone = nn.Sequential(
            conv_block(3,   32),   # 224 → 112
            conv_block(32,  64),   # 112 → 56
            conv_block(64,  128),  # 56  → 28
            conv_block(128, 256),  # 28  → 14
            conv_block(256, 512),  # 14  → 7
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_food_classes))
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)
        return self.cls_head(feat), self.reg_head(feat), None, None

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── M1: ResNet-50 (ImageNet pretrained) ───────────────────────────────────────
class ResNet50Model(nn.Module):
    """M1 — ResNet-50 pretrained on ImageNet, GAP, MSE regression."""
    def __init__(self, num_food_classes: int, dropout: float = 0.4):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_food_classes))
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, 1))

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.cls_head(feat), self.reg_head(feat), None

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── M2 / M3: ViT backbone (ImageNet or CLIP) with LoRA ───────────────────────
class ViTLoRAModel(nn.Module):
    """
    M2: ImageNet ViT-B/16 + LoRA + CLS token + MSE
    M3: CLIP ViT-B/16  + LoRA + CLS token + MSE
    Controlled by `use_clip` flag.
    """
    def __init__(self,
                 num_food_classes: int,
                 use_clip: bool = False,
                 use_cross_patch_attn: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 dropout: float = 0.4,
                 unfreeze_last_n: int = 2):
        super().__init__()
        self.use_cross_patch_attn = use_cross_patch_attn

        # load backbone
        if use_clip:
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.visual = clip.vision_model
        else:
            self.visual = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # freeze all, then unfreeze last N blocks
        for p in self.visual.parameters():
            p.requires_grad = False
        # CLIP uses .layers, HuggingFace ViT uses .layer
        enc = self.visual.encoder
        encoder_layers = enc.layers if hasattr(enc, "layers") else enc.layer
        for layer in encoder_layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # apply LoRA — CLIP uses q_proj/v_proj, HuggingFace ViT uses query/value
        target_modules = ["q_proj", "v_proj"] if use_clip else ["query", "value"]
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        self.visual = get_peft_model(self.visual, lora_cfg)

        embed_dim = 768
        if use_cross_patch_attn:
            self.patch_module = CrossPatchAttention(embed_dim=embed_dim, dropout=dropout)
        else:
            self.patch_module = None

        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_food_classes))
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        self.ord_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, NUM_ORDINAL))

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.visual(pixel_values=pixel_values)
        if self.use_cross_patch_attn:
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            feat, attn_w = self.patch_module(patch_tokens)
        else:
            feat = outputs.last_hidden_state[:, 0, :]  # CLS token
            attn_w = None

        cls_logits = self.cls_head(feat)
        reg_logits = self.reg_head(feat)   # for MSE
        ord_logits = self.ord_head(feat)   # for CORAL
        return cls_logits, reg_logits, ord_logits, attn_w

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── M6: Paired CLIP+LoRA+CrossPatch (before + after) ─────────────────────────
class PairedViTLoRAModel(nn.Module):
    """
    M6 — Shared CLIP ViT encodes before & after separately.
    Concat 392 patch tokens (196 before + 196 after) fed into CrossPatchAttention,
    which learns to attend to regions that changed between the two images.
    No classification head — pure regression on r ∈ [0,1].
    """
    def __init__(self,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 dropout: float = 0.4,
                 unfreeze_last_n: int = 2):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.visual = clip.vision_model

        for p in self.visual.parameters():
            p.requires_grad = False
        enc = self.visual.encoder
        encoder_layers = enc.layers if hasattr(enc, "layers") else enc.layer
        for layer in encoder_layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.visual = get_peft_model(self.visual, lora_cfg)

        embed_dim = 768
        self.patch_module = CrossPatchAttention(embed_dim=embed_dim, dropout=dropout)
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, before_pixels: torch.Tensor, after_pixels: torch.Tensor):
        before_patches = self.visual(pixel_values=before_pixels).last_hidden_state[:, 1:, :]
        after_patches  = self.visual(pixel_values=after_pixels).last_hidden_state[:, 1:, :]
        combined = torch.cat([before_patches, after_patches], dim=1)  # (B, 392, 768)
        feat, attn_w = self.patch_module(combined)
        r = self.reg_head(feat).squeeze(1)  # (B,)
        return r, attn_w

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(model_id: str, num_food_classes: int) -> nn.Module:
    """
    model_id: 'M0' | 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M6'
    """
    if model_id == "M0":
        return CustomCNN(num_food_classes)
    elif model_id == "M1":
        return ResNet50Model(num_food_classes)
    elif model_id == "M2":
        return ViTLoRAModel(num_food_classes, use_clip=False, use_cross_patch_attn=False)
    elif model_id == "M3":
        return ViTLoRAModel(num_food_classes, use_clip=True,  use_cross_patch_attn=False)
    elif model_id in ("M4", "M5"):
        return ViTLoRAModel(num_food_classes, use_clip=True,  use_cross_patch_attn=True)
    elif model_id == "M6":
        return PairedViTLoRAModel()
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
