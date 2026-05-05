import torch
import torch.nn as nn
from transformers import CLIPModel, ViTModel
from peft import LoraConfig, get_peft_model
import torchvision.models as tv_models

NUM_ORDINAL = 4


# ── Losses (unchanged) ────────────────────────────────────────────────────────
def coral_loss(logits: torch.Tensor, ordinal_labels: torch.Tensor) -> torch.Tensor:
    K = logits.shape[1] + 1
    targets = torch.zeros_like(logits)
    for k in range(K - 1):
        targets[:, k] = (ordinal_labels > k).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def ordinal_to_ratio(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    rank = (probs > 0.5).sum(dim=1).long()
    centres = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=logits.device)
    return centres[rank]


# ── Cross-Patch Attention (unchanged) ─────────────────────────────────────────
class CrossPatchAttention(nn.Module):
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
        return self.norm(self.W_o(out)), attn_weights.mean(dim=1).squeeze(1)


# ── M1-NoCls: ResNet-50, regression only ─────────────────────────────────────
class ResNet50NoCls(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        bb = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, 1))

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)
        return self.reg_head(feat), None  # (reg_logits, attn_w)

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── M2-M5-NoCls: ViT/CLIP + LoRA, regression only ────────────────────────────
class ViTLoRANoCls(nn.Module):
    """
    M2-NoCls: ImageNet ViT + LoRA, MSE only
    M3-NoCls: CLIP ViT   + LoRA, MSE only
    M4-NoCls: CLIP ViT   + LoRA + CrossPatch, MSE only
    M5-NoCls: CLIP ViT   + LoRA + CrossPatch, CORAL only
    """
    def __init__(self,
                 use_clip: bool = False,
                 use_cross_patch: bool = False,
                 use_coral: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 dropout: float = 0.4,
                 unfreeze_last_n: int = 2):
        super().__init__()
        self.use_cross_patch = use_cross_patch
        self.use_coral = use_coral

        if use_clip:
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.visual = clip.vision_model
            target_modules = ["q_proj", "v_proj"]
        else:
            self.visual = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            target_modules = ["query", "value"]

        for p in self.visual.parameters():
            p.requires_grad = False
        enc = self.visual.encoder
        encoder_layers = enc.layers if hasattr(enc, "layers") else enc.layer
        for layer in encoder_layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.visual = get_peft_model(self.visual, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=target_modules, lora_dropout=0.05, bias="none",
        ))

        embed_dim = 768
        self.patch_module = CrossPatchAttention(embed_dim, dropout=dropout) if use_cross_patch else None
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        self.ord_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, NUM_ORDINAL)) if use_coral else None

    def forward(self, pixel_values: torch.Tensor):
        out = self.visual(pixel_values=pixel_values)
        if self.use_cross_patch:
            feat, attn_w = self.patch_module(out.last_hidden_state[:, 1:, :])
        else:
            feat, attn_w = out.last_hidden_state[:, 0, :], None

        if self.use_coral:
            return self.ord_head(feat), attn_w   # (ord_logits, attn_w)
        return self.reg_head(feat), attn_w        # (reg_logits, attn_w)

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model_nocls(model_id: str) -> nn.Module:
    """
    model_id: 'M1' | 'M2' | 'M3' | 'M4' | 'M5'
    No classification head in any variant.
    """
    if model_id == "M1":
        return ResNet50NoCls()
    elif model_id == "M2":
        return ViTLoRANoCls(use_clip=False, use_cross_patch=False)
    elif model_id == "M3":
        return ViTLoRANoCls(use_clip=True,  use_cross_patch=False)
    elif model_id == "M4":
        return ViTLoRANoCls(use_clip=True,  use_cross_patch=True, use_coral=False)
    elif model_id == "M5":
        return ViTLoRANoCls(use_clip=True,  use_cross_patch=True, use_coral=True)
    else:
        raise ValueError(f"NoCls only covers M1-M5, got: {model_id}")
