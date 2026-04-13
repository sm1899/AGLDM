from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPModel, CLIPProcessor

from agldm.utils.images import to_zero_one


class InceptionMetrics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False, transform_input=False)
        self.model.eval()
        self._avgpool: torch.Tensor | None = None
        self.model.avgpool.register_forward_hook(self._capture_avgpool)

    def _capture_avgpool(self, _module, _inputs, output) -> None:
        self._avgpool = torch.flatten(output, 1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = F.interpolate(to_zero_one(images), size=(299, 299), mode="bilinear", align_corners=False)
        logits = self.model(images)
        if self._avgpool is None:
            msg = "Failed to capture Inception features."
            raise RuntimeError(msg)
        return self._avgpool, torch.softmax(logits, dim=-1)


class ClipScorer(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score(self, images: torch.Tensor, texts: list[str], device: torch.device) -> torch.Tensor:
        pil_images = [to_pil_image(to_zero_one(image).cpu()) for image in images]
        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(device)
        outputs = self.model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        return (image_embeds * text_embeds).sum(dim=-1)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_inception_score(probabilities: np.ndarray, eps: float = 1e-16) -> float:
    py = np.mean(probabilities, axis=0, keepdims=True)
    kl = probabilities * (np.log(probabilities + eps) - np.log(py + eps))
    return float(np.exp(np.mean(np.sum(kl, axis=1))))
