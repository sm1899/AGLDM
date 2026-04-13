from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
        }
