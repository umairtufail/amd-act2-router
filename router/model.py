"""MultiTierRouter: DistilBERT text encoder + numeric features -> tier logits.

Small on purpose (~66M params): it must make routing decisions locally, on
CPU, inside the submission container, for zero Fireworks tokens.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from router.features import NUM_CATEGORIES, NUM_NUMERIC_FEATURES

DEFAULT_ENCODER = "distilbert-base-uncased"
CATEGORY_EMBED_DIM = 8


class MultiTierRouter(nn.Module):
    def __init__(self, num_tiers: int = 4, encoder_name: str = DEFAULT_ENCODER,
                 encoder_config=None):
        """encoder_config, when given, builds the encoder architecture without
        downloading pretrained weights — used at inference where the trained
        weights come from our own checkpoint (keeps the container offline)."""
        super().__init__()
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # 768 for DistilBERT
        self.category_embed = nn.Embedding(NUM_CATEGORIES, CATEGORY_EMBED_DIM)
        self.head = nn.Sequential(
            nn.Linear(hidden + NUM_NUMERIC_FEATURES + CATEGORY_EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_tiers),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric: torch.Tensor,
        category_index: torch.Tensor,
    ) -> torch.Tensor:
        # [CLS]-position hidden state as the sentence embedding.
        text_emb = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        cat_emb = self.category_embed(category_index)
        combined = torch.cat([text_emb, numeric, cat_emb], dim=-1)
        return self.head(combined)

    @torch.no_grad()
    def predict_proba(self, **batch) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(**batch), dim=-1)

    @torch.no_grad()
    def predict(self, **batch) -> torch.Tensor:
        return self.predict_proba(**batch).argmax(dim=-1)


def pick_device() -> torch.device:
    """cuda covers ROCm too — PyTorch on AMD GPUs reports as 'cuda'."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
