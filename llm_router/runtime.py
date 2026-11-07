"""Wiring: build the real pool from config.yaml."""

from __future__ import annotations

from llm_router.config import model_spec
from llm_router.pool import TIERS, GeminiBackend, Pool


def build_pool(cfg: dict, temperature: float | None = None) -> Pool:
    """Real pool. `temperature` applies to both tiers; None keeps the API default (used for the v0 collection)."""
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    specs = {tier: model_spec(cfg, tier) for tier in TIERS}
    backends = {tier: GeminiBackend(client, specs[tier].name, temperature) for tier in TIERS}
    return Pool(backends, specs, timeout_s=float(cfg["gateway"]["request_timeout_s"]))


def build_embedder(model: str = "BAAI/bge-small-en-v1.5"):
    """Sentence embedder used by the learned router. Runs on CPU."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)

    def embed(texts: list[str]):
        return encoder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    return embed
