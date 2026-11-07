"""Config and environment loading. The API key is read from the environment and never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing variables."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def api_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    usd_per_1m_input: float
    usd_per_1m_output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.usd_per_1m_input + output_tokens * self.usd_per_1m_output) / 1_000_000


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config.yaml").read_text())


def model_spec(cfg: dict, tier: str) -> ModelSpec:
    m = cfg["models"][tier]
    missing = [k for k in ("name", "usd_per_1m_input", "usd_per_1m_output") if m.get(k) is None]
    if missing:
        raise ValueError(f"config.yaml models.{tier} is missing {missing}; fill them in first")
    return ModelSpec(m["name"], float(m["usd_per_1m_input"]), float(m["usd_per_1m_output"]))
