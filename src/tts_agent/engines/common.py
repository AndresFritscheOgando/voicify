"""Helpers shared by the model-backed engines: device selection and Hugging Face cache lookups."""

from __future__ import annotations

import logging
import os

from tts_agent.errors import ModelLoadError

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Map `auto`/`cuda`/`cpu` to the device to load on. `auto` falls back to CPU with a warning."""
    import torch

    cuda = torch.cuda.is_available()
    if requested == "cuda":
        if not cuda:
            raise ModelLoadError(
                "TTS_DEVICE=cuda but CUDA is not available. Use auto or cpu, "
                "or install a CUDA build of PyTorch."
            )
        return "cuda"
    if requested == "auto" and not cuda:
        logger.warning("cuda_unavailable fallback=cpu")
        return "cpu"
    return "cuda" if requested == "auto" else "cpu"


def hub_file(repo_id: str, filename: str) -> str:
    """Path to a file of a Hugging Face repo: from the local cache when it is there, otherwise
    downloaded once.

    `hf_hub_download` asks huggingface.co whether the file changed on every call unless
    HF_HUB_OFFLINE=1 is set. Resolving cached files ourselves means no network access at all once
    everything has been downloaded.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached
    logger.info("model_download repo=%s file=%s", repo_id, filename)
    return hf_hub_download(repo_id, filename)


def offline_hint() -> str:
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return (
            " (HF_HUB_OFFLINE is set but this file is not cached yet: unset HF_HUB_OFFLINE and"
            " run once with a network connection to download it)"
        )
    return ""
