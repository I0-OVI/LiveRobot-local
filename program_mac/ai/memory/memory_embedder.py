"""
Text embedding module for RAG memory system
Uses sentence-transformers for text vectorization
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Please install: pip install sentence-transformers")


def _hf_hub_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        return os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")
    return os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE


def _resolve_sentence_transformers_repo_id(model_name: str) -> str:
    """Match sentence-transformers hub id (bare name → sentence-transformers/…)."""
    if "/" in model_name:
        return model_name
    return "sentence-transformers/" + model_name


def _snapshot_has_embedder_files(snapshot_dir: str) -> bool:
    """True if a hub snapshot looks usable offline (ST modules.json or plain HF weights)."""
    if not os.path.isdir(snapshot_dir):
        return False
    if os.path.isfile(os.path.join(snapshot_dir, "modules.json")):
        return True
    cfg = os.path.join(snapshot_dir, "config.json")
    if not os.path.isfile(cfg):
        return False
    if os.path.isfile(os.path.join(snapshot_dir, "model.safetensors")) or os.path.isfile(
        os.path.join(snapshot_dir, "pytorch_model.bin")
    ):
        return True
    return os.path.isfile(os.path.join(snapshot_dir, "model.safetensors.index.json"))


def _hub_cache_roots() -> list[str]:
    """HF hub layout may live under HF_HUB_CACHE or SENTENCE_TRANSFORMERS_HOME."""
    roots: list[str] = []
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if st_home:
        roots.append(st_home)
    roots.append(_hf_hub_cache_dir())
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _sentence_transformer_snapshot_path(model_name: str) -> Optional[str]:
    """
    Absolute path to a hub snapshot dir for this model, or None.
    Loading SentenceTransformer(repo_id) still triggers Hub API in some tokenizer paths;
    loading from this path avoids any HTTP (same idea as Qwen local snapshot in text_generator).
    """
    repo_id = _resolve_sentence_transformers_repo_id(model_name)
    slug = "models--" + repo_id.replace("/", "--")
    for hub_root in _hub_cache_roots():
        snapshots = os.path.join(hub_root, slug, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for rev in sorted(os.listdir(snapshots)):
            snap = os.path.join(snapshots, rev)
            if os.path.isdir(snap) and _snapshot_has_embedder_files(snap):
                return os.path.abspath(snap)
    return None


class MemoryEmbedder:
    """Text embedding module using sentence-transformers"""

    # Default model for Chinese-English support
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    # Alternative Chinese-optimized model
    CHINESE_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize text embedder

        Args:
            model_name: Name of the sentence-transformers model
                       If None, uses DEFAULT_MODEL
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        self.model_name = model_name or self.DEFAULT_MODEL
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Memory] Loading embedding model: {self.model_name}")
        try:
            self._load_model_with_cache_policy(self.model_name)
        except Exception as e:
            if self.model_name != self.DEFAULT_MODEL:
                print(f"[Memory] ⚠ Failed to load {self.model_name}, trying fallback {self.DEFAULT_MODEL}...")
                self.model_name = self.DEFAULT_MODEL
                self._load_model_with_cache_policy(self.DEFAULT_MODEL)
            else:
                raise RuntimeError(
                    f"Failed to load embedding model: {e}. "
                    "If the model is not cached yet, run once while online to download, "
                    "or set HF_ENDPOINT (e.g. https://hf-mirror.com) and retry."
                ) from e

        self._cache: dict = {}
        self._use_cache = False

    def _load_model_with_cache_policy(self, name: str) -> None:
        """When hub cache exists, load from snapshot path (no Hub repo id → no model_info HTTP)."""
        snap_path = _sentence_transformer_snapshot_path(name)
        if snap_path is not None:
            print(
                "[Memory] Embedding model found in local Hugging Face cache — "
                "loading from snapshot directory (no Hub API calls)."
            )
            self.model = SentenceTransformer(
                snap_path,
                device=self.device,
                local_files_only=True,
            )
            print("[Memory] ✓ Embedding model loaded from local cache")
            return

        self.model = SentenceTransformer(name, device=self.device)
        print("[Memory] ✓ Embedding model loaded (download / cache update completed)")

    def embed_text(self, text: str, use_cache: bool = False) -> List[float]:
        """
        Convert text to embedding vector

        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.get_embedding_dim()

        # Check cache if enabled
        if use_cache and self._use_cache and text in self._cache:
            return self._cache[text]

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding_list = embedding.tolist()

            # Cache if enabled
            if use_cache and self._use_cache:
                self._cache[text] = embedding_list

            return embedding_list
        except Exception as e:
            print(f"[Memory] Error embedding text: {e}")
            # Return zero vector on error
            return [0.0] * self.get_embedding_dim()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (batch processing)

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            return [[0.0] * self.get_embedding_dim()] * len(texts)

        try:
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Convert to list of lists
            embeddings_list = embeddings.tolist()

            # Pad with zero vectors for empty texts
            result = []
            text_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings_list[text_idx])
                    text_idx += 1
                else:
                    result.append([0.0] * self.get_embedding_dim())

            return result
        except Exception as e:
            print(f"[Memory] Error embedding batch: {e}")
            # Return zero vectors on error
            return [[0.0] * self.get_embedding_dim()] * len(texts)

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embedding vectors

        Returns:
            Dimension of the embedding vector
        """
        # Get dimension by encoding a dummy text
        try:
            dummy_embedding = self.model.encode("test", convert_to_numpy=True)
            return dummy_embedding.shape[0]
        except Exception:
            # Default dimensions for common models
            if "MiniLM" in self.model_name:
                return 384
            elif "bge-small" in self.model_name:
                return 512
            else:
                return 384  # Default fallback

    def enable_cache(self, max_cache_size: int = 1000):
        """
        Enable embedding cache

        Args:
            max_cache_size: Maximum number of cached embeddings
        """
        self._use_cache = True
        self._max_cache_size = max_cache_size

    def disable_cache(self):
        """Disable embedding cache"""
        self._use_cache = False
        self._cache.clear()

    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
