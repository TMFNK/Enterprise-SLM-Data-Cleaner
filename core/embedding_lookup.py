"""
embedding_lookup.py: semantic alias resolution via bge-m3 embeddings.

A soft post-alias normalizer used by convention_spec._lookup() after the
exact-set and deterministic alias checks miss.  It embeds an unknown value
and compares it against an *expanded* label set (canonical codes + every
known alias key for that field).  Matching a readable alias label (e.g.
"nederland") returns the canonical ("NL").  Comparing only against opaque
ISO codes like "DE" does not work: short codes carry no semantics.

If the best cosine similarity is below the field's threshold the value
passes through unchanged, preserving the grounding guarantee (regions /
true unknowns stay as-is).

Design
------
* Single global model (BAAI/bge-m3) loaded lazily; no startup penalty.
* Pre-computes expanded label embeddings once per field type on first resolve().
* Activated by env var USE_EMBEDDINGS=1 (off by default for zero overhead).
* Every failure mode (ImportError, model error, OOM) degrades gracefully to
  "no result" so the deterministic pathway is never blocked.
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger(__name__)

_USE_EMBEDDINGS = os.environ.get("USE_EMBEDDINGS", "0") == "1"


class EmbeddingAliasResolver:
    """Resolve unknown controlled-vocabulary values via embedding similarity."""

    def __init__(
        self,
        field_canonicals: dict[str, set[str]],
        thresholds: dict[str, float] | None = None,
        field_aliases: dict[str, dict[str, str]] | None = None,
    ):
        self._canonicals = field_canonicals
        self._thresholds = thresholds or {}
        # field -> {alias_key_lower: canonical}; used as readable embedding anchors
        self._aliases = field_aliases or {}
        self._model = None
        # field -> {"labels": [...], "vecs": ndarray, "to_canon": {label: canon}}
        self._index: dict[str, dict] = {}
        self._enabled = _USE_EMBEDDINGS
        if self._enabled:
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                logger.warning(
                    "USE_EMBEDDINGS=1 but sentence-transformers is not installed. "
                    "Disabling embedding resolver."
                )
                self._enabled = False

    def _ensure_model(self):
        if self._model is not None or not self._enabled:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("BAAI/bge-m3")

    def _precompute(self, field_type: str):
        if field_type in self._index or not self._enabled:
            return
        self._ensure_model()

        to_canon: dict[str, str] = {}
        for canon in self._canonicals[field_type]:
            to_canon[canon] = canon
        for alias_key, canon in self._aliases.get(field_type, {}).items():
            to_canon[alias_key] = canon

        labels = sorted(to_canon.keys())
        vecs = self._model.encode(labels, normalize_embeddings=True)
        self._index[field_type] = {
            "labels": labels,
            "vecs": vecs,
            "to_canon": to_canon,
        }

    def resolve(self, field_type: str, raw_value: str) -> str | None:
        """Return the best-matching canonical value or *None* (fall through)."""
        if not self._enabled or field_type not in self._canonicals:
            return None
        threshold = self._thresholds.get(field_type, 0.85)
        try:
            self._precompute(field_type)
            import numpy as np

            index = self._index[field_type]
            vec = self._model.encode([raw_value], normalize_embeddings=True)[0]
            sims = index["vecs"] @ vec
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])
            if best_sim < threshold:
                return None
            return index["to_canon"][index["labels"][best_i]]
        except Exception as exc:
            logger.debug("embedding resolve failed for %s=%r: %s", field_type, raw_value, exc)
            return None
