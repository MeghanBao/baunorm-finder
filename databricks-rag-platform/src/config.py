"""
Central configuration for the baunorm-finder RAG platform.

Single source of truth for Unity Catalog names, Databricks Foundation Model /
embedding endpoints, Vector Search objects, and retrieval parameters. Every
notebook and module imports from here so a change (e.g. swapping the generation
model, or promoting from `dev` to `prod`) happens in exactly one place.

Zentrale Konfiguration – alle Namen (Unity Catalog, Endpoints, Vector Search)
und Modell-IDs stehen nur hier. Notebooks und Module importieren daraus.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Foundation Model + embedding endpoints (Databricks-native, pay-per-token)
# --------------------------------------------------------------------------
# Generation stays fully inside Databricks (no external Claude/OpenAI). Model is
# read from an env var so `config.py` never has to change to swap it — the DABs
# target injects GENERATION_MODEL, and unit tests can override it too.
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL", "databricks-meta-llama-3-3-70b-instruct"
)

# Embeddings: the corpus is German, so the default is a self-hosted multilingual
# model served on Databricks Model Serving. Set EMBEDDING_MODE=managed to fall
# back to the zero-setup, English-leaning managed endpoint instead.
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "self_hosted")  # self_hosted | managed
SELF_HOSTED_EMBEDDING_ENDPOINT = os.environ.get(
    "SELF_HOSTED_EMBEDDING_ENDPOINT", "baunorm-e5-multilingual"
)
SELF_HOSTED_EMBEDDING_HF_MODEL = "intfloat/multilingual-e5-large"
MANAGED_EMBEDDING_ENDPOINT = "databricks-gte-large-en"
# Both models emit 1024-dim vectors, so the index schema is stable across modes.
EMBEDDING_DIMENSION = 1024


def embedding_endpoint() -> str:
    """Resolve which Model Serving endpoint Vector Search should call for embeddings."""
    return (
        SELF_HOSTED_EMBEDDING_ENDPOINT
        if EMBEDDING_MODE == "self_hosted"
        else MANAGED_EMBEDDING_ENDPOINT
    )


@dataclass(frozen=True)
class Catalog:
    """Unity Catalog layout. `env` (dev/staging/prod) is folded into the catalog
    name so every environment is fully isolated with a single switch."""

    env: str = field(default_factory=lambda: os.environ.get("BAUNORM_ENV", "dev"))
    base_catalog: str = "baunorm"
    curated_schema: str = "curated"
    ml_schema: str = "ml"

    @property
    def catalog(self) -> str:
        # dev/staging use a suffixed catalog; prod keeps the clean name.
        return self.base_catalog if self.env == "prod" else f"{self.base_catalog}_{self.env}"

    def _fq(self, schema: str, obj: str) -> str:
        return f"{self.catalog}.{schema}.{obj}"

    # curated (corpus + retrieval)
    @property
    def normen_seed(self) -> str:
        return self._fq(self.curated_schema, "normen_seed")

    @property
    def normen(self) -> str:
        return self._fq(self.curated_schema, "normen")

    @property
    def normen_chunks(self) -> str:
        return self._fq(self.curated_schema, "normen_chunks")

    @property
    def normen_index(self) -> str:
        return self._fq(self.curated_schema, "normen_index")

    # ml (eval + registered models)
    @property
    def eval_set(self) -> str:
        return self._fq(self.ml_schema, "eval_set")

    @property
    def agent_model(self) -> str:
        return self._fq(self.ml_schema, "baunorm_rag_agent")


# --------------------------------------------------------------------------
# Vector Search + retrieval
# --------------------------------------------------------------------------
VECTOR_SEARCH_ENDPOINT = os.environ.get("VECTOR_SEARCH_ENDPOINT", "baunorm-vs")
# Column that carries the text Vector Search embeds, and the columns returned to
# the agent so it can build citations with provenance.
CHUNK_TEXT_COLUMN = "chunk"
CHUNK_ID_COLUMN = "chunk_id"
RETRIEVE_COLUMNS = ["chunk_id", "norm", "abschnitt", "chunk", "provenance", "wert"]
TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "5"))

# Serving endpoint the deployed agent lands on (also what the Flask app calls).
AGENT_SERVING_ENDPOINT = os.environ.get("AGENT_SERVING_ENDPOINT", "baunorm-rag-agent")


# --------------------------------------------------------------------------
# Provenance vocabulary — shared by synthesis, agent, and UI so the badge
# semantics never drift between layers.
# --------------------------------------------------------------------------
PROVENANCE_VERIFIED = "verified"
PROVENANCE_SYNTHETIC = "synthetic"

# Value placeholder written into synthetic rows instead of a fabricated number.
SYNTHETIC_VALUE_MARKER = "— synthetisch, nicht amtlich verifiziert —"

# German badge labels surfaced in citations and the app UI.
PROVENANCE_LABELS = {
    PROVENANCE_VERIFIED: "✅ verifiziert",
    PROVENANCE_SYNTHETIC: "⚠ synthetisch",
}


def config() -> Catalog:
    """Convenience accessor so callers can do `from src.config import config`."""
    return Catalog()
