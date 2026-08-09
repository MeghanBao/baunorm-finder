"""Config: environment isolation + embedding endpoint resolution."""
import importlib


def _fresh_config(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.config as config

    return importlib.reload(config)


def test_catalog_suffixed_in_dev_clean_in_prod(monkeypatch):
    cfg_mod = _fresh_config(monkeypatch, BAUNORM_ENV="dev")
    dev = cfg_mod.Catalog()
    assert dev.catalog == "baunorm_dev"
    assert dev.normen == "baunorm_dev.curated.normen"

    cfg_mod = _fresh_config(monkeypatch, BAUNORM_ENV="prod")
    prod = cfg_mod.Catalog()
    assert prod.catalog == "baunorm"
    assert prod.agent_model == "baunorm.ml.baunorm_rag_agent"


def test_embedding_endpoint_switches_on_mode(monkeypatch):
    cfg_mod = _fresh_config(monkeypatch, EMBEDDING_MODE="managed")
    assert cfg_mod.embedding_endpoint() == cfg_mod.MANAGED_EMBEDDING_ENDPOINT

    cfg_mod = _fresh_config(monkeypatch, EMBEDDING_MODE="self_hosted")
    assert cfg_mod.embedding_endpoint() == cfg_mod.SELF_HOSTED_EMBEDDING_ENDPOINT


def test_provenance_labels_present(monkeypatch):
    cfg_mod = _fresh_config(monkeypatch)
    assert "verifiziert" in cfg_mod.PROVENANCE_LABELS[cfg_mod.PROVENANCE_VERIFIED]
    assert "synthetisch" in cfg_mod.PROVENANCE_LABELS[cfg_mod.PROVENANCE_SYNTHETIC]
