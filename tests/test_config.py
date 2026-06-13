import os
import pytest
from unittest.mock import patch


def test_config_loads_from_env():
    env = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "CHROMA_PERSIST_DIR": ".chroma",
        "KNOWLEDGE_BASE_DIR": "data/knowledge_base",
    }
    with patch.dict(os.environ, env, clear=True):
        from importlib import reload
        import src.config as cfg
        reload(cfg)
        settings = cfg.get_settings()
        assert settings.deepseek_api_key == "sk-test"
        assert settings.deepseek_model == "deepseek-chat"


def test_config_raises_on_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        from importlib import reload
        import src.config as cfg
        with pytest.raises(Exception):
            reload(cfg)
            cfg.get_settings()
