import os
import pytest
from unittest.mock import patch, MagicMock


def _mock_settings():
    s = MagicMock()
    s.deepseek_api_key = "sk-test"
    s.deepseek_base_url = "https://api.deepseek.com/v1"
    s.deepseek_model = "deepseek-chat"
    return s


def test_build_llm_returns_chat_model():
    import src.llm.deepseek_client as mod
    with patch("src.llm.deepseek_client.get_settings", return_value=_mock_settings()):
        llm = mod.build_llm()
        assert hasattr(llm, "invoke")


def test_build_llm_uses_correct_model():
    import src.llm.deepseek_client as mod
    with patch("src.llm.deepseek_client.get_settings", return_value=_mock_settings()):
        llm = mod.build_llm()
        assert llm.model_name == "deepseek-chat"
