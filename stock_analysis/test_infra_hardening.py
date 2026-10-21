#!/usr/bin/env python3
"""
Regression tests for infrastructure hardening changes.

Covers:
- app.main._mask_secrets        : sensitive field masking in logged bodies
- app.config Settings           : CORS origin list parsing
- app.model_gateway             : per-call credentials (no global litellm state)
- app.api.evaluation            : kline window selection / date parsing
"""
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from app.main import _mask_secrets
from app.config import get_settings


# ===================================================================
# Log secret masking
# ===================================================================
class TestMaskSecrets:
    def test_masks_password_and_api_key(self):
        body = '{"username": "alice", "password": "secret123", "api_key": "sk-abc123"}'
        masked = _mask_secrets(body)
        assert "secret123" not in masked
        assert "sk-abc123" not in masked
        assert '"alice"' in masked
        assert masked.count("***MASKED***") == 2

    def test_masks_token_case_insensitive(self):
        body = '{"Token": "t-xyz", "Authorization": "Bearer abc"}'
        masked = _mask_secrets(body)
        assert "t-xyz" not in masked
        assert "Bearer abc" not in masked

    def test_leaves_normal_fields_alone(self):
        body = '{"symbol": "600519", "name": "贵州茅台", "price": 1500.5}'
        assert _mask_secrets(body) == body

    def test_handles_multiline_json(self):
        body = '{\n  "username": "bob",\n  "password": "hunter2"\n}'
        masked = _mask_secrets(body)
        assert "hunter2" not in masked
        assert "bob" in masked


# ===================================================================
# CORS configuration
# ===================================================================
class TestCorsConfig:
    def test_default_wildcard_no_credentials_semantics(self):
        settings = get_settings()
        origins = settings.cors_origins_list
        if origins == ["*"]:
            from app.main import app
            cors_mw = None
            for m in app.user_middleware:
                if m.cls.__name__ == "CORSMiddleware":
                    cors_mw = m
                    break
            assert cors_mw is not None

    def test_parsing_comma_separated_origins(self):
        from app.config import Settings
        s = Settings(CORS_ORIGINS="https://a.com, https://b.com ,")
        assert s.cors_origins_list == ["https://a.com", "https://b.com"]

    def test_empty_falls_back_to_wildcard(self):
        from app.config import Settings
        s = Settings(CORS_ORIGINS="")
        assert s.cors_origins_list == ["*"]


# ===================================================================
# Model gateway: credentials must be per-call only
# ===================================================================
class TestGatewayPerCallCredentials:
    def _make_gateway_with_two_configs(self):
        from app.model_gateway import ModelGateway, LLMConfig, LLMProvider
        gw = ModelGateway(encryption_secret="test-secret")
        cfg_a = LLMConfig(id="a", name="A", provider=LLMProvider.OPENAI,
                          api_key="key-a", base_url="https://a.example/v1",
                          model_name="gpt-test")
        cfg_b = LLMConfig(id="b", name="B", provider=LLMProvider.DEEPSEEK,
                          api_key="key-b", base_url="https://b.example/v1",
                          model_name="deepseek-chat")
        gw.add_config(cfg_a)
        gw.add_config(cfg_b)
        return gw, cfg_a, cfg_b

    def test_chat_passes_credentials_via_kwargs_not_globals(self):
        import litellm
        gw, cfg_a, cfg_b = self._make_gateway_with_two_configs()

        captured_kwargs = []

        def fake_acompletion(**kwargs):
            captured_kwargs.append(dict(kwargs))
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "ok"
            resp.usage.total_tokens = 10
            return resp

        litellm.api_key = None
        litellm.api_base = None
        try:
            with patch.object(litellm, "acompletion", side_effect=fake_acompletion):
                r1 = asyncio.run(gw.chat(messages=[{"role": "user", "content": "hi"}], config_id="a"))
                r2 = asyncio.run(gw.chat(messages=[{"role": "user", "content": "hi"}], config_id="b"))
        finally:
            litellm.api_key = None
            litellm.api_base = None

        assert r1.success and r2.success
        # Global state must stay untouched across both calls
        assert litellm.api_key is None
        assert litellm.api_base is None
        # Per-call kwargs carry the right credentials
        assert captured_kwargs[0]["api_key"] == "key-a"
        assert captured_kwargs[0]["api_base"] == "https://a.example/v1"
        assert captured_kwargs[1]["api_key"] == "key-b"
        assert captured_kwargs[1]["model"] == "deepseek/deepseek-chat"

    def test_disabled_model_returns_error(self):
        from app.model_gateway import ModelGateway, LLMConfig
        gw = ModelGateway(encryption_secret="s")
        cfg = LLMConfig(id="x", name="X", is_enabled=False)
        gw.add_config(cfg)
        resp = asyncio.run(gw.chat(messages=[{"role": "user", "content": "hi"}], config_id="x"))
        assert not resp.success
        assert "disabled" in resp.error


# ===================================================================
# Evaluation kline window logic
# ===================================================================
class _FakeKline:
    def __init__(self, date_str: str, price: float):
        self.date = date_str
        self.price = price


class TestPickKlineWindow:
    def _sample_klines(self):
        return [
            _FakeKline("2026-08-03", 10.0),
            _FakeKline("20260804", 11.0),   # compact format must parse too
            _FakeKline("2026-08-05", 12.0),
            _FakeKline("2026-08-06", 9.0),
            _FakeKline("2026-08-07", 10.5),
        ]

    def test_baseline_on_decision_date_target_n_bars_later(self):
        from app.api.evaluation import _pick_kline_window
        baseline, target = _pick_kline_window(
            self._sample_klines(), datetime(2026, 8, 4, 15, 0), 2
        )
        assert baseline.price == 11.0
        assert target.price == 9.0

    def test_insufficient_future_bars_returns_none(self):
        from app.api.evaluation import _pick_kline_window
        baseline, target = _pick_kline_window(
            self._sample_klines(), datetime(2026, 8, 7), 3
        )
        assert baseline is None and target is None

    def test_decision_before_all_bars_returns_none(self):
        from app.api.evaluation import _pick_kline_window
        baseline, target = _pick_kline_window(
            self._sample_klines(), datetime(2026, 7, 1), 1
        )
        assert baseline is None and target is None

    def test_unparseable_dates_filtered_out(self):
        from app.api.evaluation import _pick_kline_window
        kl = [_FakeKline("garbage", 1.0), _FakeKline("2026-08-03", 10.0),
              _FakeKline("2026-08-04", 11.0)]
        baseline, target = _pick_kline_window(kl, datetime(2026, 8, 3), 1)
        assert baseline.price == 10.0
        assert target.price == 11.0
