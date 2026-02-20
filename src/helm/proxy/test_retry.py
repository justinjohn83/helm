import importlib

import pytest
from retrying import RetryError

from helm.common.request import RequestResult
from helm.common.tokenization_request import TokenizationRequestResult
import helm.proxy.retry as retry_module
from helm.proxy.retry import get_retry_decorator, retry_if_request_failed


def _reload_retry_module(
    monkeypatch: pytest.MonkeyPatch,
    helm_retries: int | None = None,
    helm_tokenizer_retries: int | None = None,
):
    if helm_retries is None:
        monkeypatch.delenv("HELM_RETRIES", raising=False)
    else:
        monkeypatch.setenv("HELM_RETRIES", str(helm_retries))

    if helm_tokenizer_retries is None:
        monkeypatch.delenv("HELM_TOKENIZER_RETRIES", raising=False)
    else:
        monkeypatch.setenv("HELM_TOKENIZER_RETRIES", str(helm_tokenizer_retries))

    return importlib.reload(retry_module)


def test_retry_for_successful_request():
    @retry_module.retry_request
    def make_request() -> RequestResult:
        return RequestResult(success=True, completions=[], cached=False, embedding=[])

    result: RequestResult = make_request()
    assert result.success


def test_retry_for_failed_request():
    retry_fail_fast = get_retry_decorator(
        operation="Request",
        max_attempts=1,
        wait_exponential_multiplier_seconds=1,
        retry_on_result=retry_if_request_failed,
    )

    @retry_fail_fast
    def make_request() -> RequestResult:
        return RequestResult(success=False, completions=[], cached=False, embedding=[])

    # Should throw a `RetryError`
    try:
        make_request()
        assert False
    except Exception as e:
        assert isinstance(e, RetryError)
        result: RequestResult = e.last_attempt.value
        assert not result.success


def test_retry_request_uses_helm_retries_env(monkeypatch: pytest.MonkeyPatch):
    module = _reload_retry_module(monkeypatch=monkeypatch, helm_retries=1, helm_tokenizer_retries=None)

    attempts = 0

    @module.retry_request
    def make_request() -> RequestResult:
        nonlocal attempts
        attempts += 1
        return RequestResult(success=False, completions=[], cached=False, embedding=[])

    with pytest.raises(RetryError):
        make_request()

    assert attempts == 1

    _reload_retry_module(monkeypatch=monkeypatch, helm_retries=None, helm_tokenizer_retries=None)


def test_retry_tokenizer_request_uses_helm_tokenizer_retries_env(monkeypatch: pytest.MonkeyPatch):
    module = _reload_retry_module(monkeypatch=monkeypatch, helm_retries=5, helm_tokenizer_retries=1)

    attempts = 0

    @module.retry_tokenizer_request
    def tokenize() -> TokenizationRequestResult:
        nonlocal attempts
        attempts += 1
        return TokenizationRequestResult(success=False, cached=False, text="", tokens=[], error="failed")

    with pytest.raises(RetryError):
        tokenize()

    assert attempts == 1

    _reload_retry_module(monkeypatch=monkeypatch, helm_retries=None, helm_tokenizer_retries=None)
