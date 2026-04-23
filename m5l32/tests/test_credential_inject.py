"""T13-T16: SecureToolWrapper 单元测试。"""

import os
import pytest

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from security.credential_inject import SecureToolWrapper


class QueryInput(BaseModel):
    query: str = Field(description="查询内容")


class MockApiTool(BaseTool):
    name: str = "mock_api"
    description: str = "模拟 API 调用"
    args_schema: type[BaseModel] = QueryInput

    def _run(self, query: str, api_key: str = "") -> str:
        return f"result for {query} with key={api_key}"


# T13: 密钥注入成功
def test_credential_injected(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test-12345678")
    tool = MockApiTool()
    wrapped = SecureToolWrapper.wrap(tool, {"api_key": "TEST_API_KEY"})
    result = wrapped._run(query="hello")
    assert "sk-test-12345678" in result


# T14: 缺失环境变量报错
def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    tool = MockApiTool()
    with pytest.raises(ValueError, match="not set"):
        SecureToolWrapper.wrap(tool, {"api_key": "MISSING_KEY"})


# T15: 工具描述不含密钥信息
def test_tool_description_unchanged(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-secret")
    tool = MockApiTool()
    original_desc = tool.description
    original_name = tool.name
    wrapped = SecureToolWrapper.wrap(tool, {"api_key": "TEST_API_KEY"})
    assert wrapped.description == original_desc
    assert wrapped.name == original_name
    assert "sk-secret" not in wrapped.description


# T16: credential_status 不暴露值
def test_credential_status_no_value(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-12345678901234567890123456789012")
    status = SecureToolWrapper.get_credential_status({"api_key": "TEST_API_KEY"})
    assert status["api_key"]["is_set"] is True
    assert status["api_key"]["length"] == 35
    assert "value" not in status["api_key"]
    assert "sk-" not in str(status)
