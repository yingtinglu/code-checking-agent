"""LLM 客户端"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

# 导入前清除所有代理环境变量
for _key in list(os.environ.keys()):
    if "proxy" in _key.lower():
        os.environ.pop(_key, None)

import httpx
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError


def _print_rate_limit_headers(error: RateLimitError) -> None:
    """打印限流响应头，帮助诊断是 IP 限流还是 Key 配额耗尽。"""
    if not hasattr(error, 'response') or error.response is None:
        print("  [限流诊断] 无法获取响应头")
        return
    headers = error.response.headers
    diag = {}
    for key, value in headers.items():
        kl = key.lower()
        if 'ratelimit' in kl or 'retry' in kl or kl.startswith('x-'):
            diag[key] = value
    if diag:
        print(f"  [限流诊断] {json.dumps(diag, ensure_ascii=False)}")
    else:
        print(f"  [限流诊断] 无标准限流头, 全部头: {dict(headers)}")


@dataclass
class ChatResponse:
    """LLM 响应"""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


@dataclass
class StreamChunk:
    """流式输出的单个 chunk"""
    delta: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient:
    """大语言模型客户端"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or ""
        self.api_base = api_base or ""
        self.timeout = timeout
        self.max_retries = max_retries

        # 创建禁用代理的 httpx 客户端
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, read=timeout + 60),
            trust_env=False,
        )

        # 初始化 OpenAI 客户端（保留给 chat() 非流式使用）
        init_kwargs: Dict[str, Any] = {
            "http_client": self._http_client,
        }
        if api_key:
            init_kwargs["api_key"] = api_key
        if api_base:
            init_kwargs["base_url"] = api_base

        self.client = OpenAI(**init_kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> ChatResponse:
        """
        发送会话请求（非流式接口，内部委托 chat_stream 流式累积）。

        CCS 网关对部分模型只开放流式路由，非流式请求会返回 404，
        因此统一委托 chat_stream 流式拉取，累积成完整 ChatResponse。

        Args:
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            tools: OpenAI function calling 格式的工具列表
            model: 覆盖 self.model 的模型名，为 None 则用 self.model

        Returns:
            ChatResponse
        """
        content_parts: List[str] = []
        tool_calls_data: Optional[List[Dict[str, Any]]] = None
        finish_reason = "stop"

        for chunk in self.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            model=model,
        ):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.tool_calls:
                tool_calls_data = chunk.tool_calls
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        return ChatResponse(
            content="".join(content_parts),
            tool_calls=tool_calls_data,
            finish_reason=finish_reason,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Generator[StreamChunk, None, None]:
        """流式输出，yield StreamChunk。支持 reasoning_content 和 tool_calls 拼接。含重试和退避。"""
        use_model = model or self.model
        body: Dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None

        for attempt in range(self.max_retries):
            # 流式 tool_calls 拼接状态
            tc_accumulator: Dict[int, Dict[str, Any]] = {}
            has_content = False
            rc_buffer: List[str] = []

            try:
                with self._http_client.stream(
                    "POST", url, json=body, headers=headers,
                ) as response:
                    if response.status_code == 429:
                        raise RateLimitError(
                            message=f"Rate limited (429)",
                            response=httpx.Response(429),
                            body=None,
                        )
                    if response.status_code != 200:
                        response.read()
                        error_text = response.text[:500]
                        raise RuntimeError(f"API 错误 {response.status_code}: {error_text}")

                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            if not has_content and rc_buffer:
                                rc_text = "".join(rc_buffer)
                                if rc_text:
                                    yield StreamChunk(delta=rc_text)
                            if tc_accumulator:
                                yield StreamChunk(
                                    delta="",
                                    finish_reason="tool_calls",
                                    tool_calls=list(tc_accumulator.values()),
                                )
                            return

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        content = delta.get("content", "")
                        rc = delta.get("reasoning_content", "")

                        if content:
                            has_content = True
                            yield StreamChunk(delta=content)

                        if rc:
                            rc_buffer.append(rc)

                        tc_deltas = delta.get("tool_calls")
                        if tc_deltas:
                            for tc_delta in tc_deltas:
                                idx = tc_delta.get("index", 0)
                                if idx not in tc_accumulator:
                                    tc_accumulator[idx] = {
                                        "id": tc_delta.get("id", ""),
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                fn = tc_delta.get("function", {})
                                if fn.get("name"):
                                    tc_accumulator[idx]["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    tc_accumulator[idx]["function"]["arguments"] += fn["arguments"]
                                if tc_delta.get("id"):
                                    tc_accumulator[idx]["id"] = tc_delta["id"]

                        if finish_reason:
                            if not has_content and rc_buffer:
                                rc_text = "".join(rc_buffer)
                                if rc_text:
                                    yield StreamChunk(delta=rc_text)
                            if tc_accumulator:
                                yield StreamChunk(
                                    delta="",
                                    finish_reason=finish_reason,
                                    tool_calls=list(tc_accumulator.values()),
                                )
                            else:
                                yield StreamChunk(delta="", finish_reason=finish_reason)
                    # 流正常结束
                    return

            except RateLimitError as e:
                _print_rate_limit_headers(e)
                wait_time = 2 ** attempt
                print(f"API 限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                last_error = "API 请求频率限制"

            except APITimeoutError:
                last_error = "API 请求超时"
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"请求超时，等待 {wait_time}s 后重试 ({attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)

            except APIError as e:
                last_error = f"API 错误: {e}"
                print(f"API 错误详情: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API 错误，等待 {wait_time}s 后重试 ({attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)

            except httpx.TimeoutException:
                last_error = "HTTP 请求超时"
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"HTTP 超时，等待 {wait_time}s 后重试 ({attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)

            except Exception as e:
                if isinstance(e, RuntimeError) and "API 错误" in str(e):
                    last_error = str(e)
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"等待 {wait_time}s 后重试 ({attempt + 1}/{self.max_retries})...")
                        time.sleep(wait_time)
                else:
                    last_error = f"未知错误: {e}"
                    break

        raise RuntimeError(f"LLM 流式请求失败: {last_error}")

    def chat_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        简化的单轮对话接口

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词

        Returns:
            模型回复
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages).content
