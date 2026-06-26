"""LLM 客户端"""
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# 导入前清除所有代理环境变量
for _key in list(os.environ.keys()):
    if "proxy" in _key.lower():
        os.environ.pop(_key, None)

import httpx
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError


@dataclass
class ChatResponse:
    """LLM 响应"""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


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
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 创建禁用代理的 httpx 客户端
        # trust_env=False 禁止读取环境变量中的代理设置
        http_client = httpx.Client(
            timeout=timeout,
            trust_env=False,
        )
        
        # 初始化 OpenAI 客户端
        init_kwargs: Dict[str, Any] = {
            "http_client": http_client,
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
    ) -> ChatResponse:
        """
        发送会话请求

        Args:
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            tools: OpenAI function calling 格式的工具列表

        Returns:
            ChatResponse
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if tools:
                    kwargs["tools"] = tools

                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                tool_calls_data = None
                if message.tool_calls:
                    tool_calls_data = []
                    for tc in message.tool_calls:
                        tool_calls_data.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        })

                return ChatResponse(
                    content=message.content or "",
                    tool_calls=tool_calls_data,
                    finish_reason=choice.finish_reason or "stop",
                )
                
            except RateLimitError:
                # 限流，等后重试
                wait_time = 2 ** attempt
                print(f"API 限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                last_error = "API 请求频率限制"
                
            except APITimeoutError:
                last_error = "API 请求超时"
                if attempt < self.max_retries - 1:
                    print(f"请求超时，正重试 ({attempt + 1}/{self.max_retries})...")
                    
            except APIError as e:
                last_error = f"API 错误: {e}"
                print(f"API 错误详情: {e}")
                if attempt < self.max_retries - 1:
                    print(f"API 错误，正在重试 ({attempt + 1}/{self.max_retries})...")
                    
            except Exception as e:
                last_error = f"未知错误: {e}"
                break
        
        raise RuntimeError(f"LLM 求失败: {last_error}")
    
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
