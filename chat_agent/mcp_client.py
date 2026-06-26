"""MCP Client — 连接 MCP Server，发现工具，执行工具调用。"""
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from .async_utils import run_async


@dataclass
class MCPServerConfig:
    """单个 MCP Server 配置"""
    name: str
    transport: str  # "stdio" | "sse"
    # stdio
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # sse
    url: Optional[str] = None


class MCPClient:
    """管理多个 MCP Server 连接（同步接口）。"""

    def __init__(self, server_configs: List[MCPServerConfig]):
        self._server_configs = server_configs
        self._tools: Dict[str, Dict[str, Any]] = {}  # {server_name: {tool_name: info}}
        self._sessions: Dict[str, ClientSession] = {}
        self._exit_stack: Optional[AsyncExitStack] = None

    def connect(self) -> None:
        """连接所有 MCP Server 并发现工具。"""
        run_async(self._connect_all())

    async def _connect_all(self) -> None:
        """异步连接方法"""
        self._exit_stack = AsyncExitStack()  # 用于管理所有连接的生命周期
        await self._exit_stack.__aenter__()

        for config in self._server_configs:
            try:
                if config.transport == "stdio":
                    server_params = StdioServerParameters(
                        command=config.command,
                        args=config.args or [],
                        env=config.env,
                    )
                    read_stream, write_stream = await self._exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                elif config.transport == "sse":
                    read_stream, write_stream = await self._exit_stack.enter_async_context(
                        sse_client(config.url)
                    )
                else:
                    print(f"[MCP] 未知传输类型: {config.transport}")
                    continue

                # 创建会话并初始化
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                self._sessions[config.name] = session
                # 工具发现：收集每个server暴露的工具，将其转换为openai function calling格式存入self._tools
                tools_result = await session.list_tools()
                server_tools = {}
                for tool in tools_result.tools:
                    server_tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                        "server_name": config.name,
                    }
                self._tools[config.name] = server_tools
                print(f"[MCP] 已连接 '{config.name}'，发现 {len(server_tools)} 个工具")
            except Exception as e:
                print(f"[MCP] 连接 '{config.name}' 失败: {e}")

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """返回所有工具的 OpenAI function calling 格式。"""
        result = []
        for server_name, tools in self._tools.items():
            for tool_name, tool_info in tools.items():
                result.append({
                    "type": "function",
                    "function": {
                        "name": tool_info["name"],
                        "description": tool_info["description"],
                        "parameters": tool_info["input_schema"],
                    }
                })
        return result

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用，返回结果字符串。"""
        for server_name, tools in self._tools.items():
            if tool_name in tools:
                session = self._sessions.get(server_name)
                if session is None:
                    return f"Error: 服务器 '{server_name}' 未连接"
                try:
                    result = run_async(session.call_tool(tool_name, arguments))
                    parts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                except Exception as e:
                    return f"Error: 调用工具 '{tool_name}' 失败: {e}"
        return f"Error: 未知工具 '{tool_name}'"

    def shutdown(self) -> None:
        """关闭所有连接。"""
        if self._exit_stack:
            try:
                run_async(self._exit_stack.aclose())
            except Exception:
                pass
            self._exit_stack = None
        self._sessions.clear()
        self._tools.clear()
