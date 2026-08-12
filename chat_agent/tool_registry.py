"""统一工具注册表 — 管理原生 Python 工具和 MCP 工具，对外提供统一接口"""
import json
import re as _re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logger import agent_log

_current_conv_id = ""


def set_conv_id(conv_id: str) -> None:
    global _current_conv_id
    _current_conv_id = conv_id

Reducer = Callable[[Dict[str, Any], str], str]

if __name__ == "__main__":
    import __main__
    import os
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__main__.__file__)))
else:
    _PROJECT_ROOT = None


# 瞬时错误关键词 — 命中则重试，否则视为永久错误直接返回
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "超时",
    "connection", "connect", "reset", "broken pipe", "EOF", "断开",
    "temporarily", "临时", "unavailable", "不可用",
    "429", "rate limit", "限流", "too many requests",
    "502", "503", "504", "internal server error", "gateway",
    "retry", "再次",
)
# 永久错误关键词 — 显式标记为不可重试（优先级高于瞬时判定）
_PERMANENT_MARKERS = (
    "参数不匹配", "参数解析失败", "未知工具", "未连接",
    "not found", "permission", "权限", "forbidden", "403", "400",
    "illegal", "非法",
)

_TOOL_MAX_RETRIES = 2  # 瞬时错误额外重试次数（总尝试 = 1 + 此值）
_TOOL_RETRY_BASE_WAIT = 0.5  # 退避基数（秒）


def _classify_error(text: str) -> str:
    """返回 'transient' / 'permanent' / 'unknown'。"""
    if not text:
        return "unknown"
    low = text.lower()
    for marker in _PERMANENT_MARKERS:
        if marker.lower() in low:
            return "permanent"
    for marker in _TRANSIENT_MARKERS:
        if marker.lower() in low:
            return "transient"
    return "unknown"


def _is_error_result(result: str) -> bool:
    """ToolRegistry / MCPClient 失败时返回 错误:/Error: 前缀字符串。"""
    if not isinstance(result, str):
        return False
    low = result.lstrip()
    return low.startswith("错误:") or low.startswith("Error:")


class ToolRegistry:
    """统一工具注册表。

    - 原生工具：直接注册 Python 函数，schema 为 OpenAI function-calling 格式
    - MCP 工具：通过 import_mcp() 导入 MCPClient 中的工具
    - 对外统一：get_all_tools() 返回 OpenAI 格式列表，call_tool() 自动分派
    """

    # 合法的归档策略：决定工具结果是否落盘及存储/上下文内容格式
    _ARCHIVE_POLICIES = ("archive", "inline", "no_archive_raw", "archive_ref")

    def __init__(self, confirmation_gateway=None):
        self._native: Dict[str, Tuple[Dict[str, Any], Callable]] = {}
        self._mcp_client = None
        self._confirmation = confirmation_gateway
        self._reducers: Dict[str, Reducer] = {}
        # 工具归档策略：archive(默认,大结果落盘) / inline(不落盘,存投影) / no_archive_raw(不落盘,存原文)
        self._archive_policies: Dict[str, str] = {}

    def set_confirmation_gateway(self, gateway) -> None:
        """注入授权网关（由 ChatAgent 在初始化时调用）。"""
        self._confirmation = gateway

    def register_reducer(self, name: str, reducer: Reducer) -> None:
        """注册工具结果 reducer：(arguments, content) -> 投影字符串。

        历史轮次的 tool result 压缩时调用，保留定位/结构信息而非粗暴截断。
        未注册的工具走 _default_reduce。
        """
        self._reducers[name] = reducer

    def get_reducer(self, name: str) -> Optional[Reducer]:
        return self._reducers.get(name)

    def reduce(self, name: str, arguments: Dict[str, Any], content: str) -> str:
        """压缩 tool result。优先用注册的 reducer，否则走默认投影。"""
        reducer = self._reducers.get(name)
        if reducer:
            try:
                return reducer(arguments or {}, content or "")
            except Exception as e:
                agent_log(_current_conv_id, f"工具 {name} 的 reducer 执行失败: {e}，已回退默认投影", level="ERROR")
                print(f"ERROR: 工具 {name} 执行出错，已使用默认处理")
        return _default_reduce(arguments or {}, content or "")

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 handler: Callable, archive_policy: str = "archive") -> None:
        """注册原生工具（Python 函数）。

        Args:
            name: 工具名称（全局唯一）
            description: 工具描述（LLM 看到的说明）
            parameters: JSON Schema 格式的参数定义（与 OpenAI function-calling 的 parameters 字段一致）
            handler: Python 可调用对象，接收关键字参数，返回 str
            archive_policy: 归档策略，决定工具结果是否落盘及存储/上下文内容格式：
                "archive"(默认) — 大结果落盘，存储带 [已存档] 前缀，上下文放投影
                "inline"        — 不落盘，存储带 [已投影] 前缀，上下文放投影（如 read_file，源在磁盘可重读）
                "no_archive_raw" — 不落盘，存储带 [_原文] 前缀，上下文放原文（结果本身就是原文）
                "archive_ref"   — 不落盘，存储带 [存档引用] 前缀指向 tool_call_id，上下文放原文（如 read_artifact 取回的存档原文）
        """
        if name in self._native:
            raise ValueError(f"工具 '{name}' 已注册")
        if archive_policy not in self._ARCHIVE_POLICIES:
            raise ValueError(
                f"非法归档策略 '{archive_policy}'，合法值: {self._ARCHIVE_POLICIES}"
            )
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._native[name] = (schema, handler)
        self._archive_policies[name] = archive_policy

    def get_archive_policy(self, name: str) -> str:
        """返回工具的归档策略。未注册策略的工具（含 MCP 工具）默认 'archive'。"""
        return self._archive_policies.get(name, "archive")

    def import_mcp(self, mcp_client) -> None:
        """导入 MCP 工具。将 MCPClient 中的工具纳入统一注册表。

        MCP 工具的 schema 和执行仍然由 MCPClient 管理，
        此处只持有引用以便 get_all_tools() 合并和 call_tool() 分派。
        """
        self._mcp_client = mcp_client

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """返回统一的 OpenAI function-calling 格式工具列表。

        原生工具在前，MCP 工具在后，LLM 根据名称选择调用。
        """
        tools = [schema for schema, _ in self._native.values()]
        if self._mcp_client:
            tools.extend(self._mcp_client.get_openai_tools())
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """分派工具执行：先查原生工具，再查 MCP 工具。

        对瞬时错误（超时/连接/限流/5xx）自动重试，永久错误（参数错误/未知工具/权限）
        立即返回。重试只对工具执行层生效，不改变 LLM 决策。

        Args:
            name: 工具名称
            arguments: 工具参数（dict）

        Returns:
            工具执行结果（str）
        """
        # 0. 授权网关：高风险工具执行前询问用户
        if self._confirmation is not None:
            decision = self._confirmation.check(name, arguments)
            if decision != "allow":
                return self._confirmation.deny_message(name)

        # 1. 查原生工具
        if name in self._native:
            _, handler = self._native[name]
            return self._call_with_retry(
                lambda: handler(**arguments),
                on_type_error=lambda e: f"错误: 工具 '{name}' 参数不匹配 — {e}",
                on_error=lambda e: f"错误: 工具 '{name}' 执行失败 — {e}",
            )

        # 2. 查 MCP 工具
        if self._mcp_client:
            return self._call_with_retry(
                lambda: self._mcp_client.call_tool(name, arguments),
            )

        return f"错误: 未知工具 '{name}'"

    def _call_with_retry(
        self,
        invoke: Callable[[], str],
        on_type_error: Optional[Callable[[Exception], str]] = None,
        on_error: Optional[Callable[[Exception], str]] = None,
    ) -> str:
        """带瞬时错误重试的工具执行包装。

        - TypeError 视为参数错误（永久），立即返回 on_type_error
        - 其他异常按消息分类：transient 重试，permanent/unknown 立即返回
        - 成功返回的结果若为 错误:/Error: 前缀字符串，也按同样规则判定是否重试
        """
        last_result: Optional[str] = None
        for attempt in range(1 + _TOOL_MAX_RETRIES):
            try:
                result = invoke()
            except TypeError as e:
                if on_type_error:
                    return on_type_error(e)
                return f"错误: 参数不匹配 — {e}"
            except Exception as e:
                msg = str(e)
                kind = _classify_error(msg)
                if kind == "transient" and attempt < _TOOL_MAX_RETRIES:
                    wait = _TOOL_RETRY_BASE_WAIT * (2 ** attempt)
                    agent_log(_current_conv_id, f"工具瞬时错误，{wait:.1f}s 后重试 ({attempt + 1}/{_TOOL_MAX_RETRIES}): {msg[:120]}", level="ERROR")
                    print(f"ERROR: 工具请求出错，正在重试")
                    time.sleep(wait)
                    continue
                if on_error:
                    return on_error(e)
                return f"错误: 工具执行失败 — {e}"

            # 执行未抛异常，但可能返回错误字符串
            if _is_error_result(result):
                kind = _classify_error(result)
                if kind == "transient" and attempt < _TOOL_MAX_RETRIES:
                    wait = _TOOL_RETRY_BASE_WAIT * (2 ** attempt)
                    agent_log(_current_conv_id, f"工具瞬时错误，{wait:.1f}s 后重试 ({attempt + 1}/{_TOOL_MAX_RETRIES}): {result[:120]}", level="ERROR")
                    print(f"ERROR: 工具请求出错，正在重试")
                    time.sleep(wait)
                    last_result = result
                    continue
                return result

            return result

        return last_result if last_result is not None else "错误: 工具重试后仍失败"

    @property
    def tool_names(self) -> List[str]:
        """返回所有已注册工具名称列表（原生 + MCP）。"""
        names = list(self._native.keys())
        if self._mcp_client:
            for server_name, tools in self._mcp_client._tools.items():
                names.extend(tools.keys())
        return names

    @property
    def native_count(self) -> int:
        return len(self._native)

    @property
    def mcp_count(self) -> int:
        if not self._mcp_client:
            return 0
        total = 0
        for tools in self._mcp_client._tools.values():
            total += len(tools)
        return total


# ── 默认 reducer（未注册工具的声明式兜底）──

_DEFAULT_JSON_BUDGET = 800  # 默认投影目标 token 预算（字符近似）


def _brief_args(arguments: Dict[str, Any]) -> str:
    """精简工具参数用于投影头部。"""
    if not arguments:
        return ""
    parts = []
    for k, v in arguments.items():
        if isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}={v[:60]}...")
        elif isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
        elif v is None:
            continue
        else:
            parts.append(f"{k}=<{type(v).__name__}>")
    return ", ".join(parts)


def _default_reduce(arguments: Dict[str, Any], content: str) -> str:
    """默认投影：JSON 取顶层结构，非 JSON 保留行数 + 首部。

    目标是保留"调了什么、返回了什么形状"，让 LLM 判断是否需要重读/换参数，
    而不是 content[:120] 这种丢光定位信息的截断。
    """
    args_str = _brief_args(arguments)
    header = f"args={args_str}" if args_str else ""

    stripped = (content or "").lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            data = None
        if data is not None:
            body = _summarize_json(data)
            return "\n".join(p for p in [header, body] if p)

    lines = (content or "").split("\n")
    first_line = lines[0] if lines else ""
    tail = "\n".join(lines[1:4]) if len(lines) > 1 else ""
    summary = f"({len(lines)}行) {first_line}"
    if tail:
        summary += "\n" + tail
    return "\n".join(p for p in [header, summary] if p)


def _summarize_json(data: Any, depth: int = 0) -> str:
    """递归摘要 JSON：标量保留，集合报长度，嵌套限深。"""
    if depth > 2:
        return "..."
    if isinstance(data, dict):
        parts = []
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                sv = str(v)
                if len(sv) > 80:
                    sv = sv[:80] + "..."
                parts.append(f"{k}={sv}")
            elif isinstance(v, list):
                parts.append(f"{k}=[{len(v)}项]")
            elif isinstance(v, dict):
                parts.append(f"{k}={{...{len(v)}字段}}")
        return "\n".join(parts)
    if isinstance(data, list):
        return f"[{len(data)}项]"
    return str(data)[:200]
