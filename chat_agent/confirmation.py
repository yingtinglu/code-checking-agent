"""工具执行前的用户授权网关。

由 REPL 注入 prompt 回调，对高风险工具在执行前询问用户。
会话级"始终允许"规则只存内存，重启失效。
非交互场景不注入 gateway，默认放行（向后兼容）。
"""
from typing import Any, Callable, Dict, Optional, Set, Tuple


# 风险分级：默认需授权的工具及其风险等级
_DEFAULT_REQUIRED_TOOLS: Dict[str, str] = {
    # 高风险：改动文件系统结构或内容
    "write_file": "high",
    "edit_file": "high",
    "delete_file": "high",
    "move_file": "high",
    "create_directory": "high",
    # 中风险：复制或改变授权范围
    "copy_file": "medium",
    "add_allowed_path": "medium",
    "remove_allowed_path": "medium",
}

# 从工具参数中提取"关键参数指纹"的字段优先级
_KEY_FIELDS: Tuple[str, ...] = ("path", "source", "destination", "pattern", "command", "cwd")


def extract_key(tool_name: str, arguments: Dict[str, Any]) -> str:
    """提取工具调用的关键参数指纹，用于会话级记忆去重。

    对文件工具取 path/source/destination 等字段拼成指纹；
    无可识别字段时返回空串（此时会话记忆按工具名整体生效）。
    """
    if not isinstance(arguments, dict):
        return ""
    parts = []
    for field in _KEY_FIELDS:
        if field in arguments and arguments[field]:
            parts.append(f"{field}={arguments[field]}")
    return "|".join(parts)


def summarize_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """生成给用户看的一行调用摘要。"""
    if not isinstance(arguments, dict) or not arguments:
        return tool_name

    items = []
    for field in _KEY_FIELDS:
        if field in arguments and arguments[field]:
            val = str(arguments[field])
            if len(val) > 60:
                val = val[:57] + "..."
            items.append(f"{field}={val}")
    if not items:
        # 退化为 JSON 摘要
        import json
        raw = json.dumps(arguments, ensure_ascii=False)
        if len(raw) > 80:
            raw = raw[:77] + "..."
        return f"{tool_name}({raw})"
    return f"{tool_name}({', '.join(items)})"


class ConfirmationGateway:
    """工具执行前的授权网关。

    用法：
        gateway = ConfirmationGateway(prompt_callback=input)
        decision = gateway.check(tool_name, arguments)
        if decision == "deny":
            return "错误: 用户拒绝执行工具 ..."
    """

    def __init__(
        self,
        prompt_callback: Optional[Callable[[str], str]] = None,
        required_tools: Optional[Dict[str, str]] = None,
    ):
        self._prompt = prompt_callback
        self._required = dict(required_tools) if required_tools else dict(_DEFAULT_REQUIRED_TOOLS)
        # 会话级始终允许：(tool_name, key) 集合
        self._session_allowed: Set[Tuple[str, str]] = set()
        # 本轮取消标志：用户选 c 后本轮剩余工具一律拒绝
        self._round_cancelled = False

    @property
    def interactive(self) -> bool:
        return self._prompt is not None

    def is_required(self, tool_name: str) -> bool:
        return tool_name in self._required

    def reset_round(self) -> None:
        """每轮工具调用开始前重置"本轮取消"标志。"""
        self._round_cancelled = False

    def check(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """返回 'allow' / 'deny'。无 prompt 或工具不在需授权列表时直接放行。

        'cancel_round' 在内部转化为 deny 并置位本轮取消标志。
        """
        if not self.interactive or not self.is_required(tool_name):
            return "allow"

        if self._round_cancelled:
            return "deny"

        key = extract_key(tool_name, arguments)
        if (tool_name, key) in self._session_allowed:
            return "allow"

        summary = summarize_call(tool_name, arguments)
        if tool_name == "add_allowed_path":
            prompt = (
                f"\n请求访问目录权限：{arguments.get('path', '')}\n"
                f"[y=允许 / n=拒绝]: "
            )
        else:
            prompt = (
                f"\n请求权限：{summary}\n"
                f"[y=允许 / n=拒绝]: "
            )

        try:
            choice = self._prompt(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "deny"

        if choice in ("y", "yes", ""):
            return "allow"
        return "deny"

    def deny_message(self, tool_name: str) -> str:
        return f"错误: 用户拒绝执行工具 '{tool_name}'"
