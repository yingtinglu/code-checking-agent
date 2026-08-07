"""上下文管理 — 有状态累积对象，四层压缩，与存储分离。

存储是 append-only 完整日志；上下文是内存里持活的工作集，每轮增量追加，
累积到阈值触发 L2/L3/L4 原地压缩。L1 工具结果分流在 append 时即时判定。
上下文不落盘，进程重启时从存储一次性重建。
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .logger import agent_log
from .prompts import load_prompt

if TYPE_CHECKING:
    from .client import LLMClient
    from .storage import Message, ToolArtifactStore

# 模型上下文窗口配置（token 为单位）
MODEL_CONTEXT_CONFIG: Dict[str, Dict[str, int]] = {
    "MiniMax-M2.7": {"max_context_tokens": 163840},
    "GLM5.1": {"max_context_tokens": 160000},
    "glm-5.1": {"max_context_tokens": 160000},
    "maas-glm-5.2-aliyun": {"max_context_tokens": 1000000},
    "glm-5.2": {"max_context_tokens": 1000000},
    "default": {"max_context_tokens": 200000},
}

# 四层阈值
_L1_ARTIFACT_THRESHOLD = 4000   # 单条 tool result token，超过则落盘
_L2_TRIGGER = 0.70              # 触发 L2 历史轮截断
_L3_TRIGGER = 0.85              # L2 后仍超则 L3 远古丢弃
_L4_TRIGGER = 0.92              # L3 后仍超则 L4 全量摘要
_COMPRESS_TARGET = 0.65         # 压缩目标水位
_COMPRESS_MAX_RETRIES = 2       # 单层递归压缩次数上限

# L2' 文本压缩预算（user/assistant 文本截断，不动工具结果）
_L2_USER_BUDGET = 100           # user 文本保留首部 token
_L2_ANSWER_BUDGET = 150         # assistant 最终回答保留首+尾 token

# L3' bounded 丢弃：至少保留最近 K 轮原文，不无限丢
_L3_MIN_KEEP_TURNS = 3

# L4' 全量选择性滚动摘要参数
_L4_CHUNK_BUDGET = 30000        # 每块 token 上限（按完整轮次打包）
_L4_SUMMARY_NARRATIVE = 1500    # 滚动摘要叙事部分 token 上限
_L4_SUMMARY_FACTS = 500         # 关键事实部分 token 上限

FULL_SUMMARY_PROMPT = load_prompt(
    "full_summary.txt",
    "将以下完整会话历史压缩为结构化摘要，保留关键结论、数值、决策与工具调用脉络。"
    "输出 [历史摘要]...[/历史摘要] 格式。",
)

# tiktoken 编码器（懒加载，不可用时回退到字符估算）
_encoding = None
_tiktoken_available = None
_CHAR_TO_TOKEN_RATIO = 1.5


def _get_encoding():
    global _encoding, _tiktoken_available
    if _tiktoken_available is True:
        return _encoding
    if _tiktoken_available is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except Exception:
            _tiktoken_available = False
            agent_log("", "[提示] tiktoken 不可用，使用字符估算 token 数（安装/网络问题）")
    return _encoding if _tiktoken_available else None


def _token_count(messages: List[Dict[str, Any]]) -> int:
    """统计消息列表的 token 数。跳过 None 元素。"""
    enc = _get_encoding()
    if enc is not None:
        total = 0
        for m in messages:
            if m is None:
                continue
            total += 4
            c = m.get("content")
            if isinstance(c, str):
                total += len(enc.encode(c))
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        total += len(enc.encode(args))
                    name = func.get("name", "")
                    if isinstance(name, str):
                        total += len(enc.encode(name))
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id", "")
                if isinstance(tc_id, str):
                    total += len(enc.encode(tc_id))
                name = m.get("name", "")
                if isinstance(name, str):
                    total += len(enc.encode(name))
        total += 2
        return total
    total_chars = 0
    for m in messages:
        if m is None:
            continue
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                if isinstance(args, str):
                    total_chars += len(args)
    return int(total_chars * _CHAR_TO_TOKEN_RATIO)


def _str_token_count(text: str) -> int:
    """统计单个字符串的 token 数。"""
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    return int(len(text) / _CHAR_TO_TOKEN_RATIO)


def _str_truncate_by_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """按 token 数截断字符串，保留不超过 max_tokens 个 token。"""
    enc = _get_encoding()
    if enc is not None:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        cut = max_tokens
        while cut > 0:
            try:
                return enc.decode(tokens[:cut]) + suffix
            except Exception:
                cut -= 1
        return suffix
    max_chars = int(max_tokens * _CHAR_TO_TOKEN_RATIO)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def _merge_adjacent_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并相邻同角色消息，避免 user-user 或 assistant-assistant 相邻违反 API 约束。

    - tool 消息不可合并（tool_call_id 必须独立）
    - assistant 带 tool_calls 不可合并（结构不可拆）
    - user/user 或 assistant(纯文本)/assistant(纯文本) 合并 content
    """
    if not messages:
        return messages
    result = [messages[0]]
    for m in messages[1:]:
        prev = result[-1]
        if m.get("role") == "tool" or prev.get("role") == "tool":
            result.append(m)
            continue
        if (m.get("role") == "assistant" and m.get("tool_calls")) or \
           (prev.get("role") == "assistant" and prev.get("tool_calls")):
            result.append(m)
            continue
        if m.get("role") == prev.get("role"):
            prev_content = prev.get("content") or ""
            m_content = m.get("content") or ""
            prev["content"] = prev_content + "\n" + m_content if prev_content else m_content
            continue
        result.append(m)
    return result


# ── L1 存档标记解析 ──
# 三类前缀互斥，决定工具结果在存储 content 中的格式：
#   [已存档]   — archive 策略大结果：原文落盘，content = 前缀+存档路径+[投影]分隔+投影
#   [已投影]   — inline 策略：不落盘，content = 前缀+投影
#   [存档引用] — archive_ref 策略：不落盘，content = 前缀+tool_call_id；重建时按 id 从存档读原文
_ARCHIVED_PREFIX = "[已存档] "
_INLINE_PREFIX = "[已投影] "
_ARCHIVE_REF_PREFIX = "[存档引用] "
_PROJECTION_MARKER = "\n[投影]\n"


def _parse_prefixed_content(content: str, artifact_store=None) -> Optional[str]:
    """从带前缀的 tool result content 中提取实际内容。无前缀返回 None（走 reduce 兜底）。

    - [已存档]   → 取 [投影] 段（投影在写入时已生成，直接用）
    - [已投影]   → 取前缀后投影
    - [存档引用] → 按 path 从存档读原文（archive_ref：上下文放原文，与运行时一致）
    无前缀返回 None，由调用方走 reduce 兜底。
    """
    if not content:
        return None
    if content.startswith(_ARCHIVED_PREFIX):
        # 落盘大结果：取 [投影] 段
        if _PROJECTION_MARKER in content:
            return content.split(_PROJECTION_MARKER, 1)[1]
        return content[len(_ARCHIVED_PREFIX):]
    if content.startswith(_INLINE_PREFIX):
        return content[len(_INLINE_PREFIX):]  # 已是投影，直接用
    if content.startswith(_ARCHIVE_REF_PREFIX):
        # 存档引用：按 path 从存档读原文放回上下文（read_artifact 取回的原文）
        ref_path = content[len(_ARCHIVE_REF_PREFIX):].strip()
        if artifact_store is not None and ref_path:
            raw = artifact_store.read(ref_path)
            if raw is not None:
                return raw
        return None  # 读不到存档（可能被孤儿清理删了或存档丢失），返回 None 走兜底
    return None


class ContextManager:
    """有状态上下文累积对象。

    在 ChatAgent 上持活，跨 chat() 调用保留 _api_messages。
    增量追加 → 阈值触发 L2/L3/L4 原地压缩 → get_api_messages 供 LLM 调用。
    """

    def __init__(self, model: str, llm_client: Optional["LLMClient"] = None,
                 tool_registry=None, artifact_store=None):
        model_lower = model.lower().replace("-", "").replace(".", "")
        matched = None
        for key in MODEL_CONTEXT_CONFIG:
            key_normalized = key.lower().replace("-", "").replace(".", "")
            if key_normalized == model_lower:
                matched = key
                break
        if not matched:
            matched = "default"
        self.config = MODEL_CONTEXT_CONFIG[matched]
        self.max_context_tokens = self.config["max_context_tokens"]

        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._artifacts = artifact_store

        # 工作上下文（投影后的 api dict 列表，不含 system）
        self._api_messages: List[Dict[str, Any]] = []
        # 当前轮在 _api_messages 中的起始下标（保护范围）
        self._cur_turn_start: int = 0
        # system prompt 分离持有
        self._system_prompt: str = ""
        # L4 用：取完整存储历史的回调
        self._history_getter: Optional[Callable[[], List[Any]]] = None
        # 工具归档策略由 tool_registry.get_archive_policy 决定，此处不再硬编码集合

    # ── 装配 ──

    def set_llm_client(self, client: "LLMClient"):
        self._llm_client = client

    def set_tool_registry(self, registry) -> None:
        self._tool_registry = registry

    def set_artifact_store(self, store) -> None:
        self._artifacts = store

    def set_history_getter(self, getter: Callable[[], List[Any]]) -> None:
        self._history_getter = getter

    def set_conv_id(self, conv_id: str) -> None:
        self._conv_id = conv_id

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt or ""

    # ── 加载/重建 ──

    def load_from_history(self, messages: List[Any], system_prompt: str,
                          compress: bool = True) -> None:
        """从存储完整历史一次性重建上下文。会话加载时调用。

        tool 消息取投影（存档的取 [投影] 段，普通的当场 reduce）。
        compress=True 时重建后若超限跑一次 compress；rebuild_from_checkpoint 传
        compress=False（此时 _cur_turn_start=0，压缩必然失效白跑，留给续跑的 build_context）。
        """
        from .storage import Message
        self._api_messages = []
        self._cur_turn_start = 0
        self._system_prompt = system_prompt or ""

        for msg in messages:
            self._append_storage_message(msg)

        # 重建后若已超限，压缩一次（rebuild 传 compress=False 跳过：_cur_turn_start=0 时
        # 压缩必然失效白跑 L4，留给续跑的 build_context 在 _cur_turn_start 正确后压）
        if compress and self._api_messages and self._total_tokens() > self.max_context_tokens * _L2_TRIGGER:
            self._compress_in_place(0)

    def rebuild_from_checkpoint(self, storage_history: List[Any],
                                turn_storage_messages: List[Any],
                                system_prompt: str) -> None:
        """断点续跑时重建 context_manager 到 checkpoint 时刻状态。

        - storage_history：存储全量历史（turn 开始前状态）。
        - turn_storage_messages：本轮 checkpoint 里已累积的消息（storage Message 格式，
          由 agent 把 BaseMessage 转换得来）。tool 消息取投影（复用 _append_storage_message
          的前缀解析，不二次 reduce/归档，幂等）。
        - 重建后 _api_messages = 全量历史 + 本轮消息，信息保真；build_context 会重新
          派生 state["api_messages"] 并按需压缩，所以无需逐字节匹配中间态。

        begin_turn 在遇到本轮第一条 user 消息时调用一次（与 append_user_msg 一致）。
        不在此处压缩（compress=False）：此刻 _cur_turn_start=0，L3/L4 会把全部消息当
        "本轮"保护而空转、白跑 L4；交给续跑的 build_context 在 _cur_turn_start 正确后压缩。
        """
        self.load_from_history(storage_history, system_prompt, compress=False)  # 重置+追加历史，不压缩
        began = False
        for msg in turn_storage_messages:
            if not began and getattr(msg, "role", None) == "user":
                self.begin_turn()  # 标记当前轮起点（保护当前轮不被压缩丢）
                began = True
            self._append_storage_message(msg)

    def _append_storage_message(self, msg: Any) -> None:
        """把一条存储 Message 追加到工作上下文（tool 取投影）。"""
        role = msg.role
        if role == "system":
            # system 由 set_system_prompt 单独持有，不入 _api_messages
            return
        if role == "tool":
            content = msg.content or ""
            prefixed = _parse_prefixed_content(content, self._artifacts)
            if prefixed is not None:
                ctx_content = prefixed  # [已存档]/[已投影]/[存档引用] 均摘取，不再二次 reduce
            else:
                args = self._tc_args_for(msg.tool_call_id)
                ctx_content = self._reduce_tool_result(msg.name or "", content, args)
            self._api_messages.append({
                "role": "tool",
                "content": ctx_content,
                "tool_call_id": msg.tool_call_id or "",
                "name": msg.name or "",
            })
            return
        if role == "assistant" and msg.tool_calls:
            openai_tcs = [tc.to_dict() for tc in msg.tool_calls]
            self._api_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": openai_tcs,
            })
            return
        self._api_messages.append({"role": role, "content": msg.content or ""})

    def _tc_args_for(self, tool_call_id: Optional[str]) -> Dict[str, Any]:
        """从上下文中已有的 assistant tool_calls 找出该 tool_call_id 的参数。"""
        if not tool_call_id:
            return {}
        for m in self._api_messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc.get("id") == tool_call_id:
                        raw = tc.get("function", {}).get("arguments", "")
                        if isinstance(raw, str):
                            try:
                                return json.loads(raw) if raw else {}
                            except (json.JSONDecodeError, ValueError):
                                return {}
                        if isinstance(raw, dict):
                            return raw
        return {}

    # ── 增量追加（graph 节点调用）──

    def begin_turn(self) -> None:
        """标记新一轮开始：当前轮起始下标 = 当前工作上下文长度。"""
        self._cur_turn_start = len(self._api_messages)

    def append_user(self, content: str) -> None:
        self._api_messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str,
                         tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        if tool_calls:
            self._api_messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            })
        else:
            self._api_messages.append({"role": "assistant", "content": content or ""})

    def append_tool_result(self, tool_name: str, tool_call_id: str,
                           result: str, arguments: Dict[str, Any],
                           conv_id: str) -> str:
        """L1 工具结果分流：按归档策略决定是否落盘及存储/上下文内容格式。

        - archive（默认）：大结果落盘，存储 content = [已存档] path + [投影] 投影，上下文=投影；
                          小结果存储 content = 原文(无前缀)，上下文=投影。
        - inline：不落盘，存储 content = [已投影] 投影，上下文=投影（如 read_file）。
        - archive_ref：不落盘，存储 content = [存档引用] tool_call_id，上下文=原文
                      （如 read_artifact：原文已在存档，存储只放引用，LLM 主动取回的原文进当前轮上下文）。

        重建时 _parse_prefixed_content 按前缀摘取，与运行时上下文一致，不再二次 reduce。
        """
        policy = (self._tool_registry.get_archive_policy(tool_name)
                  if self._tool_registry else "archive")
        projection = self._reduce_tool_result(tool_name, result, arguments or {})

        # ── inline：源可廉价重读（如 read_file），不落盘，存投影 ──
        if policy == "inline":
            self._api_messages.append({
                "role": "tool",
                "content": projection,
                "tool_call_id": tool_call_id,
                "name": tool_name,
            })
            return f"{_INLINE_PREFIX}{projection}"

        # ── archive_ref：read_artifact 取回的存档原文，不落盘（已在存档），上下文放原文 ──
        # 存储只放 path 引用，重建时按 path 从存档读原文放回上下文（与运行时一致）
        if policy == "archive_ref":
            self._api_messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call_id,
                "name": tool_name,
            })
            ref_path = (arguments or {}).get("path", "")
            return f"{_ARCHIVE_REF_PREFIX}{ref_path}"

        # ── archive：大结果落盘，小结果原文 ──
        result_tokens = _str_token_count(result)
        ctx_projection = projection  # 进上下文的投影；落盘大结果时附 path 钩子
        if result_tokens > _L1_ARTIFACT_THRESHOLD and self._artifacts is not None:
            try:
                path = self._artifacts.write(conv_id, tool_call_id, result)
                # 大结果原文已落盘，投影尾部加钩子告诉 LLM 可按路径取回完整原文
                ctx_projection = f"{projection}\n[原文可取 read_artifact(path={path})]"
                stored = f"{_ARCHIVED_PREFIX}{path}{_PROJECTION_MARKER}{ctx_projection}"
                agent_log(self._conv_id, f"[L1存档] {tool_name}: {result_tokens} tokens → 存档 {path}")
            except Exception as e:
                agent_log(self._conv_id, f"[L1存档失败] {tool_name}: {e}，保留原文")
                stored = result
        else:
            stored = result
        self._api_messages.append({
            "role": "tool",
            "content": ctx_projection,
            "tool_call_id": tool_call_id,
            "name": tool_name,
        })
        return stored

    def reset(self) -> None:
        """清空上下文（clear_history 时调用）。"""
        self._api_messages = []
        self._cur_turn_start = 0

    # ── 压缩入口 ──

    def should_compress(self, tools_tokens: int = 0, hint: str = "") -> bool:
        """是否需要压缩。按 LLM 实际看到口径（system + _api_messages + hint + tools）判阈值。"""
        return self._total_tokens(hint) + tools_tokens > self.max_context_tokens * _L2_TRIGGER

    def _total_tokens(self, hint: str = "") -> int:
        """LLM 实际看到的消息 token 数：[system] + _api_messages + [hint]（与 get_api_messages 一致）。"""
        msgs: List[Dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._api_messages)
        if hint:
            msgs.append({"role": "system", "content": hint})
        return _token_count(msgs)

    def compress_if_needed(self, tools_tokens: int = 0, hint: str = "") -> bool:
        """阈值触发则原地压缩，返回是否压缩过。hint 参与 token 统计（与 get_api_messages 一致）。"""
        if not self._api_messages:
            return False
        if not self.should_compress(tools_tokens, hint):
            return False
        self._compress_in_place(tools_tokens, hint)
        return True

    def _compress_in_place(self, tools_tokens: int = 0, hint: str = "") -> None:
        """依次 L2→L3→L4 原地压缩，每层后检查是否降到目标。

        统一口径：cur_tokens = _total_tokens(hint) + tools（含 system + hint，
        与 build_context 的 [压缩后] 一致），消除"L2 后比压缩后小"的口径错觉。
        """
        limit = self.max_context_tokens
        cur_tokens = self._total_tokens(hint) + tools_tokens

        # L2 历史轮截断
        if cur_tokens > limit * _L2_TRIGGER:
            self._api_messages = self._l2_truncate_dialogs(self._api_messages)
            cur_tokens = self._total_tokens(hint) + tools_tokens
            agent_log(self._conv_id, f"[L2截断后] {cur_tokens:,} / {limit:,} tokens ({cur_tokens/limit*100:.1f}%)")

        # L3 远古丢弃
        if cur_tokens > limit * _L3_TRIGGER:
            self._api_messages = self._l3_drop_ancient(self._api_messages, tools_tokens)
            cur_tokens = self._total_tokens(hint) + tools_tokens
            agent_log(self._conv_id, f"[L3丢弃后] {cur_tokens:,} / {limit:,} tokens ({cur_tokens/limit*100:.1f}%)")

        # L4 全量摘要
        if cur_tokens > limit * _L4_TRIGGER:
            self._api_messages = self._l4_full_summary(self._api_messages, tools_tokens)
            cur_tokens = self._total_tokens(hint) + tools_tokens
            agent_log(self._conv_id, f"[L4摘要后] {cur_tokens:,} / {limit:,} tokens ({cur_tokens/limit*100:.1f}%)")

        # 合并相邻同角色（压缩可能产生）
        self._api_messages = _merge_adjacent_roles(self._api_messages)

    def get_api_messages(self, failure_hint: str = "") -> List[Dict[str, Any]]:
        """返回送入 LLM 的完整消息列表：[system] + _api_messages + ([system, hint] if hint)。"""
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend(self._api_messages)
        if failure_hint:
            result.append({"role": "system", "content": failure_hint})
        return result

    # ── L2 历史轮截断 ──

    def _l2_truncate_dialogs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """L2'：压缩历史轮的 user 文本 + 最终 assistant 回答文本，不动 tool_calls/tool 结果。

        拓宽自原"纯文本轮"——工具密集轮也压（user 描述 + assistant 结论常是大头），
        但 tool_calls 骨架和 tool 结果投影保留（L1 已保真，重要）。
        每轮压缩 user 首部 + 最终回答首尾，原地替换并标 [历史对话摘要] 防重压。
        """
        result = list(messages)
        protected_start = self._cur_turn_start
        n = min(protected_start, len(result))

        # 按轮次分组：每轮 = 一个 user + 其后到下个 user 前的全部消息
        turns = []
        i = 0
        while i < n:
            m = result[i]
            if m is None or m.get("role") != "user":
                i += 1
                continue
            uc = m.get("content") or ""
            if "[历史对话摘要]" in uc or "[历史摘要]" in uc:  # 已摘要，跳过
                i += 1
                continue
            start = i
            j = i + 1
            while j < n and (result[j] is None or result[j].get("role") != "user"):
                j += 1
            turns.append((start, j))
            i = j

        for start, end in turns:
            um = result[start]
            user_content = um.get("content") or ""
            if user_content in ("开始执行", "继续分析"):  # 退化分析执行指令，不压
                continue
            # 找轮内最后的无 tool_calls assistant（最终回答）
            final_asst_idx = None
            for k in range(start + 1, end):
                mk = result[k]
                if mk is not None and mk.get("role") == "assistant" and not mk.get("tool_calls"):
                    final_asst_idx = k
            # 压缩 user（首部）
            user_brief = _str_truncate_by_tokens(user_content, _L2_USER_BUDGET)
            result[start] = {**um, "content": f"[历史对话摘要]\n问:{user_brief}\n[/历史对话摘要]"}
            # 压缩最终 assistant 回答（首+尾，保结论）
            if final_asst_idx is not None:
                am = result[final_asst_idx]
                asst_brief = self._truncate_head_tail(am.get("content") or "", _L2_ANSWER_BUDGET)
                result[final_asst_idx] = {**am, "content": f"[历史对话摘要]\n答:{asst_brief}\n[/历史对话摘要]"}

        return [m for m in result if m is not None]

    @staticmethod
    def _truncate_head_tail(text: str, budget: int) -> str:
        """截断保留首部 + 尾部各约 budget/2 token（保结论）。"""
        if not text:
            return text
        half = max(1, budget // 2)
        head = _str_truncate_by_tokens(text, half)
        # 尾部按字符近似取后半（_str_truncate_by_tokens 只保首）
        tail_chars = int(half * _CHAR_TO_TOKEN_RATIO)
        tail = text[-tail_chars:] if len(text) > tail_chars else text
        head_chars = int(half * _CHAR_TO_TOKEN_RATIO)
        sep = "\n...\n" if len(text) > (head_chars + tail_chars) else ""
        return f"{head}{sep}{tail}"

    # ── L3 远古丢弃 ──

    def _l3_drop_ancient(self, messages: List[Dict[str, Any]], tools_tokens: int) -> List[Dict[str, Any]]:
        """从最旧 user-turn 起整段丢弃，直到降到目标。保护 system + 当前轮 + pinned。

        第一版 pinned 只从存储 history_getter 取（_api_messages 不带 pinned 标记），
        默认全 False，所以当前实际只保护当前轮。
        """
        limit = self.max_context_tokens
        target = int(limit * _COMPRESS_TARGET)
        result = list(messages)

        # system 块边界
        system_end = 0
        for i, m in enumerate(result):
            if m is not None and m.get("role") == "system":
                system_end = i + 1
            else:
                break

        protected_start = max(system_end, self._cur_turn_start)

        # 按 user-turn 分组（仅历史区）
        turns = []
        i = system_end
        while i < protected_start:
            if result[i] is not None and result[i].get("role") == "user":
                start = i
                j = i + 1
                while j < protected_start and (result[j] is None or result[j].get("role") != "user"):
                    j += 1
                turns.append((start, j))
                i = j
            else:
                i += 1

        # 从最旧开始丢，但保至少 K 轮近期原文（bounded，不无限丢）
        max_droppable = max(0, len(turns) - _L3_MIN_KEEP_TURNS)
        dropped = 0
        for start, end in turns:
            if dropped >= max_droppable:
                break  # 保 K 轮，不无限丢
            if _token_count([m for m in result if m is not None]) + tools_tokens <= target:
                break
            for idx in range(start, end):
                result[idx] = None
            dropped += 1

        result = [m for m in result if m is not None]

        # 清理悬空 tool result
        result = self._drop_dangling_tool_results(result)
        return result

    @staticmethod
    def _drop_dangling_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_tc_ids = set()
        for m in messages:
            if m is None:
                continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    valid_tc_ids.add(tc.get("id", ""))
        return [
            m for m in messages
            if m is None or m.get("role") != "tool" or m.get("tool_call_id") in valid_tc_ids
        ]

    # ── L4 全量摘要 ──

    def _l4_full_summary(self, messages: List[Dict[str, Any]], tools_tokens: int) -> List[Dict[str, Any]]:
        """L4' 全量选择性滚动摘要：基于完整存储历史滚动摘要，替换为 [历史摘要] + 当前轮。

        数据源 = conversation.messages 全量；滚动摘要双输出（叙事+关键事实），
        忽略工具 chatter，结果 compact ≪ L3 工作副本。失败回退 L3。
        """
        if self._llm_client is None or self._history_getter is None:
            return self._l3_drop_ancient(messages, tools_tokens)

        # 取完整存储历史
        try:
            history_msgs = self._history_getter() or []
        except Exception:
            history_msgs = []

        if not history_msgs:
            return messages

        summary = self._rolling_summary(history_msgs)
        if not summary:
            return self._l3_drop_ancient(messages, tools_tokens)

        # 当前轮保留
        cur_turn = messages[self._cur_turn_start:] if self._cur_turn_start < len(messages) else []
        cur_turn = [m for m in cur_turn if m is not None]

        summary_user = {
            "role": "user",
            "content": f"[历史摘要]\n{summary}\n[/历史摘要]",
        }
        return [summary_user] + cur_turn

    def _render_msg_for_summary(self, m: Any) -> str:
        """渲染单条存储 Message 为摘要用文本。"""
        role = m.role
        content = m.content or ""
        if role == "system":
            return ""
        if role == "tool":
            # L4 摘要渲染：存档引用只标存在性不读原文（防爆）；前缀取投影；无前缀截 500
            if content.startswith(_ARCHIVE_REF_PREFIX):
                tc_id = content[len(_ARCHIVE_REF_PREFIX):].strip()
                content = f"[存档引用 tool_call_id={tc_id}]"
            else:
                prefixed = _parse_prefixed_content(content)
                if prefixed is not None:
                    content = prefixed
                else:
                    content = _str_truncate_by_tokens(content, 500)
            return f"[工具结果 {m.name or ''}] {content}"
        if role == "assistant" and m.tool_calls:
            # 工具调用调度：不把工具名当叙事主线；只留回答文本，无则跳过
            return f"assistant: {content}" if content.strip() else ""
        return f"{role}: {content}"

    # ── L4 滚动分块 ──

    def _chunk_messages_by_turn(self, messages: List[Any], budget: int) -> List[List[str]]:
        """按完整轮次分块，每块渲染文本 ≤ budget token。不切开 tool_call↔result 配对。

        返回 List[List[str]]：每块是该块内各消息渲染文本的列表（已渲染，供滚动拼接）。
        """
        # 按轮次分组（user 开新轮，其后的非 user 消息归入该轮）
        turns: List[List[Any]] = []
        cur: List[Any] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user" and cur:
                turns.append(cur)
                cur = []
            cur.append(m)
        if cur:
            turns.append(cur)

        # 按预算打包成块（不切开轮次；单轮超 budget 则单独成块并警告）
        chunks: List[List[str]] = []
        cur_chunk: List[str] = []
        cur_tokens = 0
        for turn in turns:
            turn_texts = [self._render_msg_for_summary(m) for m in turn]
            turn_texts = [t for t in turn_texts if t]  # 去空（如无文本的 tool_call 调度）
            t_tokens = _str_token_count("\n".join(turn_texts))
            if cur_chunk and cur_tokens + t_tokens > budget:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_tokens = 0
            if t_tokens > budget and not cur_chunk:
                # 单轮超 budget：单独成块，警告（不切碎轮次）
                agent_log(self._conv_id, f"[L4分块] 单轮 {t_tokens} token > budget {budget}，单独成块")
            cur_chunk.extend(turn_texts)
            cur_tokens += t_tokens
        if cur_chunk:
            chunks.append(cur_chunk)
        return chunks

    def _rolling_summary(self, history_msgs: List[Any]) -> Optional[str]:
        """全量选择性滚动摘要：按轮次分块，oldest→newest 滚动累积双输出（叙事+关键事实）。

        每块产出 [叙事]+[关键事实]，与上轮合并，远古压缩、近期保真。
        任一块失败 → 返回 None（调用方回退 L3）。
        """
        chunks = self._chunk_messages_by_turn(history_msgs, _L4_CHUNK_BUDGET)
        if not chunks:
            return None
        rolling = ""
        for idx, chunk_texts in enumerate(chunks):
            chunk_text = "\n".join(chunk_texts)
            user_content = ""
            if rolling:
                user_content += f"[已有摘要（更早历史，需进一步压缩）]\n{rolling}\n[/已有摘要]\n\n"
            user_content += f"[本块原文（近期，尽量保留细节）]\n{chunk_text}\n[/本块原文]"
            rolling = self._call_llm_summary_chunk(user_content)
            if rolling is None:
                return None
            agent_log(self._conv_id, f"[L4滚动] 块 {idx + 1}/{len(chunks)} 摘要 {len(rolling)} 字符")
        return rolling

    def _call_llm_summary_chunk(self, user_content: str) -> Optional[str]:
        """单块滚动摘要调用（双输出 prompt）。max_tokens = 叙事+关键事实预算。"""
        if self._llm_client is None:
            return None
        try:
            max_tokens = _L4_SUMMARY_NARRATIVE + _L4_SUMMARY_FACTS
            full_text = ""
            for chunk in self._llm_client.chat_stream(
                messages=[
                    {"role": "system", "content": FULL_SUMMARY_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=max_tokens,
            ):
                if chunk.delta:
                    full_text += chunk.delta
            text = full_text.strip()
            return text if text else None
        except Exception as e:
            agent_log(self._conv_id, f"[L4摘要错误] {e}")
            return None

    # ── tool result 投影 ──

    def _reduce_tool_result(self, tool_name: str, content: str,
                            arguments: Optional[Dict[str, Any]] = None) -> str:
        """委托 tool_registry.reduce。"""
        if self._tool_registry is not None:
            return self._tool_registry.reduce(tool_name, arguments or {}, content)
        from .tool_registry import _default_reduce
        return _default_reduce(arguments or {}, content)
