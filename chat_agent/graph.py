"""LangGraph 对话图 — state["messages"] 只存本轮新增消息，context_manager 持活历史。"""
import copy
import hashlib
import json
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .context import ContextManager, _token_count, _str_token_count, _str_truncate_by_tokens
from .guardrails import sanitize_input, detect_sensitive_info, redact_pii, redact_credentials, filter_harmful_content
from .llm_adapter import _langchain_to_api_messages, _openai_tc_to_langchain
from .logger import agent_log
from .prompts import load_prompt
from .storage import Storage
from .tool_registry import ToolRegistry


# 检索三件套：受去重硬闸门和轮次预算约束，防止循环撑爆上下文
_SEARCH_TOOLS = ("list_files", "grep", "read_file")
# 会撑爆上下文的工具集合（含 bash 统计）：全部参与去重 + 轮次预算，防止 LLM 换工具绕过硬机制
_BUDGET_TOOLS = ("list_files", "grep", "read_file", "bash")
# 检索/统计轮次预算默认值：超过则强制 final_answer 收尾
_DEFAULT_MAX_SEARCH_ITERATIONS = 15

# ReAct 反思 prompt（每轮工具后决定 continue/replan；最终回答时机由 call_llm 决定）
REFLECT_PROMPT = load_prompt(
    "reflect.txt",
    "基于计划与已执行工具结果，输出 DECISION: continue|replan + 理由。",
)
# 计划指令（注入 call_llm 首次调用，要求先出 [计划] 再调工具）
PLAN_INSTRUCTION = (
    "如果这是本轮首次行动（尚无工具结果），先输出 [计划]分步路线图[/计划]"
    "（每步：目标|工具意图|理由，限 500 token），再调用第一个工具。"
    "若问题简单无需工具，直接回答即可（不输出 [计划]）。"
)
# replan 指令（reflect 决定 replan 时注入，要求重出 [计划]）
REPLAN_INSTRUCTION = (
    "上一份计划需要修订。请重新输出 [计划]修订后的分步路线图[/计划]，"
    "说明修订原因，再调用下一个工具。"
)


def _search_arg_hash(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """对受预算约束的工具关键参数生成稳定哈希，用于去重。

    不同工具取不同关键字段；返回 None 表示该工具不参与去重。
    bash 按 command+cwd 去重（同命令不重复跑 powershell）。
    """
    if tool_name not in _BUDGET_TOOLS:
        return None
    if not isinstance(arguments, dict):
        return None
    if tool_name == "list_files":
        key = {"pattern": arguments.get("pattern"), "path": arguments.get("path", ".")}
    elif tool_name == "grep":
        files = arguments.get("files") or []
        files_sorted = sorted(files) if isinstance(files, list) else files
        key = {
            "pattern": arguments.get("pattern"),
            "files": files_sorted,
            "context": arguments.get("context", 0),
        }
    elif tool_name == "bash":
        key = {"command": arguments.get("command"), "cwd": arguments.get("cwd")}
    else:  # read_file
        key = {
            "path": arguments.get("path"),
            "offset": arguments.get("offset", 0),
            "limit": arguments.get("limit", 2000),
        }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return f"{tool_name}:{hashlib.md5(raw.encode("utf-8")).hexdigest()}"


class ChatAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 本轮新增消息（每次 invoke 从空开始）
    system_prompt: str
    conversation_id: str
    title: str
    model: str
    temperature: float
    max_tokens: int
    tool_iteration: int
    max_tool_iterations: int
    tool_calls_result: Optional[List[Dict[str, Any]]]
    api_messages: List[Dict[str, Any]]
    response_text: str
    user_input: str
    canary: str
    tool_failure_ids: List[str]
    tool_failure_history: List[str]
    tool_failure_names: List[str]
    tool_failure_streaks: Dict[str, int]
    cooldown_tools: Dict[str, int]
    history_count: int  # 本轮开始前存储中的消息数（供 save_response 更新索引）
    # 检索收尾硬机制
    search_iteration: int  # 已消耗的检索轮次（每轮调任意检索工具+1，去重跳过的不计）
    max_search_iterations: int  # 检索轮次预算上限，到顶强制 final_answer
    search_called_tools: Dict[str, str]  # arg_hash -> tool_call_id，跨轮持久的去重表
    # 长期记忆
    _add_allowed_path_called: bool  # 本轮 execute_tools 是否调了 add_allowed_path
    # ReAct 反思
    reflect_decision: str  # continue|replan（reflect 节点产出；最终回答时机由 call_llm 决定）
    reflect_reasoning: str  # 反思理由（trace 用）


def append_user_msg(state: ChatAgentState, config) -> dict:
    context_manager: ContextManager = config["configurable"]["context_manager"]
    raw_input = state.get("user_input", "")

    # 输入预处理（控制字符清除、长度限制、SQL 注入拦截）
    cleaned, warning = sanitize_input(raw_input)
    if warning and "拦截" in warning:
        return {
            "messages": [HumanMessage(content="[输入已拦截]")],
            "tool_iteration": 0,
            "tool_calls_result": None,
            "response_text": warning,
            "tool_failure_ids": [],
            "tool_failure_history": [],
            "tool_failure_names": [],
            "tool_failure_streaks": {},
            "cooldown_tools": {},
            "search_iteration": 0,
            "max_search_iterations": _DEFAULT_MAX_SEARCH_ITERATIONS,
            "search_called_tools": {},
            "_add_allowed_path_called": False,
            "reflect_decision": "",
            "reflect_reasoning": "",
        }
    if warning:
        agent_log(state["conversation_id"], f"[输入防护] {warning}")

    # 敏感信息检测（交互式确认）
    sensitive_warning = detect_sensitive_info(cleaned)
    if sensitive_warning:
        agent_log(state["conversation_id"], f"[输入防护] {sensitive_warning}")
        try:
            confirm = input("  确认发送? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "n"
        if confirm not in ("y", "yes"):
            return {
                "messages": [HumanMessage(content="[用户取消发送]")],
                "tool_iteration": 0,
                "tool_calls_result": None,
                "response_text": "用户取消发送包含敏感信息的消息。",
                "tool_failure_ids": [],
                "tool_failure_history": [],
                "tool_failure_names": [],
                "tool_failure_streaks": {},
                "cooldown_tools": {},
                "search_iteration": 0,
                "max_search_iterations": _DEFAULT_MAX_SEARCH_ITERATIONS,
                "search_called_tools": {},
                "_add_allowed_path_called": False,
                "reflect_decision": "",
                "reflect_reasoning": "",
            }

    # 增量追加到上下文
    context_manager.begin_turn()
    context_manager.append_user(cleaned)

    user_msg = HumanMessage(content=cleaned)
    return {
        "messages": [user_msg],
        "tool_iteration": 0,
        "tool_calls_result": None,
        "response_text": "",
        "tool_failure_ids": [],
        "tool_failure_history": [],
        "tool_failure_names": [],
        "tool_failure_streaks": {},
        "cooldown_tools": {},
        "search_iteration": 0,
        "max_search_iterations": _DEFAULT_MAX_SEARCH_ITERATIONS,
        "search_called_tools": {},
        "_add_allowed_path_called": False,
        "reflect_decision": "",
        "reflect_reasoning": "",
    }


def load_memory(state: ChatAgentState, config) -> dict:
    """加载记忆节点：注入指令让 LLM 判断项目后调用 search_memory。"""
    from .tools.filesystem import ALLOWED_ROOTS
    if not ALLOWED_ROOTS:
        return {}
    instruction = (
        "你有 search_memory 工具可用。请先判断当前对话涉及的项目路径，"
        "然后调用 search_memory(project_path, query) 搜索与用户任务相关的记忆，"
        "query 填写用户当前任务或问题的描述。"
    )
    context_manager: ContextManager = config["configurable"]["context_manager"]
    context_manager.append_user(instruction)
    return {}

def push_queue(state: ChatAgentState, config) -> dict:
    """入队节点：save_response 完成后，如果 ALLOWED_ROOTS 有目录，
    深拷贝本轮对话片段连同模型参数一起作为快照放入记忆队列。
    """
    from .tools.filesystem import ALLOWED_ROOTS
    if not ALLOWED_ROOTS:
        return {}
    project_root = ALLOWED_ROOTS[0]

    context_manager: ContextManager = config["configurable"]["context_manager"]
    memory_manager = config["configurable"].get("memory_manager")
    if memory_manager is None:
        return {}

    # 深拷贝本轮对话片段
    turn_start = context_manager._cur_turn_start
    turn_messages = copy.deepcopy(context_manager._api_messages[turn_start:])

    from .memory import MemorySnapshot
    snapshot = MemorySnapshot(
        api_messages=turn_messages,
        project_root=project_root,
        model=state.get("model", ""),
        temperature=state.get("temperature", 0.7),
        max_tokens=state.get("max_tokens", 8192),
        conv_id=state.get("conversation_id", ""),
    )
    memory_manager.enqueue(snapshot)
    agent_log(state["conversation_id"], f"[记忆] 已入队快照 (项目: {project_root})")
    return {}


def _build_failure_hint(state: ChatAgentState) -> str:
    """构建工具失败缺口提示，注入到 LLM 上下文。

    只陈述事实（哪些工具刚失败、哪些在冷却中、还剩几轮解禁），
    不下"禁止调用"的永久禁令——工具恢复后 LLM 可重新尝试。
    """
    cooldown = state.get("cooldown_tools") or {}
    all_failed = state.get("tool_failure_names") or []
    if not all_failed and not cooldown:
        return ""

    lines = ["[工具调用状态提示]"]

    # 本轮新失败的工具（去重保序）
    if all_failed:
        seen = set()
        unique_failed = []
        for n in all_failed:
            if n not in seen:
                seen.add(n)
                unique_failed.append(n)
        lines.append(
            f"以下工具近期调用失败: {', '.join(unique_failed)}。"
            "若需相关数据，可换参数或换工具；也可在冷却结束后重试。"
        )

    # 冷却中的工具（仍可尝试，但提示 LLM 优先走其他路径）
    if cooldown:
        items = [f"{name}(还剩{remain}轮解禁)" for name, remain in cooldown.items() if remain > 0]
        if items:
            lines.append(
                f"以下工具处于冷却中: {', '.join(items)}，本轮不可调用。"
                "冷却结束后会自动恢复可用。"
            )

    lines.append(
        "回答时只能引用已成功获取的数据，对未获取部分明确说明\"未获取到相关数据\"，不要编造。"
    )
    return "\n".join(lines)


# 失败连续次数达到此阈值则进入冷却（而非永久剔除）
_TOOL_COOLDOWN_THRESHOLD = 2
# 进入冷却后的持续轮数（含进入当轮）；冷却结束后自动解禁可重试
_TOOL_COOLDOWN_ROUNDS = 2


def call_llm(state: ChatAgentState, config) -> dict:
    from .llm_adapter import CCSChatModel
    llm: CCSChatModel = config["configurable"]["llm"]
    tool_registry: ToolRegistry = config["configurable"].get("tool_registry", ToolRegistry())
    openai_tools = tool_registry.get_all_tools() if tool_registry else None

    # 临时屏蔽冷却中的工具（仅本轮不暴露给 LLM；冷却结束后自动恢复）
    cooldown = state.get("cooldown_tools") or {}
    cooling_names = {n for n, remain in cooldown.items() if remain > 0}
    if cooling_names and openai_tools:
        filtered = [
            t for t in openai_tools
            if t.get("function", {}).get("name") not in cooling_names
        ]
        if filtered:
            openai_tools = filtered
            agent_log(state["conversation_id"], f"[冷却] 本轮屏蔽工具: {', '.join(sorted(cooling_names))}")
        # 若过滤后无工具可用，保留原列表让 LLM 走自然语言回答

    full_text = ""
    tool_calls_result = None

    # 注入计划/反思指令（仅本次调用，不入 context_manager 存储，不破 turn 切分）
    # - 首次行动（tool_iteration==0）：要求先出 [计划] 再调第一个工具
    # 注入计划/反思指令（仅本次调用，不入 context_manager 存储，不破 turn 切分）
    # 优先级：replan > 首次(tool_iteration==0) > continue。replan 是显式修订意图，优先。
    msgs = list(state["api_messages"])
    if state.get("reflect_decision") == "replan":
        msgs.append({"role": "user", "content": REPLAN_INSTRUCTION})
    elif state.get("tool_iteration", 0) == 0:
        msgs.append({"role": "user", "content": PLAN_INSTRUCTION})
    elif state.get("reflect_decision") == "continue":
        rr = state.get("reflect_reasoning") or ""
        msgs.append({"role": "user", "content": f"上轮反思判定继续，理由如下。请据此执行下一步：\n{rr}"})

    client = llm._get_client()
    for chunk in client.chat_stream(
        messages=msgs,
        temperature=state.get("temperature", 0.7),
        max_tokens=state.get("max_tokens", 8192),
        tools=openai_tools,
    ):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
            full_text += chunk.delta
        if chunk.tool_calls:
            tool_calls_result = chunk.tool_calls

    if full_text:
        print()

    return {
        "response_text": full_text,
        "tool_calls_result": tool_calls_result,
    }


def final_answer(state: ChatAgentState, config) -> dict:
    """迭代到上限时的收尾：不暴露工具，让 LLM 基于已有数据输出最终报告。

    复用 build_context 已构建的 api_messages，追加一条收尾提示，
    然后调 LLM（无 tools）生成报告。失败时回退到 save_response 的兜底。
    """
    from .llm_adapter import CCSChatModel
    llm: CCSChatModel = config["configurable"]["llm"]
    client = llm._get_client()

    api_messages = list(state.get("api_messages") or [])
    # 收尾原因：检索/统计轮次预算用尽 vs 全局工具迭代到上限（reflect 不再产 done）
    search_iter = state.get("search_iteration") or 0
    max_search = state.get("max_search_iterations") or _DEFAULT_MAX_SEARCH_ITERATIONS
    hit_iter_limit = state["tool_iteration"] >= state["max_tool_iterations"]
    if search_iter >= max_search and not hit_iter_limit:
        wrap_hint = (
            "检索/统计轮次预算已用尽，不要再调用检索或统计工具（list_files/grep/read_file/bash）。"
            "请基于已收集的检索结果直接输出完整、详尽的分析报告，覆盖用户问题的各个方面，不要简短。"
            "未检索到的部分如实说明，但已检索部分要充分展开。"
        )
        agent_log(state["conversation_id"], f"[收尾] 检索/统计轮次预算用尽 ({search_iter}/{max_search})，让 LLM 基于已有数据输出报告")
    else:
        wrap_hint = (
            "已达到工具调用次数上限，不再调用工具。请基于已收集的全部信息，"
            "直接输出完整、详尽的分析报告，覆盖用户问题的各个方面，不要简短。"
            "不要表示遗憾或说无法完成，直接给出结论。"
        )
        agent_log(state["conversation_id"], "[收尾] 工具迭代到上限，让 LLM 基于已有数据输出报告")
    api_messages.append({"role": "user", "content": wrap_hint})
    full_text = ""
    try:
        for chunk in client.chat_stream(
            messages=api_messages,
            temperature=state.get("temperature", 0.7),
            max_tokens=state.get("max_tokens", 8192),
            tools=None,
        ):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                full_text += chunk.delta
        if full_text:
            print()
    except Exception as e:
        agent_log(state["conversation_id"], f"[收尾失败] {e}")

    return {
        "response_text": full_text,
        "tool_calls_result": None,
    }


def reflect(state: ChatAgentState, config) -> dict:
    """ReAct 反思节点：execute_tools 后，基于计划+已执行工具结果决定 continue/replan/done。

    无工具 LLM 调用（tools=None），输出首行 DECISION + 理由。注入 [反思] 到 context_manager。
    受硬约束：检索/迭代预算到顶时不可 continue（route_response 会挡到 final_answer）。
    """
    from .llm_adapter import CCSChatModel
    llm: CCSChatModel = config["configurable"]["llm"]
    context_manager: ContextManager = config["configurable"]["context_manager"]
    client = llm._get_client()

    # 构建反思输入：当前 api（含计划+工具结果）+ 反思指令
    api = context_manager.get_api_messages("")
    msgs = list(api) + [{"role": "user", "content": REFLECT_PROMPT}]

    full_text = ""
    try:
        for chunk in client.chat_stream(
            messages=msgs,
            temperature=0,
            max_tokens=512,
            tools=None,
        ):
            if chunk.delta:
                full_text += chunk.delta
    except Exception as e:
        agent_log(state["conversation_id"], f"[反思失败] {e}，默认 continue")
        full_text = "DECISION: continue\n（反思调用失败，默认继续）"

    # 解析首行 DECISION（只认 continue/replan；无 done——最终回答时机由 call_llm 决定）
    decision = "continue"
    for line in full_text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("continue", "replan"):
                decision = val
            break

    # 不在此处强制 done：检索/迭代预算到顶由 route_response 在 call_llm→execute 闸口守
    # （强制 final_answer）。reflect 只判 continue/replan，避免过早收尾导致答案简短不全。

    agent_log(state["conversation_id"], f"[反思] decision={decision} | {full_text[:120]}")

    # 不注入 context_manager：[反思] 作 user 会破 L2/L3 按 user 分轮 + push_queue 取 turn 切分；
    # 作 assistant-无-tool_calls 会使下个 call_llm 输入以 assistant 结尾 → LLM 不再思考/停调工具。
    # 故只存 state，由 call_llm 经临时指令（不入存储）读取 reflect_reasoning 引导下一步。
    return {
        "reflect_decision": decision,
        "reflect_reasoning": full_text,
    }


def route_after_reflect(state: ChatAgentState) -> str:
    """reflect 后路由：replan→build_context（call_llm 重出计划）/ continue→build_context
    （或 load_memory if add_allowed_path）。无 done——最终回答由 call_llm 自身决定
    （不再调工具即回答 → save_response），避免 reflect 过早判 done 导致答案简短不全。"""
    decision = state.get("reflect_decision", "continue")
    if decision == "replan":
        return "build_context"  # → call_llm（检测 reflect_decision==replan 注入 REPLAN_INSTRUCTION）
    # continue：保留 add_allowed_path 重触发 load_memory 的逻辑
    return _route_after_execute(state)


def _has_search_tool_call(tool_calls_result: Optional[List[Dict]]) -> bool:
    """判断本轮工具调用是否含受预算约束的工具（检索三件套 + bash）。

    LLM 不能靠换工具绕过轮次预算——三件套和 bash 都在此集合内。
    """
    if not tool_calls_result:
        return False
    for tc in tool_calls_result:
        name = tc.get("function", {}).get("name", "")
        if name in _BUDGET_TOOLS:
            return True
    return False


def route_response(state: ChatAgentState) -> str:
    tool_calls = state.get("tool_calls_result")
    if tool_calls and state["tool_iteration"] < state["max_tool_iterations"]:
        # 硬闸门 B：检索/统计轮次预算到顶且本轮仍想调受预算工具 → 强制收尾
        search_iter = state.get("search_iteration") or 0
        max_search = state.get("max_search_iterations") or _DEFAULT_MAX_SEARCH_ITERATIONS
        if search_iter >= max_search and _has_search_tool_call(tool_calls):
            agent_log(state["conversation_id"], f"[收尾] 检索/统计轮次预算用尽 ({search_iter}/{max_search})，强制基于已有数据输出")
            return "final_answer"
        return "execute_tools"
    # 到迭代上限且 LLM 还想调工具 — 给一次收尾机会，让 LLM 基于已有数据输出报告
    if tool_calls and state["tool_iteration"] >= state["max_tool_iterations"]:
        return "final_answer"
    return "save_response"


def build_context(state: ChatAgentState, config) -> dict:
    context_manager: ContextManager = config["configurable"]["context_manager"]
    tool_registry: ToolRegistry = config["configurable"].get("tool_registry", ToolRegistry())
    openai_tools = tool_registry.get_all_tools() if tool_registry else None

    # 设置子模块 conv_id，使其日志写入当前会话日志文件
    from .guardrails import set_conv_id as _gd_set_conv_id
    _gd_set_conv_id(state["conversation_id"])

    # 工具失败缺口注入
    failure_hint = _build_failure_hint(state)

    # 压缩（L2/L3/L4 原地，L1 已在 append_tool_result 时完成）
    tools_tokens = _str_token_count(json.dumps(openai_tools, ensure_ascii=False)) if openai_tools else 0
    limit = context_manager.max_context_tokens
    compressed = context_manager.compress_if_needed(tools_tokens, failure_hint)

    api_messages = context_manager.get_api_messages(failure_hint)

    tokens = _token_count(api_messages)
    total_tokens = tokens + tools_tokens
    tag = "[压缩后]" if compressed else "[上下文]"
    agent_log(state["conversation_id"], f"{tag} {total_tokens:,} / {limit:,} tokens ({total_tokens/limit*100:.1f}%)")

    # 硬裁剪兜底
    if tokens + tools_tokens > limit:
        agent_log(state["conversation_id"], f"[警告] 压缩后仍超限 ({tokens + tools_tokens:,} > {limit:,})，执行硬裁剪")
        api_messages = _hard_truncate(api_messages, limit - tools_tokens)
        tokens = _token_count(api_messages)
        agent_log(state["conversation_id"], f"[硬裁剪后] {tokens + tools_tokens:,} / {limit:,} tokens ({(tokens + tools_tokens)/limit*100:.1f}%)")

    return {"api_messages": api_messages}


def _normalize_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> tuple:
    """归一化工具参数中的日期字段为 YYYYMM。

    识别 month / base_month / month_from / month_to 等字段，以及 month_to=2028-12 这类写法。
    返回 (归一化后的 arguments, {字段: (原值, 归一化值)})。
    """
    from .degradation.agent import _normalize_month

    date_fields = ("month", "base_month", "month_from", "month_to",
                   "start_month", "end_month")
    changes = {}
    if not isinstance(arguments, dict):
        return arguments, changes
    for field in date_fields:
        if field in arguments and arguments[field]:
            original = arguments[field]
            if isinstance(original, str) and re.match(r"^\d{6}$", original):
                continue  # 已是标准格式
            normalized = _normalize_month(original)
            if normalized and normalized != original:
                arguments[field] = normalized
                changes[field] = (original, normalized)
    return arguments, changes


def execute_tools(state: ChatAgentState, config) -> dict:
    tool_registry: ToolRegistry = config["configurable"].get("tool_registry", ToolRegistry())
    context_manager = config["configurable"].get("context_manager")
    tool_calls_result = state["tool_calls_result"] or []
    conv_id = state["conversation_id"]

    # 设置子模块 conv_id，使其日志写入当前会话日志文件
    from .tool_registry import set_conv_id as _tr_set_conv_id
    from .guardrails import set_conv_id as _gd_set_conv_id
    _tr_set_conv_id(conv_id)
    _gd_set_conv_id(conv_id)

    # ── 临时测试钩子：断点续跑测试（测完删除此块）──
    # 放 data/_chkpt_break 标记文件即在第 2+ 次 execute_tools 抛异常模拟崩溃；
    # 触发后自毁标记文件，重启续跑不会重复触发。
    import os as _os
    _break_marker = _os.path.join("data", "_chkpt_break")
    if _os.path.exists(_break_marker) and state.get("tool_iteration", 0) >= 1:
        try:
            _os.remove(_break_marker)
        except OSError:
            pass
        agent_log(state["conversation_id"], f"[测试钩子] 模拟崩溃 @ execute_tools (tool_iteration={state.get('tool_iteration')})")
        raise RuntimeError("CHKPT_TEST: 模拟崩溃，用于测试断点续跑")
    # ── 测试钩子结束 ──

    # 每轮工具调用开始前重置"本轮取消"标志
    if tool_registry._confirmation is not None:
        tool_registry._confirmation.reset_round()

    new_messages = []
    failed_tool_ids = []
    failed_tool_names = []
    # 检索去重表（跨轮持久）：复制本回合状态，循环中增量更新
    search_called_tools = dict(state.get("search_called_tools") or {})
    round_searched = False  # 本轮是否有新的（非去重跳过的）检索调用

    # Assistant 消息（带 tool_calls）—— 同步追加到上下文
    lc_tool_calls = _openai_tc_to_langchain(tool_calls_result)
    assistant_msg = AIMessage(
        content=state["response_text"] or "",
        tool_calls=lc_tool_calls,
        additional_kwargs={"tool_calls": tool_calls_result},
    )
    new_messages.append(assistant_msg)
    if context_manager is not None:
        context_manager.append_assistant(state["response_text"] or "", tool_calls=tool_calls_result)

    # 执行每个工具调用
    for tc in tool_calls_result:
        tool_name = tc["function"]["name"]
        try:
            arguments = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError) as e:
            agent_log(state["conversation_id"], f"[工具参数解析失败] {tool_name}: {e}")
            tool_msg = ToolMessage(
                content=f"错误: 工具参数 JSON 解析失败 — {e}",
                tool_call_id=tc["id"],
                name=tool_name,
            )
            new_messages.append(tool_msg)
            if context_manager is not None:
                context_manager.append_tool_result(
                    tool_name, tc["id"],
                    f"错误: 工具参数 JSON 解析失败 — {e}",
                    {}, conv_id,
                )
            failed_tool_ids.append(tc["id"])
            if tool_name not in failed_tool_names:
                failed_tool_names.append(tool_name)
            continue

        # 参数归一化：把工具参数里的日期字段修正为标准格式
        arguments, normalized_fields = _normalize_tool_arguments(tool_name, arguments)
        if normalized_fields:
            agent_log(state["conversation_id"], f"[参数归一化] {tool_name}: {normalized_fields}")

        # 硬闸门 A：受预算工具去重（检索三件套 + bash）——同 arg_hash 已执行过则不重复执行
        arg_hash = _search_arg_hash(tool_name, arguments)
        if arg_hash is not None:
            prev_tc_id = search_called_tools.get(arg_hash)
            if prev_tc_id is not None:
                agent_log(state["conversation_id"], f"[检索去重] {tool_name}({json.dumps(arguments, ensure_ascii=False)[:100]}) 跳过重复调用，见历史 tool_call_id={prev_tc_id}")
                result = (
                    f"已执行过相同参数（见历史 tool_call_id={prev_tc_id}）。"
                    "不要重复调用同一检索/统计，请基于已有结果继续分析或停止。"
                )
                # 去重命中：不执行、不计数、直接构造 tool message
                tool_msg = ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                    name=tool_name,
                )
                new_messages.append(tool_msg)
                if context_manager is not None:
                    context_manager.append_tool_result(
                        tool_name, tc["id"], result, arguments, conv_id,
                    )
                continue
            # 未命中：正常执行，事后登记 arg_hash
            round_searched = True

        result = tool_registry.call_tool(tool_name, arguments)
        agent_log(state["conversation_id"], f"[工具调用] {tool_name}({json.dumps(arguments, ensure_ascii=False)})")
        display = result[:200] + ("..." if len(result) > 200 else "")
        agent_log(state["conversation_id"], f"[工具结果] {display}")

        # 检索工具执行成功后登记到去重表（arg_hash -> 本次 tool_call_id）
        if arg_hash is not None and not (
            isinstance(result, str) and (
                result.lstrip().startswith("错误:") or result.lstrip().startswith("Error:")
            )
        ):
            search_called_tools[arg_hash] = tc["id"]

        # 识别工具执行失败
        if isinstance(result, str) and (
            result.lstrip().startswith("错误:") or result.lstrip().startswith("Error:")
        ):
            failed_tool_ids.append(tc["id"])
            if tool_name not in failed_tool_names:
                failed_tool_names.append(tool_name)

        # L1 工具结果分流：大结果落盘，返回存储 content；上下文放投影
        if isinstance(result, str) and result and context_manager is not None:
            stored = context_manager.append_tool_result(
                tool_name, tc["id"], result, arguments, conv_id,
            )
        else:
            stored = result
            if context_manager is not None:
                context_manager.append_tool_result(
                    tool_name, tc["id"], str(result), arguments, conv_id,
                )

        tool_msg = ToolMessage(
            content=stored,
            tool_call_id=tc["id"],
            name=tool_name,
        )
        new_messages.append(tool_msg)

    # 硬闸门 B：本轮有新的检索调用（去重跳过的不计）则消耗 1 次检索预算
    new_search_iter = (state.get("search_iteration") or 0) + (1 if round_searched else 0)

    # 检测 add_allowed_path
    add_allowed_path_called = False
    for tc in tool_calls_result:
        if tc["function"]["name"] == "add_allowed_path":
            add_allowed_path_called = True
            break

    return {
        "messages": new_messages,
        "tool_iteration": state["tool_iteration"] + 1,
        "tool_failure_ids": failed_tool_ids,
        "tool_failure_history": (state.get("tool_failure_history") or []) + failed_tool_ids,
        "tool_failure_names": (state.get("tool_failure_names") or []) + failed_tool_names,
        "tool_failure_streaks": _update_streaks(state, tool_calls_result, failed_tool_names),
        "cooldown_tools": _update_cooldown(state, tool_calls_result, failed_tool_names),
        "search_iteration": new_search_iter,
        "search_called_tools": search_called_tools,
        "_add_allowed_path_called": add_allowed_path_called,
    }


def _update_streaks(state: ChatAgentState, tool_calls_result: List[Dict], failed_names: List[str]) -> Dict[str, int]:
    """更新各工具的连续失败计数：失败 +1，本轮被调用且成功则归零。

    streak 只在"被调用"时变化——未被调用的工具保持原值。
    """
    streaks = dict(state.get("tool_failure_streaks") or {})
    called_names = {tc["function"]["name"] for tc in tool_calls_result}
    failed_set = set(failed_names)
    for name in called_names:
        if name in failed_set:
            streaks[name] = streaks.get(name, 0) + 1
        else:
            streaks[name] = 0  # 调用成功，归零
    return streaks


def _update_cooldown(state: ChatAgentState, tool_calls_result: List[Dict], failed_names: List[str]) -> Dict[str, int]:
    """维护冷却计数：

    - 先把已有冷却计数 -1（消耗一轮）
    - 连续失败次数达到阈值的工具，进入冷却（设置剩余轮数）
    - 冷却到 0 自动移除（解禁）
    """
    cooldown = {n: max(0, r - 1) for n, r in (state.get("cooldown_tools") or {}).items()}

    streaks = dict(state.get("tool_failure_streaks") or {})
    # 用更新后的 streaks 判定是否新进入冷却
    failed_set = set(failed_names)
    for name in failed_set:
        streak = streaks.get(name, 0) + 1  # 本轮失败后的 streak
        if streak >= _TOOL_COOLDOWN_THRESHOLD and cooldown.get(name, 0) == 0:
            cooldown[name] = _TOOL_COOLDOWN_ROUNDS
            agent_log(state["conversation_id"], f"[冷却] {name} 连续失败 {streak} 次，进入 {_TOOL_COOLDOWN_ROUNDS} 轮冷却")

    # 移极解禁：若某工具本轮被调用且成功，立即清掉冷却与 streak
    called_names = {tc["function"]["name"] for tc in tool_calls_result}
    for name in called_names:
        if name not in failed_set:
            cooldown.pop(name, None)

    # 清理计数到 0 的项
    return {n: r for n, r in cooldown.items() if r > 0}


def save_response(state: ChatAgentState, config) -> dict:
    storage: Storage = config["configurable"]["storage"]
    context_manager = config["configurable"].get("context_manager")

    failure_history = state.get("tool_failure_history") or []
    response_text = state.get("response_text") or ""
    hit_iter_limit = state["tool_iteration"] >= state["max_tool_iterations"] and state.get("tool_calls_result")

    if hit_iter_limit and not response_text:
        content = "抱歉，工具调用次数达到上限，无法完成任务。"
    elif hit_iter_limit and response_text:
        content = response_text
    elif failure_history and not _has_usable_tool_data(state):
        content = _build_tool_failure_notice(state)
    else:
        content = response_text

    # canary 检测
    canary = state.get("canary", "")
    if canary and canary in content:
        agent_log(state["conversation_id"], "[canary] 检测到系统提示词泄露，已截断输出")
        content = "我无法透露系统配置。"

    # 凭证泄露防护
    content = redact_credentials(content)
    # PII 脱敏
    content = redact_pii(content)
    # 有害内容过滤
    content = filter_harmful_content(content)

    assistant_msg = AIMessage(content=content)

    # 同步追加到上下文
    if context_manager is not None:
        context_manager.append_assistant(content)

    conv_id = state["conversation_id"]
    title = state.get("title", "新对话")
    # 消息数 = 本轮开始前的历史数 + 本轮新增（含此条 assistant）
    history_count = state.get("history_count", 0)
    msg_count = history_count + len(state["messages"]) + 1
    storage.update_index(conv_id, title, msg_count)

    return {"messages": [assistant_msg]}


def _has_usable_tool_data(state: ChatAgentState) -> bool:
    """判断本轮是否有至少一个成功的工具结果可作为回答依据。

    只要存在一条非错误前缀的 ToolMessage，且其对应的 assistant tool_call 不在失败历史中，
    即认为有可用数据。LLM 仍可能基于部分成功数据回答，这是允许的。
    """
    failure_ids = set(state.get("tool_failure_history") or [])
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            if (msg.tool_call_id or "") in failure_ids:
                continue
            content = (msg.content or "").lstrip()
            if content and not content.startswith("错误:") and not content.startswith("Error:"):
                return True
    return False


def _build_tool_failure_notice(state: ChatAgentState) -> str:
    """工具调用失败时的如实告知，不编撰内容。"""
    failure_ids = set(state.get("tool_failure_history") or [])
    failed_tools = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and (msg.tool_calls or msg.additional_kwargs.get("tool_calls")):
            tcs = msg.tool_calls or msg.additional_kwargs.get("tool_calls", [])
            for tc in tcs:
                tc_id = tc.get("id") or tc.get("function", {}).get("name", "")
                if tc_id in failure_ids:
                    name = tc.get("name") or tc.get("function", {}).get("name", "")
                    if name and name not in failed_tools:
                        failed_tools.append(name)

    tool_part = f"（失败工具: {', '.join(failed_tools)}）" if failed_tools else ""
    return (
        f"所需的数据未能成功获取{tool_part}，因此无法给出可靠回答。\n"
        "可能的原因：工具服务不可用、参数错误或数据源暂时无法访问。\n"
        "请稍后重试，或检查工具服务状态后再次提问。"
    )


def _hard_truncate(messages: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """硬裁剪兜底：优先丢弃与当前任务无关的完整 user-assistant 轮次（最旧优先），
    当前 turn 自身过大时再按 tool-iteration 子单元从最旧丢起。

    一个完整轮次 = user 需求 + 之后一系列 assistant/tool 操作，直到 assistant 给出最终结果。
    保护：system + 当前 turn（最后一个 user-turn）。
    丢弃优先级：历史 turn 最旧优先（≈ 和当前任务最无关）→ 当前 turn 内最旧 tool-iteration。
    """
    if not messages:
        return messages

    messages = [m for m in messages if m is not None]
    if not messages:
        return messages

    # system 块边界
    system_end = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system_end = i + 1
        else:
            break

    # 按 user-turn 分组：每个 user 消息开始，到下一个 user 之前为一个完整轮次
    turns = []  # (start, end_exclusive)
    i = system_end
    while i < len(messages):
        if messages[i].get("role") == "user":
            start = i
            j = i + 1
            while j < len(messages) and messages[j].get("role") != "user":
                j += 1
            turns.append((start, j))
            i = j
        else:
            i += 1

    result = list(messages)

    # 阶段 1：历史 turn 从最旧开始丢（当前 turn = turns[-1] 保护）
    if len(turns) > 1:
        for start, end in turns[:-1]:
            if _token_count([m for m in result if m is not None]) <= limit:
                break
            for idx in range(start, end):
                result[idx] = None

    # 阶段 2：仍超限 → 当前 turn 自身过大，按 tool-iteration 子单元从最旧丢起，
    # 保留最后一个 iteration（最近的数据）+ 最终 assistant 文本
    if _token_count([m for m in result if m is not None]) > limit and turns:
        cur_start, cur_end = turns[-1]
        iter_units = []
        k = cur_start
        while k < cur_end:
            m = result[k]
            if m is None:
                k += 1
                continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                unit = [k]
                tc_ids = {tc["id"] for tc in m["tool_calls"]}
                j = k + 1
                while j < cur_end:
                    mj = result[j]
                    if mj is None:
                        j += 1
                        continue
                    if mj.get("role") == "tool" and mj.get("tool_call_id") in tc_ids:
                        unit.append(j)
                        j += 1
                    else:
                        break
                iter_units.append(unit)
                k = j
            else:
                iter_units.append([k])
                k += 1
        # 保留最后一个 iteration，丢更老的
        for unit in iter_units[:-1]:
            if _token_count([m for m in result if m is not None]) <= limit:
                break
            for idx in unit:
                result[idx] = None

    result = [m for m in result if m is not None]

    # 清理悬空 tool result（其 assistant(tool_calls) 被丢的）
    valid_tc_ids = set()
    for m in result:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                valid_tc_ids.add(tc["id"])
    result = [
        m for m in result
        if m.get("role") != "tool" or m.get("tool_call_id") in valid_tc_ids
    ]

    # 最后兜底：仍超限则截断最后一条 user 内容
    if _token_count(result) > limit:
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                others = [m for j, m in enumerate(result) if j != i]
                budget = max(0, limit - _token_count(others) - 100)
                content = result[i].get("content") or ""
                if _token_count([{"role": "user", "content": content}]) > budget:
                    result[i] = {**result[i], "content": _str_truncate_by_tokens(content, budget, "\n...(内容过长已截断)")}
                break

    return result


def _route_after_append(state: ChatAgentState) -> str:
    """条件边1：append_user_msg 后，ALLOWED_ROOTS 有目录则 load_memory，否则 build_context。"""
    from .tools.filesystem import ALLOWED_ROOTS
    if ALLOWED_ROOTS:
        return "load_memory"
    return "build_context"


def _route_after_execute(state: ChatAgentState) -> str:
    """条件边2：execute_tools 后，调了 add_allowed_path 则 load_memory，否则 build_context。"""
    if state.get("_add_allowed_path_called"):
        return "load_memory"
    return "build_context"


def build_chat_graph(
    llm,
    storage: Storage,
    context_manager: ContextManager,
    tool_registry: Optional[ToolRegistry] = None,
    memory_manager=None,
    checkpointer=None,
):
    """构建对话状态图。

    checkpointer：传入 LangGraph checkpointer（如 SqliteSaver）启用节点级状态持久化 +
    断点续跑；为 None 则无 checkpoint（退化原行为）。thread_id 由调用方在
    config["configurable"]["thread_id"] 传入，每轮独立以避免跨轮消息累积。
    """
    graph = StateGraph(ChatAgentState)

    graph.add_node("append_user_msg", append_user_msg)
    graph.add_node("load_memory", load_memory)
    graph.add_node("build_context", build_context)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("reflect", reflect)
    graph.add_node("final_answer", final_answer)
    graph.add_node("save_response", save_response)
    graph.add_node("push_queue", push_queue)

    graph.add_edge(START, "append_user_msg")
    # 条件边1：append_user_msg 后根据 allowed_path 路由
    graph.add_conditional_edges("append_user_msg", _route_after_append, {
        "load_memory": "load_memory",
        "build_context": "build_context",
    })
    graph.add_edge("load_memory", "build_context")
    graph.add_edge("build_context", "call_llm")
    graph.add_conditional_edges("call_llm", route_response, {
        "execute_tools": "execute_tools",
        "final_answer": "final_answer",
        "save_response": "save_response",
    })
    # execute_tools → reflect（ReAct 反思：决定 continue/replan/done）
    graph.add_edge("execute_tools", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {
        "build_context": "build_context",     # continue / replan → 重建 api → call_llm
        "load_memory": "load_memory",          # continue + add_allowed_path
    })
    graph.add_edge("final_answer", "save_response")
    graph.add_edge("save_response", "push_queue")
    graph.add_edge("push_queue", END)

    compile_kwargs = {"checkpointer": checkpointer} if checkpointer is not None else {}
    return graph.compile(**compile_kwargs)
