"""Fork Agent 独立 LangGraph — 后台运行，提取和存储长期记忆。

流程：START → fork_build_context → fork_call_llm → fork_route
    ├─ fork_execute_tools → fork_call_llm (循环)
    └─ fork_finish → END

项目路径由 LLM 根据对话内容判断：
  - 平台级话题 → 不执行记忆操作
  - 非平台 → 以 project_root 作为项目路径执行记忆提取
"""
import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..context import _token_count, _str_token_count, _str_truncate_by_tokens
from ..logger import agent_log
from ..prompts import load_prompt
from . import store

# Fork Agent 工具定义（OpenAI function-calling 格式）
_FORK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fork_read_file",
            "description": "读取记忆文件内容。路径相对于项目的 .agent_memory/ 目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于 .agent_memory/ 的文件路径，如 'memory_index.md' 或 'model_insight/标题.md'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fork_write_memory",
            "description": "写入一条记忆文件。自动处理时间戳、索引更新。"
            "去重由系统自动完成（Jaccard 相似度预筛）：如果发现相似已有记忆，系统会返回候选内容供你判断；"
            "你只需关注提取质量。当系统返回相似候选时，请在下一轮用 mode='update' 和 update_path 更新，"
            "或用 mode='create' 强制创建新记忆。无候选反馈时默认创建。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "记忆分类：model_insight 或 user_feedback",
                    },
                    "title": {
                        "type": "string",
                        "description": "记忆标题，简短明确，只含中文/字母/数字",
                    },
                    "content": {
                        "type": "string",
                        "description": "记忆完整内容",
                    },
                    "scope": {
                        "type": "string",
                        "description": "作用域：'project' 或 'directory:相对路径'，可选",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["create", "update"],
                        "description": "create=创建新记忆（默认），update=更新已有记忆（需同时提供 update_path）",
                    },
                    "update_path": {
                        "type": "string",
                        "description": "update 模式时，要更新的记忆文件相对路径（如 'model_insight/标题.md'）",
                    },
                },
                "required": ["category", "title", "content"],
            },
        },
    },
]


class ForkAgentState(TypedDict):
    api_messages: List[Dict[str, Any]]
    project_root: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    tool_calls_result: Optional[List[Dict[str, Any]]]
    response_text: str
    tool_iteration: int
    max_tool_iterations: int
    stats: Dict[str, int]
    conv_id: str


def fork_build_context(state: ForkAgentState, config) -> dict:
    """构建 fork agent 的 LLM 上下文。

    系统提示词 = memory_extraction.txt + 当前授权项目路径 + 已有记忆索引内容
    + 截断后的对话片段 + 任务指令（含平台判断）。
    """
    project_root = state["project_root"]

    # 系统提示词
    system_prompt = load_prompt(
        "memory_extraction.txt",
        "你是记忆提取代理，在后台整理对话中的长期记忆。",
    )
    if project_root:
        system_prompt += f"\n\n## 当前授权项目\n{project_root}"

    # 截断 api_messages
    truncated = _truncate_messages(state["api_messages"])

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(truncated)

    # 追加任务指令：LLM 从对话内容判断项目，平台话题则跳过
    messages.append({
        "role": "user",
        "content": (
            "请根据对话内容判断当前讨论是否属于平台级话题"
            "（平台、分发中台、算法中台、上架安全、增长中台、联运、客户端、预装、push商业化、AppTouch、其他、生态等），"
            "如果是则只输出 PLATFORM_SKIP，不要调用任何工具；"
            "否则按照系统提示词规则提取适合长期保存的记忆，调用 fork_write_memory 写入。"
            "去重由系统自动处理：如果发现与已有记忆相似，系统会返回候选内容供你判断 create 还是 update；"
            "你不需要手动 fork_read_file 做去重比对。"
        ),
    })

    return {"api_messages": messages}


def _truncate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """截断 tool 内容到 200 token，总计不超过 1000 token，超出的从最早开始丢弃 tool。"""
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool" and isinstance(content, str):
            if _str_token_count(content) > 200:
                content = _str_truncate_by_tokens(content, 200)
                msg = {**msg, "content": content}
        result.append(msg)

    while _token_count(result) > 1000:
        removed = False
        for i, m in enumerate(result):
            if m.get("role") == "tool":
                result.pop(i)
                removed = True
                break
        if not removed:
            break
    return result


def fork_call_llm(state: ForkAgentState, config) -> dict:
    """调用 LLM（独立 LLMClient，不经过 CCSChatModel）。"""
    llm_client = config["configurable"]["llm_client"]

    full_text = ""
    tool_calls_result = None

    for chunk in llm_client.chat_stream(
        messages=state["api_messages"],
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
        tools=_FORK_TOOLS,
        model=state["model"],
    ):
        if chunk.delta:
            full_text += chunk.delta
        if chunk.tool_calls:
            tool_calls_result = chunk.tool_calls

    # 平台级话题检测：LLM 输出 PLATFORM_SKIP 则将 project_root 置为 None
    if "PLATFORM_SKIP" in full_text and not tool_calls_result:
        agent_log(state.get("conv_id", ""), "[记忆] fork 检测到平台级话题，跳过记忆操作")
        return {"response_text": full_text, "tool_calls_result": None, "project_root": None}

    agent_log(state.get("conv_id", ""), f"[记忆] fork LLM 响应: {full_text[:200]}")
    return {
        "response_text": full_text,
        "tool_calls_result": tool_calls_result,
    }


def fork_route(state: ForkAgentState) -> str:
    """条件边：project_root 为 None（平台话题）或无 tool_calls 则结束，否则执行工具。"""
    if not state.get("project_root"):
        return "fork_finish"
    if state.get("tool_calls_result") and state["tool_iteration"] < state["max_tool_iterations"]:
        return "fork_execute_tools"
    return "fork_finish"


def fork_execute_tools(state: ForkAgentState, config) -> dict:
    """执行 fork_read_file / fork_write_memory 工具调用。"""
    project_root = state["project_root"]
    tool_calls = state["tool_calls_result"] or []
    stats = dict(state.get("stats", {"created": 0, "updated": 0, "skipped": 0}))

    new_messages = []

    # Assistant message with tool_calls
    new_messages.append({
        "role": "assistant",
        "content": state["response_text"] or "",
        "tool_calls": tool_calls,
    })

    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            new_messages.append({
                "role": "tool",
                "content": "错误: 工具参数 JSON 解析失败",
                "tool_call_id": tc["id"],
            })
            stats["skipped"] += 1
            agent_log(state.get("conv_id", ""), f"[记忆] fork 工具调用失败: {name} 参数解析错误")
            continue

        if name == "fork_read_file":
            path = args.get("path", "")
            agent_log(state.get("conv_id", ""), f"[记忆] fork 工具调用: fork_read_file({path})")
            content = store.read_memory_file(project_root, path)
            if content is None:
                result = f"文件不存在: {path}"
                agent_log(state.get("conv_id", ""), f"[记忆] fork 工具结果: {path} 不存在")
            else:
                result = content
                agent_log(state.get("conv_id", ""), f"[记忆] fork 工具结果: {path}")
            new_messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tc["id"],
            })

        elif name == "fork_write_memory":
            category = args.get("category", "")
            title = args.get("title", "")
            content = args.get("content", "")
            scope = args.get("scope", "")
            mode = args.get("mode", "create")
            update_path = args.get("update_path", "")

            if not category or not title or not content:
                new_messages.append({
                    "role": "tool",
                    "content": "错误: category、title、content 为必填字段",
                    "tool_call_id": tc["id"],
                })
                stats["skipped"] += 1
                agent_log(state.get("conv_id", ""), "[记忆] fork 工具调用失败: fork_write_memory 字段缺失")
                continue

            agent_log(state.get("conv_id", ""), f"[记忆] fork 工具调用: fork_write_memory({category}/{title}, mode={mode})")
            full_content = f"[scope: {scope}]\n{content}" if scope else content

            # mode=update：LLM 确认更新指定记忆
            if mode == "update" and update_path:
                old_content = store.read_memory_file(project_root, update_path) or ""
                merged = store.merge_content(old_content, full_content)
                store.update_memory_file(project_root, update_path, merged)
                new_messages.append({
                    "role": "tool",
                    "content": f"已更新记忆: {update_path}",
                    "tool_call_id": tc["id"],
                })
                stats["updated"] += 1
                agent_log(state.get("conv_id", ""), f"[记忆] fork 工具结果: 更新 {update_path}")
                continue

            # mode=create 或无 mode：Jaccard 预筛
            similar = store.search_similar_memories(project_root, category, full_content)

            if similar:
                # 返回高相似候选内容，让 LLM 判断 create 还是 update
                candidates_info = []
                for path, score in similar[:3]:  # 最多返回3个候选
                    old = store.read_memory_file(project_root, path) or ""
                    candidates_info.append(
                        f"--- 相似记忆 (相似度 {score:.0%}, 路径 {path}) ---\n{old}"
                    )
                new_messages.append({
                    "role": "tool",
                    "content": (
                        f"发现 {len(similar)} 条相似已有记忆，请判断：\n"
                        f"1. 如果语义相同或包含关系 → 请用 fork_write_memory 的 mode='update' "
                        f"和 update_path='{similar[0][0]}' 更新\n"
                        f"2. 如果语义不同 → 请用 fork_write_memory 的 mode='create' 创建新记忆\n\n"
                        + "\n\n".join(candidates_info)
                    ),
                    "tool_call_id": tc["id"],
                })
                stats["skipped"] += 1  # 本轮未写入，等 LLM 下轮决策
                agent_log(state.get("conv_id", ""),
                          f"[记忆] fork Jaccard 发现 {len(similar)} 条相似记忆，返回给 LLM 判断")
            else:
                # 无相似记忆，直接创建
                rel_path = store.write_memory_file(project_root, category, title, full_content)
                new_messages.append({
                    "role": "tool",
                    "content": f"已创建记忆: {rel_path}",
                    "tool_call_id": tc["id"],
                })
                stats["created"] += 1
                agent_log(state.get("conv_id", ""), f"[记忆] fork 工具结果: 创建 {rel_path}")

    return {
        "api_messages": state["api_messages"] + new_messages,
        "tool_iteration": state["tool_iteration"] + 1,
        "stats": stats,
    }


def fork_finish(state: ForkAgentState, config) -> dict:
    """完成节点：打印日志。"""
    project_root = state.get("project_root")
    stats = state.get("stats", {"created": 0, "updated": 0, "skipped": 0})
    if project_root is None:
        agent_log(state.get("conv_id", ""), "[记忆] fork agent 完成: 平台级话题，跳过记忆操作")
    else:
        agent_log(state.get("conv_id", ""), f"[记忆] fork agent 完成 (项目: {project_root}): "
              f"创建 {stats.get('created', 0)} 条, "
              f"更新 {stats.get('updated', 0)} 条, "
              f"跳过 {stats.get('skipped', 0)} 条")
    return {}


def build_fork_graph():
    """构建 fork agent 的 LangGraph。"""
    graph = StateGraph(ForkAgentState)

    graph.add_node("fork_build_context", fork_build_context)
    graph.add_node("fork_call_llm", fork_call_llm)
    graph.add_node("fork_execute_tools", fork_execute_tools)
    graph.add_node("fork_finish", fork_finish)

    graph.add_edge(START, "fork_build_context")
    graph.add_edge("fork_build_context", "fork_call_llm")
    graph.add_conditional_edges("fork_call_llm", fork_route, {
        "fork_execute_tools": "fork_execute_tools",
        "fork_finish": "fork_finish",
    })
    graph.add_edge("fork_execute_tools", "fork_call_llm")
    graph.add_edge("fork_finish", END)

    return graph.compile()
