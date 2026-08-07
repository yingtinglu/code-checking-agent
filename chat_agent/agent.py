"""ChatAgent 核心模块 — 委托给 LangGraph 图执行。context_manager 持活工作上下文。"""
import json
import secrets
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .client import LLMClient, ChatResponse
from .config import Config
from .context import ContextManager
from .graph import ChatAgentState, build_chat_graph
from .llm_adapter import CCSChatModel
from .storage import Storage, Conversation, Message, ToolCall, ToolArtifactStore
from .tool_registry import ToolRegistry
from .tools import register_all_tools, register_read_artifact_tool
from .memory import MemoryManager

if TYPE_CHECKING:
    from .storage import Message


def _base_to_storage_messages(messages: List[BaseMessage]) -> List[Message]:
    """将本轮新增的 LangChain BaseMessage 列表转为 Storage Message 列表（1:1）。"""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append(Message(role="system", content=msg.content))
        elif isinstance(msg, HumanMessage):
            result.append(Message(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    ToolCall(id=tc["id"], name=tc["name"], arguments=json.dumps(tc["args"], ensure_ascii=False))
                    for tc in msg.tool_calls
                ]
            elif msg.additional_kwargs.get("tool_calls"):
                tool_calls = [
                    ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                    for tc in msg.additional_kwargs["tool_calls"]
                ]
            result.append(Message(role="assistant", content=msg.content or "", tool_calls=tool_calls))
        elif isinstance(msg, ToolMessage):
            result.append(Message(role="tool", content=msg.content, tool_call_id=msg.tool_call_id, name=msg.name))
        else:
            result.append(Message(role="user", content=msg.content))
    return result


class ChatAgent:
    """聊天 Agent — 委托给 LangGraph 图执行。context_manager 持活工作上下文。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        conversation_id: Optional[str] = None,
        storage: Optional[Storage] = None,
        mcp_client=None,
        max_tool_iterations: int = 20,
        confirmation_gateway=None,
        checkpointer=None,
    ):
        config = Config.from_env()

        self.model = model or config.model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.storage = storage or Storage()

        # 初始化或加载会话
        if conversation_id:
            self.conversation = self.storage.load_conversation(conversation_id)
            if self.conversation:
                self.system_prompt = self.conversation.system_prompt
            else:
                self.conversation = self.storage.create_conversation(system_prompt=system_prompt or config.system_prompt)
                self.system_prompt = self.conversation.system_prompt
        else:
            self.conversation = self.storage.create_conversation(system_prompt=system_prompt or config.system_prompt)
            self.system_prompt = self.conversation.system_prompt

        # canary
        self.canary = f"CANARY:{secrets.token_hex(4)}"

        # LLM 适配器
        self._llm = CCSChatModel(
            model=self.model,
            api_key=api_key or config.api_key,
            api_base=api_base or config.api_base,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

        # MCP 工具
        self.mcp_client = mcp_client
        self.max_tool_iterations = max_tool_iterations

        # 统一工具注册表
        self.tool_registry = ToolRegistry(confirmation_gateway=confirmation_gateway)
        register_all_tools(self.tool_registry)
        if mcp_client:
            self.tool_registry.import_mcp(mcp_client)
        from .degradation import register_degradation_reducers
        register_degradation_reducers(self.tool_registry)
        # read_artifact 需要存档存储，装配后注册
        register_read_artifact_tool(
            self.tool_registry,
            self.storage.artifacts,
        )

        # 上下文压缩（有状态累积对象）
        self.context_manager = ContextManager(
            self.model,
            llm_client=self._llm._get_client(),
            tool_registry=self.tool_registry,
            artifact_store=self.storage.artifacts,
        )
        self.context_manager.set_system_prompt(f"{self.system_prompt}\n\n{self.canary}")
        self.context_manager.set_conv_id(self.conversation_id)
        self.context_manager.set_history_getter(lambda: self.conversation.messages)
        # 加载已有会话时从存储重建上下文
        if self.conversation.messages:
            self.context_manager.load_from_history(
                self.conversation.messages,
                f"{self.system_prompt}\n\n{self.canary}",
            )

        # 记忆管理器（fork agent 后台线程，独立 LLMClient，创建一次复用）
        self.memory_manager = MemoryManager(
            api_key=config.api_key,
            api_base=config.api_base,
            model=self.model,
        )

        # checkpointer（SqliteSaver 等）启用节点级状态持久化 + 断点续跑；None 则无 checkpoint
        self._checkpointer = checkpointer

        # 构建 LangGraph 图
        self.graph = build_chat_graph(
            llm=self._llm,
            storage=self.storage,
            context_manager=self.context_manager,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            checkpointer=checkpointer,
        )

        # 加载已有会话后，检测是否有未完成 turn 的 checkpoint，提示续跑
        self._detect_and_offer_resume()

    def chat(self, message: str, save: bool = True) -> str:
        # 保险：若上一轮未完成（崩溃/续跑失败遗留 active_thread），先处理，避免新轮覆盖 → 旧 turn 被遗弃无法恢复
        if self._checkpointer and self.conversation.active_thread:
            print(f"\n[续跑] 检测到上一轮未完成（turn {self.conversation.turn_seq}，保留断点）")
            try:
                confirm = input("y=续跑 / n=丢弃并处理本次新消息: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                confirm = "n"
            if confirm in ("y", "yes"):
                return self.resume_turn()
            # n：用户主动开新轮 → 永久丢弃旧 turn（单 active_thread 无法同时保留两个未完成 turn）
            self._discard_incomplete_turn(self.conversation.active_thread)

        initial_state = {
            "messages": [],  # 本轮新增，从空开始；历史在 context_manager 里
            "system_prompt": f"{self.system_prompt}\n\n{self.canary}",
            "conversation_id": self.conversation.id,
            "title": self.conversation.title,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tool_iteration": 0,
            "max_tool_iterations": self.max_tool_iterations,
            "tool_calls_result": None,
            "api_messages": [],
            "response_text": "",
            "user_input": message,
            "canary": self.canary,
            "tool_failure_ids": [],
            "tool_failure_history": [],
            "tool_failure_names": [],
            "tool_failure_streaks": {},
            "cooldown_tools": {},
            "history_count": len(self.conversation.messages),
            "memory_content": "",
            "_add_allowed_path_called": False,
            "reflect_decision": "",
            "reflect_reasoning": "",
            "search_iteration": 0,
            "max_search_iterations": 15,
            "search_called_tools": {},
        }

        # per-turn thread_id：每轮独立，避免 add_messages 跨轮累积；持久化以支持崩溃续跑
        self.conversation.turn_seq += 1
        thread_id = f"{self.conversation.id}#{self.conversation.turn_seq}"
        self.conversation.active_thread = thread_id
        if save:
            self.storage.save_conversation(self.conversation)

        result = self.graph.invoke(
            initial_state,
            config={"configurable": {
                "llm": self._llm,
                "storage": self.storage,
                "context_manager": self.context_manager,
                "tool_registry": self.tool_registry,
                "memory_manager": self.memory_manager,
                "thread_id": thread_id,
            }},
        )

        # turn 完成：清 active_thread（不再触发续跑检测），保留 turn_seq 计数
        self.conversation.active_thread = ""

        # 本轮新增消息追加到存储
        new_messages = _base_to_storage_messages(result["messages"])
        self.conversation.messages.extend(new_messages)
        self.system_prompt = result.get("system_prompt", self.system_prompt)
        self.conversation.system_prompt = self.system_prompt

        if save:
            self.storage.save_conversation(self.conversation)

        return result.get("response_text", "")

    # ── 断点续跑 ──

    def _make_config(self, thread_id: str) -> dict:
        """构造 graph invoke/get_state 用的 config（含 thread_id 与各运行时依赖）。"""
        return {"configurable": {
            "llm": self._llm,
            "storage": self.storage,
            "context_manager": self.context_manager,
            "tool_registry": self.tool_registry,
            "memory_manager": self.memory_manager,
            "thread_id": thread_id,
        }}

    def _detect_and_offer_resume(self) -> None:
        """加载会话后检测未完成 turn 的 checkpoint，提示用户是否续跑。

        无 checkpointer / 无 active_thread / checkpoint 已到 END → 静默跳过。
        """
        if not self._checkpointer:
            return
        thread_id = self.conversation.active_thread
        if not thread_id:
            return
        config = self._make_config(thread_id)
        try:
            snap = self.graph.get_state(config)
        except Exception as e:
            print(f"  [续跑检测] 查询 checkpoint 失败: {e}，忽略")
            return
        if snap is None or not snap.next:
            # 已完成或无 checkpoint：清理残留 active_thread
            self.conversation.active_thread = ""
            self.storage.save_conversation(self.conversation)
            return
        print(f"\n[续跑] 检测到未完成的任务（turn {self.conversation.turn_seq}，停在节点 {list(snap.next)}）")
        try:
            confirm = input("是否续跑? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "n"
        if confirm in ("y", "yes"):
            self.resume_turn()
        else:
            self._defer_incomplete_turn(thread_id)

    def _discard_incomplete_turn(self, thread_id: str) -> None:
        """永久丢弃未完成 turn：清 active_thread（sqlite checkpoint 留为孤儿，不再可续跑）。

        仅用于用户明确要开新轮、主动放弃断点的场景（chat 入口选 n）。
        崩溃/load 提示选 n 用 _defer_incomplete_turn（保留可续）。
        """
        self.conversation.active_thread = ""
        self.storage.save_conversation(self.conversation)
        print(f"  [续跑] 已丢弃未完成 turn {thread_id}（无法再续跑）")

    def _defer_incomplete_turn(self, thread_id: str) -> None:
        """推迟续跑：保留 active_thread + checkpoint，稍后可输 resume 或下次 load 续跑。"""
        # active_thread 不清，checkpoint 不删，保持可续跑
        print(f"  [续跑] 已推迟未完成 turn {thread_id}（可输 resume 续跑，或下次 load 时续跑）")

    def resume_turn(self) -> str:
        """从最后 checkpoint 续跑当前会话未完成的 turn。

        重建 context_manager 到 checkpoint 时刻状态（load_from_history + 重放本轮消息），
        然后 graph.invoke(None, config) 续跑。完成后追加消息、清 active_thread。
        """
        if not self._checkpointer:
            print("  [续跑] 无可用 checkpointer，不支持续跑")
            return ""
        thread_id = self.conversation.active_thread
        if not thread_id:
            print("  [续跑] 当前没有未完成的任务")
            return ""
        config = self._make_config(thread_id)
        try:
            snap = self.graph.get_state(config)
        except Exception as e:
            print(f"  [续跑] 查询 checkpoint 失败: {e}")
            return ""
        if snap is None or not snap.next:
            self.conversation.active_thread = ""
            self.storage.save_conversation(self.conversation)
            print("  [续跑] 该任务已完成或无 checkpoint，已清理标记")
            return ""

        # 重建 context_manager：全量存储历史（turn 前）+ 重放本轮 checkpoint 消息（幂等）
        turn_base_msgs = (snap.values or {}).get("messages", []) or []
        turn_storage_msgs = _base_to_storage_messages(turn_base_msgs)
        sys_prompt = f"{self.system_prompt}\n\n{self.canary}"
        try:
            self.context_manager.rebuild_from_checkpoint(
                self.conversation.messages, turn_storage_msgs, sys_prompt
            )
        except Exception as e:
            print(f"  [续跑] context_manager 重建失败: {e}")
            return ""

        print(f"  [续跑] 从节点 {list(snap.next)} 续跑 turn {thread_id}")
        result = self.graph.invoke(None, config)  # input=None = 从最后 checkpoint 续跑

        # turn 完成：清 active_thread + 追加本轮全部消息（checkpoint 的 + 续跑新增的）
        self.conversation.active_thread = ""
        new_messages = _base_to_storage_messages(result.get("messages", []))
        self.conversation.messages.extend(new_messages)
        self.system_prompt = result.get("system_prompt", self.system_prompt)
        self.conversation.system_prompt = self.system_prompt
        self.storage.save_conversation(self.conversation)
        return result.get("response_text", "")

    def clear_history(self, save: bool = True):
        self.conversation.messages = []
        self.conversation.title = "新对话"
        self.context_manager.reset()
        if save:
            self.storage.save_conversation(self.conversation)

    def get_history(self) -> List["Message"]:
        return list(self.conversation.messages)

    def get_full_history(self) -> List["Message"]:
        return self.conversation.messages

    def set_system_prompt(self, prompt: str, save: bool = True):
        self.system_prompt = prompt
        self.conversation.system_prompt = prompt
        self.context_manager.set_system_prompt(f"{prompt}\n\n{self.canary}")
        if save:
            self.storage.save_conversation(self.conversation)

    def rename_conversation(self, title: str) -> bool:
        return self.storage.rename_conversation(self.conversation.id, title)

    def set_title_from_first_message(self):
        for msg in self.conversation.messages:
            if msg.role == "user":
                self.conversation.title = msg.content[:20] + ("..." if len(msg.content) > 20 else "")
                self.storage.save_conversation(self.conversation)
                break

    @property
    def conversation_id(self) -> str:
        return self.conversation.id

    @property
    def title(self) -> str:
        return self.conversation.title

    @property
    def history_count(self) -> int:
        return len(self.conversation.messages)

    @property
    def client(self) -> LLMClient:
        return self._llm._get_client()

    def __repr__(self) -> str:
        return f"ChatAgent(id={self.conversation_id}, model={self.model}, history={self.history_count} messages)"
