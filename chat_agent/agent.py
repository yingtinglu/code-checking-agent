"""ChatAgent 核心模块"""
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .client import LLMClient, ChatResponse
from .config import Config
from .storage import Storage, Conversation, Message, ToolCall

if TYPE_CHECKING:
    from .storage import Message


class ChatAgent:
    """聊天 Agent"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        conversation_id: Optional[str] = None,
        storage: Optional[Storage] = None,
        mcp_client=None,
        max_tool_iterations: int = 10,
    ):
        # 加载配置
        config = Config.from_env()
        
        self.model = model or config.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化存储
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
        
        # 初始化 LLM 客户端
        self.client = LLMClient(
            model=self.model,
            api_key=api_key or config.api_key,
            api_base=api_base or config.api_base,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

        # MCP 工具
        self.mcp_client = mcp_client
        self.max_tool_iterations = max_tool_iterations
    
    def chat(self, message: str, save: bool = True) -> str:
        self.conversation.add_message("user", message)

        openai_tools = None
        if self.mcp_client:
            openai_tools = self.mcp_client.get_openai_tools()

        iteration = 0
        while iteration < self.max_tool_iterations:
            api_messages = self._build_api_messages()

            response = self.client.chat(
                messages=api_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=openai_tools,
            )

            if not response.tool_calls:
                self.conversation.add_message("assistant", response.content)
                break

            # LLM 请求调用工具
            tool_calls_objs = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                for tc in response.tool_calls
            ]
            self.conversation.add_message(
                "assistant",
                response.content or "",
                tool_calls=tool_calls_objs,
            )

            for tc in response.tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                print(f"  [工具调用] {tool_name}({json.dumps(arguments, ensure_ascii=False)})")
                result = self.mcp_client.call_tool(tool_name, arguments)
                display = result[:200] + ("..." if len(result) > 200 else "")
                print(f"  [工具结果] {display}")

                self.conversation.add_message(
                    "tool",
                    result,
                    tool_call_id=tc["id"],
                    name=tool_name,
                )

            iteration += 1

        if iteration >= self.max_tool_iterations:
            self.conversation.add_message(
                "assistant",
                "抱歉，工具调用次数达到上限，无法完成任务。"
            )

        if save:
            self.storage.save_conversation(self.conversation)

        return self.conversation.messages[-1].content

    def _build_api_messages(self) -> List[Dict[str, Any]]:
        """构建发送给 LLM 的消息列表，包含 tool_calls 和 tool 角色消息。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self.conversation.messages:
            if msg.role == "tool":
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [tc.to_dict() for tc in msg.tool_calls],
                })
            else:
                messages.append({"role": msg.role, "content": msg.content})

        return messages
    
    def clear_history(self, save: bool = True):
        self.conversation.messages = []
        self.conversation.title = "新对话"
        if save:
            self.storage.save_conversation(self.conversation)
    
    def get_history(self) -> List["Message"]:
        return self.conversation.messages
    
    def set_system_prompt(self, prompt: str, save: bool = True):
        self.system_prompt = prompt
        self.conversation.system_prompt = prompt
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
    
    def __repr__(self) -> str:
        return f"ChatAgent(id={self.conversation_id}, model={self.model}, history={self.history_count} messages)"
