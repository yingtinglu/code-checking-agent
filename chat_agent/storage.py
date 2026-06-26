"""对话存储模块"""
import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ToolCall:
    """LLM 工具调用"""
    id: str
    name: str
    arguments: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        func = data.get("function", {})
        return cls(
            id=data["id"],
            name=func.get("name", data.get("name", "")),
            arguments=func.get("arguments", data.get("arguments", "")),
        )


@dataclass
class Message:
    """对话消息"""
    role: str          # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Conversation:
    """对话会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "新对话"
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    system_prompt: str = "你是一个有帮助的AI助手，请用简洁、清晰的语言回答用户的问题。"
    
    def add_message(self, role: str, content: str,
                    tool_calls: Optional[List[ToolCall]] = None,
                    tool_call_id: Optional[str] = None,
                    name: Optional[str] = None) -> Message:
        """添加消息"""
        msg = Message(role=role, content=content,
                      tool_calls=tool_calls, tool_call_id=tool_call_id, name=name)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "system_prompt": self.system_prompt,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        conv = cls(
            id=data["id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            system_prompt=data.get("system_prompt", ""),
        )
        for msg_data in data.get("messages", []):
            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = [ToolCall.from_dict(tc) for tc in msg_data["tool_calls"]]
            conv.messages.append(Message(
                role=msg_data["role"],
                content=msg_data.get("content", ""),
                timestamp=msg_data.get("timestamp", time.time()),
                tool_calls=tool_calls,
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
            ))
        return conv


class Storage:
    """对话存储管理"""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self._ensure_index()
    
    def _ensure_index(self):
        """确保索引文件存在"""
        if not self.index_file.exists():
            self._save_index({})
    
    def _load_index(self) -> Dict[str, Dict]:
        """加载索引"""
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save_index(self, index: Dict[str, Dict]):
        """保存索引"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def _get_file_path(self, conv_id: str) -> Path:
        """获取会话文件路径"""
        return self.storage_dir / f"{conv_id}.json"
    
    def create_conversation(self, title: str = None, system_prompt: str = None) -> Conversation:
        """创建新对话"""
        conv = Conversation(
            title=title or "新对话",
            system_prompt=system_prompt or "你是一个有帮助的AI助手，请用简洁、清晰的语言回答用户的问题。"
        )
        self.save_conversation(conv)
        return conv
    
    def save_conversation(self, conv: Conversation):
        """保存对话"""
        conv.updated_at = time.time()
        file_path = self._get_file_path(conv.id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 更新索引
        index = self._load_index()
        index[conv.id] = {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "message_count": len(conv.messages),
        }
        self._save_index(index)
    
    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """加载对话"""
        file_path = self._get_file_path(conv_id)
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Conversation.from_dict(data)
    
    def delete_conversation(self, conv_id: str) -> bool:
        """删除对话"""
        file_path = self._get_file_path(conv_id)
        if file_path.exists():
            file_path.unlink()
            index = self._load_index()
            if conv_id in index:
                del index[conv_id]
                self._save_index(index)
            return True
        return False
    
    def list_conversations(self) -> List[Dict]:
        """列出所有对话"""
        index = self._load_index()
        return sorted(index.values(), key=lambda x: x["updated_at"], reverse=True)
    
    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        """重命名对话"""
        conv = self.load_conversation(conv_id)
        if conv is None:
            return False
        conv.title = new_title
        self.save_conversation(conv)
        return True
