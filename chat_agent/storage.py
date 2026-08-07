"""对话存储模块"""
import hashlib
import os
import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

_VALID_ID_PATTERN = re.compile(r"^[a-f0-9-]+$")
# 工具调用 id 可能含大写/特殊字符，落盘前归一化为文件名安全字符
_ARTIFACT_ID_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]")


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
    """对话消息（存储层只记录事实，不承载压缩状态）"""
    role: str          # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None     # 稳定消息 id（跨保存/加载不变）
    pinned: bool = False         # 不可复得信息标记（L3 丢弃时保护，默认 False，留扩展点）

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.id:
            d["id"] = self.id
        if self.pinned:
            d["pinned"] = True
        return d


@dataclass
class Conversation:
    """对话会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    title: str = "新对话"
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    system_prompt: str = ""
    # 断点续跑支持：turn_seq 单调递增轮次序号；active_thread 标记进行中的 thread_id（非空=有未完成 turn）
    turn_seq: int = 0
    active_thread: str = ""

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
            "turn_seq": self.turn_seq,
            "active_thread": self.active_thread,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        conv = cls(
            id=data["id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            system_prompt=data.get("system_prompt", ""),
            turn_seq=data.get("turn_seq", 0),
            active_thread=data.get("active_thread", ""),
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
                id=msg_data.get("id"),
                pinned=msg_data.get("pinned", False),
            ))
        return conv


class ToolArtifactStore:
    """大工具结果落盘存储：data/tool_artifacts/<conv_id>/<safe_tc_id>.json

    L1 工具结果分流：单条结果超阈值时原文存档，上下文与存储消息只保留存档路径 + 投影。
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tool_artifacts")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_artifact_id(tool_call_id: str) -> str:
        """把 tool_call_id 归一化为文件名安全字符。空或全非法时用 sha256 摘要。"""
        if not tool_call_id:
            return hashlib.sha256(b"empty").hexdigest()[:16]
        cleaned = _ARTIFACT_ID_SANITIZE.sub("_", tool_call_id)[:64]
        if not cleaned or cleaned == "_" * len(cleaned):
            return hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()[:16]
        return cleaned

    def _conv_dir(self, conv_id: str) -> Path:
        if not _VALID_ID_PATTERN.match(conv_id):
            raise ValueError(f"非法会话 ID: {conv_id!r}")
        return self.base_dir / conv_id

    def write(self, conv_id: str, tool_call_id: str, content: str) -> str:
        """落盘工具结果原文，返回文件绝对路径（存入存储消息 content 的 [已存档] 标记）。"""
        conv_dir = self._conv_dir(conv_id)
        conv_dir.mkdir(parents=True, exist_ok=True)
        file_path = conv_dir / f"{self._safe_artifact_id(tool_call_id)}.json"
        payload = json.dumps({
            "tool_call_id": tool_call_id,
            "content": content,
            "saved_at": time.time(),
        }, ensure_ascii=False, indent=2)
        Storage._atomic_write(file_path, payload)
        return str(file_path)

    def read(self, path: str) -> Optional[str]:
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("content")

    def delete_for_conversation(self, conv_id: str) -> bool:
        """删除某会话的全部工具结果存档。"""
        conv_dir = self._conv_dir(conv_id)
        if conv_dir.exists():
            import shutil
            shutil.rmtree(conv_dir, ignore_errors=True)
            return True
        return False

    def cleanup_orphans(self, conversations_dir) -> int:
        """清理孤儿 artifact 目录：会话文件已不存在但 artifact 目录残留的。

        扫描 base_dir 下每个子目录，若 conversations_dir/<conv_id>.json 不存在则删除。
        返回清理的目录数。目录名必须符合 conv_id 格式校验（防误删/路径遍历）。
        """
        if not self.base_dir.exists():
            return 0
        import shutil
        removed = 0
        for entry in self.base_dir.iterdir():
            if not entry.is_dir():
                continue
            conv_id = entry.name
            # 校验目录名是合法 conv_id 格式，防 .DS_Store/__pycache__/临时目录被误删或路径遍历
            if not _VALID_ID_PATTERN.match(conv_id):
                continue
            conv_file = Path(conversations_dir) / f"{conv_id}.json"
            if not conv_file.exists():
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
                print(f"  [清理] 孤儿 artifact 目录: {conv_id}")
        return removed


class Storage:
    """对话存储管理"""
    
    def __init__(self, storage_dir: str = None,
                 artifact_store: Optional["ToolArtifactStore"] = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self.artifacts = artifact_store or ToolArtifactStore()
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
        self._atomic_write(self.index_file, json.dumps(index, ensure_ascii=False, indent=2))

    @staticmethod
    def _atomic_write(path: Path, content: str):
        """原子写入：先写临时文件再 rename，防止崩溃导致文件损坏。"""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".tmp_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(path))
        except BaseException:
            # 写入失败时清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    
    def _get_file_path(self, conv_id: str) -> Path:
        """获取会话文件路径。校验 ID 格式防止路径遍历。"""
        if not _VALID_ID_PATTERN.match(conv_id):
            raise ValueError(f"非法会话 ID: {conv_id!r}")
        return self.storage_dir / f"{conv_id}.json"
    
    def create_conversation(self, title: str = None, system_prompt: str = None) -> Conversation:
        """创建新对话"""
        conv = Conversation(
            title=title or "新对话",
            system_prompt=system_prompt or "",
        )
        self.save_conversation(conv)
        return conv
    
    def save_conversation(self, conv: Conversation):
        """保存对话"""
        conv.updated_at = time.time()
        file_path = self._get_file_path(conv.id)
        self._atomic_write(file_path, json.dumps(conv.to_dict(), ensure_ascii=False, indent=2))
        
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
        """删除对话及其工具结果存档"""
        file_path = self._get_file_path(conv_id)
        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        # 清理工具结果存档
        self.artifacts.delete_for_conversation(conv_id)
        index = self._load_index()
        if conv_id in index:
            del index[conv_id]
            self._save_index(index)
            deleted = True
        return deleted

    def cleanup_orphan_artifacts(self) -> int:
        """清理孤儿 artifact 目录（对应会话文件已不存在的）。启动时调用一次。"""
        return self.artifacts.cleanup_orphans(self.storage_dir)

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

    def update_index(self, conv_id: str, title: str = None,
                     message_count: int = None, updated_at: float = None):
        """轻量更新索引元数据（不重写会话文件）。用于 SqliteSaver checkpoint 后同步。"""
        index = self._load_index()
        if conv_id not in index:
            # 新会话，创建索引条目
            index[conv_id] = {
                "id": conv_id,
                "title": title or "新对话",
                "created_at": time.time(),
                "updated_at": updated_at or time.time(),
                "message_count": message_count or 0,
            }
        else:
            entry = index[conv_id]
            if title is not None:
                entry["title"] = title
            if message_count is not None:
                entry["message_count"] = message_count
            entry["updated_at"] = updated_at or time.time()
        self._save_index(index)
