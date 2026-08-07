"""记忆管理器 — 持有内存队列和 daemon 后台线程，实现 fork agent 的提取和存储。

Fork agent 以独立 LangGraph 运行：
  1. push_queue: 深拷贝本轮对话片段 + 模型参数 → 快照入队
  2. 后台线程取出快照 → fork_graph.invoke() → 完成
"""
import queue
import threading
from typing import Any, Dict, List, Optional

from .fork_graph import build_fork_graph


class MemorySnapshot:
    """快照：本轮对话片段 + 模型参数。"""

    def __init__(
        self,
        api_messages: List[Dict[str, Any]],
        project_root: str,
        model: str,
        temperature: float,
        max_tokens: int,
        conv_id: str = "",
    ):
        self.api_messages = api_messages
        self.project_root = project_root
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conv_id = conv_id


class MemoryManager:
    """记忆管理器：队列 + daemon 后台线程，一次只处理一个快照，顺序执行。"""

    def __init__(self, api_key: str, api_base: str, model: str = "glm-5.1"):
        from ..client import LLMClient
        self._llm_client = LLMClient(
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self._queue: queue.Queue[Optional[MemorySnapshot]] = queue.Queue(maxsize=10)
        self._fork_graph = build_fork_graph()
        self._daemon = threading.Thread(target=self._run, name="fork-agent", daemon=True)
        self._daemon.start()

    def enqueue(self, snapshot: MemorySnapshot) -> None:
      """将快照放入队列。队列满时拒绝新快照。"""
      from ..logger import agent_log
      try:
          self._queue.put_nowait(snapshot)
      except queue.Full:
          agent_log(snapshot.conv_id or "", "[记忆] 队列已满，丢弃当前快照")

    def _run(self) -> None:
        """后台线程主循环。"""
        from ..logger import agent_log
        while True:
            try:
                snapshot = self._queue.get()
                if snapshot is None:
                    break
                self._process_with_retry(snapshot)
            except Exception as e:
                agent_log("", f"[记忆] 后台线程异常: {e}")

    def _process_with_retry(self, snapshot: MemorySnapshot) -> None:
        """处理单个快照，最多重试 3 次。"""
        from ..logger import agent_log
        project = snapshot.project_root or "None"
        conv_id = snapshot.conv_id or ""
        for attempt in range(1, 4):
            try:
                self._process(snapshot)
                return
            except Exception as e:
                agent_log(conv_id, f"[记忆] fork agent 处理失败 (项目: {project}, 第{attempt}次): {e}")
                if attempt >= 3:
                    agent_log(conv_id, f"[记忆] fork agent 已达最大重试次数 (项目: {project})，跳过此快照")

    def _process(self, snapshot: MemorySnapshot) -> None:
        """调用 fork graph 处理快照。"""
        from ..logger import agent_log
        self._fork_graph.invoke(
            {
                "api_messages": snapshot.api_messages,
                "project_root": snapshot.project_root,
                "model": snapshot.model,
                "temperature": snapshot.temperature,
                "max_tokens": snapshot.max_tokens,
                "tool_calls_result": None,
                "response_text": "",
                "tool_iteration": 0,
                "max_tool_iterations": 5,
                "stats": {"created": 0, "updated": 0, "skipped": 0},
                "conv_id": snapshot.conv_id or "",
            },
            config={"configurable": {"llm_client": self._llm_client}},
        )
        agent_log(snapshot.conv_id or "", "[记忆] 快照已处理完成")
