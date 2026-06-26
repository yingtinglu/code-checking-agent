"""Async/sync 桥接 — 在同步代码中运行 MCP 异步操作。"""
import asyncio
import threading
from typing import TypeVar, Coroutine

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _ensure_loop():
    """启动后台事件循环（如未运行）。"""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        return

    _loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _thread = threading.Thread(target=_run_loop, daemon=True, name="mcp-async")
    _thread.start()


def run_async(coro: Coroutine[None, None, T]) -> T:
    """提交异步协程到后台循环，阻塞等待结果。"""
    _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


def shutdown():
    """停止后台事件循环。"""
    global _loop, _thread
    if _loop is not None:
        _loop.call_soon_threadsafe(_loop.stop)
        if _thread is not None:
            _thread.join(timeout=5)
        _loop = None
        _thread = None
