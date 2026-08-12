"""内部日志工具：所有调试输出写文件，不污染对话终端。"""
import os
from datetime import datetime


def agent_log(conv_id: str, msg: str, level: str = "INFO") -> None:
    """写入 data/logs/{conv_id}.log，带时间戳和级别。"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}\n"
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{conv_id}.log"), "a", encoding="utf-8") as f:
        f.write(line)
