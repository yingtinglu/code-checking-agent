"""统一提示词加载模块 — 所有 prompt 从 prompts/ 目录加载，不存在时用 fallback"""
import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def load_prompt(filename: str, fallback: str = "") -> str:
    """从 prompts/ 目录加载提示词文件，不存在则返回 fallback。"""
    path = os.path.join(_PROMPTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return fallback
