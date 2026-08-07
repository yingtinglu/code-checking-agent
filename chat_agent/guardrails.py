"""防护模块 — 输入预处理、PII 脱敏、有害内容过滤"""
import re
from typing import Optional, Tuple

from .logger import agent_log

_current_conv_id = ""


def set_conv_id(conv_id: str) -> None:
    global _current_conv_id
    _current_conv_id = conv_id

# 控制字符（保留 \n \r \t）
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# ANSI 转义序列
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# SQL 注入特征
_SQL_INJECTION_PATTERNS = [
    re.compile(r"(?:';\s*(?:DROP|ALTER|CREATE|DELETE|INSERT|UPDATE)\s)", re.IGNORECASE),
    re.compile(r"(?:OR\s+1\s*=\s*1)", re.IGNORECASE),
    re.compile(r"(?:UNION\s+SELECT)", re.IGNORECASE),
    re.compile(r"(?:;\s*--)", re.IGNORECASE),
]

# 敏感信息检测（收紧正则：需上下文关键词提示，减少对数值数据的误伤）
_ID_CARD_RE = re.compile(r"(?:身份证|ID|id_card|身份号)\s*[：:号]?\s*(\d{17}[\dXx])")
_PHONE_RE = re.compile(r"(?:手机|电话|联系方式|phone|tel)\s*[：:号]?\s*(1[3-9]\d{9})")

# 单条消息 token 上限（约 8000 tokens ≈ 12000 中文字符）
_MAX_INPUT_CHARS = 12000


def sanitize_input(text: str) -> Tuple[str, Optional[str]]:
    """输入预处理：返回 (处理后文本, 警告信息或None)。

    处理项：
    1. 控制字符清除（保留换行回车制表符）
    2. ANSI 转义序列清除
    3. 输入长度限制（超出截断）
    4. SQL 注入硬拦截
    """
    warnings = []

    # 1. 控制字符清除
    cleaned = _CONTROL_CHAR_RE.sub("", text)

    # 2. ANSI 转义序列清除
    cleaned = _ANSI_ESCAPE_RE.sub("", cleaned)

    # 3. 输入长度限制
    if len(cleaned) > _MAX_INPUT_CHARS:
        cleaned = cleaned[:_MAX_INPUT_CHARS]
        warnings.append(f"输入过长，已截断至 {_MAX_INPUT_CHARS} 字符")

    # 4. SQL 注入硬拦截
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return "", "检测到潜在 SQL 注入攻击，输入已被拦截"

    warning_str = "; ".join(warnings) if warnings else None
    return cleaned, warning_str


def detect_sensitive_info(text: str) -> Optional[str]:
    """检测输入中的敏感信息，返回警告信息或 None。"""
    findings = []
    if _ID_CARD_RE.search(text):
        findings.append("身份证号")
    if _PHONE_RE.search(text):
        findings.append("手机号")
    if findings:
        return f"检测到敏感信息（{', '.join(findings)}），是否确认发送？"
    return None


# ── 输出护栏 ──

# PII 脱敏：只替换捕获组内的数字部分，保留上下文关键词
_PII_PATTERNS = [
    (_ID_CARD_RE, r"***"),    # group(1) = 身份证号数字
    (_PHONE_RE, r"***"),      # group(1) = 手机号数字
]


def redact_pii(text: str) -> str:
    """将输出中的 PII（身份证号、手机号）替换为脱敏标记。

    只替换捕获组内的数字，保留"身份证""手机"等上下文关键词。
    例如："身份证号 110101199003071234" → "身份证号 ***"
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# 有害内容关键词（收紧：避免与正常技术用语冲突）
_HARMFUL_KEYWORDS = [
    # 暴力 — 保留完整短语，不保留单字
    "炸弹制作", "爆炸物配方",
    # 歧视
    "劣等种族",
    # 自伤 — 保留完整短语
    "自杀方法", "如何自残",
]

# 凭证泄露检测正则
_CREDENTIAL_PATTERNS = [
    # OpenAI API Key
    (re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'), '[凭证已脱敏]'),
    # 通用 Bearer Token
    (re.compile(r'\bBearer\s+[a-zA-Z0-9\-._~+/]+=*', re.IGNORECASE), 'Bearer [凭证已脱敏]'),
    # 环境变量密钥赋值
    (re.compile(r'\b(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*["\']\S+["\']', re.IGNORECASE), r'\1=[凭证已脱敏]'),
]


def redact_credentials(text: str) -> str:
    """凭证泄露防护：检测输出中的 API Key、Bearer Token、环境变量密钥赋值，替换匹配项。"""
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def filter_harmful_content(text: str) -> str:
    """有害内容过滤：命中关键词时记录警告，不替换原始内容（记录模式）。

    内部工具场景下误报代价 > 漏报代价，故仅记录不替换。
    """
    for keyword in _HARMFUL_KEYWORDS:
        if keyword in text:
            agent_log(_current_conv_id, f"内容过滤：检测到可能的有害关键词 '{keyword}'", level="ERROR")
            print(f"ERROR: 输入包含敏感内容，已过滤")
            return text
    return text
