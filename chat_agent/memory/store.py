"""记忆存储 — 管理 .agent_memory/ 目录下的文件读写。

目录结构：
  current_project/.agent_memory/
    memory_index.md
    user_feedback/
      标题.md
    model_insight/
      标题.md
"""
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


def _normalize_title(title: str) -> str:
    """标题归一化：去空格、标点、转小写，用于去重比较。"""
    return re.sub(r'[^\w]', '', title).lower()


def _safe_filename(title: str) -> str:
    """将标题转为安全文件名：只保留中文、字母、数字，其余替换为下划线。"""
    return re.sub(r'[^\w]', '_', title)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _memory_dir(project_root: str) -> str:
    return os.path.join(project_root, ".agent_memory")


def _index_path(project_root: str) -> str:
    return os.path.join(_memory_dir(project_root), "memory_index.md")


def read_index(project_root: str) -> Dict[str, List[str]]:
    """读取 memory_index.md，返回 {category: [file_path, ...]}。"""
    idx_path = _index_path(project_root)
    if not os.path.isfile(idx_path):
        return {}
    result: Dict[str, List[str]] = {}
    current_category = None
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## "):
                current_category = line[3:].strip()
                if current_category not in result:
                    result[current_category] = []
            elif line.startswith("- ") and current_category is not None:
                result[current_category].append(line[2:].strip())
    return result


def write_index(project_root: str, index: Dict[str, List[str]]) -> None:
    """将索引字典写回 memory_index.md。"""
    _ensure_dir(_memory_dir(project_root))
    lines = []
    for category in sorted(index.keys()):
        lines.append(f"## {category}")
        for fp in index[category]:
            lines.append(f"- {fp}")
        lines.append("")
    with open(_index_path(project_root), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_memory_file(project_root: str, relative_path: str) -> Optional[str]:
    """读取一条记忆文件的内容。"""
    full_path = os.path.join(_memory_dir(project_root), relative_path)
    if not os.path.isfile(full_path):
        return None
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def write_memory_file(project_root: str, category: str, title: str, content: str) -> str:
    """写入一条记忆文件，返回相对路径。在 .agent_memory/{category}/ 下创建文件并在索引中追加。
    文件首行自动添加时间戳。
    """
    cat_dir = os.path.join(_memory_dir(project_root), category)
    _ensure_dir(cat_dir)
    filename = _safe_filename(title) + ".md"
    filepath = os.path.join(cat_dir, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_content = f"[saved: {timestamp}]\n{content}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(file_content)
    relative_path = f"{category}/{filename}"
    index = read_index(project_root)
    if category not in index:
        index[category] = []
    if relative_path not in index[category]:
        index[category].append(relative_path)
    write_index(project_root, index)
    return relative_path


def update_memory_file(project_root: str, relative_path: str, content: str) -> None:
    """更新已有记忆文件的内容。更新时间戳。"""
    full_path = os.path.join(_memory_dir(project_root), relative_path)
    if os.path.isfile(full_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        # 如果内容已有时间戳行则替换，否则在前面加
        if content.startswith("[saved:"):
            file_content = content
        else:
            file_content = f"[saved: {timestamp}]\n{content}"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(file_content)


def search_duplicate(project_root: str, category: str, title: str) -> Optional[str]:
    """在同 category 下查找标题 normalize 后重复的记忆，返回其相对路径，无则 None。"""
    index = read_index(project_root)
    entries = index.get(category, [])
    norm_target = _normalize_title(title)
    for entry in entries:
        entry_filename = os.path.basename(entry)
        entry_title = os.path.splitext(entry_filename)[0]
        if _normalize_title(entry_title) == norm_target:
            return entry
    return None


def _strip_timestamp(text: str) -> str:
    """去掉首行时间戳，返回纯内容。"""
    if text.startswith("[saved:"):
        lines = text.split("\n", 1)
        return lines[1] if len(lines) > 1 else ""
    return text


def merge_content(old_content: str, new_content: str) -> str:
    """合并两条记忆内容，保留内容更丰富的版本。合并后不带时间戳（由 update_memory_file 补上）。"""
    old_body = _strip_timestamp(old_content)
    new_body = _strip_timestamp(new_content)
    if len(new_body) >= len(old_body):
        return new_body
    return old_body


def get_all_memory_content(project_root: str) -> str:
    """读取所有记忆文件内容，拼成文本供 load_memory 的 LLM 选择使用。"""
    index = read_index(project_root)
    if not index:
        return ""
    parts = []
    for category in sorted(index.keys()):
        parts.append(f"## {category}")
        for fp in index[category]:
            content = read_memory_file(project_root, fp)
            if content is not None:
                parts.append(f"- {fp}: {content.strip()}")
        parts.append("")
    return "\n".join(parts)


# ── Jaccard 相似度 ──

def _tokenize(text: str) -> set:
    """将文本分词为 token 集合：中文逐字拆分，英文/数字按空格/下划线分词。"""
    tokens = set()
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            tokens.add(ch)
    for word in re.findall(r'[a-zA-Z0-9_]+', text.lower()):
        tokens.add(word)
    return tokens


def jaccard_similarity(s1: str, s2: str) -> float:
    """计算两个字符串的 Jaccard 相似度（token 集合交集/并集）。"""
    t1, t2 = _tokenize(s1), _tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


_JACCARD_THRESHOLD = 0.4  # 预筛阈值，宽松一些让 LLM 做最终判断


def search_similar_memories(
    project_root: str, category: str, content: str,
    threshold: float = _JACCARD_THRESHOLD,
) -> List[Tuple[str, float]]:
    """在同 category 下查找与 content Jaccard 相似度 >= threshold 的已有记忆。

    返回 [(相对路径, 相似度分数)]，按相似度降序排列。
    """
    index = read_index(project_root)
    entries = index.get(category, [])
    results: List[Tuple[str, float]] = []
    for entry in entries:
        old_content = read_memory_file(project_root, entry)
        if old_content is None:
            continue
        clean = _strip_timestamp(old_content)
        score = jaccard_similarity(content, clean)
        if score >= threshold:
            results.append((entry, score))
    results.sort(key=lambda x: -x[1])
    return results


# ── 48 小时硬判决 ──

def get_recent_memory_entries(project_root: str, hours: int = 48) -> List[Dict]:
    """获取指定小时内保存的记忆条目摘要（标题+路径+保存时间）。

    供 load_memory 程序硬判决使用，不依赖 LLM 判断时间。
    """
    index = read_index(project_root)
    cutoff = datetime.now() - timedelta(hours=hours)
    results: List[Dict] = []
    for category, entries in index.items():
        for entry in entries:
            content = read_memory_file(project_root, entry)
            if content is None:
                continue
            m = re.match(r'\[saved: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]', content)
            if not m:
                continue
            try:
                saved_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if saved_time >= cutoff:
                title = os.path.splitext(os.path.basename(entry))[0]
                results.append({
                    "category": category,
                    "title": title,
                    "path": entry,
                    "saved": m.group(1),
                })
    return results


# ── 主 Agent 记忆搜索 ──

def search_memory(project_root: str, query: str,
                  hours: int = 48, top_k: int = 5) -> List[Dict]:
    """48小时硬判决 + Jaccard 相似度排序，返回与 query 最相关的 top_k 条记忆。

    流程：
    1. get_recent_memory_entries 做48小时硬判决
    2. jaccard_similarity 计算每条记忆内容与 query 的相似度
    3. 按相似度降序返回 top_k 条
    """
    recent = get_recent_memory_entries(project_root, hours)
    if not recent:
        return []
    scored = []
    for entry in recent:
        content = read_memory_file(project_root, entry["path"])
        if content is None:
            continue
        clean = _strip_timestamp(content)
        score = jaccard_similarity(query, clean)
        scored.append({**entry, "content": clean, "score": score})
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]
