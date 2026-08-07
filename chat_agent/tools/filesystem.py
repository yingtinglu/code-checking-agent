"""文件系统原生工具 — 文件操作 + 路径管理，路径白名单控制访问范围"""
import fnmatch
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional

# 项目根目录：chat_agent/tools/filesystem.py → tools/ → chat_agent/ → 项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 允许访问的根目录白名单 — 默认为空，必须由用户通过 add_allowed_path 显式授权
# 环境变量 FS_ALLOWED_ROOTS: 逗号或分号分隔的绝对路径列表
# 例如: export FS_ALLOWED_ROOTS="/home/user/projects;D:/code"
_extra_roots = os.environ.get("FS_ALLOWED_ROOTS", "")
ALLOWED_ROOTS: List[str] = []
if _extra_roots:
    for p in _extra_roots.replace(";", ",").split(","):
        p = p.strip()
        if p and os.path.isdir(p):
            ALLOWED_ROOTS.append(os.path.realpath(p))

# 禁止操作的目录名（即使路径在白名单内，进入这些目录仍被拒绝）
BLOCKED_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".idea", ".claude"}

# 限制常量
MAX_READ_SIZE = 100 * 1024       # read_file 最大读取 100KB
MAX_SEARCH_FILE_SIZE = 10 * 1024 * 1024  # search_files 跳过 >10MB 的文件
MAX_MATCHES = 50                 # search_files 最多返回 50 条匹配
MAX_LIST_RESULTS = 100           # list_files 硬性上限：无论 max_results 传多大，最多返回 100 条


def validate_path(user_path: str) -> str:
    """校验路径是否在白名单目录内，返回绝对路径或抛 ValueError。

    支持两种输入：
    - 相对路径：依次尝试在项目根目录、以及每个已授权根目录下解析。
      list_files 返回的相对路径是相对授权目录的，必须在此处兼容，
      否则 LLM 照抄相对路径调 read_file 会被拦，导致检索链断裂。
    - 绝对路径：直接使用，但必须位于某个白名单根目录下

    安全：无论相对基准是谁，最终绝对路径必须落在某个白名单根目录内
    （os.path.realpath 会解析 .. 越界，由 commonpath 校验兜底拦截）。
    """
    if os.path.isabs(user_path):
        candidates = [os.path.realpath(user_path)]
    else:
        # 相对路径：先试项目根（兼容旧用法），再试每个授权根目录（兼容 list_files 输出）
        candidates = [os.path.realpath(os.path.join(PROJECT_ROOT, user_path))]
        for allowed_root in ALLOWED_ROOTS:
            candidates.append(os.path.realpath(os.path.join(allowed_root, user_path)))

    # 检查每个候选路径是否落在某个允许的根目录下
    for abs_path in candidates:
        for allowed_root in ALLOWED_ROOTS:
            real_root = os.path.realpath(allowed_root)
            try:
                common = os.path.commonpath([real_root, abs_path])
            except ValueError:
                continue  # 不同盘符（Windows），跳过
            if common == real_root:
                return abs_path

    # 构造友好的错误信息
    if ALLOWED_ROOTS:
        roots_str = ", ".join(ALLOWED_ROOTS)
        hint = f"允许的根目录: [{roots_str}]。"
    else:
        hint = "当前没有已授权的目录。"
    raise ValueError(
        f"安全错误: 路径 '{user_path}' 不在允许的目录范围内。"
        f"{hint}"
        f"用户未明确要求操作文件时，不要尝试读取或列出项目目录内容。"
        f"如需访问，请先调用 add_allowed_path 添加授权目录。"
    )


def is_path_blocked(abs_path: str) -> bool:
    """检查路径是否在受保护目录内（如 .git、.venv）。"""
    for allowed_root in ALLOWED_ROOTS:
        try:
            rel = os.path.relpath(abs_path, allowed_root)
        except ValueError:
            continue
        # rel 不以 .. 开头说明在白名单目录内
        if not rel.startswith(".."):
            parts = rel.replace("\\", "/").split("/")
            return any(part in BLOCKED_DIRS for part in parts)
    return True


def is_binary(file_path: str) -> bool:
    """检测文件是否为二进制（读前 8KB 检查 null 字节）。"""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (IOError, OSError):
        return False


def _is_overly_broad_path(abs_path: str) -> bool:
    """拒绝过宽路径：盘符根目录、Unix 根目录、用户主目录。"""
    real = os.path.realpath(abs_path)
    # Windows 盘符根目录 (C:\, D:\ 等)
    drive = os.path.splitdrive(real)[0]
    if drive and real in (drive + "\\", drive + "/"):
        return True
    # Unix 根目录
    if real == "/":
        return True
    # 用户主目录
    home = os.path.expanduser("~")
    if os.path.realpath(home) == real:
        return True
    return False


def _get_display_path(abs_path: str) -> str:
    """根据匹配的 allowed root 生成显示用相对路径。"""
    for root in ALLOWED_ROOTS:
        real_root = os.path.realpath(root)
        try:
            rel = os.path.relpath(abs_path, real_root)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    return abs_path


# ========== 工具实现 ==========

def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """读取文件内容，带行号输出。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许访问此目录(.git, .venv 等)"

    if not os.path.isfile(abs_path):
        return f"错误: 文件不存在: {path}"

    if is_binary(abs_path):
        return f"错误: 文件 '{path}' 是二进制文件，不支持读取"

    file_size = os.path.getsize(abs_path)
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total_lines = len(lines)
    selected = lines[offset: offset + limit]

    # 按体积截断
    result_lines = []
    current_size = 0
    for i, line in enumerate(selected, start=offset):
        encoded_len = len(line.encode("utf-8"))
        if current_size + encoded_len > MAX_READ_SIZE:
            result_lines.append(f"\n... [文件过大，已在 {current_size} 字节处截断]")
            break
        result_lines.append(f"{i}: {line.rstrip()}")
        current_size += encoded_len

    header = f"文件: {path} ({total_lines} 行, {file_size} 字节)"
    if offset > 0 or limit < total_lines:
        end_line = min(offset + len(selected), total_lines) - 1
        header += f" [显示 {offset}-{end_line} 行]"

    return header + "\n" + "\n".join(result_lines)


def write_file(path: str, content: str) -> str:
    """写入文件，自动创建父目录。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许写入此目录"

    parent = os.path.dirname(abs_path)
    os.makedirs(parent, exist_ok=True)

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    size = len(content.encode("utf-8"))
    return f"成功写入 {path} ({size} 字节)"


def list_directory(path: str = ".") -> str:
    """列出目录内容，显示类型、大小、修改时间。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许列出此目录"

    if not os.path.isdir(abs_path):
        return f"错误: 目录不存在: {path}"

    entries = []
    for entry in sorted(os.listdir(abs_path)):
        if entry in BLOCKED_DIRS:
            continue
        full = os.path.join(abs_path, entry)
        try:
            if os.path.isdir(full):
                entries.append(f"  [DIR]  {entry}/")
            elif os.path.isfile(full):
                size = os.path.getsize(full)
                mtime = os.path.getmtime(full)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                entries.append(f"  [FILE] {entry}  ({size} bytes, {mtime_str})")
            else:
                entries.append(f"  [OTHER] {entry}")
        except (PermissionError, OSError):
            entries.append(f"  [???]  {entry}")

    header = f"目录: {path or '.'} ({len(entries)} 项)"
    if not entries:
        return header + "\n  (空目录)"
    return header + "\n" + "\n".join(entries)


def search_files(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    """递归搜索文件内容（类似 grep），支持正则和文件名 glob 过滤。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许搜索此目录"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"错误: 无效正则表达式: {e}"

    matches = []

    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in BLOCKED_DIRS]

        for fname in files:
            if not fnmatch.fnmatch(fname, file_pattern):
                continue
            fpath = os.path.join(root, fname)
            try:
                if os.path.getsize(fpath) > MAX_SEARCH_FILE_SIZE:
                    continue
                if is_binary(fpath):
                    continue
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = _get_display_path(fpath)
                            matches.append(f"{rel}:{line_no}: {line.rstrip()}")
                            if len(matches) >= MAX_MATCHES:
                                matches.append(f"\n... [结果过多，已截断，共显示 {MAX_MATCHES} 条]")
                                return "\n".join(matches)
            except (PermissionError, OSError):
                continue

    if not matches:
        return f"未找到匹配 '{pattern}' 的内容"
    return "\n".join(matches)


def create_directory(path: str) -> str:
    """递归创建目录（类似 mkdir -p）。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许在受保护目录下创建目录"

    os.makedirs(abs_path, exist_ok=True)
    return f"成功创建目录: {path}"


def delete_file(path: str) -> str:
    """删除文件或空目录。禁止删除已授权根目录本身。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许删除受保护目录中的文件"

    real_abs = os.path.realpath(abs_path)
    for root in ALLOWED_ROOTS:
        if real_abs == os.path.realpath(root):
            return "错误: 不允许删除已授权的根目录本身"

    if os.path.isfile(abs_path):
        os.remove(abs_path)
        return f"成功删除文件: {path}"
    elif os.path.isdir(abs_path):
        if os.listdir(abs_path):
            return f"错误: 目录 '{path}' 不为空，不能删除"
        os.rmdir(abs_path)
        return f"成功删除空目录: {path}"
    else:
        return f"错误: 路径不存在: {path}"


def move_file(source: str, destination: str) -> str:
    """移动或重命名文件/目录。"""
    src_abs = validate_path(source)
    dst_abs = validate_path(destination)
    if is_path_blocked(src_abs) or is_path_blocked(dst_abs):
        return "错误: 不允许移动受保护目录中的文件"

    if not os.path.exists(src_abs):
        return f"错误: 源路径不存在: {source}"

    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    shutil.move(src_abs, dst_abs)
    return f"成功移动: {source} -> {destination}"


def copy_file(source: str, destination: str) -> str:
    """复制文件，保留元数据。自动创建目标父目录。"""
    src_abs = validate_path(source)
    dst_abs = validate_path(destination)
    if is_path_blocked(src_abs) or is_path_blocked(dst_abs):
        return "错误: 不允许复制受保护目录中的文件"

    if not os.path.isfile(src_abs):
        return f"错误: 源文件不存在或不是文件: {source}"

    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    shutil.copy2(src_abs, dst_abs)
    return f"成功复制: {source} -> {destination}"


def find_files(pattern: str, path: str = ".") -> str:
    """按文件名模式递归查找文件和目录（类似 find）。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许搜索此目录"

    if not os.path.isdir(abs_path):
        return f"错误: 目录不存在: {path}"

    results = []
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in BLOCKED_DIRS]
        # 匹配目录名
        for d in dirs:
            if fnmatch.fnmatch(d, pattern):
                full = os.path.join(root, d)
                rel = _get_display_path(full)
                results.append(f"[DIR]  {rel}/")
        # 匹配文件名
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                full = os.path.join(root, f)
                rel = _get_display_path(full)
                try:
                    size = os.path.getsize(full)
                    results.append(f"[FILE] {rel}  ({size} bytes)")
                except OSError:
                    results.append(f"[FILE] {rel}")
        if len(results) >= MAX_MATCHES:
            results.append(f"\n... [结果过多，已截断，共显示 {MAX_MATCHES} 条]")
            break

    if not results:
        return f"未找到匹配 '{pattern}' 的文件或目录"
    header = f"查找 '{pattern}' 在 {path} 下 ({len([r for r in results if not r.startswith('...')])} 项)"
    return header + "\n" + "\n".join(results)


def list_files(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """按文件名 glob 递归列出候选文件清单（含路径+大小+行数），供 grep/read_file 后续定位。

    职责是"缩小检索范围的第一步"：把检索从全仓收窄到相关文件列表。
    硬性上限 MAX_LIST_RESULTS=100：无论 max_results 传多大，最多返回 100 条，防上下文撑爆。
    """
    # 硬性上限：LLM 传大于 100 的值也截到 100
    max_results = min(int(max_results or 100), MAX_LIST_RESULTS)
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许搜索此目录"
    if not os.path.isdir(abs_path):
        return f"错误: 目录不存在: {path}"

    results = []  # (size, line_count, rel_path)
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in BLOCKED_DIRS]
        for fname in files:
            if not fnmatch.fnmatch(fname, pattern):
                continue
            fpath = os.path.join(root, fname)
            try:
                if is_binary(fpath):
                    continue
                size = os.path.getsize(fpath)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    line_count = sum(1 for _ in f)
            except (PermissionError, OSError):
                continue
            rel = _get_display_path(fpath)
            results.append((size, line_count, rel))
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    total = len(results)
    header = f"查找 '{pattern}' 在 {path} 下"
    if total == 0:
        return header + "\n未找到匹配的文件"
    truncated = len(results) >= max_results
    lines = [header + (f" (前 {max_results} 项)" if truncated else "")]
    for size, lc, rel in sorted(results, key=lambda x: -x[0]):
        lines.append(f"  {rel}  ({size:,} 字节, {lc} 行)")
    return "\n".join(lines)


def grep(pattern: str, files: List[str], context: int = 0, max_matches: int = 50) -> str:
    """在指定文件列表内搜索正则，输出 path:line:content（按文件分组，命中行用 > 标记）。

    files 必填——搜索范围由调用方控制，不做全仓递归，避免返回爆炸。
    context>0 时命中行前后各显示 context 行，上下文行用空格前缀、命中行用 > 前缀。
    per-file max_matches 上限（context 行不计入），无全局总上限。
    """
    if not files:
        return "错误: files 参数必填，请先用 list_files 找候选文件再搜索"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"错误: 无效正则表达式: {e}"

    if context < 0:
        context = 0

    sections = []
    total_hits = 0
    skipped = []

    for fpath in files:
        try:
            abs_path = validate_path(fpath)
        except ValueError as e:
            skipped.append(f"{fpath}: 路径不在白名单")
            continue
        if is_path_blocked(abs_path):
            skipped.append(f"{fpath}: 受保护目录")
            continue
        if not os.path.isfile(abs_path):
            skipped.append(f"{fpath}: 文件不存在")
            continue
        if is_binary(abs_path):
            skipped.append(f"{fpath}: 二进制文件")
            continue
        try:
            if os.path.getsize(abs_path) > MAX_SEARCH_FILE_SIZE:
                skipped.append(f"{fpath}: 文件过大(>{MAX_SEARCH_FILE_SIZE // 1024 // 1024}MB)")
                continue
        except OSError:
            pass

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except (PermissionError, OSError) as e:
            skipped.append(f"{fpath}: 读取失败 {e}")
            continue

        rel = _get_display_path(abs_path)
        file_hits = 0
        file_lines_out = []
        for line_no, line in enumerate(all_lines, 1):
            if not regex.search(line):
                continue
            file_hits += 1
            total_hits += 1
            if file_hits <= max_matches:
                if context > 0:
                    start = max(0, line_no - 1 - context)
                    end = min(len(all_lines), line_no + context)
                    for j in range(start, end):
                        ln = j + 1
                        content = all_lines[j].rstrip("\n")
                        if ln == line_no:
                            file_lines_out.append(f"> {ln}: {content}")
                        else:
                            file_lines_out.append(f"  {ln}: {content}")
                else:
                    file_lines_out.append(f"{rel}:{line_no}: {line.rstrip()}")
            if file_hits >= max_matches:
                break

        sec = [f"== {rel} =="]
        if context > 0:
            sec.append(f"  (命中 {file_hits} 处{'，已截断显示前 ' + str(max_matches) if file_hits >= max_matches else ''})")
        sec.extend(file_lines_out)
        if file_hits >= max_matches:
            remaining = file_hits - max_matches
            if remaining > 0:
                sec.append(f"  (该文件共 {file_hits} 处命中，仅显示前 {max_matches} 处)")
        sections.append("\n".join(sec))

    parts = [f"搜索 '{pattern}' 在 {len(files)} 个文件中 (共 {total_hits} 处命中)"]
    parts.extend(sections)
    if skipped:
        parts.append("跳过文件:")
        parts.extend(f"  {s}" for s in skipped)
    return "\n\n".join(parts)


def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """精准替换文件中的文本片段。先读取文件，替换指定内容后写回。"""
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许编辑此目录中的文件"

    if not os.path.isfile(abs_path):
        return f"错误: 文件不存在: {path}"

    if is_binary(abs_path):
        return f"错误: 文件 '{path}' 是二进制文件，不支持编辑"

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    count = content.count(old_text)
    if count == 0:
        return f"未找到要替换的内容。文件共 {len(content.splitlines())} 行，请检查 old_text 是否精确匹配。"

    if count > 1 and not replace_all:
        # 多处匹配但未指定全部替换，显示上下文帮助用户精确定位
        lines = content.splitlines(True)
        contexts = []
        pos = 0
        for i, line in enumerate(lines):
            if old_text in line:
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                ctx = "".join(f"{j}: {lines[j].rstrip()}" for j in range(start, end))
                contexts.append(f"第 {i} 行附近:\n{ctx}")
                if len(contexts) >= 3:
                    break
        return (
            f"找到 {count} 处匹配。如需替换所有，请设置 replace_all=true。\n"
            f"匹配位置:\n" + "\n---\n".join(contexts)
        )

    if replace_all:
        new_content = content.replace(old_text, new_text)
        actual = count
    else:
        new_content = content.replace(old_text, new_text, 1)
        actual = 1

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    old_lines = len(content.splitlines())
    new_lines = len(new_content.splitlines())
    return f"成功替换 {path}: {actual} 处匹配已替换 ({old_lines} → {new_lines} 行)"


def project_stats(path: str = ".", max_depth: int = 3, top_n: int = 20) -> str:
    """一次性返回代码库结构统计，避免逐个 find_files + read_file 摸结构。

    返回：按扩展名计数、top-N 最大文件（路径+大小+行数）、目录树、main/test 文件数比。
    """
    abs_path = validate_path(path)
    if is_path_blocked(abs_path):
        return "错误: 不允许统计此目录"
    if not os.path.isdir(abs_path):
        return f"错误: 目录不存在: {path}"

    ext_counts: Dict[str, list] = {}  # ext -> [count, total_bytes]
    file_list = []  # (size, rel_path, ext)
    main_count = 0
    test_count = 0

    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in BLOCKED_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if is_binary(fpath):
                continue
            rel = _get_display_path(fpath)
            ext = os.path.splitext(fname)[1].lower() or "(无扩展)"
            ext_counts.setdefault(ext, [0, 0])
            ext_counts[ext][0] += 1
            ext_counts[ext][1] += size
            file_list.append((size, rel, ext))
            if "/src/main/" in rel.replace("\\", "/"):
                main_count += 1
            if "/src/test/" in rel.replace("\\", "/"):
                test_count += 1

    # 扩展名统计（按总字节降序，取前 15）
    ext_lines = []
    for ext, (cnt, total) in sorted(ext_counts.items(), key=lambda x: -x[1][1])[:15]:
        ext_lines.append(f"  {ext}: {cnt} 文件, {total:,} 字节")

    # top-N 最大文件
    file_list.sort(reverse=True)
    top_lines = []
    for size, rel, ext in file_list[:top_n]:
        line_count = ""
        try:
            with open(os.path.join(abs_path, rel), "r", encoding="utf-8", errors="replace") as f:
                line_count = f"{sum(1 for _ in f)} 行"
        except OSError:
            pass
        top_lines.append(f"  {size:>10,} 字节  {line_count:>10}  {rel}")

    # 目录树（限深，只列目录名 + 直接子项数）
    tree_lines = _dir_tree(abs_path, max_depth)

    parts = [
        f"统计: {path}",
        f"文件总数: {len(file_list)}  main: {main_count}  test: {test_count}",
        "",
        "按扩展名（前15，按总字节降序）:",
        *ext_lines,
        "",
        f"最大文件 top-{top_n}:",
        *top_lines,
        "",
        f"目录树（深 {max_depth}）:",
        *tree_lines,
    ]
    return "\n".join(parts)


def _dir_tree(root: str, max_depth: int) -> List[str]:
    """生成限深目录树，每行: 缩进 + 目录名 + (N 子项)。"""
    lines = []
    root_name = os.path.basename(root) or root

    def walk(dir_path: str, depth: int, prefix: str):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return
        dirs = [e for e in entries if e not in BLOCKED_DIRS and os.path.isdir(os.path.join(dir_path, e))]
        for d in dirs:
            full = os.path.join(dir_path, d)
            try:
                child_count = len([e for e in os.listdir(full) if e not in BLOCKED_DIRS])
            except OSError:
                child_count = 0
            lines.append(f"{prefix}{d}/ ({child_count} 项)")
            walk(full, depth + 1, prefix + "  ")

    lines.append(f"{root_name}/")
    walk(root, 1, "  ")
    return lines


def _validate_bash_command(command: str, cwd: str) -> tuple:
    """校验命令安全性（Windows 走 powershell 只读 cmdlet）。返回 (ok, 错误信息)。

    - 危险 cmdlet/子串硬禁：变更、网络、任意代码、重定向
    - cwd 必须是已授权根目录
    """
    if not command or not command.strip():
        return False, "错误: 空命令"
    if not cwd:
        return False, "错误: bash 需指定 cwd（须为已授权目录）"

    # cwd 必须在白名单内
    cwd_abs = os.path.realpath(cwd)
    in_whitelist = False
    for root in ALLOWED_ROOTS:
        try:
            if os.path.realpath(root) == cwd_abs or os.path.commonpath([os.path.realpath(root), cwd_abs]) == os.path.realpath(root):
                in_whitelist = True
                break
        except ValueError:
            continue
    if not in_whitelist:
        return False, f"错误: cwd '{cwd}' 不在已授权目录内，先调 add_allowed_path"

    # 危险 cmdlet 与子串硬禁（powershell 变更/网络/任意代码/重定向）
    # 注意：Write-Output / Write-Host 是只读输出 cmdlet，不禁（LLM 用它打印汇总统计）
    _HARD_BLOCKED = (
        # 变更类 cmdlet
        "Remove-Item", "Set-Content", "Add-Content", "Out-File", "New-Item",
        "Move-Item", "Copy-Item", "Clear-Content", "Clear-Item", "Clear-ItemProperty",
        "Set-Item", "Set-ItemProperty", "Rename-Item", "Write-AllBytes",
        # 网络
        "Invoke-WebRequest", "Invoke-RestMethod", "Start-BitsTransfer", "Net.WebClient",
        "DownloadFile", "DownloadString",
        # 任意代码 / 调用
        "Invoke-Expression", " iex ", "Start-Process", "Start-Job",
        # 外部解释器
        "python", "node", "cmd ", "powershell ", "pwsh ", "bash ", "sh ",
        # git 变更
        "git reset", "git checkout", "git rm", "git push", "git rebase", "git merge",
        "git commit", "git stash", "git clean",
    )
    low = " " + command + " "
    for marker in _HARD_BLOCKED:
        if marker.lower() in low.lower():
            return False, f"错误: 禁止的操作: '{marker.strip()}'（仅限只读分析）"

    # 重定向检测：只禁明确的写文件重定向，避免误伤 "Files >500 lines" 这类字符串字面量。
    # - >> / 1> / 2> / 2>&1 / 1>&2 带数字流前缀或双 >，是明确重定向
    # - > 后跟 .txt/.log/.csv 等文件扩展名，是重定向写文件
    # - 普通裸 > 不禁（>500 这类比较/字符串常见，重定向写文件需配合 Out-File/Set-Content 已禁）
    import re as _re
    _REDIRECT_RE = _re.compile(r"(?<!\w)[12](?:>&[12]|>)|(?<![<\w-])>>\s*\S|(?<![<\w-])>\s*\S+\.(?:txt|log|csv|json|xml|tmp|out)\b")
    if _REDIRECT_RE.search(command):
        return False, "错误: 禁止重定向（>> 1> 2> 2>&1），仅限只读分析"
    # & 调用操作符（& "path" 或 & {script}）用于执行任意命令，禁；但不误伤合法 &
    if _re.search(r"(?<!\w)&\s*[\"{]", command):
        return False, "错误: 禁止 & 调用操作符执行任意命令，仅限只读分析"

    # 只读 cmdlet 白名单（命令里出现的 cmdlet 必须都在此集合）
    _ALLOWED_CMDLETS = {
        # 探测
        "Get-ChildItem", "Get-Item", "Get-Location", "Test-Path", "Resolve-Path",
        "Get-Content", "Get-ItemProperty", "Get-Acl", "Get-FileInfo",
        # 搜索 / 统计
        "Select-String", "Measure-Object", "Group-Object", "Select-Object",
        "Sort-Object", "Where-Object", "ForEach-Object", "Compare-Object",
        # 格式化 / 输出（只读，写 stdout 不改文件）
        "Format-Table", "Format-List", "Format-Custom", "ConvertTo-Csv",
        "Write-Output", "Write-Host",
        # git 只读
        "git",
    }
    # 提取命令中所有 cmdlet（形如 Xxx-Yyy 的 token）
    import re as _re
    cmdlet_re = _re.compile(r"\b[A-Z][a-z]+-[A-Z][a-zA-Z]+\b")
    found_cmdlets = set(cmdlet_re.findall(command))
    disallowed = found_cmdlets - _ALLOWED_CMDLETS
    if disallowed:
        return False, f"错误: cmdlet {sorted(disallowed)} 不在白名单（仅限只读分析 cmdlet）"

    return True, ""


def bash(command: str, cwd: str, timeout: int = 30) -> str:
    """执行只读命令做代码分析（Windows 走 powershell，用 Select-String/Get-ChildItem/Measure-Object 等）。

    安全：危险 cmdlet 硬禁 + 只读 cmdlet 白名单 + cwd 须在已授权目录。
    输出截断到 8000 字符（头 6000 + 尾 2000）。
    """
    ok, err = _validate_bash_command(command, cwd)
    if not ok:
        return err

    import subprocess
    # Windows 用 powershell 执行；非 Windows 退回 shell=True
    # capture_output 不分配真实终端，powershell Format-Table 退回 80 字符宽度，
    # 长路径表格被截断成乱码（LLM 误判统计失败）。执行前设宽缓冲兜底。
    try:
        if os.name == "nt":
            # 强制 powershell 按 UTF-8 输出，避免中文文件名/内容被默认 GBK 编码后
            # subprocess 用 GBK 解码失败（UnicodeDecodeError in _readerthread）
            _PRELUDE = (
                "$ErrorActionPreference='SilentlyContinue';"
                "try{[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                "$OutputEncoding=[System.Text.Encoding]::UTF8;}catch{};"
                "try{$host.UI.RawUI.BufferSize="
                "New-Object System.Management.Automation.Host.Size(4096,5000)}catch{};"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", _PRELUDE + command],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        else:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return f"错误: 命令超时（{timeout}s）"
    except Exception as e:
        return f"错误: 执行失败 — {e}"

    output = proc.stdout
    if proc.stderr:
        output += ("\n[stderr]\n" if output else "") + proc.stderr

    if len(output) > 8000:
        output = output[:6000] + f"\n... [输出过长，截断，共 {len(output)} 字符] ...\n" + output[-2000:]
    return output or "(无输出)"


# ========== 路径管理工具 ==========

def add_allowed_path(path: str) -> str:
    """将一个目录添加到允许访问的白名单中。只保留一条，添加前先清空已有目录。"""
    abs_path = os.path.realpath(path)

    if not os.path.isdir(abs_path):
        return f"错误: 路径不存在或不是目录: {path}"

    if _is_overly_broad_path(abs_path):
        return "错误: 不允许添加过宽路径（盘符根目录、用户主目录等）作为允许目录"

    dirname = os.path.basename(abs_path)
    if dirname in BLOCKED_DIRS:
        return f"错误: 不允许添加受保护目录: {dirname}"

    real_allowed = [os.path.realpath(r) for r in ALLOWED_ROOTS]
    if abs_path in real_allowed and len(ALLOWED_ROOTS) == 1:
        return f"该目录已在允许列表中: {path}"

    # 清空已有，只保留一条
    ALLOWED_ROOTS.clear()
    ALLOWED_ROOTS.append(abs_path)
    return f"已添加允许目录: {path}\n当前允许的目录: {list_allowed_paths()}"


def list_allowed_paths() -> str:
    """列出当前所有允许访问的根目录。"""
    lines = []
    if not ALLOWED_ROOTS:
        return "允许访问的目录: (空 — 尚未授权任何目录，需先调用 add_allowed_path)"
    for i, root in enumerate(ALLOWED_ROOTS, 1):
        lines.append(f"  {i}. {root}")
    return "允许访问的目录:\n" + "\n".join(lines)


def remove_allowed_path(path: str) -> str:
    """从允许访问的白名单中移除一个目录。"""
    abs_path = os.path.realpath(path)

    for i, root in enumerate(ALLOWED_ROOTS):
        if os.path.realpath(root) == abs_path:
            del ALLOWED_ROOTS[i]
            return f"已移除允许目录: {path}\n当前允许的目录: {list_allowed_paths()}"

    return f"错误: 该目录不在允许列表中: {path}"


# ========== 注册函数 ==========

_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "读取文件内容，带行号输出。支持行号范围和分页读取。自动检测并拒绝二进制文件。大文件(>100KB)会被截断。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，如 'main.py'（相对路径）或 'D:/code/MyProject/main.py'（已授权目录下的绝对路径）"},
                "offset": {"type": "integer", "description": "起始行号(从0开始)，可选，默认0"},
                "limit": {"type": "integer", "description": "最多读取行数，可选，默认2000"},
            },
            "required": ["path"],
        },
        "handler": read_file,
        "archive_policy": "inline",  # 源在磁盘可重读，结果不落盘，存投影
    },
    {
        "name": "write_file",
        "description": "写入文件内容。自动创建父目录。覆盖已存在的文件。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，相对路径或已授权目录下的绝对路径"},
                "content": {"type": "string", "description": "要写入的文件内容"},
            },
            "required": ["path", "content"],
        },
        "handler": write_file,
    },
    {
        "name": "list_directory",
        "description": "列出目录内容，显示文件类型(DIR/FILE)、大小和修改时间。自动跳过.git等受保护目录。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。注意：未授权时不要用此工具窥探项目目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，须为已授权目录。支持相对路径或已授权目录下的绝对路径"},
            },
            "required": [],
        },
        "handler": list_directory,
    },
    {
        "name": "create_directory",
        "description": "递归创建目录（类似 mkdir -p）。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，相对路径或已授权目录下的绝对路径"},
            },
            "required": ["path"],
        },
        "handler": create_directory,
    },
    {
        "name": "delete_file",
        "description": "删除文件或空目录。目录必须为空才能删除。禁止删除项目根目录。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件或空目录路径，相对路径或已授权目录下的绝对路径"},
            },
            "required": ["path"],
        },
        "handler": delete_file,
    },
    {
        "name": "move_file",
        "description": "移动或重命名文件或目录。自动创建目标父目录。源和目标路径均支持相对路径（相对于项目根目录）或已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "源路径，相对路径或已授权目录下的绝对路径"},
                "destination": {"type": "string", "description": "目标路径，相对路径或已授权目录下的绝对路径"},
            },
            "required": ["source", "destination"],
        },
        "handler": move_file,
    },
    {
        "name": "copy_file",
        "description": "复制文件，保留元数据。自动创建目标父目录。源和目标路径均支持相对路径（相对于项目根目录）或已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "源文件路径，相对路径或已授权目录下的绝对路径"},
                "destination": {"type": "string", "description": "目标文件路径，相对路径或已授权目录下的绝对路径"},
            },
            "required": ["source", "destination"],
        },
        "handler": copy_file,
    },
    {
        "name": "add_allowed_path",
        "description": "将一个目录添加到文件工具的允许访问白名单中。当用户要求操作某个文件夹下的文件时，如果该文件夹不在默认允许目录中，必须先调用此工具添加，然后才能使用其他文件工具操作该目录。不允许添加过宽路径（如盘符根目录C:\\、用户主目录等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要添加的目录绝对路径，如 'D:/code/MyProject'"},
            },
            "required": ["path"],
        },
        "handler": add_allowed_path,
    },
    {
        "name": "list_allowed_paths",
        "description": "列出当前所有允许文件工具访问的根目录。用于确认哪些目录已被授权。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": list_allowed_paths,
    },
    {
        "name": "remove_allowed_path",
        "description": "从允许访问的白名单中移除一个目录（不允许移除项目根目录）。当用户不再需要操作某个文件夹时可调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要移除的目录绝对路径"},
            },
            "required": ["path"],
        },
        "handler": remove_allowed_path,
    },
    {
        "name": "edit_file",
        "description": "精准替换文件中的文本片段。指定 old_text 和 new_text，将文件中匹配的文本替换为新内容后写回。默认只替换第一处匹配；如需替换所有匹配，设置 replace_all=true。当有多处匹配但未设置 replace_all 时，会显示匹配位置帮助精确定位。路径可以是相对路径（相对于项目根目录），也可以是已授权目录下的绝对路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，相对路径或已授权目录下的绝对路径"},
                "old_text": {"type": "string", "description": "要替换的原始文本（必须精确匹配，包括缩进和空格）"},
                "new_text": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {"type": "boolean", "description": "是否替换所有匹配，默认false只替换第一处"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        "handler": edit_file,
    },
    {
        "name": "list_files",
        "description": "按文件名 glob 递归列出候选文件清单（含路径+大小+行数），用于缩小检索范围、为 grep 提供候选文件列表。支持 shell 通配符（如 *Mapper.java、*.py、*Controller*）。分析本地代码仓的第一步：先 list_files 收窄到相关文件，再 grep 搜关键字。不要用它做全仓逐个读取。路径须为已授权目录（先调 add_allowed_path）。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "文件名 glob，如 '*Mapper.java'、'*.py'、'*Controller*'"},
                "path": {"type": "string", "description": "搜索起始目录，须为已授权目录。默认 '.'（项目根）"},
                "max_results": {"type": "integer", "description": "最多返回文件数，默认100"},
            },
            "required": ["pattern"],
        },
        "handler": list_files,
        "archive_policy": "inline",  # 文件清单是定位信息非原文，不落盘存投影；reducer 全留路径供 grep 复用
    },
    {
        "name": "grep",
        "description": "在指定文件列表内搜索正则，输出 path:line:content 格式（按文件分组，命中行用 > 标记，可选上下文行）。files 必填——搜索范围由调用方控制，不做全仓递归。典型用法：先 list_files 找候选文件，再把路径列表传给本工具的 files 搜关键字（如 @Autowired、extends\\s\\w+、TODO|FIXME）。context 控制每个命中行前后显示的上下文行数（0=只命中行）。per-file max_matches 控制每文件命中上限（默认50）。路径须在白名单内。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式，如 '@Autowired'、'extends\\s\\w+'、'TODO|FIXME'"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要搜索的文件列表（相对或绝对路径，须在白名单内）。必填，控制搜索范围",
                },
                "context": {"type": "integer", "description": "每个命中行前后显示的上下文行数，默认0（只命中行）"},
                "max_matches": {"type": "integer", "description": "每个文件最多返回的命中数（context行不计入），默认50"},
            },
            "required": ["pattern", "files"],
        },
        "handler": grep,
        "archive_policy": "archive",  # 结果可能是大列表，大结果落盘，上下文放投影
    },
    {
        "name": "project_stats",
        "description": "一次性返回代码库结构统计（按扩展名计数、top-N 最大文件含行数、目录树、main/test 文件数比），避免逐个 find_files + read_file 摸结构。分析代码质量时优先调此工具拿到全貌，再按大小排序只深读 top-5 最大文件 + 2-3 个代表性文件，其余只统计不读。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "统计的起始目录，须为已授权目录。支持相对路径或已授权目录下的绝对路径"},
                "max_depth": {"type": "integer", "description": "目录树最大深度，默认3"},
                "top_n": {"type": "integer", "description": "最大文件返回数量，默认20"},
            },
            "required": [],
        },
        "handler": project_stats,
    },
    {
        "name": "bash",
        "description": "执行只读命令做代码分析。Windows 走 powershell，用 Select-String(≈grep)/Get-ChildItem(≈find)/Measure-Object(≈wc)/Group-Object(≈uniq -c)/Sort-Object/Where-Object/ForEach-Object 等只读 cmdlet。禁止变更/网络/任意代码 cmdlet(Remove-Item/Set-Content/Out-File/New-Item/Move-Item/Copy-Item/Invoke-WebRequest/Invoke-Expression/Start-Process 等)及重定向(> >>)。cwd 必须是已授权目录。输出超 8000 字符自动截断。用于跑真静态分析（import 热点、大文件分布、循环依赖线索、DI 密度等），比逐个 read_file 准确。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "只读分析命令（powershell 语法）。如 \"Get-ChildItem -Path src -Recurse -Filter *.java | Select-String -Pattern '@Autowired' | Measure-Object | Select-Object Count\""},
                "cwd": {"type": "string", "description": "执行目录，必须是已授权目录（先调 add_allowed_path）"},
                "timeout": {"type": "integer", "description": "超时秒数，默认30"},
            },
            "required": ["command", "cwd"],
        },
        "handler": bash,
    },
    {
        "name": "read_artifact",
        "description": "读回此前工具调用结果的原件（L1 存档）。仅用于取回不可廉价重算工具（bash 统计、search_files 大仓 grep、远程 MCP 查询）的历史快照原文。read_file 的结果请直接重新调用 read_file(path, offset) 重读，不要用本工具。path 可从历史 tool 结果投影末尾的 [原文可取 read_artifact(path=...)] 标记找到。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "存档文件路径（落盘大结果投影里 [原文可取 read_artifact(path=...)] 给出的路径）"},
            },
            "required": ["path"],
        },
        "handler": None,  # 见下：handler 在注册时从 context_manager.artifacts 装配
        "archive_policy": "archive_ref",  # 取回存档原文给 LLM 看：上下文放原文，存储只存 [存档引用] path，不落盘
    },
]


def _search_memory_handler(project_path: str, query: str) -> str:
    """执行 search_memory 工具：48小时硬判决 + Jaccard 相似度排序。"""
    from ..memory import store
    results = store.search_memory(project_path, query)
    if not results:
        return "未找到48小时内与当前任务相关的记忆"
    parts = []
    for r in results:
        parts.append(
            f"## {r['title']} (分类: {r['category']}, 相似度: {r['score']:.0%}, "
            f"保存于 {r['saved']})\n{r['content']}"
        )
    return "\n\n".join(parts)


# search_memory 工具定义（handler 须在列表之前定义）
_SEARCH_MEMORY_DEF = {
    "name": "search_memory",
    "description":
        "搜索与当前任务相关的项目记忆。"
        "根据项目根目录下 .agent_memory/memory_index.md 索引中的每条记忆文件，"
        "先筛选48小时内保存的记忆条目（程序硬判决，读取文件头的 [saved: YYYY-MM-DD HH:MM] 判断时间），"
        "再通过 Jaccard 相似度计算每条记忆文件内容与查询文本的相似程度，"
        "返回相似度最高的最多5条记忆的完整内容。"
        "调用前请先判断当前对话涉及的项目路径。",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "项目根目录的绝对路径"
            },
            "query": {
                "type": "string",
                "description": "用户当前任务或问题的描述，用于与记忆内容做相似度匹配"
            }
        },
        "required": ["project_path", "query"]
    },
    "handler": _search_memory_handler,
    "archive_policy": "no_archive_raw",
}


# ========== Reducer（历史轮次 tool result 投影）==========

# 匹配 read_file 带行号输出中的签名行：'  12: def foo(' / '  34: class Bar'
_SIG_LINE_RE = re.compile(r"^\s*\d+:\s*(async\s+def\s|def\s|class\s)\s*\w+")


def _read_file_reducer(args: Dict, content: str) -> str:
    """read_file 投影：路径 + 行数 + 签名行 + 首尾片段。

    LLM 据此判断"读过这个文件、结构是什么、要不要带 offset 重读细节"，
    而不是 content[:120] 丢光定位信息。
    """
    path = args.get("path", "?")
    offset = args.get("offset", 0)
    lines = content.split("\n") if content else []
    header = lines[0] if lines else f"文件: {path}"

    body = lines[1:] if len(lines) > 1 else []
    sigs = [ln for ln in body if _SIG_LINE_RE.match(ln)]
    head_sample = body[:15]
    tail_sample = body[-15:] if len(body) > 30 else []

    parts = [header, f"[投影] path={path} offset={offset} 签名{len(sigs)}处"]
    if sigs:
        parts.append("--- 签名 ---")
        parts.extend(sigs[:30])
    if head_sample:
        parts.append("--- 首部 ---")
        parts.extend(head_sample)
    if tail_sample:
        parts.append("--- 尾部 ---")
        parts.extend(tail_sample)
    return "\n".join(parts)


def _search_reducer(args: Dict, content: str) -> str:
    """search_files 投影：匹配数 + 前 30 条匹配。"""
    pattern = args.get("pattern", "?")
    lines = content.split("\n") if content else []
    matches = [ln for ln in lines if ":" in ln and not ln.startswith("...")]
    parts = [f"[投影] search pattern={pattern} 匹配{len(matches)}条"]
    parts.extend(matches[:30])
    if len(matches) > 30:
        parts.append(f"... 还有 {len(matches) - 30} 条")
    return "\n".join(parts)


def _find_reducer(args: Dict, content: str) -> str:
    """find_files 投影：前 30 条结果。"""
    pattern = args.get("pattern", "?")
    lines = content.split("\n") if content else []
    results = [ln for ln in lines if ln.startswith("[DIR]") or ln.startswith("[FILE]")]
    parts = [f"[投影] find pattern={pattern} 命中{len(results)}项"]
    parts.extend(results[:30])
    if len(results) > 30:
        parts.append(f"... 还有 {len(results) - 30} 项")
    return "\n".join(parts)


def _list_files_reducer(args: Dict, content: str) -> str:
    """list_files 投影：保留全部路径（行数/字节压成简短后缀），不截断路径。

    list_files 的全部意义是给 grep 喂候选文件列表，reducer 截断路径会让链断。
    max_results=100 已控量，reducer 只压数字细节。
    """
    pattern = args.get("pattern", "?")
    path = args.get("path", ".")
    lines = content.split("\n") if content else []
    parts = [f"[投影] list_files pattern={pattern} path={path}"]
    # header 行（首行）已含命中数；其余行是 "  相对路径  (N 字节, M 行)"
    if lines:
        parts.append(lines[0])
    for ln in lines[1:]:
        # 压缩 "(8421 字节, 234 行)" → "234行"
        m = re.match(r"^(\s*)(\S.*)\s+\(([\d,]+)\s*字节,\s*(\d+)\s*行\)(\s*\(.*\))?\s*$", ln)
        if m:
            parts.append(f"{m.group(1)}{m.group(2)}  ({m.group(4)}行){m.group(5) or ''}")
        elif ln.startswith("(前"):
            parts.append(ln)
    return "\n".join(parts)


def _grep_reducer(args: Dict, content: str) -> str:
    """grep 投影：全留每条命中的 path:line + 命中行前 60 字符，丢弃 context 上下文行。

    grep 命中行是 LLM 决定 read_file(offset=...) 精读位置的依据，
    path:line 一个都不能丢，否则后续轮无法定位要重 grep。只压 context 行。
    支持两种输出模式：context=0 扁平(rel:line: content)、context>0 分组(> line: content + 上下文行)。
    """
    pattern = args.get("pattern", "?")
    lines = content.split("\n") if content else []
    parts = [f"[投影] grep pattern={pattern}"]
    # 首行含命中总数
    if lines and lines[0].startswith("搜索"):
        parts.append(lines[0])

    cur_file = None
    hits = 0
    # context=0 模式命中行形如 "rel:42: content"（含两个冒号，第二个冒号后是行号）
    flat_hit_re = re.compile(r"^(.+?):(\d+):\s*(.*)$")
    for ln in lines:
        # 文件分组头 "== path =="（仅 context>0 模式出现）
        if ln.startswith("== ") and ln.endswith(" =="):
            cur_file = ln[3:-3]
            continue
        # context>0 模式命中行："> 42: content"
        if ln.startswith("> "):
            hits += 1
            rest = ln[2:]  # "42: content"
            snippet = rest[:60] + ("..." if len(rest) > 60 else "")
            parts.append(f"{cur_file}:{snippet}" if cur_file else rest)
            continue
        # context=0 模式命中行：扁平 "rel:42: content"（排除分组头/摘要/上下文行）
        m = flat_hit_re.match(ln)
        if m and not ln.startswith("  "):
            hits += 1
            path, lineno, txt = m.group(1), m.group(2), m.group(3)
            snippet = txt[:60] + ("..." if len(txt) > 60 else "")
            parts.append(f"{path}:{lineno}: {snippet}")
        # 其余行（context 上下文行 "  41: ..."、摘要括号行）跳过
    parts[0] = f"{parts[0]} 共{hits}处命中(已留全部path:line)"
    return "\n".join(parts)


def _list_dir_reducer(args: Dict, content: str) -> str:
    """list_directory 投影：header + entry 名（去 size/mtime）。"""
    lines = content.split("\n") if content else []
    parts = [lines[0]] if lines else []
    for ln in lines[1:]:
        m = re.match(r"^\s*\[(DIR|FILE|OTHER|\?\?\?)\]\s*(.+?)(?:\s+\(.*\))?$", ln)
        if m:
            parts.append(f"[{m.group(1)}] {m.group(2).rstrip('/')}")
        else:
            parts.append(ln)
    return "\n".join(parts)


def _passthrough_reducer(args: Dict, content: str) -> str:
    """状态类工具：返回值本就是一行状态，不压。"""
    return content


def _project_stats_reducer(args: Dict, content: str) -> str:
    """project_stats 投影：保留 header + 扩展名统计 + top 文件名（去行数/字节细节）。"""
    lines = content.split("\n") if content else []
    parts = []
    for ln in lines:
        if ln.startswith("  ") and "字节" in ln and "行" in ln:
            # top-N 行：只留路径，去字节/行数
            seg = ln.split()
            if seg:
                parts.append(f"  {seg[-1]}")
        else:
            parts.append(ln)
    return "\n".join(parts)


def _bash_reducer(args: Dict, content: str) -> str:
    """bash 投影：保留命令 + 输出头 30 行 + 尾 10 行。"""
    cmd = args.get("command", "?")
    lines = content.split("\n") if content else []
    parts = [f"[投影] bash: {cmd} ({len(lines)}行)"]
    if len(lines) <= 40:
        parts.extend(lines)
    else:
        parts.extend(lines[:30])
        parts.append(f"... 中间省略 {len(lines) - 40} 行 ...")
        parts.extend(lines[-10:])
    return "\n".join(parts)


_REDUCERS = {
    "read_file": _read_file_reducer,
    "write_file": _passthrough_reducer,
    "edit_file": _passthrough_reducer,
    "delete_file": _passthrough_reducer,
    "move_file": _passthrough_reducer,
    "copy_file": _passthrough_reducer,
    "create_directory": _passthrough_reducer,
    "add_allowed_path": _passthrough_reducer,
    "list_allowed_paths": _passthrough_reducer,
    "remove_allowed_path": _passthrough_reducer,
    "project_stats": _project_stats_reducer,
    "bash": _bash_reducer,
    "list_files": _list_files_reducer,
    "grep": _grep_reducer,
    "search_memory": _passthrough_reducer,
}


def register_filesystem_tools(registry) -> None:
    """将文件系统工具（含路径管理工具）注册到 ToolRegistry。

    read_artifact 的 handler 需要访问 ToolArtifactStore，由 ChatAgent 在
    装配 context_manager 后通过 registry.register_artifact_store 注入。
    """
    for tool_def in _TOOL_DEFINITIONS:
        if tool_def.get("handler") is None:
            continue  # read_artifact 延后注册
        registry.register(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            handler=tool_def["handler"],
            archive_policy=tool_def.get("archive_policy", "archive"),
        )
    # search_memory 工具（handler 在 _TOOL_DEFINITIONS 之后定义，单独注册）
    registry.register(
        name=_SEARCH_MEMORY_DEF["name"],
        description=_SEARCH_MEMORY_DEF["description"],
        parameters=_SEARCH_MEMORY_DEF["parameters"],
        handler=_SEARCH_MEMORY_DEF["handler"],
        archive_policy=_SEARCH_MEMORY_DEF.get("archive_policy", "archive"),
    )
    for name, reducer in _REDUCERS.items():
        registry.register_reducer(name, reducer)


def _make_read_artifact_handler(artifact_store):
    """构造 read_artifact handler：按存档文件路径直接读回原文。"""
    def read_artifact(path: str) -> str:
        if artifact_store is None:
            return "错误: 未装配存档存储"
        content = artifact_store.read(path)
        if content is None:
            return f"错误: 未找到存档或读取失败（路径 {path}）"
        return content
    return read_artifact


def register_read_artifact_tool(registry, artifact_store) -> None:
    """注册 read_artifact 工具（需 artifact_store 装配后调用）。"""
    tool_def = next(t for t in _TOOL_DEFINITIONS if t["name"] == "read_artifact")
    registry.register(
        name="read_artifact",
        description=tool_def["description"],
        parameters=tool_def["parameters"],
        handler=_make_read_artifact_handler(artifact_store),
        archive_policy=tool_def.get("archive_policy", "archive_ref"),
    )
    # read_artifact 结果是原文，历史轮再压一次投影意义不大；用 passthrough
    registry.register_reducer("read_artifact", _passthrough_reducer)
