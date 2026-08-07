#!/usr/bin/env python3
"""ChatAgent 命令行入口"""
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

from chat_agent import ChatAgent, Storage, Config
from chat_agent.mcp_client import MCPClient, MCPServerConfig
from chat_agent.prompts import load_prompt


def print_welcome():
    """打印欢迎信息"""
    print("=" * 50)
    print("       ChatAgent - 智能对话助手")
    print("=" * 50)
    print()
    print("命令：")
    print("  - list    : 列出所有对话")
    print("  - new     : 创建新对话")
    print("  - load    : 加载指定对话")
    print("  - delete  : 删除指定对话")
    print("  - clear   : 清空当前对话历史")
    print("  - title   : 重命名当前对话")
    print("  - resume  : 续跑当前会话未完成的任务（断点续跑）")
    print("  - analyze : 退化分析模式")
    print("  - reports : 列出退化分析报告")
    print("  - back    : 退出追问模式，恢复通用对话")
    print("  - folders : 查看当前允许访问的文件夹列表")
    print("  - quit    : 退出程序")
    print()


def print_response(response: str):
    """打印 Agent 响应"""
    print()
    print(f"【助手】{response}")
    print()


def list_conversations(storage: Storage):
    """列出所有对话"""
    conversations = storage.list_conversations()
    if not conversations:
        print("暂无对话记录")
        return
    
    print("\n对话列表：")
    print("-" * 50)
    for i, conv in enumerate(conversations, 1):
        # 格式化时间
        from datetime import datetime
        time_str = datetime.fromtimestamp(conv["updated_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"{i}. [{conv['id']}] {conv['title']}")
        print(f"   消息数: {conv['message_count']} | 更新时间: {time_str}")
    print()


def select_by_index(items, prompt="请选择: "):
    """通用编号选择：输入编号选择列表项，越界或非数字要求重新输入，留空返回 None"""
    while True:
        try:
            choice = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not choice:
            return None
        try:
            idx = int(choice) - 1
        except ValueError:
            print("请输入数字编号，或留空取消")
            continue
        if 0 <= idx < len(items):
            return items[idx]
        print(f"编号超出范围（1-{len(items)}），请重新输入")


def load_conversation(storage: Storage) -> str:
    """加载指定对话"""
    conversations = storage.list_conversations()
    if not conversations:
        print("暂无对话记录")
        return None

    print("\n选择要加载的对话：")
    for i, conv in enumerate(conversations, 1):
        print(f"  {i}. [{conv['id']}] {conv['title']}")

    result = select_by_index(conversations, prompt="\n请输入编号: ")
    return result["id"] if result else None


def create_new_conversation() -> bool:
    """询问是否创建新对话"""
    try:
        choice = input("是否创建新对话? (y/n): ").strip().lower()
        return choice in ["y", "yes"]
    except (KeyboardInterrupt, EOFError):
        return False


def main():
    # 加载配置
    config = Config.from_env()

    # 初始化存储
    storage = Storage()

    # 启动时清理孤儿 artifact 目录（对应会话文件已不存在的，多为历史 bug 遗留或绕过 delete 命令直接删文件）
    removed = storage.cleanup_orphan_artifacts()
    if removed:
        print(f"[启动清理] 移除 {removed} 个孤儿 artifact 目录")

    # 节点级 checkpoint Saver：启用断点续跑（节点状态持久化到 sqlite，崩溃可续）
    checkpointer = None
    _ckpt_conn = None
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs("data", exist_ok=True)
        _ckpt_conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(_ckpt_conn)
    except Exception as e:
        print(f"[警告] SqliteSaver 初始化失败，断点续跑不可用: {e}")

    # 初始化 MCP 客户端
    mcp_client = None
    if config.mcp_enabled and config.mcp_servers:
        server_configs = [
            MCPServerConfig(
                name=cfg.get("name", "unknown"),
                transport=cfg.get("transport", "stdio"),
                command=cfg.get("command"),
                args=cfg.get("args"),
                env=cfg.get("env"),
                url=cfg.get("url"),
                headers=cfg.get("headers"),
                timeout=cfg.get("timeout", 30),
                sse_read_timeout=cfg.get("sse_read_timeout", 300),
                enabled=cfg.get("enabled", True),
            )
            for cfg in config.mcp_servers
        ]
        mcp_client = MCPClient(server_configs)
        print("正在连接 MCP 服务器...")
        mcp_client.connect()

    # 默认创建新对话
    from chat_agent.confirmation import ConfirmationGateway
    confirmation_gateway = ConfirmationGateway(prompt_callback=input)
    agent = ChatAgent(
        storage=storage,
        mcp_client=mcp_client,
        max_tool_iterations=config.max_tool_iterations,
        confirmation_gateway=confirmation_gateway,
        checkpointer=checkpointer,
    )

    # 报告工具发现情况
    native_count = agent.tool_registry.native_count
    mcp_tool_count = agent.tool_registry.mcp_count
    print(f"已注册 {native_count + mcp_tool_count} 个工具 (原生: {native_count}, MCP: {mcp_tool_count})")
    for name in agent.tool_registry.tool_names:
        prefix = "[原生]" if name in agent.tool_registry._native else "[MCP]"
        print(f"  - {prefix} {name}")
    
    print_welcome()
    print(f"当前对话 ID: {agent.conversation_id}")
    print(f"对话标题: {agent.title}")
    print(f"使用模型: {agent.model}")
    print("-" * 50)
    
    # 追问模式标记：analyze 后置 True，back 后置 False
    in_followup = False

    # 对话循环
    while True:
        try:
            user_input = input("【用户】: ").strip()
        except (KeyboardInterrupt, EOFError):
            if mcp_client:
                mcp_client.shutdown()
            print("\n\n再见！")
            break
        
        if not user_input:
            continue
        
        # 处理命令
        if user_input.lower() in ["quit", "exit", "q"]:
            if mcp_client:
                mcp_client.shutdown()
            if _ckpt_conn is not None:
                try: _ckpt_conn.close()
                except Exception: pass
            print("再见！")
            break
        
        elif user_input.lower() == "list":
            list_conversations(storage)
            continue

        elif user_input.lower() == "resume":
            # 断点续跑：从最后 checkpoint 续跑当前会话未完成的 turn
            agent.resume_turn()
            continue

        elif user_input.lower() == "new":
            agent = ChatAgent(storage=storage, mcp_client=mcp_client,
                              max_tool_iterations=config.max_tool_iterations,
                              confirmation_gateway=confirmation_gateway,
                              checkpointer=checkpointer)
            print(f"\n已创建新对话")
            print(f"当前对话 ID: {agent.conversation_id}")
            print(f"对话标题: {agent.title}")
            continue

        elif user_input.lower() == "load":
            conv_id = load_conversation(storage)
            if conv_id:
                agent = ChatAgent(conversation_id=conv_id, storage=storage,
                                  mcp_client=mcp_client,
                                  max_tool_iterations=config.max_tool_iterations,
                                  confirmation_gateway=confirmation_gateway,
                                  checkpointer=checkpointer)
                print(f"\n已加载对话: [{agent.conversation_id}] {agent.title}")
                print(f"历史消息: {agent.history_count} 条")
            continue
        
        elif user_input.lower() == "delete":
            conv_id = load_conversation(storage)
            if conv_id:
                try:
                    confirm = input(f"定要删除对话 [{conv_id}] 吗? (y/n): ").strip().lower()
                    if confirm in ["y", "yes"]:
                        storage.delete_conversation(conv_id)
                        print(f"已删除对话 [{conv_id}]")
                        if agent.conversation_id == conv_id:
                            agent = ChatAgent(storage=storage, mcp_client=mcp_client,
                                              max_tool_iterations=config.max_tool_iterations,
                                              confirmation_gateway=confirmation_gateway,
                                              checkpointer=checkpointer)
                            print("已自动切换到新对话")
                except (KeyboardInterrupt, EOFError):
                    pass
            continue
        
        elif user_input.lower() == "clear":
            agent.clear_history()
            print("已清空对话历史")
            continue
        
        elif user_input.lower() == "title":
            try:
                new_title = input("请输入新标题: ").strip()
                if new_title:
                    agent.rename_conversation(new_title)
                    print(f"修改标题为: {new_title}")
            except (KeyboardInterrupt, EOFError):
                pass
            continue

        elif user_input.lower() == "analyze":
            run_degradation_analyze(config, mcp_client, agent)
            in_followup = True
            continue

        elif user_input.lower() == "back":
            if in_followup:
                agent.set_system_prompt(load_prompt("default.txt", "你是一个有帮助的AI助手，请用简洁、清晰的语言回答用户的问题。"))
                in_followup = False
                print("已退出追问模式，恢复通用对话。")
            else:
                print("当前不在追问模式。")
            continue

        elif user_input.lower() == "reports":
            list_degradation_reports()
            continue

        elif user_input.lower() == "folders":
            from chat_agent.tools.filesystem import list_allowed_paths
            print(list_allowed_paths())
            continue

        elif user_input.lower().startswith("addfolder "):
            folder = user_input[len("addfolder "):].strip()
            from chat_agent.tools.filesystem import add_allowed_path
            result = add_allowed_path(folder)
            print(result)
            continue
        
        # 发送消息前：输入防护
        from chat_agent.guardrails import sanitize_input, detect_sensitive_info

        # SQL 注入拦截
        _, block_warning = sanitize_input(user_input)
        if block_warning and "拦截" in block_warning:
            print(f"\n【助手】{block_warning}\n")
            continue

        # 敏感信息检测，让用户确认
        sensitive_warning = detect_sensitive_info(user_input)
        if sensitive_warning:
            print(f"⚠ {sensitive_warning}")
            try:
                confirm = input("继续发送? (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                continue
            if confirm not in ("y", "yes"):
                print("已取消发送")
                continue

        try:
            response = agent.chat(user_input)
            # 流式输出已在 chat() 内部实时打印，此处只打印换行分隔
            print()

            # 如果是第一条消息，自动设置标题
            user_msg_count = sum(1 for m in agent.get_history() if m.role == "user")
            if user_msg_count == 1:
                agent.set_title_from_first_message()
                print(f"对话已自动命名为: {agent.title}")

        except Exception as e:
            print(f"\n错误: {e}")
            # 本轮崩溃：active_thread 仍非空（chat() 在 invoke 前已保存）。
            # 立即提示续跑，避免用户直接输新消息覆盖 active_thread → 崩溃 turn 被遗弃无法恢复。
            if agent.conversation.active_thread:
                print(f"[续跑] 检测到本轮未完成（turn {agent.conversation.turn_seq}，已保留断点）")
                try:
                    confirm = input("是否续跑? (y/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    confirm = "n"
                if confirm in ("y", "yes"):
                    try:
                        agent.resume_turn()
                    except Exception as re:
                        print(f"\n续跑失败: {re}\n可稍后输 resume 重试，或输新消息开始新轮（将提示丢弃中断的 turn）。")
                else:
                    # n = 推迟：保留 active_thread + checkpoint，稍后可 resume 或下次 load 续跑
                    agent._defer_incomplete_turn(agent.conversation.active_thread)
                    print("可输 resume 续跑，或直接输新消息（会提示丢弃中断的 turn 后处理）。")


def list_degradation_reports():
    """列出退化分析报告，支持查看详情和删除"""
    from chat_agent.degradation import DegradationStorage, ReportFormatter
    storage = DegradationStorage()
    formatter = ReportFormatter()
    reports = storage.list_reports()
    if not reports:
        print("暂无退化分析报告")
        return
    print("\n退化分析报告列表：")
    print("-" * 50)
    for i, r in enumerate(reports, 1):
        from datetime import datetime
        time_str = datetime.fromtimestamp(r.get("created_at", 0)).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. [{r['id']}] {r.get('repo_path', '')} "
              f"({r.get('ref_from', '')}→{r.get('ref_to', '')}) "
              f"健康度:{r.get('overall_score', 0):.0f} "
              f"趋势:{r.get('overall_trend', '')} "
              f"{time_str}")
    print()
    print("输入编号查看详情，编号前加 d 删除（如 d1），留空返回")
    while True:
        try:
            choice = input("请选择: ").strip()
        except (KeyboardInterrupt, EOFError):
            return
        if not choice:
            return
        # 删除操作
        if choice.lower().startswith("d"):
            try:
                idx = int(choice[1:]) - 1
            except ValueError:
                print("无效输入，请输入 d+编号（如 d1），或留空取消")
                continue
            if 0 <= idx < len(reports):
                report_id = reports[idx]["id"]
                try:
                    confirm = input(f"确定删除报告 [{report_id}] 吗? (y/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    return
                if confirm in ["y", "yes"]:
                    if storage.delete_report(report_id):
                        print(f"已删除报告 [{report_id}]")
                    else:
                        print("删除失败，报告文件不存在")
                else:
                    print("已取消")
            else:
                print(f"编号超出范围（1-{len(reports)}），请重新输入")
                continue
            return
        # 查看详情
        try:
            idx = int(choice) - 1
        except ValueError:
            print("请输入数字编号，或留空取消")
            continue
        if 0 <= idx < len(reports):
            report_id = reports[idx]["id"]
            report = storage.load_report(report_id)
            if report:
                print()
                print(formatter.format_report(report))
            else:
                print("报告加载失败")
        else:
            print(f"编号超出范围（1-{len(reports)}），请重新输入")
            continue
        return


def run_degradation_analyze(config: Config, mcp_client, agent: ChatAgent):
    """运行退化分析"""
    from chat_agent.degradation import DegradationAgent, DegradationStorage, ReportFormatter, ParamValidationError, DataCollectionError

    # 域名称输入校验：获取域列表，输入不在范围内则要求重输
    valid_domains = None
    try:
        from chat_agent.degradation.data_provider import QualityPlatformProvider
        _provider = QualityPlatformProvider(api_base=config.quality_api_base)
        valid_domains = _provider.get_domain_info(type=1)
    except Exception:
        pass  # 获取失败则跳过校验

    while True:
        repo_path = input("请输入域/微服务名称 (留空分析全部): ").strip() or None
        if repo_path is None:
            break
        if valid_domains is not None and repo_path not in valid_domains:
            print(f"域 '{repo_path}' 不在范围内，请重新输入")
            continue
        break
    month_from = input("请输入对比起始月份 (YYYYMM，留空自动选择): ").strip() or None
    month_to = input("请输入对比结束月份 (YYYYMM，默认最近月份): ").strip() or None

    deg_storage = DegradationStorage()
    # 复用现有的 LLMClient
    from chat_agent.client import LLMClient
    llm_client = LLMClient(
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    deg_agent = DegradationAgent(
        storage=deg_storage,
        mcp_client=mcp_client,
        llm_client=llm_client,
        max_tool_iterations=config.degradation_max_iterations,
    )

    print("\n正在分析...")
    try:
        report = deg_agent.analyze(
            repo_path=repo_path,
            month_from=month_from,
            month_to=month_to,
        )
    except ParamValidationError as e:
        # 参数错误：直接告知用户，不生成报告、不进入追问模式
        print(f"\n{e}")
        return
    except DataCollectionError as e:
        # 数据收集失败：只告知失败，不生成报告、不问导出、不进追问
        print(f"\n分析失败: {e}")
        return
    except Exception as e:
        print(f"\n分析失败: {e}")
        return

    try:
        formatter = ReportFormatter()
        print()
        print(formatter.format_report(report))

        # 推理路径
        if report.reasoning_trace:
            print("推理路径: ", end="")
            steps = [f"{s.tool_calls}" for s in report.reasoning_trace if s.tool_calls]
            print(" → ".join(steps[:6]))

        # 询问是否导出报告文件
        _maybe_export_report_file(report, formatter)

        # 获取追问上下文，注入到当前 ChatAgent 继续对话
        ctx = deg_agent.get_followup_context()
        if not ctx:
            return

        # 切换 system prompt 为追问专用
        agent.set_system_prompt(ctx["system_prompt"])

        # 注入报告上下文到对话历史 + 上下文管理器
        for msg in ctx["messages"]:
            agent.conversation.add_message(msg["role"], msg["content"])
            if msg["role"] == "user":
                agent.context_manager.append_user(msg["content"])
            else:
                agent.context_manager.append_assistant(msg["content"])

        # 更新标题
        agent.conversation.title = f"退化分析: {repo_path or '全域'} {month_from or ''}→{month_to or ''}"
        agent.storage.save_conversation(agent.conversation)

        print(f"\n已进入退化分析追问模式，当前对话 ID: {agent.conversation_id}")
        print("直接输入问题即可追问。输入 back 退出追问模式，恢复通用对话。")

    except Exception as e:
        print(f"\n分析失败: {e}")


def _maybe_export_report_file(report, formatter):
    """报告生成完毕后询问用户是否导出 txt 文件。默认存 D 盘，可自定义路径。"""
    import os
    from datetime import datetime

    print()
    choice = input("是否将报告导出为 txt 文件? (Y/n): ").strip().lower()
    if choice in ("n", "no", "否"):
        return

    # 默认文件名: 退化报告_域_起始-结束_时间戳.txt
    repo_part = report.repo_path or "全域"
    # 去掉文件名非法字符
    safe_repo = "".join(c for c in repo_part if c not in r'\/:*?"<>|')
    time_str = datetime.fromtimestamp(report.created_at).strftime("%Y%m%d_%H%M%S")
    default_name = f"退化报告_{safe_repo}_{report.ref_from}-{report.ref_to}_{time_str}.txt"
    default_dir = "D:\\"
    default_path = os.path.join(default_dir, default_name)

    path_input = input(
        f"存储路径 (留空使用默认: {default_path}): "
    ).strip()
    target = path_input or default_path

    # 确保目录存在
    target_dir = os.path.dirname(target)
    if target_dir and not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            print(f"目录创建失败: {e}")
            return

    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(formatter.format_report(report))
            if report.reasoning_trace:
                f.write("\n\n推理路径:\n")
                for step in report.reasoning_trace:
                    f.write(
                        f"  步骤{step.step} [{step.skill}]: "
                        f"{step.tool_results_summary} — {step.reasoning}\n"
                    )
        print(f"报告已导出: {target}")
    except OSError as e:
        print(f"报告导出失败: {e}")


if __name__ == "__main__":
    main()
