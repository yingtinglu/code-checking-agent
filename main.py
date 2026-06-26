#!/usr/bin/env python3
"""ChatAgent 命令行入口"""
import os
import sys
from chat_agent import ChatAgent, Storage, Config
from chat_agent.mcp_client import MCPClient, MCPServerConfig


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


def load_conversation(storage: Storage) -> str:
    """加载指定对话"""
    conversations = storage.list_conversations()
    if not conversations:
        print("暂无对话记录")
        return None
    
    print("\n选择要加载的对话：")
    for i, conv in enumerate(conversations, 1):
        print(f"  {i}. [{conv['id']}] {conv['title']}")
    
    try:
        choice = input("\n请输入编号: ").strip()
        if not choice:
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(conversations):
            return conversations[idx]["id"]
    except (ValueError, KeyboardInterrupt):
        pass
    return None


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
            )
            for cfg in config.mcp_servers
        ]
        mcp_client = MCPClient(server_configs)
        print("正在连接 MCP 服务器...")
        mcp_client.connect()
        tools = mcp_client.get_openai_tools()
        print(f"已发现 {len(tools)} 个工具")
        for tool in tools:
            desc = tool["function"]["description"][:60]
            print(f"  - {tool['function']['name']}: {desc}")

    # 默认创建新对话
    agent = ChatAgent(
        storage=storage,
        mcp_client=mcp_client,
        max_tool_iterations=config.max_tool_iterations,
    )
    
    print_welcome()
    print(f"当前对话 ID: {agent.conversation_id}")
    print(f"对话标题: {agent.title}")
    print(f"使用模型: {agent.model}")
    print("-" * 50)
    
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
            print("再见！")
            break
        
        elif user_input.lower() == "list":
            list_conversations(storage)
            continue
        
        elif user_input.lower() == "new":
            agent = ChatAgent(storage=storage, mcp_client=mcp_client,
                              max_tool_iterations=config.max_tool_iterations)
            print(f"\n已创建新对话")
            print(f"当前对话 ID: {agent.conversation_id}")
            print(f"对话标题: {agent.title}")
            continue
        
        elif user_input.lower() == "load":
            conv_id = load_conversation(storage)
            if conv_id:
                agent = ChatAgent(conversation_id=conv_id, storage=storage,
                                  mcp_client=mcp_client,
                                  max_tool_iterations=config.max_tool_iterations)
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
                                              max_tool_iterations=config.max_tool_iterations)
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
        
        # 发送消息
        try:
            response = agent.chat(user_input)
            print_response(response)
            
            # 如果是第一条消息，自动设置标题
            user_msg_count = sum(1 for m in agent.get_history() if m.role == "user")
            if user_msg_count == 1:
                agent.set_title_from_first_message()
                print(f"对话已自动命名为: {agent.title}")
                
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
