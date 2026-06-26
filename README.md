# ChatAgent

智能话 Agent，支持多轮对话、上下文记忆和对话历史持久化。

## 功能特性

### Phase 1 - 核心功能

- ✅ 单轮对话
- ✅ 多轮对话（自动维护上下文）
- ✅ 系统提示词（角色设定）
- ✅ 对话历史管理

### Phase 2 - 数据持久化

- ✅ 对话会话保存（JSON 文件）
- ✅ 对话会话加载
- ✅ 会话管理（创建、删除、列表）
- ✅ 自动命名对话标题

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行

```cmd
python main.py
```

## 使用方法

### 基本命令

| 命令     | 说明             |
| -------- | ---------------- |
| `list`   | 列出所有对话     |
| `new`    | 创建新对话       |
| `load`   | 加载指定对话     |
| `delete` | 删除指定对话     |
| `clear`  | 清空当前对话历史 |
| `title`  | 重名当前对话     |
| `quit`   | 退出程序         |

### 对话历史

对话自动保存到 `data/conversations/` 目录，格式为 JSON 文件。关闭程序后再次运行可以加载历史对话继续交流。

## 项目结构

```
ChatAgent/
├── chat_agent/          # 核心包
│   ├── __init__.py
│   ├── agent.py         # Agent 核逻辑
│   ├── client.py        # LLM 客户端封装
│   ├── config.py        # 配置管理
│   └─ storage.py       # 对话存储管理
├── prompts/             # Prompt 模板
├── data/                # 数据目录
│   └── conversations/   # 对话历史存储
├── main.py              # 入口文件
├── requirements.txt     # 依赖
└── README.md
```

## 配置

代码默认使用 `config.py` 中的内置配置：

- **模型**: MiniMax-M2.7
- **API 地址**: http://100.85.219.23:28081/v1
- **API Key**: sk-123（默认）

## License

MIT
