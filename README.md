# code-checking-agent
# 聊天对话 Agent 开发需求

## 1. 项目概述

**项目名称**: ChatAgent  
**项目类型**: AI 对话助手  
**核心功能**: 于大语言模型的智能对话系统，支持多轮对话、上下文记忆、多种对话场景  
**目标用户**: 需智能对话能力的终端用户或集成方

---

## 2. 功能需求

### 2.1 核心功能

| 功能模块     | 描述                                                     | 优先级 |
| ------------ | -------------------------------------------------------- | ------ |
| 单轮对话     | 用户发送消息，Agent 返回回复                             | P0     |
| 多轮对话     | 支持上下文连续对话，Agent 能记住对话历史                 | P0     |
| 对话历史管理 | 保存、加载、清除对记录                                   | P1     |
| 角色设定     | 支持自定义 Agent 角色/人格（如助手、翻译官、代码审查员） | P1     |

### 2.2 扩展功能（可选）

| 功能模块      | 描述                           | 优先级 |
| ------------- | ------------------------------ | ------ |
| 流式输出      | 支打字机效果的流式响应         | P2     |
| 对话导出      | 将会话导出为 Markdown/TXT 文件 | P2     |
| 快捷指令      | 预义常用 Prompt 模板，一键使用 | P2     |
| 多 Agent 切换 | 支持创建多个不同角色的 Agent   | P2     |

---

## 3. 技术架构

### 3.1 技术选型

| 层级           | 技术选型                              | 说明                       |
| -------------- | ------------------------------------- | -------------------------- |
| **编程语言**   | Python 3.10+                          | 主流 AI 开发语言，生态丰富 |
| **LLM 接口**   | OpenAI API / Anthropic API / 本地模型 | 支持主流大模型             |
| **SDK**        | `openai` / `anthropic` / `langchain`  | 简化 API 调用              |
| **会话管理**   | SQLite / JSON 文件                    | 持久化存储对话历史         |
| **命令行界面** | `rich` / `prompt_toolkit`             | 美化终端交互               |
| **Web 界面**   | Gradio / Streamlit (可选)             | 提供可视化界面             |

### 3.2 项目结构（推荐）

```
ChatAgent/
├── chat_agent/              # 核心包
│   ├── __init__.py
│   ├── agent.py             # Agent 核心逻辑
│   ├── client.py            # LLM 客户端封装
│   ├── storage.py           # 对话存储
│   └── config.py            # 配置管理
├── prompts/                 # Prompt 模板
│   └── default.txt
├── data/                    # 数据目录
│   └── conversations/       # 对话历史存储
├─ tests/                   # 单元测试
├── main.py                  # 入口文件
├── requirements.txt         # 依赖
└── README.md
```

---

## 4. 数据模型

### 4.1 对话消息

```python
class Message:
    role: str        # "user" | "assistant" | "system"
    content: str     # 消息内容
    timestamp: float # 时间戳
```

### 4.2 对话会话

```python
class Conversation:
    id: str                     # 会话唯一 ID
    title: str                  # 会话标题
    messages: List[Message]     # 消息列表
    created_at: float           # 创建时间
    updated_at: float           # 更新时间
    system_prompt: str          # 系提示词（角色设定）
```

---

## 5. 接口设计

### 5.1 核心接口

```python
class ChatAgent:
    def __init__(self, model: str, api_key: str, system_prompt: str = None)
    
    def chat(self, message: str) -> str
        """发送消息并返回回复"""
    
    def clear_history(self) -> None
        """清除对话历史"""
    
    def get_history(self) -> List[Message]
        """获取对话历史"""
    
    def set_system_prompt(self, prompt: str) -> None
        """设置系统提示词"""
```

---

## 6. 非功能需求

| 需求类型     | 描述                                       |
| ------------ | ------------------------------------------ |
| **性能**     | 首次响应时间 < 5秒（取决于 LLM API）       |
| **可靠性**   | API 调用失败时支持重试和错误提示           |
| **可扩展性** | 便集成新的 LLM 提供商                      |
| **易用性**   | 清晰的命令行交互，错误提示友好             |
| **安全性**   | API Key 硬编码，通过环境变量或配置文件注入 |

---

## 7. 开发里程碑

| 阶段               | 内容         | 产出                         |
| ------------------ | ------------ | ---------------------------- |
| **Phase 1**        | 基础对话功能 | 单轮对话、多轮对话、基础配置 |
| **Phase 2**        | 数据持久化   | 对话历史保存与加载           |
| **Phase 3**        | 角色与模板   | 系统提示词、快捷指令         |
| **Phase 4** (可选) | Web 界面     | Gradio/Streamlit 可视化界面  |

---

## 8. 快速开始示例

```python
from chat_agent import ChatAgent

# 初始化 Agent（使用默认助手角色）
agent = ChatAgent(
    model="gpt-4",
    api_key="your-api-key",
    system_prompt="你是一个有帮助的助手"
)

# 对话
while True:
    user_input = input("你: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.chat(user_input)
    print(f"助手: {response}")
```

---

## 9. 后续可扩展方向

- 支持多 LLM 后端（OpenAI、Claude、本地 Ollama）
- 支持 Function Calling / Tool Use
- 支持多模态（图片理解、音交互）
- 支持 Agent 协作（多个 Agent 分工）
- Web API 服务化部署

---

> **建议**: 如果是第一次开发 Agent，建议从 Phase 1 开始，先实现核心对话功能，再逐步迭代扩展。
