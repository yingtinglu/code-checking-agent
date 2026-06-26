# ChatAgent MCP Client 实现原理

## 一、整体架构

```
用户输入
  │
  ▼
main.py (REPL 循环)
  │
  ├─ Config.from_env() ←─ mcp_servers.json
  │
  ├─ MCPClient.connect() ──► daemon 线程事件循环 ──► MCP Server 子进程
  │
  └─ ChatAgent.chat() ──► LLMClient.chat(tools=...) ──► CCS API
           │                                              │
           │         ◄─── ChatResponse(tool_calls=...) ◄──┘
           │
           └─ MCPClient.call_tool() ──► MCP Server ──► 工具结果
                │
                └─ LLMClient.chat() ──► CCS API (带工具结果)
                     │
                     └─ 最终回复
```

核心思路：**LLM 决定是否调工具 → MCP Client 执行工具 → 结果回传 LLM → 继续推理**，即 ReAct（Reasoning + Acting）循环。

## 二、启动阶段

```
main.py 启动
  │
  ├─ Config.from_env() 读取环境变量
  │    └─ _load_mcp_config() 发现 MCP_SERVERS_CONFIG=mcp_servers.json
  │         └─ 解析 JSON → config.mcp_servers = [{name, transport, command, args}]
  │            config.mcp_enabled = True
  │
  ├─ 构建 MCPServerConfig 列表 → 创建 MCPClient
  │
  └─ mcp_client.connect()
       │
       └─ 调用 run_async(self._connect_all())  ← 进入异步世界
            │
            ├─ 创建 AsyncExitStack 管理所有异步上下文
            │
            └─ 对每个 server 配置：
                 ├─ stdio 传输 → stdio_client() 启动子进程（test_mcp_server.py）
                 │               子进程的 stdin/stdout 作为通信管道
                 ├─ 创建 ClientSession(read_stream, write_stream)
                 ├─ session.initialize()  ← MCP 协议握手
                 └─ session.list_tools()  ← 发现工具，存入 self._tools
```

## 三、async/sync 桥接

MCP SDK 是纯异步的（基于 asyncio），而项目代码全是同步的。`async_utils.py` 通过 **daemon 线程 + 常驻事件循环** 桥接两者：

```
同步代码（main.py, agent.py）          异步世界（MCP SDK）
         │                                    │
         │  mcp_client.connect()              │
         │  ──── run_async(coro) ──────────►  │  后台 daemon 线程中
         │       │                             │  的事件循环执行协程
         │       │  asyncio.run_coroutine_     │
         │       │  threadsafe(coro, loop)     │
         │       │                             │
         │       ◄── future.result() ─────────│  协程完成，返回结果
         │  （阻塞等待）                       │
```

关键代码：

```python
# async_utils.py

def _ensure_loop():
    """启动后台事件循环（如未运行）。"""
    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_run_loop, daemon=True)
    _thread.start()

def run_async(coro):
    """提交异步协程到后台循环，阻塞等待结果。"""
    _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()  # 阻塞当前线程，直到协程完成
```

为什么不用 `asyncio.run()`？因为每次调用会创建新的事件循环，且不能在已有事件循环内嵌套调用。daemon 线程方案只需创建一次循环，所有异步操作复用。

## 四、MCP Client 核心逻辑

### 4.1 连接与工具发现

```python
# mcp_client.py

async def _connect_all(self):
    self._exit_stack = AsyncExitStack()
    await self._exit_stack.__aenter__()

    for config in self._server_configs:
        # 1. 建立传输通道
        if config.transport == "stdio":
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        elif config.transport == "sse":
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(config.url)
            )

        # 2. 创建会话并握手
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        # 3. 发现工具
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            # 每个工具有 name, description, inputSchema
            server_tools[tool.name] = { ... }
```

`AsyncExitStack` 确保所有连接（子进程、网络会话）在 `shutdown()` 时按正确顺序关闭。

### 4.2 工具格式转换

MCP 工具的 schema 需要转换成 OpenAI function calling 格式，才能传给 LLM：

```python
# MCP 工具格式（来自 session.list_tools()）
Tool(name="get_weather", description="获取天气", inputSchema={...})

# ↓ 转换为 OpenAI 格式

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": { "type": "object", "properties": {"city": ...} }
    }
}
```

### 4.3 工具调用

```python
# mcp_client.py

def call_tool(self, tool_name, arguments):
    # 找到工具所属的 server session
    for server_name, tools in self._tools.items():
        if tool_name in tools:
            session = self._sessions[server_name]
            # 通过 MCP 协议调用工具
            result = run_async(session.call_tool(tool_name, arguments))
            # 提取文本内容
            return "\n".join(item.text for item in result.content)
```

## 五、ReAct 循环（核心流程）

`ChatAgent.chat()` 实现了 ReAct 循环——LLM 推理与工具调用交替执行：

```
用户输入: "北京天气怎么样？"
  │
  ▼
ChatAgent.chat("北京天气怎么样？")
  │
  ├─ conversation.add_message("user", "北京天气怎么样？")
  │
  ├─ mcp_client.get_openai_tools()  ← 获取 OpenAI 格式的工具列表
  │
  └─ while 循环 (最多 max_tool_iterations 次)：
       │
       ├─ _build_api_messages() 构建消息列表
       │
       ├─ client.chat(messages, tools=openai_tools)  ← 发给 LLM
       │
       │    ◄── LLM 返回 ChatResponse:
       │        content="",
       │        tool_calls=[{
       │          "id": "call_abc123",
       │          "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}
       │        }]
       │
       ├─ 检测到 tool_calls 不为空 → 执行工具
       │
       │    ├─ 保存 assistant 消息（带 tool_calls）到对话历史
       │    │
       │    └─ 对每个 tool_call:
       │         ├─ 解析 arguments JSON → {"city": "北京"}
       │         ├─ mcp_client.call_tool("get_weather", {"city": "北京"})
       │         │    └─ run_async(session.call_tool(...))
       │         │         └─ MCP 协议发到子进程 → 执行 → 返回
       │         │              "晴天，25°C，微风"
       │         │
       │         └─ 保存 tool 角色消息到对话历史:
       │            role="tool", content="晴天，25°C，微风",
       │            tool_call_id="call_abc123"
       │
       │    打印: [工具调用] get_weather({"city": "北京"})
       │    打印: [工具结果] 晴天，25°C，微风
       │
       └─ 继续循环 ──► 第二轮调用 LLM
            │
            ├─ _build_api_messages() 现在包含:
            │    [system, user消息, assistant(带tool_calls), tool消息(结果)]
            │
            ├─ client.chat(messages, tools=openai_tools)
            │
            │    ◄── LLM 看到工具结果后生成最终回复:
            │        content="北京今天天气晴朗，温度25°C，微风。",
            │        tool_calls=None,
            │        finish_reason="stop"
            │
            └─ tool_calls 为空 → 保存 assistant 回复，退出循环
                 返回: "北京今天天气晴朗，温度25°C，微风。"
```

## 六、消息在对话历史中的结构

一次带工具调用的对话，`conversation.messages` 长这样：

```json
[
  {
    "role": "user",
    "content": "北京天气怎么样？",
    "timestamp": 1719400000.0
  },
  {
    "role": "assistant",
    "content": "",
    "timestamp": 1719400001.0,
    "tool_calls": [{
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"北京\"}"
      }
    }]
  },
  {
    "role": "tool",
    "content": "晴天，25°C，微风",
    "timestamp": 1719400002.0,
    "tool_call_id": "call_abc123",
    "name": "get_weather"
  },
  {
    "role": "assistant",
    "content": "北京今天天气晴朗，温度25°C，微风，比较舒适。",
    "timestamp": 1719400003.0
  }
]
```

这些会持久化到 `data/conversations/` 下的 JSON 文件，下次加载对话时能完整还原上下文（包括工具调用历史）。

## 七、消息序列化

`_build_api_messages()` 负责将内部消息模型转为 OpenAI API 需要的格式：

```python
def _build_api_messages(self):
    messages = []
    if self.system_prompt:
        messages.append({"role": "system", "content": self.system_prompt})

    for msg in self.conversation.messages:
        if msg.role == "tool":
            # tool 角色消息：必须带 tool_call_id
            messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
        elif msg.tool_calls:
            # assistant 消息带工具调用：content 可以为 None
            messages.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [tc.to_dict() for tc in msg.tool_calls],
            })
        else:
            # 普通消息
            messages.append({"role": msg.role, "content": msg.content})
```

## 八、关键设计决策

| 设计点           | 选择                                | 原因                                          |
| ---------------- | ----------------------------------- | --------------------------------------------- |
| async/sync 桥接  | daemon 线程 + 常驻事件循环          | 避免 `asyncio.run()` 的重复创建和嵌套问题     |
| 工具发现时机     | 启动时一次性 `list_tools()`         | 工具列表通常不会动态变化                      |
| 工具格式转换     | MCP schema → OpenAI function format | MiniMax-M2.7 支持 OpenAI 的 `tools` 参数      |
| 连接生命周期     | `AsyncExitStack` 统一管理           | 确保子进程和网络连接在 shutdown 时正确关闭    |
| 工具调用错误处理 | 返回错误字符串作为 tool 结果        | 让 LLM 看到错误并自行调整策略，而不是中断循环 |
| 最大迭代次数     | 默认 10 次，可配置                  | 防止 LLM 陷入无限工具调用循环                 |
| 消息向后兼容     | 新字段用 `.get()` 读取              | 旧对话 JSON 无 tool_calls 字段也能正常加载    |

## 九、文件清单

| 文件                        | 角色                                           |
| --------------------------- | ---------------------------------------------- |
| `chat_agent/async_utils.py` | async/sync 桥接，daemon 线程管理               |
| `chat_agent/mcp_client.py`  | MCP 连接、工具发现、格式转换、工具执行         |
| `chat_agent/client.py`      | LLMClient + ChatResponse，支持 tools 参数      |
| `chat_agent/agent.py`       | ChatAgent，ReAct 循环，消息构建                |
| `chat_agent/storage.py`     | ToolCall + Message 扩展，工具消息持久化        |
| `chat_agent/config.py`      | MCP server 配置加载                            |
| `main.py`                   | 启动时初始化 MCP，退出时清理                   |
| `mcp_servers.json`          | MCP server 配置文件                            |
| `test_mcp_server.py`        | 本地测试 MCP Server（get_weather + calculate） |

