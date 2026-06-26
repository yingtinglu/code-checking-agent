"""配置管理模块"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    """Agent 配置类"""
    
    # LLM 配置
    model: str = "MiniMax-M2.7"  # CCS 默认模型
    api_key: Optional[str] = None  # 不设置默认值，通过环境变量或参数传入
    api_base: Optional[str] = None  # 支持自定义 API 地址（如 CCS）
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # 对话配置
    system_prompt: str = "你是一个有帮助的AI助手，请用简洁、清晰的语言回答用的问题。"
    
    # 请求配置
    timeout: int = 60
    max_retries: int = 3

    # MCP 配置
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    mcp_enabled: bool = False
    max_tool_iterations: int = 10
    
    def __post_init__(self):
        """自动从环境变量加载配置（优先级：环境变量 > 默认值）"""
        if self.api_key is None:
            self.api_key = (
                os.environ.get("CCS_API_KEY") or
                os.environ.get("ANTHROPIC_AUTH_TOKEN") or
                os.environ.get("OPENAI_API_KEY") or
                "sk-123"  # 最终兜底默认值
            )
        if self.api_base is None:
            self.api_base = (
                os.environ.get("CCS_API_BASE") or
                os.environ.get("ANTHROPIC_BASE_URL") or
                os.environ.get("OPENAI_API_BASE") or
                "http://100.85.219.23:28081/v1"  # 底默认值（SDK 会自动添加 /chat/completions）
            )
        self._load_mcp_config()
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置（优先级：CCS > OpenAI）"""
        return cls(
            api_key=(
                os.environ.get("CCS_API_KEY") or
                os.environ.get("ANTHROPIC_AUTH_TOKEN") or
                os.environ.get("OPENAI_API_KEY") or
                "sk-123"
            ),
            api_base=(
                os.environ.get("CCS_API_BASE") or
                os.environ.get("ANTHROPIC_BASE_URL") or
                os.environ.get("OPENAI_API_BASE") or
                "http://100.85.219.23:28081/v1"
            ),
            model=(
                os.environ.get("CCS_MODEL") or
                os.environ.get("ANTHROPIC_MODEL") or
                os.environ.get("OPENAI_MODEL", "MiniMax-M2.7")
            ),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
            max_tool_iterations=int(os.environ.get("MCP_MAX_ITERATIONS", "10")),
        )

    def _load_mcp_config(self):
        """从 JSON 文件加载 MCP Server 配置。"""
        mcp_config_path = os.environ.get("MCP_SERVERS_CONFIG")
        if mcp_config_path and os.path.exists(mcp_config_path):
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            servers = data.get("mcpServers", data.get("servers", []))
            if isinstance(servers, dict):
                # Claude Desktop 格式: {"mcpServers": {"name": {...}, ...}}
                server_list = []
                for name, cfg in servers.items():
                    entry = {"name": name, **cfg}
                    server_list.append(entry)
                servers = server_list
            self.mcp_servers = servers
            self.mcp_enabled = bool(servers)
