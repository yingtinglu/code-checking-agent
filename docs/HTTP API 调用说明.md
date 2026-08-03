# HTTP API 调用说明

## 1. 完整调用链路

```
用户: analyze
  │
  ▼
DegradationAgent.analyze()
  │  构造消息列表: [system, user("分析支付域")]
  │  调用 LLMClient.chat(messages, tools=...)
  │
  ▼
LLM 返回 tool_calls: [{"function": {"name": "get_code_info", "arguments": "{\"month\":\"202606\",\"base_month\":\"202601\"}"}}]
  │
  ▼
DegradationAgent 解析 tool_calls
  │  遍历每个 tool_call:
  │    tool_name = "get_code_info"
  │    arguments = {"month": "202606", "base_month": "202601"}
  │    result = mcp_client.call_tool(tool_name, arguments)
  │
  ▼
MCPClient.call_tool("get_code_info", {"month": "202606", "base_month": "202601"})
  │  查找 "get_code_info" 属于哪个 MCP Server → degradation-server
  │  获取对应 session
  │  调用 session.call_tool("get_code_info", arguments)  # MCP SDK async 调用
  │  通过 stdio 发送请求到子进程
  │
  ▼
degradation_mcp_server.py (子进程)
  │  @server.call_tool() 处理函数被触发
  │  name = "get_code_info", arguments = {"month": "202606", "base_month": "202601"}
  │
  │  raw = provider.get_code_info(month="202606", base_month="202601")
  │       │
  │       ▼
  │  QualityPlatformProvider.get_code_info(month="202606", base_month="202601")
  │       │  构造 HTTP 请求
  │       │  resp = self.client.get(f"{self.api_base}/show/code-info",
  │       │                          params={"month": "202606", "baseMonth": "202601"})
  │       │  resp.raise_for_status()
  │       │  return resp.json()
  │       │
  │       ▼
  │  CodeQualityShow 后端 (Spring Boot, 端口 8081)
  │       │  GET /show/code-info?month=202606&baseMonth=202601
  │       │  查询 MySQL, 聚合数据, 返回 JSON
  │       │
  │       ▼
  │  HTTP 响应 (原始 JSON, 约 2000 token)
  │       [
  │         {"domain": "分发中台", "subDomainDatas": [
  │           {"domain": "CMS", "codeNumber": 8343147, "codeNumberPre": 8512323,
  │            "codeNumberBaseOver": 273696, "circularDependencies": 376,
  │            "circularDependenciesPre": 385, "circularDependenciesBaseOver": 182, ...}
  │         ]},
  │         ...
  │       ]
  │
  │  回到 MCP Server:
  │  result = simplify_code_info(raw)   # 调用 metrics.py 精简
  │  return [TextContent(type="text", text=json.dumps(result))]
  │
  ▼
MCPClient 收到结果 (精简后约 400 token)
  │  {"month": "202606", "base_month": "202601",
  │   "high_attention": [...], "needs_attention": [...], "overview": {...}}
  │
  ▼
DegradationAgent 收到结果
  │  将 tool 结果加入对话: messages.append({"role": "tool", "content": result})
  │  继续下一轮 LLM 调用...
```

## 2. QualityPlatformProvider — httpx 调用细节

### 2.1 初始化

```python
import os
import httpx
from typing import Optional, List, Dict

class QualityPlatformProvider:
    def __init__(self, api_base: str = None):
        self.api_base = api_base or os.environ.get(
            "QUALITY_API_BASE", "http://localhost:8081"
        )
        # trust_env=False: 不读取环境变量中的代理设置
        # 和现有 LLMClient 的 httpx 用法一致
        self.client = httpx.Client(
            base_url=self.api_base,
            timeout=30.0,
            trust_env=False,
        )
```

### 2.2 get_code_info — 核心数据获取

```python
def get_code_info(self, month: str, base_month: Optional[str] = None) -> List[Dict]:
    """获取指定月份的代码质量数据。

    对应平台 API: GET /show/code-info

    Args:
        month: 查询月份，格式 YYYYMM，如 "202606"
        base_month: 基比月份，格式 YYYYMM，如 "202601"。
                    不传则使用平台默认（上年12月）

    Returns:
        按域分组的质量数据列表。每个域包含 subDomainDatas 子列表。
        嵌套结构: [{domain, subDomainDatas: [{domain, codeNumber, ...}]}]

    实际响应示例 (已简化):
        [
          {
            "domain": "分发中台",
            "codeNumber": 12962555,
            "codeNumberPre": 13196054,
            "codeNumberBaseOver": 445126,
            "circularDependencies": 985,
            "circularDependenciesPre": 1007,
            "circularDependenciesBaseOver": 416,
            "codeLineDuplicationRate": 0.0378,
            "codeLineDuplicationRatePre": 0.0372,
            "codeLineDuplicationRateBaseOver": -0.0016,
            ...
            "subDomainDatas": [
              {
                "domain": "CMS",
                "codeNumber": 8343147,
                "circularDependencies": 376,
                "circularDependenciesBaseOver": 182,
                ...
              }
            ]
          }
        ]
    """
    params = {"month": month}
    if base_month:
        params["baseMonth"] = base_month

    resp = self.client.get("/show/code-info", params=params)
    resp.raise_for_status()
    return resp.json()
```

**关键点**：

- `base_month` 可选，不传时后端默认基比为上年12月
- 响应是嵌套结构：顶层是域，`subDomainDatas` 是分组
- 每个指标有4个变体：当前值、`*Pre`（环比上期）、`*BaseOver`（基比变化量）、`*PreOver`（环比变化量）
- `codeLineDuplicationRate` 返回的是小数（如 0.0478 表示 4.78%），不是百分比

### 2.3 get_code_history — 12个月历史趋势

```python
def get_code_history(self, domains: List[str], month: Optional[str] = None) -> Dict[str, List]:
    """获取指定域的 12 个月历史趋势。

    对应平台 API: GET /show/code-info/history

    Args:
        domains: 域名称列表，如 ["CMS"] 或 ["分发中台", "算法中台"]
        month: 截至月份，可选

    Returns:
        按域分组的历史数据字典。
        {"CMS": [{month: "202507", codeNumber: ..., ...}, ...12条]}

    实际响应示例:
        {
          "CMS": [
            {"domain": "CMS", "month": "202507", "codeNumber": 7800000,
             "circularDependencies": 300, ...},
            {"domain": "CMS", "month": "202508", ...},
            ...共12个月
          ]
        }
    """
    params = [("domain", d) for d in domains]  # 同名参数用列表
    if month:
        params.append(("month", month))

    resp = self.client.get("/show/code-info/history", params=params)
    resp.raise_for_status()
    return resp.json()
```

**关键点**：

- `domain` 参数是 List，httpx 传同名多参数：`?domain=CMS&domain=算法中台`
- 返回的 PeriodData 中**没有** `*Pre`/`*BaseOver` 变体，只有当月值——历史对比需要自己从12条数据中取
- 返回中包含 `techDebtRate` 字段（code-info 中没有）

### 2.4 get_domain_info — 域/分组/SL组列表

```python
def get_domain_info(self, type: int) -> List[str]:
    """获取域/分组/SL组列表。

    对应平台 API: GET /show/domain/info

    Args:
        type: 1=域列表, 2=分组列表, 3=SL组列表 (必填)

    Returns:
        名称字符串列表

    实际响应示例 (type=1):
        ["算法中台", "上架安全", "预装", "分发中台", "push商业化",
         "增长中台", "AppTouch", "其他", "联运", "生态", "客户端"]
    """
    resp = self.client.get("/show/domain/info", params={"type": type})
    resp.raise_for_status()
    return resp.json()
```

**关键点**：

- `type` 必填，后端没有默认值
- 返回的是纯字符串列表，不是对象

### 2.5 get_domain_detail — 按维度下钻

```python
def get_domain_detail(
    self,
    domain: str,
    month: str,
    dimension: str,
    base_month: Optional[str] = None,
) -> List[Dict]:
    """按维度下钻详情，含基比变化和贡献占比。

    对应平台 API: GET /show/domain/detail

    Args:
        domain: 域名称 (必填)
        month: 查询月份 YYYYMM (必填)
        dimension: "group" | "subgroup" | "microservice" (必填，默认 group)
        base_month: 基比月份 (可选)

    Returns:
        各维度的指标值 + BaseOver + Ratio

    实际响应示例:
        [
          {
            "dimensionName": "CMS",
            "codeNumber": 8343147,
            "circularDependencies": 376,
            "circularDependenciesBaseOver": 182,
            "circularDependenciesRatio": 43.8,
            "circularDependenciesPreOver": -9,
            "circularDependenciesPreRatio": 100.0,
            "reviewNumber": 0,
            ...
          },
          ...
        ]
    """
    params = {
        "domain": domain,
        "month": month,
        "dimension": dimension,
    }
    if base_month:
        params["baseMonth"] = base_month

    resp = self.client.get("/show/domain/detail", params=params)
    resp.raise_for_status()
    return resp.json()
```

**关键点**：

- `dimension` 决定按什么维度分组：
  - `group` → 按 groupName 分组
  - `subgroup` → 按 subgroupName 分组
  - `microservice` → 按微服务名分组
- 每个指标有 `*Ratio` 和 `*PreRatio` 字段：表示该维度增长占总增长的比例
- Ratio 对定位核心问题区域至关重要（如"支付核心组循环依赖 Ratio=67%"说明它贡献了67%的增长）
- `dimensionName` 是该维度条目的名称

### 2.6 get_microservices — 微服务级数据

```python
def get_microservices(self, month: str, domain: str) -> List[Dict]:
    """获取微服务级质量数据。

    对应平台 API: GET /show/code-info/microservices

    Args:
        month: 查询月份 YYYYMM (必填)
        domain: 域名称 (必填)

    Returns:
        微服务列表，每个含 microserviceName + 所有质量指标

    实际响应示例:
        [
          {
            "domain": "CMS",
            "microserviceName": "AppGalleryCmsAppLifecycleManageService",
            "subdomainName": "CMS",
            "codeNumber": 123456,
            "circularDependencies": 5,
            ...
          },
          ...共47条
        ]
    """
    resp = self.client.get(
        "/show/code-info/microservices",
        params={"month": month, "domain": domain},
    )
    resp.raise_for_status()
    return resp.json()
```

**关键点**：

- 两个参数都必填
- 返回格式和 code-info 类似，但多了 `microserviceName` 和 `subdomainName`
- 没有嵌套结构，直接是平铺列表

### 2.7 get_code_review — 代码审查数据

```python
def get_code_review(self, month: str) -> List[Dict]:
    """获取代码审查数据。

    对应平台 API: GET /show/code-review

    Args:
        month: 查询月份 YYYYMM (必填)

    Returns:
        按域分组的审查统计

    实际响应示例:
        [
          {"domain": "CMS", "reviewNumber": 150, "reviewNumberPre": 120},
          ...
        ]
    """
    resp = self.client.get("/show/code-review", params={"month": month})
    resp.raise_for_status()
    return resp.json()
```

## 3. 错误处理

```python
def _request(self, method: str, path: str, params: Dict = None) -> Dict | List:
    """统一请求方法，处理错误。"""
    try:
        resp = self.client.request(method, path, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        # 4xx/5xx 错误
        if e.response.status_code == 500:
            raise RuntimeError(f"平台服务内部错误: {path}") from e
        raise RuntimeError(f"API 请求失败: {e.response.status_code} {path}") from e
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接平台 API: {self.api_base}") from e
    except httpx.TimeoutException:
        raise RuntimeError(f"平台 API 请求超时: {path}") from e
```

## 4. 为什么请求后端而不是前端

前端 (localhost:5174) 是 Vite 开发服务器，只是一个 Vue SPA 静态页面 + API 代理。它通过 vite.config.js 的 proxy 配置将 `/show`、`/api`、`/management`、`/code-review` 路径转发到后端 (localhost:8081)：

```javascript
// vite.config.js
server: {
  proxy: {
    '/show': { target: 'http://localhost:8081', changeOrigin: true },
    '/api':  { target: 'http://localhost:8081', changeOrigin: true },
    // ...
  }
}
```

前端本身不产生数据，只是个透传代理。生产部署时 Vite 开发服务器不存在，后端直接提供服务。所以 MCP Server 直接请求后端 8081 是正确的——少一跳代理，更快更可靠。

如果后端 8081 端口未对 Agent 所在机器暴露，可设置 `QUALITY_API_BASE` 指向可达地址（如前端的代理地址 `http://localhost:5174`，但仅限开发环境）。

## 5. 常见陷阱

| 陷阱                                     | 说明                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| `codeLineDuplicationRate` 是小数         | 返回 0.0478 表示 4.78%，不是 4.78。展示和计算时需 ×100       |
| `domain/detail` 的 domain 参数要精确匹配 | 必须和 `domain/info` 返回的名称完全一致，含空格和大小写      |
| `domain` 参数在 history 中是同名多值     | httpx 用 `params=[("domain", d) for d in domains]` 传多值    |
| `subDomainDatas` 可能为 null             | code-info 的子域数据可能为 null 而非空列表，需 `item.get("subDomainDatas") or []` |
| `domain/detail` 当前返回 500             | 后端 bug，域名匹配逻辑可能有问题，待修复                     |
| `dangerousFunctions` 通常为 0            | 大部分域此指标为 0，精简时可跳过                             |
