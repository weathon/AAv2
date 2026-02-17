# 代码审查：主目录 (OpenAI Agent SDK) vs .mcp_version (MCP)

## 架构差异总览

| 方面 | 主目录 (Agent SDK) | MCP 版本 |
|---|---|---|
| **框架** | OpenAI Agents SDK (`Agent`, `Runner`, `@function_tool`) | MCP 协议 (`FastMCP` 服务端 + 手动客户端循环) |
| **拓扑** | 单进程，工具在进程内调用 | 客户端-服务端，通过 SSE 传输 |
| **LLM 模型** | `google/gemini-3-flash-preview` | `moonshotai/kimi-k2.5` |
| **图片描述模型** | `moonshotai/kimi-k2.5` | `qwen/qwen3-vl-30b-a3b-instruct` |
| **追踪系统** | `wandb` | `weave` (W&B Weave) |
| **资源加载** | 即时加载（import 时） | 延迟加载（通过 `init()` 工具） |
| **图片返回类型** | `ToolOutputImage` (base64 PNG) | `MCPImage` (二进制 PNG) |

---

## 逐文件差异

### 1. Agent 入口 (`agent_main.py` vs `client.py`)

- **主目录**：一行 `Runner.run_sync(agent, "Psychedelic art", max_turns=100)` —— SDK 处理整个 agent 循环。
- **MCP**：约 300 行手动 agent 循环，包含：
  - 通过 `session.list_tools()` 发现 MCP 工具
  - LLM 重试 + 指数退避 + 120 秒超时
  - 图片压缩流水线（WebP 优先，JPEG 回退，1536px）
  - 旧消息图片渐进降级（超过 10 条消息后从 1536px 降至 1024px）
  - sample 日志强制执行（每次 `sample` 后必须调用 `log_actions`）
  - 采样产物保存（PNG + markdown 到 `sample_logs/`）
  - OpenRouter provider 路由提示：`extra_body={"provider": {"order": ["moonshotai/int4"]}}`
  - 图片从 tool 结果中分离到 user 消息（兼容不支持 tool 结果中图片的模型）

### 2. 工具实现 (`search_tools.py` + `commit_tools.py` vs `server.py`)

- **主目录**：工具分布在 2 个文件，使用 `@function_tool` 装饰器
- **MCP**：所有工具合并在 `server.py`，使用 `@mcp.tool()` 装饰器
- **MCP 新增**：`test_image` 工具（主目录没有）、每个操作的 weave 追踪、所有工具的 `_require_init()` 守卫
- **主目录独有**：`sample` 工具中 `wandb.log({"sample_result": wandb.Image(...)})` 记录图片（MCP 没有）
- **核心搜索/采样/提交逻辑功能上完全一致**（相同的余弦相似度、阈值逻辑、路径构建）

### 3. `dataset_loader.py`

- **主目录**：`sys.path.append("../")` —— 相对路径
- **MCP**：`sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))` —— 基于 `__file__` 的路径（更健壮）
- **MCP 新增**：`dataset_loader_summary()` 函数用于元数据报告

### 4. `image_utils.py`

- **主目录**：有 `encode()` 返回 `ToolOutputImage`；错误信息写的是 "Need 0 or more images"（错误——应该是 "1 or more"）
- **MCP**：有 `encode_to_base64()` 返回纯 base64 字符串；错误信息正确写为 "Need 1 or more images"
- `grid_stack`、`vstack`、`hstack` **完全一致**

### 5. `system_prompt.md`

- 两个版本**完全相同**（同一个文件）。这会导致问题——见下方 bug 列表。

---

## MCP 服务端潜在问题 / Bug

### 大概率 Bug

**1. `log_actions`、`status`、`undo_commit` 不应该要求 `init()`**
- 位置：`server.py:793`、`server.py:705`、`server.py:677`

这些工具只写日志文件或操作 `dataset_commits` 字典 + JSON 文件，完全不使用任何初始化资源（model、embeddings、HPSv3）。但它们都调用了 `_require_init()`，意味着 agent **在 `init()` 完成前甚至不能记录日志或查看状态**。系统提示词指示 agent 先记录计划再调用 init，但这会失败。

**2. `commit` 使用 `>=` 阈值，但系统提示词说"严格大于"**
- 位置：`server.py:634`

```python
mask = res >= threshold  # 按系统提示词应该是 res > threshold
```

系统提示词写的是"相似度**严格大于** `threshold`"。两个版本都用了 `>=`。阈值边界上的图片行为与文档不一致。

**3. `search` 工具没有返回相似度分数**
- 位置：`server.py:452-455`

系统提示词说 search "返回前t个预览结果及其**相似度分数**。同时返回所有图片的**相似度分布**"。但两个版本都只返回图片网格和一条文本消息，不返回分数。Agent 无法根据分数来确定 `commit`/`sample` 的合适阈值。

**4. 主目录缺少 `init` 工具，但共享的系统提示词要求调用它**

共享的 `system_prompt.md` 说"必须先调用一次 `init()`"，但主目录的 `agent_main.py` 没有暴露 `init` 作为工具（8 个工具中没有 init）。Agent 尝试调用时会报错。MCP 版本正确处理了这一点。

### 潜在问题

**5. 可变默认参数**
- 位置：`server.py:418`、`server.py:469`、`server.py:533`、`server.py:596`

```python
def search(query, dataset, negative_prompts: list[str] = [], ...):
```

使用可变默认值 (`[]`) 是经典的 Python 反模式。FastMCP 可能每次调用都创建新实例，所以实际上可能不会出 bug，但仍有风险。建议改为 `None` 并在函数内处理。

**6. 图片描述模型不一致导致美学评分不同**

服务端用 `qwen/qwen3-vl-30b-a3b-instruct`（`server.py:194`），主目录用 `moonshotai/kimi-k2.5`（`search_tools.py:47`）。由于 HPSv3 评分依赖于 caption 质量，相同图片在两个实现中会得到不同的美学分数。

**7. 硬编码用户路径**
- 位置：`server.py:321`、`server.py:639`

```python
f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}"
```

两个版本都硬编码了这个路径，不可移植。

**8. 工具返回类型不一致**

`search`、`sample`、`sample_from_committed` 正常时返回 `list`（图片 + 文本），错误/无结果时返回 `str`。类型不一致可能导致 MCP 客户端解析混乱。

**9. `_search_impl` 有三种返回类型**
- 位置：`server.py:335-337`

无图片时返回 `None`，正常时返回 PIL Image，`return_paths=True` 时返回路径列表。一个函数三种返回类型不够清晰。

**10. 图片双重编码开销**

服务端通过 `_pil_to_mcp_image()` 编码为 PNG，通过 SSE 传输，然后客户端又通过 `_compress_image_for_llm()` 重新编码为 WebP。PNG 步骤是浪费的，因为最终格式是 WebP。可以在服务端直接压缩以节省带宽。

**11. 主目录没有 caption 空值检查**
- 位置：`search_tools.py:65-67`

```python
caption = completion.choices[0].message.content
return caption  # 可能是 None！
```

主目录不检查 `None` 或空 caption 就直接返回。MCP 版本正确做了检查（`server.py:222`）。

**12. 主目录 `image_utils.py` 错误信息矛盾**
- 位置：`image_utils.py:10`

```python
raise ValueError("Need 0 or more images")  # 应该是 "Need 1 or more"
```

检查 `if len(images) == 0` 时抛出这个错误，但消息说 "0 or more" 与逻辑矛盾。MCP 版本修正为 "1 or more"。

---

## MCP 版本相比主目录的改进

- 延迟初始化（`init()` 工具）
- 图片压缩流水线（WebP + 渐进降级）
- sample 日志强制执行机制
- weave 追踪（每个操作都有记录）
- 更健壮的路径处理（`__file__` 基准）
- 修正了错误信息
- caption 空值检查
- LLM 重试 + 超时机制
- 采样产物持久化（PNG + markdown）
