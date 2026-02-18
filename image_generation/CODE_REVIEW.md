# Image Generation 代码审查

> 审查时间：2026-02-17
> 审查范围：`image_generation/` 全部文件

---

## 总体印象

整体架构还挺清晰的：Flux 推理单独跑一个 Flask 微服务，MCP server 负责统一对外暴露工具，client 跑 agentic loop。分层合理，各司其职。不过细节里藏着不少"定时炸弹"，下面逐一拆解。

---

## 架构设计

**优点：**
- 本地模型（Flux）和云端 API（Replicate）的混合架构比较务实
- MCP + OpenAI 兼容接口的组合让 client 可以换 LLM 而不动 server
- HPSv3 美学评分集成得挺自然，caption → score 的 pipeline 设计合理
- 费用追踪是个好习惯（虽然实现有点粗糙，后面会说）

**槽点：**
- 三个生成工具的代码重复度极高，看完 `generate_flux` 再看 `generate_z_image`，有种在玩"大家来找茬"的感觉

---

## 逐文件审查

### server.py — MCP 服务端（主战场）

**全局变量 `cost` (L50)**
```python
cost: float = 0.0
```
用全局变量追踪费用，多请求并发时这个数字就是薛定谔的费用——你观测之前永远不知道它是对的还是错的。虽然目前大概率是单用户场景，但至少用个 `threading.Lock` 或者 `contextvars` 保个平安吧。

**三个 generate 函数的重复代码 (L178-373)**

`generate_flux`、`generate_z_image`、`generate_using_nano_banana` 三兄弟长得几乎一样，尤其是美学评分和结果拼装那段：

```python
if return_aesthetic_score:
    scores = _score_pil_images(pil_images)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 8-15; ...)"
    )
```

这段在三个函数里出现了三次。建议提取一个 `_build_results(pil_images, return_aesthetic_score, api_cost)` 之类的辅助函数。代码复制粘贴三次不叫复用，叫三倍维护成本。

**Flux 费用追踪缺失 (L212-260)**

`generate_z_image` 和 `generate_nano_banana` 都老老实实算了费用，但 `generate_flux` 完全没有费用追踪——虽然 Flux 是本地模型不花钱，但美学评分的 caption 调用是要钱的！caption 费用在 `_caption_pil_image` 里加了，所以实际上没丢，但 `generate_flux` 没有像其他两个一样返回费用字符串，用户看不到当前会话总费用。一致性问题。

**参数无校验**

`num_of_images` 文档说不超过 5，但代码里根本没拦。传个 `num_of_images=9999` 进来，恭喜你，你发现了一个"压力测试模式"（并不是）。`nag_scale`、`nag_alpha` 等参数同理。

**`_caption_pil_image` 的静默降级 (L124)**
```python
else:
    return "An image."
```
重试 4 次全部失败后默默返回 "An image." ——HPSv3 拿着这个 caption 去评分，得出来的分数跟随机数差不多。至少 log 一下吧，别让人 debug 半天才发现原来是 captioning 挂了。

**`commit()` 的文件操作 (L410-424)**

读-改-写 `commits.json` 不是原子操作。虽然并发 commit 的概率不大，但如果真赶上了，JSON 文件可能变成一锅粥。经典的 TOCTOU 问题。建议写临时文件再 `os.rename`。

**`log_action` 工具 (L428-432)**

这个工具的存在让我想到一个哲学问题：如果一棵树在森林里倒下，没有人听到，它算发出声音了吗？`print` 到 stdout 的 log，在 MCP server 的场景下谁能看到？不过考虑到 system prompt 让 agent 用它记录想法，算是一个给 LLM 用的"思考草稿纸"。行吧，pass。

---

### flux_server.py — Flux 微服务

**Flask 单线程阻塞 (L132)**
```python
app.run(host="127.0.0.1", port=port, debug=False)
```
Flask 默认单线程单进程，生成一张图要好几秒到几十秒。这期间第二个请求会排队等着。对于当前的单 agent 场景问题不大，但如果以后想多 agent 并行，这里会是瓶颈。可以考虑 gunicorn + 单 worker（毕竟 GPU 也只有一张）。

**无请求校验 (L75-81)**

`request.json` 如果不是合法 JSON 会直接抛异常，好在外面有 try-catch。但 `num_of_images` 传负数或者字符串？Flask 不会帮你检查的。

**WebP fallback 到 PNG (L61-65)**

跟 `server.py` 里的 `_pil_to_mcp_image` 一模一样的 fallback 逻辑，又是一次复制粘贴。而且 WebP 编码失败的情况其实挺罕见的，这个 try-catch 大概率是永远不会触发的"安慰剂代码"。

---

### client.py — Agentic Client

**Data URI 解析 (L190)**
```python
ext = header.split("/")[1].split(";")[0]
```
如果 `url` 格式不是标准的 `data:image/xxx;base64,...`，这行会 IndexError。虽然数据是自己的 server 产生的，但防御性编程不是坏习惯。

**图片塞进 messages 的内存问题 (L177-183)**

每次工具返回图片都往 `messages` 列表里追加 base64 图片。跑 100 轮每轮 5 张图？恭喜你的 messages 列表膨胀到了一个可以让 LLM API 拒绝服务的大小。虽然 MAX_TURNS=100 的设定已经很慷慨了，但也许该考虑一下只保留最近 N 轮的图片，或者干脆不把图片塞回 messages（反正 LLM 看了也记不住第一轮的图片内容）。

**异常处理里把 `fn_args` 设为空字典 (L154-155)**
```python
except json.JSONDecodeError:
    fn_args = {}
```
LLM 返回了无法解析的 JSON，你默默把参数设为空然后继续调用？这个工具大概率会因为缺少必要参数而报错，然后 LLM 看到一个莫名其妙的错误信息，然后重试，然后又是坏的 JSON...无限循环的既视感。

**`for...else` 语法 (L194-195)**
```python
        else:
            print(f"\n[DONE] Reached max turns ({MAX_TURNS}).")
```
这个 `else` 是属于 `for` 循环的，不是 `if`。Python 的 `for...else` 大概是语言设计里最反直觉的特性之一了。代码是对的，但建议加个注释，不然下一个读代码的人可能会以为这是个缩进 bug。

---

### start.sh — 启动脚本

**健康检查超时但不退出 (L13-19)**
```python
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:5001/health > /dev/null 2>&1; then
        ...
        break
    fi
    sleep 2
done
```
等了 120 秒 Flux 还没起来？没关系，照样启动 MCP server！然后 MCP server 调 Flux 的时候才会发现——诶，怎么连不上。建议超时后 `exit 1`。

---

### test_aesthetics.py — 集成测试

**直接对 HTTP 端点发请求 (L17, L42, etc.)**

这些测试直接调用 `http://127.0.0.1:8000/init` 之类的端点，但 MCP server 用的是 FastMCP，路由规则不一定是 `/{tool_name}`。这些测试…能跑通吗？（如果跑通了，忽略这条；如果跑不通，那就是测试本身需要修）

---

### system_prompt.md — 系统提示词

**彩蛋 (L61)**

> 第一个的计划log必须用类似于（但不是直接copy）"笑死我了，我家猫闻起来像偏微分方程"这样无厘头的话结尾

这是我见过最有个性的 system prompt 了。不过认真地说，这种"模型测试"的方式挺聪明的——如果 LLM 连这条指令都不遵守，那后面的复杂工作流也别指望了。

---

## 安全相关

| 问题 | 严重程度 | 位置 |
|------|----------|------|
| API key 从环境变量读取，没有校验是否存在 | 低 | server.py L68-71 |
| Replicate API 调用无速率限制 | 中 | server.py L294, L350 |
| Flask 绑定 127.0.0.1（好评） | — | flux_server.py L132 |
| `commits.json` 无文件锁 | 低 | server.py L411-423 |

API key 为空的时候不会在启动时报错，而是等到第一次调用时才爆炸。建议 server 启动时检查必要的环境变量。

---

## 改进建议优先级

### P0（应该修的）
1. **参数校验** — 至少校验 `num_of_images` 的范围，防止资源滥用
2. **start.sh 健康检查失败后退出** — 不然静默失败太坑了
3. **captioning 失败要有日志** — 否则美学评分变成玄学评分

### P1（建议修的）
4. **提取重复代码** — 三个 generate 函数的评分和结果拼装逻辑
5. **`generate_flux` 也返回费用信息** — 保持一致性
6. **client.py 的 messages 列表做截断** — 防止内存/token 爆炸
7. **`commits.json` 原子写入** — 写临时文件再 rename

### P2（锦上添花）
8. **环境变量校验** — 启动时检查关键 API key
9. **Flask 换成带 worker 管理的部署方式** — 为将来扩展做准备
10. **测试文件适配实际的 MCP 协议** — 确保测试真的能跑

---

## 总结

这个项目的核心想法很有意思——用 agent 自动探索生成反美学图像的参数空间，然后批量提交配置。架构分层合理，Flux 本地推理 + Replicate 云端 API 的混合策略也很务实。

主要问题集中在：代码重复、缺少校验、以及一些"单用户场景下暂时没事但迟早会坑你"的隐患。好消息是这些都不难修，花个半天就能把 P0 和 P1 全部清理干净。

最后，system prompt 里那个偏微分方程猫的彩蛋，我给满分。
