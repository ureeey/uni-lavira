`x-dashscope-session-cache: enable` 是 DashScope 平台提供的一种**临时提示词缓存机制**（也称为 Session 缓存），用于在多轮对话场景中自动缓存已处理的上下文，从而**降低推理延迟、减少重复计算、节省 Token 消耗成本**。该功能无需用户手动管理缓存逻辑，只需在请求头中启用即可生效。

以下从多个维度详细说明其工作原理、使用方式及限制条件：

---

### 一、核心作用与优势

- **自动缓存对话前缀**：服务端会将满足条件的提示词（prompt）缓存起来，后续请求若包含相同或延续性前缀，可直接复用已计算的中间状态（如注意力 Key/Value 向量的等效表示），避免重复推理。
- **降低首 Token 延迟**（TTFT）：尤其在长上下文或多轮对话中，第二轮及以后的响应速度显著提升。
- **节省计费 Token 数量**：缓存命中的部分不计入本次请求的 `prompt_tokens`，从而降低调用成本。
- **无代码侵入性**：业务系统只需添加请求头，无需重构消息历史管理逻辑。

---

### 二、启用方式

在调用 DashScope API 时，**在 HTTP 请求头中添加**：

```http
x-dashscope-session-cache: enable
```

> - 默认值为 `disable`，即不启用缓存；
> - 设置为 `enable` 表示开启；
> - 该 Header 对所有支持的模型和接口均有效，包括标准 Generation API 和 OpenAI 兼容的 `/responses` 接口。

#### 示例（cURL）

```bash
curl -X POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/responses \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-dashscope-session-cache: enable" \
  -d '{
    "model": "qwen3.7-plus",
    "input": "人工智能是计算机科学的一个重要分支..."
  }'
```

#### 示例（Node.js + OpenAI SDK）

```javascript
const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
  defaultHeaders: { "x-dashscope-session-cache": "enable" }
});
```

---

### 三、缓存触发条件与机制

#### 1. **最小缓存长度**
- 仅当**累计提示词长度 ≥ 1024 Token** 时，系统才会创建缓存。
- 若首轮对话不足 1024 Token，但多轮累积后超过该阈值，**后续请求仍可触发缓存创建**。

#### 2. **缓存有效期**
- 缓存有效期为 **5 分钟**，超时后自动失效。
- 在有效期内，同一会话的后续请求若延续上下文，将自动尝试命中缓存。

#### 3. **缓存粒度**
- 缓存基于**完整提示词前缀**进行匹配，非模糊匹配；
- 因此，**必须保持消息历史顺序和内容一致**，否则无法命中。

#### 4. **多轮对话衔接方式**
- 在 `/responses` 接口（OpenAI 兼容模式）中，可通过 `previous_response_id` 显式关联上一轮响应，服务端据此构建连续上下文并判断是否命中缓存。
- 在标准 Generation API 中，需**手动传入完整 messages 数组**，系统根据内容判断缓存可用性。

---

### 四、支持的模型范围

Session 缓存功能**并非全模型支持**，当前明确支持的模型包括：

- `qwen3-max`、`qwen3.7-max`、`qwen3.7-max-2026-05-20`、`qwen3.7-max-2026-06-08`
- `qwen3.7-plus`、`qwen3.7-plus-2026-05-26`
- `qwen3.6-plus`、`qwen3.5-plus`
- `qwen3.6-flash`、`qwen3.5-flash`
- `qwen-plus`、`qwen-flash`
- `qwen3-coder-plus`、`qwen3-coder-flash`

> **注意**：`qwen-vl`、`qwen-audio` 等多模态模型**未列入支持列表**，可能不生效。

---

### 五、效果验证方式

缓存命中后，响应中的 `usage` 字段会包含额外信息，例如：

```json
{
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "prompt_tokens_details": {
      "cached_tokens": 1024   // 表示有 1024 Token 来自缓存
    }
  }
}
```

通过检查 `cached_tokens` 字段是否大于 0，即可确认缓存是否生效。

---

### 六、重要限制与注意事项

- **不等于 KV Cache**：Session 缓存是 DashScope 托管服务提供的**应用层缓存机制**，与 PAI-EAS 平台的底层 KV Cache 技术不同，后者需自建服务且不可通过公共 API 启用。
- **仅适用于托管 API 调用**：包括 `dashscope.aliyuncs.com` 和 `{WorkspaceId}.maas.aliyuncs.com` 域名下的所有接口。
- **不保证 100% 命中**：若上下文发生微小变动（如标点、空格差异），可能导致缓存失效。
- **不适用于流式中断重试**：若流式请求中途断开，缓存状态不会保留至下一次请求。

---

综上，`x-dashscope-session-cache: enable` 是一种轻量、易用且高效的上下文缓存优化手段，特别适合**长文本交互或多轮客服对话**场景。建议在符合条件的模型上调用时默认开启，以提升性能与性价比。 

相关链接 
DashScope Java代码 https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope
角色扮演（Qwen-Character） 场景特殊需求 启用session cache提升缓存命中 https://help.aliyun.com/zh/model-studio/role-play
创建响应 https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses
OpenAI兼容-Responses 代码示例 Session 缓存 https://help.aliyun.com/zh/model-studio/compatibility-with-openai-responses-api
新版智能体应用 API 调用方式 https://help.aliyun.com/zh/model-studio/new-agent-application-api-reference
HTTP API DashScope同步调用（Fun-ASR-Flash） 请求头 https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api
调用智能体应用 核心功能 流式输出 curl https://help.aliyun.com/zh/model-studio/call-single-agent-application/
流式输出 如何使用 步骤二：发起流式请求 DashScope https://help.aliyun.com/zh/model-studio/stream
工作流与旧版智能体应用 API 调用方式 https://help.aliyun.com/zh/model-studio/agent-and-workflow-application-api-reference
代码解释器 使用方式  DashScope https://help.aliyun.com/zh/model-studio/qwen-code-interpreter
上下文缓存 显式缓存 快速开始 DashScope https://help.aliyun.com/zh/model-studio/context-cache
以加密的方式接入模型推理功能 DashScope SDK调用（自动加密·开箱即用） 接入流程 https://help.aliyun.com/zh/model-studio/encrypted-access-to-model-inference
DashScope SDK连接复用配置 Python SDK HTTP同步调用方式 代码示例 https://help.aliyun.com/zh/model-studio/connection-multiplexing-configuration
关于dashscope-sdk-java启动时找不到类 https://developer.aliyun.com/ask/695759
Python SDK 请求参数 https://help.aliyun.com/zh/model-studio/qwen-tts-realtime-python-sdk
Java SDK 请求参数 https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-java-sdk
千问-文生图 异步接口 DashScope SDK调用 https://help.aliyun.com/zh/model-studio/qwen-image-api