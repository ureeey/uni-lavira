import os
import time
from openai import OpenAI
import httpx

# ── Timing hooks ──────────────────────────────────────────────
timings = {}

def _on_request(request):
    timings["request_start"] = time.monotonic()

def _on_response(response):
    timings["response_headers"] = time.monotonic()

http_client = httpx.Client(
    event_hooks={"request": [_on_request], "response": [_on_response]},
)

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key=os.getenv("QWEN_API_KEY"),
        # 以下为华北2（北京）地域的URL，各地域的URL不同。调用时请将{WorkspaceId}替换为真实的业务空间ID。
        base_url=os.getenv("QWEN_BASE_URL"),
        http_client=http_client,
    )

    t0 = time.monotonic()
    completion = client.chat.completions.create(
        model="qwen3.7-max",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    t1 = time.monotonic()

    print(completion.choices[0].message.content)

    # ── Timing report ─────────────────────────────────────────
    http_total = timings.get("response_headers", 0) - timings.get("request_start", 0)
    download   = t1 - timings.get("response_headers", t0)
    wall       = t1 - t0

    print()
    print("=" * 54)
    print("  Network Timing Breakdown")
    print("=" * 54)
    print(f"  Connection + Upload + Server Queue : {http_total:8.2f}s")
    print(f"  Response Download + Parse          : {download:8.2f}s")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total (wall clock)                  : {wall:8.2f}s")
    print("=" * 54)

    # Token usage (if available)
    usage = getattr(completion, "usage", None)
    if usage:
        print(f"  Prompt tokens : {usage.prompt_tokens}")
        print(f"  Output tokens : {usage.completion_tokens}")
        print(f"  Total tokens  : {usage.total_tokens}")

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")