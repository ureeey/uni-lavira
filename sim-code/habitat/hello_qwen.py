import os
import sys
import time
import json
import socket
import argparse
import subprocess
from datetime import datetime, timezone
from urllib.parse import urlparse
from openai import OpenAI
import httpx

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen API smoke test with timing")
parser.add_argument("--force-ipv4", action="store_true", default=False,
                    help="Monkey-patch socket.getaddrinfo → AF_INET only, "
                         "bypassing IPv6 to avoid Happy Eyeballs timeout.")
args = parser.parse_args()

# ── NTP 校时 ─────────────────────────────────────────────────
def ntp_sync():
    commands = [
        ["chronyc", "-a", "makestep"],
        ["ntpdate", "-u", "ntp.aliyun.com"],
        ["sntp", "-s", "ntp.aliyun.com"],
    ]
    for cmd in commands:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                print(f"[NTP] sync OK via {' '.join(cmd[:2])}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            continue
    print("[NTP] ⚠ no working NTP tool; install:  sudo apt install ntpsec-utils",
          file=sys.stderr)
    return False

print("=" * 64)
print("  NTP time sync ...")
ntp_sync()
print("=" * 64)
print()

# ── 环境诊断 ──────────────────────────────────────────────────
print("=" * 64)
print("  Environment")
print("=" * 64)
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "NO_PROXY", "no_proxy"]:
    val = os.environ.get(var, "")
    print(f"  {var} = {val!r}")
print()

# ── IPv4 策略 ─────────────────────────────────────────────────
# 阿里云 NLB 的 IPv6 (2408:400a:…) 在本机 ISP 下不可达，
# httpx 默认 Happy Eyeballs 会先尝试 IPv6 导致 connect 超时。
# --force-ipv4 开启后 monkey-patch socket.getaddrinfo → AF_INET only。
_orig_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

ipv4_enabled = args.force_ipv4
if ipv4_enabled:
    socket.getaddrinfo = _getaddrinfo_ipv4
    print("[IPv4] socket.getaddrinfo → AF_INET only  (--force-ipv4 ON)")
else:
    print("[IPv4] default (IPv6 + IPv4),  pass --force-ipv4 to force IPv4-only")
print()

# ── DNS 信息 ──────────────────────────────────────────────────
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
hostname = urlparse(BASE_URL).hostname
print(f"  Target: {hostname}")
try:
    addrs = _orig_getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
    ips = set(a[4][0] for a in addrs)
    print(f"  IPv4: {ips}")
except Exception:
    pass
try:
    addrs6 = _orig_getaddrinfo(hostname, 443, socket.AF_INET6, socket.SOCK_STREAM)
    ips6 = set(a[4][0] for a in addrs6)
    tag = "  (已 bypass)" if ipv4_enabled else ""
    print(f"  IPv6: {ips6}{tag}")
except Exception:
    pass
print()

# ── Timing hooks ──────────────────────────────────────────────
timings = {}
req_info = {}
res_info = {}

def _on_request(request):
    timings["request_start_mono"] = time.monotonic()
    timings["request_start_wall"] = datetime.now(timezone.utc)
    req_info["method"] = request.method
    req_info["url"] = str(request.url)
    req_info["headers"] = dict(request.headers)
    try:
        req_info["body"] = request.content.decode("utf-8", errors="replace")
    except Exception:
        req_info["body"] = "(unable to read)"

def _on_response(response):
    timings["response_headers_mono"] = time.monotonic()
    timings["response_headers_wall"] = datetime.now(timezone.utc)
    res_info["status"] = f"{response.status_code} {response.reason_phrase}"
    res_info["http_version"] = response.http_version
    res_info["headers"] = dict(response.headers)

http_client = httpx.Client(
    http2=True,
    event_hooks={"request": [_on_request], "response": [_on_response]},
)

MAX_BODY = 2000

try:
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=BASE_URL,
        http_client=http_client,
    )

    t0_mono = time.monotonic()
    t0_wall = datetime.now(timezone.utc)
    completion = client.chat.completions.create(
        model="qwen3.7-max",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    t1_mono = time.monotonic()
    t1_wall = datetime.now(timezone.utc)

    print(completion.choices[0].message.content)

    # ── 耗时统计 ──────────────────────────────────────────────
    http_total = timings.get("response_headers_mono", 0) - timings.get("request_start_mono", 0)
    download   = t1_mono - timings.get("response_headers_mono", t0_mono)
    wall       = t1_mono - t0_mono

    print()
    print("=" * 64)
    print("  Network Timing Breakdown")
    print("=" * 64)
    print(f"  Connection + Upload + Server Queue : {http_total:8.2f}s")
    print(f"  Response Download + Parse          : {download:8.2f}s")
    print(f"  ───────────────────────────────────────────")
    print(f"  Total (wall clock)                  : {wall:8.2f}s")

    usage = getattr(completion, "usage", None)
    if usage:
        print(f"  Prompt tokens : {usage.prompt_tokens}")
        print(f"  Output tokens : {usage.completion_tokens}")
        print(f"  Total tokens  : {usage.total_tokens}")

    # ── 服务端时间戳 ────────────────────────────────────────
    server_headers = res_info.get("headers", {})
    print()
    print("=" * 64)
    print("  Server-Side Timing Headers  (from HTTP response)")
    print("=" * 64)
    for k in ["date", "x-request-id", "x-envoy-upstream-service-time",
              "req-cost-time", "req-arrive-time", "resp-start-time"]:
        v = server_headers.get(k, "")
        if v:
            print(f"  {k}: {v}")

    arrive_ms = server_headers.get("req-arrive-time")
    resp_ms   = server_headers.get("resp-start-time")
    if arrive_ms:
        try:
            dt = datetime.fromtimestamp(int(arrive_ms)/1000, tz=timezone.utc)
            print(f"    → req-arrive-time  = {dt.isoformat()}")
        except Exception:
            pass
    if resp_ms:
        try:
            dt = datetime.fromtimestamp(int(resp_ms)/1000, tz=timezone.utc)
            print(f"    → resp-start-time  = {dt.isoformat()}")
        except Exception:
            pass
    if arrive_ms and resp_ms:
        try:
            server_elapsed = (int(resp_ms) - int(arrive_ms)) / 1000.0
            print(f"    → server elapsed   = {server_elapsed:.2f}s  (arrive → resp_start)")
        except Exception:
            pass

    # ── Request ──────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  >>> HTTP Request >>>")
    print("=" * 64)
    print(f"  {req_info.get('method','?')} {req_info.get('url','?')}")
    print(f"  Content-Type: {req_info.get('headers',{}).get('content-type','?')}")
    body = req_info.get("body", "")
    try:
        parsed = json.loads(body)
        print(f"  model: {parsed.get('model','?')}")
        print(f"  messages ({len(parsed.get('messages',[]))}):")
        for m in parsed.get("messages", []):
            content = str(m.get("content", ""))[:200].replace("\n", "\\n")
            print(f"    [{m.get('role','?')}] {content}")
    except (json.JSONDecodeError, TypeError):
        pass

    # ── Response ──────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  <<< HTTP Response <<<")
    print("=" * 64)
    print(f"  Protocol: {res_info.get('http_version','?')}")
    print(f"  Status :  {res_info.get('status','?')}")
    print(f"  Content-Type: {res_info.get('headers',{}).get('content-type','?')}")
    for i, c in enumerate(completion.choices):
        content = (c.message.content or "")[:300].replace("\n", "\\n")
        role = getattr(c.message, "role", "?")
        print(f"  choices[{i}] {role}: {content}")
    if usage:
        print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    try:
        raw = completion.model_dump_json(indent=2)
        if len(raw) > MAX_BODY:
            raw = raw[:MAX_BODY] + f"\n... (truncated)"
        print(f"  ── raw JSON ──\n{raw}")
    except Exception:
        pass
    print("=" * 64)

    # ── 时间戳 ──────────────────────────────────────────────
    def _fmt(dt):
        return f"{dt.isoformat()}  (unix={dt.timestamp():.6f})"

    print()
    print("=" * 64)
    print("  Client-Side Wall-Clock Timestamps  (UTC)")
    print("=" * 64)
    print(f"  Client ready   : {_fmt(t0_wall)}")
    if "request_start_wall" in timings:
        print(f"  Request sent   : {_fmt(timings['request_start_wall'])}")
    if "response_headers_wall" in timings:
        print(f"  Response recv  : {_fmt(timings['response_headers_wall'])}")
    print(f"  Client done    : {_fmt(t1_wall)}")
    print("=" * 64)

    # ── 差距分析 ──────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  Client ↔ Server Gap Analysis")
    print("=" * 64)
    if arrive_ms and "request_start_wall" in timings:
        arrive_s = int(arrive_ms) / 1000.0
        send_s = timings["request_start_wall"].timestamp()
        print(f"  Request in-flight  (client send → server arrive) : {arrive_s - send_s:+.3f}s")
    if resp_ms and "response_headers_wall" in timings:
        resp_start_s = int(resp_ms) / 1000.0
        recv_s = timings["response_headers_wall"].timestamp()
        print(f"  Response in-flight (server start → client recv) : {recv_s - resp_start_s:+.3f}s")
    if arrive_ms and resp_ms and "response_headers_wall" in timings:
        server_total = (int(resp_ms) - int(arrive_ms)) / 1000.0
        client_total = (timings["response_headers_wall"] - timings["request_start_wall"]).total_seconds()
        print(f"  Network RTT (client_total - server_elapsed)     : {client_total - server_total:+.3f}s")
    print(f"  --force-ipv4 : {'ON' if ipv4_enabled else 'OFF'}")
    print("=" * 64)

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")