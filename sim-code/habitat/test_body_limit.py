#!/usr/bin/env python3
"""Compare HTTP body size limits across 3 endpoints.

    MaaS / OpenAI       — OpenAI-compatible format, dedicated workspace
    MaaS / DashScope    — DashScope native format on the SAME dedicated workspace
    DashScope public    — DashScope native format on public shared endpoint

Usage:
    source .env.local && python test_body_limit.py [--min-mb 10] [--max-mb 30] [--step-mb 2]
"""

import argparse, base64, io, json, os, socket, sys, time, urllib.request, urllib.error
from PIL import Image
import numpy as np

# ── Force IPv4 (same as api.py) ─────────────────────────────────────
_orig_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _getaddrinfo_ipv4

# ── Config ───────────────────────────────────────────────────────────
API_KEY = os.environ.get("DASHSCOPE_API_KEY",
             os.environ.get("VA_API_KEY", os.environ.get("LA_API_KEY", ""))).strip()

_maas_base = os.environ.get("VA_BASE_URL",
              "https://ws-gs1ofh9fdpuo75kq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
_maas_base = _maas_base.rstrip("/")
_ws_root   = _maas_base.replace("/compatible-mode/v1", "")

MAAS_OAI_URL = _maas_base + "/chat/completions"
MAAS_DS_URL  = _ws_root + "/api/v1/services/aigc/multimodal-generation/generation"
DS_PUB_URL   = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

MODEL = os.environ.get("VA_MODEL_NAME", "qwen3.6-plus")
TEXT  = "Describe this image in one word."

# ── Image generator ──────────────────────────────────────────────────
def _make_png_url(size_kb: int) -> str:
    side = max(64, int((size_kb * 1024 / 3) ** 0.5))
    img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (side, side, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Message builders ─────────────────────────────────────────────────
def _make_openai_messages(target_bytes: int):
    ref_url = _make_png_url(200)
    ref_text = {"type": "text", "text": TEXT}
    ref_img  = {"type": "image_url", "image_url": {"url": ref_url}}
    per_pair = len(json.dumps(ref_text).encode()) + len(json.dumps(ref_img).encode())
    base = len(json.dumps([{"role": "user", "content": [ref_text]}]).encode())
    n = max(0, (target_bytes - base) // max(1, per_pair))
    content = [{"type": "text", "text": TEXT}]
    for i in range(n):
        content.append({"type": "image_url", "image_url": {"url": ref_url}})
        content.append({"type": "text", "text": f"View {i+1}"})
    return [{"role": "user", "content": content}], n

def _make_dashscope_messages(target_bytes: int):
    ref_url = _make_png_url(200)
    ref_text = {"text": TEXT}
    ref_img  = {"image": ref_url}
    per_pair = len(json.dumps(ref_text).encode()) + len(json.dumps(ref_img).encode())
    base = len(json.dumps([{"role": "user", "content": [ref_text]}]).encode())
    n = max(0, (target_bytes - base) // max(1, per_pair))
    content = [{"text": TEXT}]
    for i in range(n):
        content.append({"image": ref_url})
        content.append({"text": f"View {i+1}"})
    return [{"role": "user", "content": content}], n

# ── HTTP ─────────────────────────────────────────────────────────────
def test(url: str, body: bytes, timeout=60):
    """Return (status_code, body_text, elapsed_sec, headers_dict)."""
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")[:500], time.time() - t0, dict(r.headers)
    except urllib.error.HTTPError as e:
        hdrs = {k.lower(): v for k, v in e.headers.items()}
        return e.code, e.read().decode(errors="replace")[:500], time.time() - t0, hdrs
    except Exception as e:
        return 0, str(e)[:500], time.time() - t0, {}

# ── Output ───────────────────────────────────────────────────────────
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; C = "\033[36m"; B = "\033[1m"; X = "\033[0m"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-mb", type=float, default=10)
    p.add_argument("--max-mb", type=float, default=30)
    p.add_argument("--step-mb", type=float, default=2)
    args = p.parse_args()
    if not API_KEY: print(f"{R}Set DASHSCOPE_API_KEY{X}"); sys.exit(1)

    sizes = []; s = args.min_mb
    while s <= args.max_mb + 0.01: sizes.append(s); s += args.step_mb

    print(f"{B}{'='*70}{X}")
    print(f"{B}HTTP Body Limit — 3 endpoints compared{X}")
    print(f"{B}{'='*70}{X}")
    print(f"  Model:          {MODEL}")
    print(f"  MaaS / OpenAI:  {MAAS_OAI_URL}")
    print(f"  MaaS / DS:      {MAAS_DS_URL}")
    print(f"  DS public:      {DS_PUB_URL}\n")

    hdr = f"{'Size':>5s}  {'Imgs':>4s}  {'MaaS / OpenAI':<34s}  {'MaaS / DashScope':<34s}  {'DashScope public':<34s}"
    print(hdr); print("-" * len(hdr))

    lim_oai = None; lim_mds = None; lim_pub = None

    for mb in sizes:
        target = int(mb * 1024 * 1024)
        print(f"{mb:.0f}MB  building...", end=" ", flush=True)
        oai_msg, n = _make_openai_messages(target)
        ds_msg,  _ = _make_dashscope_messages(target)
        oai_body = json.dumps({"model": MODEL, "messages": oai_msg, "max_tokens": 1}).encode()
        ds_body  = json.dumps({"model": MODEL, "input": {"messages": ds_msg}, "parameters": {"max_tokens": 1}}).encode()
        print(f"{n} imgs, {len(oai_body)/1024:.0f}KB", flush=True)

        print(f"       MaaS/OAI...", end=" ", flush=True)
        c1, t1_txt, t1, h1 = test(MAAS_OAI_URL, oai_body)
        print(f"{c1} {t1:.1f}s", flush=True)

        print(f"       MaaS/DS ...", end=" ", flush=True)
        c2, t2_txt, t2, h2 = test(MAAS_DS_URL,  ds_body)
        print(f"{c2} {t2:.1f}s", flush=True)

        print(f"       DS pub  ...", end=" ", flush=True)
        c3, t3_txt, t3, h3 = test(DS_PUB_URL,   ds_body)
        print(f"{c3} {t3:.1f}s", flush=True)

        if lim_oai is None and c1 == 413: lim_oai = len(oai_body) / 1024 / 1024
        if lim_mds is None and c2 == 413: lim_mds = len(ds_body)  / 1024 / 1024
        if lim_pub is None and c3 == 413: lim_pub = len(ds_body)  / 1024 / 1024

        def r(c, t):
            return f"{G}✓ 200{X}  {t:5.1f}s" if 200 <= c < 300 else (f"{R}✗ 413{X}  {t:5.1f}s" if c == 413 else f"{Y}? {c}{X}  {t:5.1f}s")

        print(f"{mb:4.0f}MB  {n:4d}  {r(c1, t1):<40s}  {r(c2, t2):<40s}  {r(c3, t3):<40s}")

        for label, c, txt, h in [("MaaS/OAI", c1, t1_txt, h1), ("MaaS/DS", c2, t2_txt, h2), ("DS pub", c3, t3_txt, h3)]:
            if c == 413:
                _server = h.get('server', '?')
                if not txt or not txt.strip():
                    print(f"       {C}{label} 413:{X} (empty body)  server={_server}")
                else:
                    try: e = json.loads(txt); print(f"       {C}{label} 413:{X} {e.get('error',{}).get('message','')[:120]}  server={_server}")
                    except: print(f"       {C}{label} 413:{X} {txt[:120]}  server={_server}")
            elif c not in (200, 413):
                _server = h.get('server', '?')
                if not txt or not txt.strip():
                    print(f"       {C}{label} {c}:{X} (empty body)  server={_server}")
                else:
                    try: e = json.loads(txt); print(f"       {C}{label} {c}:{X} {json.dumps(e)[:200]}  server={_server}")
                    except: print(f"       {C}{label} {c}:{X} {txt[:200]}  server={_server}")

        time.sleep(0.5)

    def L(v): return f"≈ {v:.1f} MB" if v else f"> {sizes[-1]:.0f} MB"
    print(f"\n{B}{'='*70}{X}")
    print(f"{B}Conclusion{X}")
    print(f"{B}{'='*70}{X}")
    print(f"  MaaS / OpenAI:     body limit {L(lim_oai)}")
    print(f"  MaaS / DashScope:  body limit {L(lim_mds)}")
    print(f"  DashScope public:  body limit {L(lim_pub)}")

if __name__ == "__main__":
    main()
