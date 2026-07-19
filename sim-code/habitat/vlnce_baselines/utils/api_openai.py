from openai import OpenAI
from PIL import Image
from habitat import logger
import io
import base64
import os
import json
import time
import sys
import socket
import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

# ── Force IPv4 ─────────────────────────────────────────────────────
# Alibaba Cloud NLB returns AAAA (IPv6) records that are often unreachable
# from certain ISPs.  httpx's Happy Eyeballs tries IPv6 first, and the
# default TCP connect timeout (10 s) means each failed v6 attempt wastes
# ~10 s before falling back to v4.  Monkey-patching socket.getaddrinfo to
# always request AF_INET eliminates that overhead.
_orig_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _getaddrinfo_ipv4

# ── Logging verbosity controls ───────────────────────────────────────────────
# Delegated to vlnce_baselines.utils.logging (three-layer: evaluator / agent / api).
from .logging import (  # noqa: F401  — re-exported for backward compatibility
    LOG_BODY,
    LOG_NETWORK,
    LOG_PROGRESS_BAR,
    log_body,
    log_network,
    log_req as log_prompt,
    log_resp as log_response,
    log_plan as log_verbose,   # log_plan replaces the old log_verbose
)


def _clear_proxy_env():
    """Temporarily remove proxy env vars that httpx might choke on (e.g. socks://).

    Returns a dict of popped values so they can be restored.
    Only clears the SOCKS / catch-all proxies; HTTP/HTTPS proxies are left intact
    because the VA endpoint (aliyun MAAS) may require them.
    """
    _proxy_keys = ('ALL_PROXY', 'all_proxy')
    backup = {}
    for k in _proxy_keys:
        if k in os.environ:
            backup[k] = os.environ.pop(k)
    return backup


def _restore_proxy_env(backup):
    """Restore proxy env vars cleared by _clear_proxy_env."""
    os.environ.update(backup)


def _build_http_client():
    """Build an httpx.Client with network-diagnostic event hooks.

    Returns None when LAVIRA_LOG_NETWORK is off (OpenAI uses its defaults).
    When enabled, the hooks log every HTTP request/response pair so slow
    requests can be pinpointed (DNS, TCP, TLS, or server-side wait).
    """
    if not (LOG_NETWORK or LOG_BODY) or httpx is None:
        return None

    _req_start = {}

    def _on_request(request):
        _req_start[id(request)] = time.time()
        _payload = len(request.content) if request.content else 0
        log_network(f"HTTP → {request.method} {request.url}  ({_payload / 1024:.0f} KB)")
        if LOG_BODY:
            ct = request.headers.get('content-type', '?')
            logger.info(f"[BODY HTTP] {request.method} {request.url}  "
                        f"body={_payload/1024:.0f} KB  content-type={ct}")

    def _on_response(response):
        req_id = id(response.request)
        _start = _req_start.pop(req_id, time.time())
        _elapsed = time.time() - _start
        _slow = "  ⚠ SLOW!" if _elapsed > 30 else ""
        log_network(f"HTTP ← {response.http_version} {response.status_code}  "
                    f"{_elapsed:.1f}s{_slow}")
        # Also log response headers that may help diagnose issues
        if _elapsed > 30:
            for key in ('x-request-id', 'x-ratelimit-remaining', 'retry-after'):
                if key in response.headers:
                    log_network(f"  header {key}: {response.headers[key]}")

    return httpx.Client(
        http2=True,
        event_hooks={'request': [_on_request], 'response': [_on_response]},
        timeout=httpx.Timeout(2000.0, connect=10.0),
    )


class _LatencyEstimator:
    """Online estimation of API latency model:  latency = x + output_tokens / k.

    Uses decoupled exponential moving average updates after each call.
    """

    def __init__(self, x0=3.0, k0=30.0, alpha=0.3):
        self.x = x0       # fixed overhead (seconds)
        self.k = k0       # token generation speed (tokens / second)
        self.alpha = alpha
        self.x_max = 0.0          # peak overhead (excludes initial x0)
        self.k_min = float('inf') # slowest speed (excludes initial k0)

    def update(self, output_tokens: int, elapsed_sec: float):
        """Incorporate a new observation (n output tokens, t seconds)."""
        n = max(output_tokens, 1)
        t = max(elapsed_sec, 0.1)

        # ── Update x (overhead) ──────────────────────────────────────
        x_obs = t - n / max(self.k, 1)
        x_obs = max(0.1, min(60.0, x_obs))
        self.x = self.alpha * x_obs + (1 - self.alpha) * self.x
        if self.x > self.x_max:
            self.x_max = self.x

        # ── Update k (token speed) ───────────────────────────────────
        # Only update k when the model is physically consistent (x < t).
        # When x >= t, the overhead estimate exceeds the total time,
        # making k_obs = n/0.1 meaningless — skip to avoid a vicious
        # cycle where high x → high k → small n/k → x stays high.
        net = t - self.x
        if net > 0.1:
            k_obs = n / net
            k_obs = max(5.0, min(500.0, k_obs))
            self.k = self.alpha * k_obs + (1 - self.alpha) * self.k
            if self.k < self.k_min:
                self.k_min = self.k


class LaViRA_OpenAI_API:

    def __init__(self, la_api_key=None, la_base_url=None, la_model_name="gpt-4-vision-preview",
                 va_model_name=None, va_api_key=None, va_base_url=None):
        # Log proxy env state before clearing (controlled by LAVIRA_LOG_NETWORK).
        log_network(f"Proxy env at init: "
                    f"ALL_PROXY={os.environ.get('ALL_PROXY','')!r}  "
                    f"all_proxy={os.environ.get('all_proxy','')!r}  "
                    f"HTTP_PROXY={os.environ.get('HTTP_PROXY','')!r}  "
                    f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')!r}  "
                    f"NO_PROXY={os.environ.get('NO_PROXY','')!r}")

        # Clear proxy env vars so httpx doesn't choke on unsupported schemes (e.g. socks://).
        # The API endpoint already encodes the real routing target via base_url.
        _proxy_backup = _clear_proxy_env()
        try:
            _http_client = _build_http_client()
            self.la_client = OpenAI(
                api_key=la_api_key,
                base_url=la_base_url,
                timeout=2000,
                http_client=_http_client,
            )
            self.la_model_name = la_model_name
            log_network(f"LA client created: base_url={la_base_url}  model={la_model_name}")

            if va_model_name:
                self.va_client = OpenAI(
                    api_key=va_api_key,
                    base_url=va_base_url,
                    timeout=2000,
                    http_client=_build_http_client(),
                )
                self.va_model_name = va_model_name
                log_network(f"VA client created: base_url={va_base_url}  model={va_model_name}")
            else:
                self.va_client = None
                self.va_model_name = None
        finally:
            _restore_proxy_env(_proxy_backup)

        self.reset_stats()
        self._la_round = 0
        self._va_round = 0
        self._lat_la = _LatencyEstimator(x0=3.0, k0=30.0, alpha=0.3)
        self._lat_va = _LatencyEstimator(x0=3.0, k0=30.0, alpha=0.3)

    def image_to_base64(self, image):
        """Convert a PIL Image or numpy array to a base64-encoded string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_debug_info(self, log_path, messages, response_text):
        if not log_path:
            return
        
        try:
            os.makedirs(log_path, exist_ok=True)
            
            # Save images and create a clean message list for saving
            saved_messages = []
            img_count = 0
            
            for msg in messages:
                new_msg = {'role': msg['role'], 'content': []}
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            if url.startswith('data:image/'):
                                try:
                                    # Extract base64
                                    header, encoded = url.split(',', 1)
                                    data = base64.b64decode(encoded)
                                    img_filename = f"image_{img_count}.png"
                                    img_path = os.path.join(log_path, img_filename)
                                    with open(img_path, 'wb') as f:
                                        f.write(data)
                                    
                                    new_msg['content'].append({
                                        'type': 'image_url',
                                        'image_url': {'url': img_filename}
                                    })
                                    img_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to save image: {e}")
                                    new_msg['content'].append({'type': 'image_url', 'image_url': {'url': 'FAILED_TO_SAVE'}})
                            else:
                                new_msg['content'].append(item)
                        else:
                            new_msg['content'].append(item)
                else:
                     new_msg['content'] = msg['content']
                saved_messages.append(new_msg)
                
            # Save prompt text
            with open(os.path.join(log_path, 'prompt.json'), 'w') as f:
                json.dump(saved_messages, f, indent=2)
                
            # Save response
            with open(os.path.join(log_path, 'response.txt'), 'w') as f:
                f.write(str(response_text))
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_la=False, log_path=None, retries=0, max_retries=5, label=None, **kwargs):
        """
        Mimic the original model.generate interface.
        Args:
            messages: list of text messages
            images: list of images
            max_new_tokens: max number of tokens to generate
            temperature: sampling temperature
            use_la: whether to use the second (LA) model
            log_path: Path to save debug logs (images and prompt)
            retries: Current retry count
            max_retries: Maximum number of retries
            label: Optional override for log prefix and stats key.
                   When set (e.g. "V2"), all stats go into a single bucket
                   regardless of use_la.
        """
        # use_la = False
        t = time.time()
        # select the client and model to use
        if use_la and self.la_client:
            client = self.la_client
            model_name = self.la_model_name
        else:
            client = self.va_client
            model_name = self.va_model_name

        # When a custom label is provided (e.g. "V2"), unify stats under that key
        # instead of splitting into 'Language Action Model' / 'Vision Action Model'.
        if label:
            stats_key = label
        elif use_la:
            stats_key = 'Language Action Model'
        else:
            stats_key = 'Vision Action Model'

        # Ensure stats slot exists for custom labels (LA/VA are pre-allocated).
        if stats_key not in self.stats:
            self.stats[stats_key] = {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'elapsed_total': 0.0, 'elapsed_max': 0.0, 'elapsed_min': float('inf'),
            }

        # Disable explicit thinking on the Qwen3 family for both LA and VA paths.
        # (Gemini-3.x still reasons internally; this only affects providers that
        # honour the enable_thinking flag, e.g. self-hosted Qwen.)
        extra_body = kwargs.pop('extra_body', None) or {}
        extra_body.setdefault('enable_thinking', False)

        # Estimate payload size for debugging.
        _msg_str = json.dumps(messages, ensure_ascii=False)
        _payload_kb = len(_msg_str.encode('utf-8')) / 1024
        if label:
            # Custom label — use a dict-based round counter keyed by label
            if not hasattr(self, '_custom_rounds'):
                self._custom_rounds = {}
            self._custom_rounds[label] = self._custom_rounds.get(label, 0) + 1
            label = f"{label} #{self._custom_rounds[label]}"
        elif use_la:
            self._la_round += 1
            label = f"LA #{self._la_round}"
        else:
            self._va_round += 1
            label = f"VA #{self._va_round}"
        # When the progress bar is active (quiet mode), its \r leaves the cursor
        # mid-line.  Emit a newline so subsequent output starts on a clean line.
        if LOG_PROGRESS_BAR:
            sys.stdout.write("\n")
            sys.stdout.flush()

        _t0 = time.time()
        _proxy_state = (f"HTTP_PROXY={os.environ.get('HTTP_PROXY','')!r}  "
                        f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')!r}")
        log_network(f"{label}  target={model_name}  payload={_payload_kb:.0f}KB  {_proxy_state}")
        bar = "─" * 40
        logger.info(f"▐ {label} → {model_name}  {_payload_kb:.0f} KB")

        try:
            log_body(messages, f"{label} OpenAI msg format")
            _call_t = time.time()
            if stats_key == 'Language Action Model':
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=120,
                    extra_body=extra_body,
                    **kwargs
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=120,
                    reasoning_effort='low',
                    extra_body=extra_body,
                    **kwargs
                )
            _elapsed = time.time() - _call_t
            log_verbose(str(response))
            # Handle case where response is a string (e.g. from some proxies or raw returns)
            if isinstance(response, str):
                logger.info(f"API returned string response for {model_name}")
                self.stats[stats_key]['calls'] += 1
                self._save_debug_info(log_path, messages, response)
                return response

            # update usage statistics
            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                log_verbose(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0
                # Track elapsed time
                s = self.stats[stats_key]
                s['elapsed_total'] += _elapsed
                if _elapsed > s['elapsed_max']:
                    s['elapsed_max'] = _elapsed
                if _elapsed < s['elapsed_min']:
                    s['elapsed_min'] = _elapsed

                # Update latency model:  t = x + out/k
                _lat = self._lat_la if use_la else self._lat_va
                _lat.update(response.usage.completion_tokens or 0, _elapsed)
                logger.info(f"▐ {label}  ✓ {_elapsed:.1f}s  |  "
                            f"in:{response.usage.prompt_tokens}  out:{response.usage.completion_tokens}  "
                            f"total:{response.usage.total_tokens}  "
                            f"x={_lat.x:.1f}s  k={_lat.k:.0f}tok/s")
                _slow = "  ⚠ SLOW!" if _elapsed > 30 else ""
                log_network(f"{label}  ✓ {_elapsed:.1f}s  "
                            f"in={response.usage.prompt_tokens}  out={response.usage.completion_tokens}{_slow}")
            content = response.choices[0].message.content
            self._save_debug_info(log_path, messages, content)
            return content

        except Exception as e:
            err_str = str(e)
            _elapsed = time.time() - t
            _proxy_at_err = (f"HTTP_PROXY={os.environ.get('HTTP_PROXY','')!r}  "
                             f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')!r}")
            logger.error(f"[{stats_key}] API error after {_elapsed:.1f}s "
                         f"model={model_name} | {type(e).__name__}: {e}")
            log_network(f"{label}  ✗ FAIL {_elapsed:.1f}s  "
                        f"{type(e).__name__}: {str(e)[:200]}  {_proxy_at_err}")
            # Non-recoverable errors — retrying won't help; short-circuit to keep run time bounded.
            non_recoverable = (
                'data_inspection_failed' in err_str or
                'DataInspectionFailed' in err_str or
                'inappropriate content' in err_str or
                'invalid_request_error' in err_str
            )
            if non_recoverable:
                logger.error(f"Non-recoverable error detected; skipping retries.")
                return "Error: API rejected request (non-recoverable)"
            if retries >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                return "Error: Failed to get response from API after max retries"

            logger.info(f'Forcing retry ({retries + 1}/{max_retries})..')
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_la, log_path=log_path, retries=retries + 1, max_retries=max_retries, label=label, **kwargs)

    def get_model_info(self):
        """Return information about the configured models."""
        info = {
            "primary_model": self.la_model_name,
            "secondary_model": self.va_model_name if self.va_client else None,
            "has_secondary": self.va_client is not None
        }
        return info

    def get_usage_stats(self):
        """Return a copy of the current usage statistics."""
        return self.stats.copy()

    def print_usage_stats(self):
        """Log detailed usage statistics and return a summary dict."""
        def _fmt_time(s):
            if s == 0 or s == float('inf'):
                return '    -'
            return f'{s:5.1f}s'

        def _print_section(title, model_name, s, lat):
            n = s['calls']
            if n == 0:
                return
            logger.info(f"  {title} ({model_name}):")
            logger.info(f"    Calls:          {n:>6d}")
            logger.info(f"    Time total:     {_fmt_time(s['elapsed_total'])}")
            logger.info(f"    Time avg:       {_fmt_time(s['elapsed_total'] / n)}")
            logger.info(f"    Time max:       {_fmt_time(s['elapsed_max'])}")
            logger.info(f"    Time min:       {_fmt_time(s['elapsed_min'] if s['elapsed_min'] != float('inf') else 0)}")
            logger.info(f"    Input tokens:   {s['input_tokens']:>8,}")
            logger.info(f"    Output tokens:  {s['output_tokens']:>8,}")
            logger.info(f"    Total tokens:   {s['total_tokens']:>8,}")
            logger.info(f"    Avg out / call: {s['output_tokens'] / n:>8.0f}")
            logger.info(f"    Max latency x:  {_fmt_time(lat.x_max)}")
            logger.info(f"    Min latency k:  {lat.k_min if lat.k_min != float('inf') else 0:>5.0f} tok/s")

        logger.info("==================== MODEL USAGE STATISTICS ====================")

        # Print each stats bucket.  Determine model + latency tracker by key.
        for key in sorted(self.stats.keys()):
            s = self.stats[key]
            model_name = self.la_model_name if key == 'Language Action Model' else self.va_model_name
            lat = self._lat_la if key == 'Language Action Model' else self._lat_va
            _print_section(key, model_name, s, lat)

        # ── Combined ─────────────────────────────────────────────────
        total_calls = sum(s['calls'] for s in self.stats.values())
        total_tokens = sum(s['total_tokens'] for s in self.stats.values())
        total_elapsed = sum(s['elapsed_total'] for s in self.stats.values())
        total_out = sum(s['output_tokens'] for s in self.stats.values())
        elapsed_values = [s['elapsed_max'] for s in self.stats.values()]
        elapsed_min_values = [s['elapsed_min'] for s in self.stats.values() if s['elapsed_min'] != float('inf')]
        comb_max = max(elapsed_values) if elapsed_values else 0
        comb_min = min(elapsed_min_values) if elapsed_min_values else 0

        logger.info(f"  ───────────────────────────────────────────")
        logger.info(f"  COMBINED:")
        logger.info(f"    Calls:          {total_calls:>6d}")
        logger.info(f"    Time total:     {_fmt_time(total_elapsed)}")
        logger.info(f"    Time avg:       {_fmt_time(total_elapsed / max(1, total_calls))}")
        logger.info(f"    Time max:       {_fmt_time(comb_max)}")
        logger.info(f"    Time min:       {_fmt_time(comb_min)}")
        logger.info(f"    Input tokens:   {sum(s['input_tokens'] for s in self.stats.values()):>8,}")
        logger.info(f"    Output tokens:  {total_out:>8,}")
        logger.info(f"    Total tokens:   {total_tokens:>8,}")
        logger.info(f"    Avg out / call: {total_out / max(1, total_calls):>8.0f}")
        logger.info("================================================================")
        # Build backward-compatible return (see merge_model_usage_stats in stats.py).
        _result = {'total_calls': total_calls, 'total_tokens': total_tokens}
        for _k, _v in self.stats.items():
            _result[_k] = _v.copy()
        # Ensure 'la' / 'va' keys always exist for backward compat.
        _result.setdefault('la', self.stats.get('Language Action Model', {}).copy())
        _result.setdefault('va', self.stats.get('Vision Action Model', {}).copy())
        return _result

    def reset_stats(self):
        """Reset usage statistics to zero."""
        self.stats = {
            'Language Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'elapsed_total': 0.0, 'elapsed_max': 0.0, 'elapsed_min': float('inf'),
            },
            'Vision Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'elapsed_total': 0.0, 'elapsed_max': 0.0, 'elapsed_min': float('inf'),
            },
        }

    def eval(self):
        """Compatibility no-op method."""
