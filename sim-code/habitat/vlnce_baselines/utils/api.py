"""LaViRA API client — unified router.

Delegates to either the OpenAI-compatible backend or the DashScope-native
backend depending on the ``use_dashscope`` flag passed at construction time.

Both backends expose the same public interface, so upstream code (agent.py)
needs no changes.
"""

from .api_openai import LaViRA_OpenAI_API, log_prompt, log_response, log_verbose, log_network
from .api_dashscope import LaViRA_DashScope_API


class LaViRA_API:
    """Unified API client for LaViRA.

    Parameters
    ----------
    use_dashscope : bool
        If True, use DashScope native SDK; otherwise OpenAI-compatible HTTP.
    la_api_key : str
        Language Agent API key (ignored in DashScope mode).
    la_base_url : str
        Language Agent base URL (ignored in DashScope mode).
    la_model_name : str
        Language Agent model name.
    va_model_name : str or None
        Vision Agent model name.
    va_api_key : str
        Vision Agent API key (ignored in DashScope mode).
    va_base_url : str
        Vision Agent base URL (ignored in DashScope mode).
    dashscope_api_key : str
        DashScope API key (only used when ``use_dashscope=True``).
    """

    def __init__(self, use_dashscope=False, dashscope_api_key=None,
                 dashscope_base_url=None,
                 la_api_key=None, la_base_url=None, la_model_name="gpt-4-vision-preview",
                 va_model_name=None, va_api_key=None, va_base_url=None):
        self.use_dashscope = use_dashscope

        if use_dashscope:
            self._backend = LaViRA_DashScope_API(
                dashscope_api_key=dashscope_api_key or la_api_key or va_api_key or '',
                la_model_name=la_model_name,
                va_model_name=va_model_name,
                dashscope_base_url=dashscope_base_url,
            )
        else:
            self._backend = LaViRA_OpenAI_API(
                la_api_key=la_api_key,
                la_base_url=la_base_url,
                la_model_name=la_model_name,
                va_model_name=va_model_name,
                va_api_key=va_api_key,
                va_base_url=va_base_url,
            )

    def __getattr__(self, name):
        """Delegate all attribute/method access to the active backend."""
        # __getattr__ is only called for attributes NOT found on the instance,
        # so 'use_dashscope' and '_backend' return normally.
        return getattr(self._backend, name)
