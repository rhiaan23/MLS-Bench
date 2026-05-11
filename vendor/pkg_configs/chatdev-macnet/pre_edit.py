"""Pre-edit operations for chatdev-macnet package.

Applied to the workspace before any task-specific mid_edit:
1. Patch config.yaml to use GPT_4O_MINI enum key (model name resolved
   at runtime from MACNET_MODEL env var).
2. Patch model_backend.py run() to: (a) use MACNET_MODEL env var for
   actual model name, (b) fallback tiktoken to cl100k_base, (c) clamp
   max_tokens to [4096, 8192] for non-OpenAI providers.
3. Suppress imgcat calls (not available in headless environments).
4. Fix log_macnet call signatures (utils.py expects role+content, but
   graph.py sometimes passes a single string).
"""

# ── Op 1: Use GPT_4O_MINI in config.yaml (line 2) ───────────────────
# The actual model name is resolved at runtime from $MACNET_MODEL
# (set by the per-setting launcher script, e.g. deepseek-chat or qwen2.5-72b-instruct).

_MODEL_FIX = """\
Model: "GPT_4O_MINI" # actual model resolved from $MACNET_MODEL env var at runtime
"""

# ── Op 1c: Patch entire run() method body in model_backend.py (lines 68-149)
# Uses $MACNET_MODEL env var for the actual model name, with fallbacks for
# tiktoken encoding and max_tokens range.
_MODEL_BACKEND_RUN_BODY = """\
        import os as _os
        actual_model = _os.environ.get("MACNET_MODEL", self.model_type.value)
        try:
            encoding = tiktoken.encoding_for_model(actual_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        num_max_token_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k-0613": 16384,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 100000,
            "gpt-4o": 4096,
            "gpt-4o-mini": 16384,
        }
        num_max_token = num_max_token_map.get(actual_model, 16384)
        num_max_completion_tokens = min(max(num_max_token - num_prompt_tokens, 4096), 8192)

        if openai_new_api:
            if BASE_URL:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=BASE_URL,
                )
            else:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY
                )

            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = client.chat.completions.create(*args, **kwargs, model=actual_model,
                                                      **self.model_config_dict)

            cost = prompt_cost(
                actual_model,
                num_prompt_tokens=response.usage.prompt_tokens,
                num_completion_tokens=response.usage.completion_tokens
            )

            if not isinstance(response, ChatCompletion):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
        else:
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = openai.ChatCompletion.create(*args, **kwargs, model=actual_model,
                                                    **self.model_config_dict)

            cost = prompt_cost(
                actual_model,
                num_prompt_tokens=response["usage"]["prompt_tokens"],
                num_completion_tokens=response["usage"]["completion_tokens"]
            )

            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
"""

# ── Op 2: Stub out imgcat in graph.py (line 301) ────────────────────
# The display_image_with_imgcat method calls imgcat which is not available.
# Replace with a no-op.

_IMGCAT_STUB_GRAPH = """\
    def display_image_with_imgcat(self, image_path):
        \"\"\"Display the image with imgcat (stubbed for headless env)\"\"\"
        pass
"""

# ── Op 3: Stub out imgcat in generate_graph.py (line 20-22) ─────────

_IMGCAT_STUB_GENGRAPH = """\
    def display_image_with_imgcat(self, image_path):
        \"\"\"Display the image with imgcat (stubbed for headless env)\"\"\"
        pass
"""

# ── Op 4: Add 'annotations' field to BaseMessage (line 57) ──────────
# New OpenAI API responses include 'annotations' which BaseMessage doesn't have.
_ANNOTATIONS_FIELD = """\
    refusal: Optional[ChatCompletionContentPartRefusalParam] = None
    annotations: Optional[list] = None
"""

# ── Op 6: Patch OpenAI client base_url to read from env ─────────────
# company_hr.py line 7-10 and chat_env.py line 12-14 have base_url=""
# which prevents OPENAI_BASE_URL env var from working. Patch to use
# os.environ.get("BASE_URL") or None.

_COMPANY_HR_CLIENT = """\
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=os.environ.get("BASE_URL") or None,
)
"""

_CHAT_ENV_CLIENT = """\
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=os.environ.get("BASE_URL") or None,
)
"""

OPS = [
    {
        "op": "replace",
        "file": "chatdev-macnet/config.yaml",
        "start_line": 2,
        "end_line": 2,
        "content": _MODEL_FIX,
    },
    # Op 1c: Replace entire run() method body to support non-OpenAI models
    {
        "op": "replace",
        "file": "chatdev-macnet/camel/model_backend.py",
        "start_line": 68,
        "end_line": 149,
        "content": _MODEL_BACKEND_RUN_BODY,
    },
    {
        "op": "replace",
        "file": "chatdev-macnet/graph.py",
        "start_line": 299,
        "end_line": 302,
        "content": _IMGCAT_STUB_GRAPH,
    },
    {
        "op": "replace",
        "file": "chatdev-macnet/generate_graph.py",
        "start_line": 20,
        "end_line": 22,
        "content": _IMGCAT_STUB_GENGRAPH,
    },
    # Op 4: Add annotations field to BaseMessage (replace line 57 to add field after refusal)
    {
        "op": "replace",
        "file": "chatdev-macnet/camel/messages/base.py",
        "start_line": 57,
        "end_line": 57,
        "content": _ANNOTATIONS_FIELD,
    },
    # Op 5: Filter unknown fields when creating ChatMessage from API response
    # chat_agent.py L245-248: **dict(choice.message) passes all fields including
    # audio, reasoning, annotations etc. that ChatMessage doesn't accept.
    {
        "op": "replace",
        "file": "chatdev-macnet/camel/agents/chat_agent.py",
        "start_line": 244,
        "end_line": 248,
        "content": "            _known_fields = {'content','refusal','role','name','tool_call_id','function_call','tool_calls','annotations'}\n            output_messages = [\n                ChatMessage(role_name=self.role_name, role_type=self.role_type,\n                            meta_dict=dict(), **{k:v for k,v in dict(choice.message).items() if k in _known_fields})\n                for choice in response.choices\n            ]\n",
    },
    # Op 6: Patch company_hr.py OpenAI client base_url (lines 7-10)
    {
        "op": "replace",
        "file": "chatdev-macnet/chatdev/company_hr.py",
        "start_line": 7,
        "end_line": 10,
        "content": _COMPANY_HR_CLIENT,
    },
    # Op 7: Patch chat_env.py OpenAI client base_url (lines 12-14)
    {
        "op": "replace",
        "file": "chatdev-macnet/chatdev/chat_env.py",
        "start_line": 12,
        "end_line": 14,
        "content": _CHAT_ENV_CLIENT,
    },
]
