import json
import re
import time


def _strip_code_fence(content, style):
    if style == "regex":
        return re.sub(r"^```json\n|```$", "", content)
    if style == "backtick":
        content = content.strip("`")
        return content.replace("json\n", "", 1).strip()
    return content


def call_llm_json(
    client,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    max_retries=3,
    required_keys=None,
    strip_code_fence=None,
    decode_unicode_escape=False,
    raise_on_exhausted=False,
):
    """Call `client.chat.completions.create` up to `max_retries` times,
    returning the first response that parses as JSON (and, if
    `required_keys` is given, whose top-level keys exactly match it).

    Returns None if no attempt produced valid JSON, unless
    `raise_on_exhausted` is set, in which case the last JSONDecodeError is
    re-raised (matching the session-generation scripts' original behavior
    of failing loudly rather than silently dropping a session).
    """
    last_error = None
    for attempt in range(max_retries):
        kwargs = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()

        if strip_code_fence:
            content = _strip_code_fence(content, strip_code_fence)
        if decode_unicode_escape:
            content = content.encode("utf-8").decode("unicode_escape")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            last_error = e
            if raise_on_exhausted and attempt == max_retries - 1:
                raise
            continue

        if required_keys is not None and set(parsed.keys()) != required_keys:
            continue
        return parsed

    return None


def call_with_backoff(fn, max_retries=5):
    """Call `fn()` (a zero-arg callable wrapping an API call), retrying with
    `(2 ** attempt) + (0.1 * attempt)` second backoff on any exception.
    After `max_retries` failed attempts, the final exception is re-raised.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                print(f"Call failed ({e}), retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
    raise last_exception
