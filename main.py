#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

The server reads its brief (the provider directory, the sealed quality table
and the workload manifest, all in $MAKEFUN_PROMPT_FILE), probes every
provider's catalog route in that provider's own wire, registers one catalog
with the env, then answers the client's twenty requests. Each request is
routed to the highest-rated free model that meets the request's requirements,
speaking the provider's wire (openai, anthropic or gemini), and the answer is
returned in the client's wire (OpenAI Chat Completions or Anthropic
Messages), as JSON or as server-sent events. A provider refusal (402, 429 or
5xx) moves the request to the next candidate; a provider that refuses twice
in a row is skipped while any other candidate remains.

Everything the server emits is a pure function of the brief, the requests and
the providers' answers: fixed timestamps, no clock, no randomness, so a
replayed trial reproduces the same bytes.

Run with --list to print the catalog document and exit 0.
"""

import json
import os
import signal
import sys
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROBE_TIMEOUT_S = 5
CALL_TIMEOUT_S = 8
MAX_ATTEMPTS = 6
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
SSE_PIECE = 24

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# No proxy for trial traffic: the env and its providers are dialled directly.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg):
    sys.stderr.write(f"aggregator: {msg}\n")
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _s(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


class Provider:
    def __init__(self, row):
        self.id = _s(row.get("id"))
        self.name = _s(row.get("name")) or self.id
        self.wire = row.get("wire")
        self.auth = row.get("auth") if row.get("auth") in (
            "bearer", "x-api-key", "x-goog-api-key", "none") else "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _s(row.get(key)):
                base = _s(row.get(key))
                break
        self.base = base
        self.key_env = _s(row.get("key_env"))
        self.key = _s(row.get("key"))


class Brief:
    def __init__(self):
        self.providers = []
        self.ratings = {}       # (provider, model) -> row dict
        self.manifest = []      # workload rows in arrival order
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief.error = f"cannot read {path}: {err}"
        return brief
    docs = documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = f"the brief at {path} holds no {PROVIDERS_SCHEMA} document"
        return brief
    for row in directory["providers"]:
        if not is_record(row):
            continue
        prov = Provider(row)
        if prov.id and prov.wire in ("openai", "anthropic", "gemini"):
            brief.providers.append(prov)
    if ratings_doc is not None:
        for row in ratings_doc["ratings"]:
            if not is_record(row):
                continue
            model, provider = _s(row.get("model")), _s(row.get("provider"))
            if not model or not provider:
                continue
            brief.ratings[(provider, model)] = {
                "rating_milli": row.get("rating_milli")
                if isinstance(row.get("rating_milli"), (int, float)) else 0,
                "context_length": _num_or_none(row.get("context_length")),
                "tools": row.get("tools") is True,
                "vision": row.get("vision") is True,
                "json": row.get("json") is True,
                "free": row.get("free") is True,
            }
    if workload_doc is not None:
        for row in workload_doc["manifest"]:
            if not is_record(row) or not _s(row.get("id")):
                continue
            req = row.get("requires") if is_record(row.get("requires")) else {}
            requires = {
                "tools": req.get("tools") is True,
                "vision": req.get("vision") is True,
                "json": req.get("json") is True,
            }
            if isinstance(req.get("min_context"), (int, float)):
                requires["min_context"] = req["min_context"]
            brief.manifest.append({
                "id": _s(row.get("id")),
                "api": "anthropic" if row.get("api") == "anthropic" else "openai",
                "stream": row.get("stream") is True,
                "weight": 2 if row.get("weight") == 2 else 1,
                "requires": requires,
            })
    return brief


def rating_of(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


# --------------------------------------------------------------------------
# HTTP plumbing and the catalog probe.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith(("http://", "https://")):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if not b:
        return prefix
    return f"{prefix}{'' if b.startswith('/') else '/'}{b.rstrip('/')}"


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": f"Bearer {key}"}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers=None, body=None, timeout=CALL_TIMEOUT_S):
    """One HTTP call; returns (status, headers dict, body bytes). Never raises
    for an HTTP error status; raises OSError/URLError for transport failures."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with OPENER.open(req, timeout=timeout) as res:
            return res.status, dict(res.headers.items()), res.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items() if err.headers else {}), err.read()


class ProbedModel:
    def __init__(self, model_id, display_name, free, context_length, tools, vision, json_cap):
        self.id = model_id
        self.display_name = display_name
        self.free = free
        self.context_length = context_length
        self.tools = tools
        self.vision = vision
        self.json = json_cap


def _zero_priced(row):
    pricing = row.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for v in pricing.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
        elif isinstance(v, str) and v.strip():
            try:
                values.append(float(v))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, row):
    if not is_record(row):
        return None
    if wire == "gemini":
        model_id = _s(row.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = _s(row.get("id"))
    if not model_id:
        return None
    context = _num_or_none(row.get("context_length"))
    if context is None:
        context = _num_or_none(row.get("context_window"))
    if context is None:
        context = _num_or_none(row.get("inputTokenLimit"))
    free = _bool_or_none(row.get("free"))
    if free is None:
        free = _zero_priced(row)
    display = _s(row.get("display_name")) or _s(row.get("displayName")) or model_id
    return ProbedModel(model_id, display, free, context,
                       _bool_or_none(row.get("tools")),
                       _bool_or_none(row.get("vision")),
                       _bool_or_none(row.get("json")))


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider.base)
    url = f"{base}/v1beta/models" if provider.wire == "gemini" else f"{base}/v1/models"
    headers = {"accept": "application/json", **auth_headers(provider.auth, provider.key)}
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, raw = http_request("GET", url, headers=headers, timeout=timeout)
    except Exception as err:  # transport failure: the provider sits this trial out
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": f"GET {url} failed: {err}"}
    if status < 200 or status >= 300:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": f"GET {url} answered HTTP {status}"}
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": f"GET {url} answered a non-JSON body"}
    listing = body.get("models") if provider.wire == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for row in listing:
            model = _model_in(provider.wire, row)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "status": status, "models": models, "error": None}


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model.free if model.free is not None else False


def capabilities_of(model, row):
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": model.context_length, "tools": model.tools,
            "vision": model.vision, "json": model.json}


def meets_requirements(requires, caps):
    if requires.get("tools") and caps["tools"] is not True:
        return False
    if requires.get("vision") and caps["vision"] is not True:
        return False
    if requires.get("json") and caps["json"] is not True:
        return False
    if requires.get("min_context") is not None:
        if caps["context_length"] is None or caps["context_length"] < requires["min_context"]:
            return False
    return True


def build_catalog(probes, brief):
    """The catalog document: the register body and the --list output are this
    one object, built once. base_url is the directory's own path, never the
    path joined to the env URL, so a replay reproduces it byte for byte."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m.id):
            if model.id in seen:
                continue
            seen.add(model.id)
            row = rating_of(brief, probe["provider"].id, model.id)
            context = row["context_length"] if row is not None else model.context_length
            models.append({"id": model.id, "free": model_free(model, row),
                           "context_length": context})
        providers.append({"id": probe["provider"].id, "base_url": probe["provider"].base,
                          "key_env": probe["provider"].key_env,
                          "wire": probe["provider"].wire, "models": models})
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire translation. Pure functions of the request body: the same body and
# the same provider answer give the same bytes on a replay.
# --------------------------------------------------------------------------

def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _part_of(block):
    """One Anthropic content block as an OpenAI content part, or None."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def arguments_of(value):
    """A tool call's arguments as an object; {} when it is not a JSON object."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if is_record(parsed):
            return parsed
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({"type": "function", "function": {
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                else {"type": "object", "properties": {}},
            }})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _image_block(source_url):
    """An OpenAI image_url as an Anthropic image block, or None."""
    if not isinstance(source_url, str) or not source_url:
        return None
    if source_url.startswith("data:") and ";base64," in source_url:
        media, data = source_url[5:].split(";base64,", 1)
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": media or "image/png", "data": data}}
    return {"type": "image", "source": {"type": "url", "url": source_url}}


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            block = {"type": "tool_result",
                     "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                     "content": text_of(content) if not isinstance(content, str) else content}
            if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) \
                    and messages[-1]["content"] and messages[-1]["content"][0].get("type") == "tool_result":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content) if not isinstance(content, str) else content
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({"type": "tool_use",
                               "id": call.get("id") if isinstance(call.get("id"), str) else "",
                               "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                               "input": arguments_of(fn.get("arguments"))})
            messages.append({"role": "assistant", "content": blocks if blocks else ""})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    block = _image_block(part["image_url"].get("url"))
                    if block is not None:
                        blocks.append(block)
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = stop
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({"name": fn["name"],
                          "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                          "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and is_record(choice.get("function")) \
            and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(url):
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        media, data = url[5:].split(";base64,", 1)
        return {"inlineData": {"mimeType": media or "image/png", "data": data}}
    if isinstance(url, str) and url:
        return {"fileData": {"fileUri": url}}
    return None


def to_gemini(body, api, model, json_mode):
    """A client request (either wire) as a Gemini generateContent body."""
    contents = []
    system_text = ""
    tool_names = {}   # tool call id -> function name, for functionResponse parts
    declarations = []

    def note_tool_call(call_id, name):
        if isinstance(call_id, str) and call_id:
            tool_names[call_id] = name

    if api == "openai":
        system_parts = []
        for raw in body.get("messages") or []:
            if not is_record(raw):
                continue
            role = raw.get("role")
            content = raw.get("content")
            if role == "system":
                text = text_of(content)
                if text:
                    system_parts.append(text)
                continue
            if role == "tool":
                text = content if isinstance(content, str) else text_of(content)
                call_id = raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else ""
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": tool_names.get(call_id, call_id), "response": {"result": text}}}]})
                continue
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not is_record(part):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                        image = _gemini_image_part(part["image_url"].get("url"))
                        if image is not None:
                            parts.append(image)
            if role == "assistant":
                for call in raw.get("tool_calls") or []:
                    if not is_record(call) or not is_record(call.get("function")):
                        continue
                    fn = call["function"]
                    name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                    note_tool_call(call.get("id"), name)
                    parts.append({"functionCall": {"name": name, "args": arguments_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        system_text = "\n".join(system_parts)
        for raw in body.get("tools") or []:
            if is_record(raw) and is_record(raw.get("function")) and isinstance(raw["function"].get("name"), str):
                fn = raw["function"]
                declarations.append({"name": fn["name"],
                                     "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                                     "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                                     else {"type": "object", "properties": {}}})
        config = {}
        max_tokens = body.get("max_tokens")
        if not isinstance(max_tokens, (int, float)):
            max_tokens = body.get("max_completion_tokens")
        if isinstance(max_tokens, (int, float)):
            config["maxOutputTokens"] = max_tokens
        stop = body.get("stop")
        if isinstance(stop, str) and stop:
            config["stopSequences"] = [stop]
        elif isinstance(stop, list) and stop:
            config["stopSequences"] = stop
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            config["topP"] = body["top_p"]
    else:  # anthropic client
        system_text = text_of(body.get("system"))
        for raw in body.get("messages") or []:
            if not is_record(raw):
                continue
            role = raw.get("role")
            content = raw.get("content")
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image" and is_record(block.get("source")):
                        source = block["source"]
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                            parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                        elif source.get("type") == "url":
                            image = _gemini_image_part(source.get("url"))
                            if image is not None:
                                parts.append(image)
                    elif block.get("type") == "tool_use":
                        name = block.get("name") if isinstance(block.get("name"), str) else ""
                        note_tool_call(block.get("id"), name)
                        parts.append({"functionCall": {"name": name,
                                                       "args": block.get("input") if is_record(block.get("input")) else {}}})
                    elif block.get("type") == "tool_result":
                        call_id = block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else ""
                        parts.append({"functionResponse": {
                            "name": tool_names.get(call_id, call_id),
                            "response": {"result": _tool_result_text(block.get("content"))}}})
            if parts:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        for raw in body.get("tools") or []:
            if is_record(raw) and isinstance(raw.get("name"), str):
                declarations.append({"name": raw["name"],
                                     "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                                     "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                                     else {"type": "object", "properties": {}}})
        config = {}
        if isinstance(body.get("max_tokens"), (int, float)):
            config["maxOutputTokens"] = body["max_tokens"]
        if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
            config["stopSequences"] = body["stop_sequences"]
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            config["topP"] = body["top_p"]

    out = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Provider calls. One call yields either a refusal (status) or the provider's
# answer reduced to (text, usage in, usage out, stop reason string).
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_FINISH = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOP = ["end_turn", "max_tokens", "tool_use", "refusal"]


def stop_family(reason):
    if not isinstance(reason, str):
        return 0
    return STOP_FAMILIES.get(reason, 0)


def sse_frames(text):
    """(event, data) pairs from an SSE body."""
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" ") if line.startswith("data: ") else line[len("data:"):])
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _json_or_none(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def parse_openai_answer(raw, stream):
    """(text, in, out, stop) from an OpenAI-wire answer, JSON or SSE."""
    if stream:
        text, usage_in, usage_out, finish = "", None, None, None
        for _event, data in sse_frames(raw.decode("utf-8", "replace")):
            if data.strip() == "[DONE]":
                continue
            chunk = _json_or_none(data.encode("utf-8"))
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                usage_in = chunk["usage"].get("prompt_tokens")
                usage_out = chunk["usage"].get("completion_tokens")
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices and is_record(choices[0]):
                delta = choices[0].get("delta")
                if is_record(delta) and isinstance(delta.get("content"), str):
                    text += delta["content"]
                if isinstance(choices[0].get("finish_reason"), str):
                    finish = choices[0]["finish_reason"]
        return text, usage_in, usage_out, finish
    body = _json_or_none(raw)
    if not is_record(body):
        return "", None, None, None
    choices = body.get("choices")
    text, finish = "", None
    if isinstance(choices, list) and choices and is_record(choices[0]):
        message = choices[0].get("message")
        if is_record(message):
            content = message.get("content")
            text = content if isinstance(content, str) else text_of(content)
        finish = choices[0].get("finish_reason")
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens"), finish


def parse_anthropic_answer(raw, stream):
    if stream:
        text, usage_in, usage_out, stop = "", None, None, None
        for event, data in sse_frames(raw.decode("utf-8", "replace")):
            chunk = _json_or_none(data.encode("utf-8"))
            if not is_record(chunk):
                continue
            if event == "message_start" and is_record(chunk.get("message")) \
                    and is_record(chunk["message"].get("usage")):
                usage_in = chunk["message"]["usage"].get("input_tokens")
            elif event == "content_block_delta" and is_record(chunk.get("delta")) \
                    and chunk["delta"].get("type") == "text_delta":
                text += chunk["delta"].get("text") or ""
            elif event == "message_delta":
                if is_record(chunk.get("usage")):
                    usage_out = chunk["usage"].get("output_tokens")
                if is_record(chunk.get("delta")):
                    stop = chunk["delta"].get("stop_reason")
        return text, usage_in, usage_out, stop
    body = _json_or_none(raw)
    if not is_record(body):
        return "", None, None, None
    text = ""
    if isinstance(body.get("content"), list):
        text = "".join(b["text"] for b in body["content"]
                       if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return text, usage.get("input_tokens"), usage.get("output_tokens"), body.get("stop_reason")


def parse_gemini_answer(raw, stream):
    def from_obj(obj):
        text = ""
        candidates = obj.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content")
            if is_record(content) and isinstance(content.get("parts"), list):
                text = "".join(p["text"] for p in content["parts"]
                               if is_record(p) and isinstance(p.get("text"), str))
            finish = candidates[0].get("finishReason")
        else:
            finish = None
        usage = obj.get("usageMetadata") if is_record(obj.get("usageMetadata")) else {}
        return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), finish

    if stream:
        text, usage_in, usage_out, finish = "", None, None, None
        for _event, data in sse_frames(raw.decode("utf-8", "replace")):
            obj = _json_or_none(data.encode("utf-8"))
            if not is_record(obj):
                continue
            t, i, o, f = from_obj(obj)
            text += t
            if i is not None:
                usage_in = i
            if o is not None:
                usage_out = o
            if f is not None:
                finish = f
        return text, usage_in, usage_out, finish
    body = _json_or_none(raw)
    if not is_record(body):
        return "", None, None, None
    return from_obj(body)


def build_provider_call(provider, model_id, client_api, client_body, stream, json_mode):
    """(url, headers, body dict) for one provider, in its own wire."""
    base = join_base(ENV_URL, provider.base)
    headers = dict(auth_headers(provider.auth, provider.key))
    if provider.wire == "openai":
        if client_api == "openai":
            body = dict(client_body)
            body["model"] = model_id
            if stream:
                body["stream"] = True
                options = dict(body.get("stream_options") or {}) \
                    if is_record(body.get("stream_options")) else {}
                options["include_usage"] = True
                body["stream_options"] = options
        else:
            body = anthropic_to_openai(client_body, model_id)
        if stream:
            headers["accept"] = "text/event-stream"
        return f"{base}/v1/chat/completions", headers, body
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if client_api == "anthropic":
            body = dict(client_body)
            body["model"] = model_id
        else:
            body = openai_to_anthropic(client_body, model_id)
        if stream:
            headers["accept"] = "text/event-stream"
        return f"{base}/v1/messages", headers, body
    # gemini wire
    body = to_gemini(client_body, client_api, model_id, json_mode)
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{base}/v1beta/models/{model_id}:{action}"
    if stream:
        headers["accept"] = "text/event-stream"
    return url, headers, body


def call_provider(provider, model_id, client_api, client_body, stream, json_mode):
    """Returns (status, answer) where answer is (text, in, out, stop) on 2xx,
    else None. Transport failures come back as status 0."""
    url, headers, body = build_provider_call(provider, model_id, client_api,
                                             client_body, stream, json_mode)
    try:
        status, _hdrs, raw = http_request("POST", url, headers=headers, body=body)
    except Exception as err:
        log(f"call to {provider.id} failed in transport: {err}")
        return 0, None
    if status < 200 or status >= 300:
        return status, None
    if provider.wire == "openai":
        return status, parse_openai_answer(raw, stream)
    if provider.wire == "anthropic":
        return status, parse_anthropic_answer(raw, stream)
    return status, parse_gemini_answer(raw, stream)


# --------------------------------------------------------------------------
# Rendering the client's answer from (text, usage, stop family).
# --------------------------------------------------------------------------

def _pieces(text):
    if text == "":
        return [""]
    return [text[i:i + SSE_PIECE] for i in range(0, len(text), SSE_PIECE)]


def _sse(event, data):
    return f"{f'event: {event}' + chr(10) if event else ''}data: {json.dumps(data)}\n\n"


def render_answer(client_api, stream, model_id, text, usage_in, usage_out, family):
    """(content_type, body str) in the client's wire."""
    usage_in = usage_in if isinstance(usage_in, int) else 0
    usage_out = usage_out if isinstance(usage_out, int) else 0
    if client_api == "openai":
        finish = OPENAI_FINISH[family]
        if not stream:
            body = {
                "id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": finish, "logprobs": None}],
                "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                          "total_tokens": usage_in + usage_out},
            }
            return "application/json", json.dumps(body)
        frames = []

        def chunk(delta, finish_reason, usage=None):
            obj = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": model_id,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if usage is not None:
                obj["choices"] = []
                obj["usage"] = usage
            return _sse(None, obj)

        frames.append(chunk({"role": "assistant", "content": ""}, None))
        for piece in _pieces(text):
            frames.append(chunk({"content": piece}, None))
        frames.append(chunk({}, finish))
        frames.append(chunk({}, None, {"prompt_tokens": usage_in,
                                       "completion_tokens": usage_out,
                                       "total_tokens": usage_in + usage_out}))
        frames.append("data: [DONE]\n\n")
        return "text/event-stream", "".join(frames)
    # anthropic client
    stop = ANTHROPIC_STOP[family]
    if not stream:
        body = {
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
        }
        return "application/json", json.dumps(body)
    frames = [_sse("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0}}})]
    frames.append(_sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _pieces(text):
        frames.append(_sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(_sse("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames)


def openai_model_list(models):
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
        for m in models]}


def anthropic_model_list(models):
    return {"data": [
        {"type": "model", "id": m["id"], "display_name": m["display_name"],
         "created_at": FIXED_CREATED_AT} for m in models], "has_more": False}


def error_body(client_api, status, err_type, message):
    if client_api == "anthropic":
        return {"type": "error", "error": {"type": err_type, "message": message}}
    return {"error": {"message": message, "type": err_type, "code": None}}


# --------------------------------------------------------------------------
# Routing and the HTTP server.
# --------------------------------------------------------------------------

def requirements_from_body(body):
    vision = False
    for message in body.get("messages") or []:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format")
    json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision, "json": json_mode}


def body_json_mode(client_api, body):
    if client_api != "openai":
        return False
    fmt = body.get("response_format")
    return is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")


class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.manifest_index = 0
        self.refusals = {}   # provider id -> consecutive refusal count
        self.served_count = {}   # provider id -> requests it has answered
        self.lock = threading.Lock()

    def served_models(self):
        out = []
        seen = set()
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in sorted(probe["models"], key=lambda m: m.id):
                row = rating_of(self.brief, probe["provider"].id, model.id)
                if not model_free(model, row):
                    continue
                if model.id in seen:
                    continue
                seen.add(model.id)
                out.append({"id": model.id, "provider": probe["provider"].id,
                            "display_name": model.display_name})
        return out

    def candidates(self, requires, client_api, json_mode):
        found = []
        for index, probe in enumerate(self.probes):
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            # The anthropic wire cannot carry an OpenAI JSON-mode request: its
            # echo would report json_mode false and the request would fail.
            if json_mode and provider.wire == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of(self.brief, provider.id, model.id)
                if not model_free(model, row):
                    continue
                if not meets_requirements(requires, capabilities_of(model, row)):
                    continue
                rating = row["rating_milli"] if row is not None else -1
                # Rating decides; on a tie (the same model often sits on two
                # providers at one rating) the provider that has answered
                # fewer requests goes first, so no single provider's budget
                # is drained while an equal twin idles.
                found.append((-rating, self.served_count.get(provider.id, 0),
                              index, model.id, probe, model))
        found.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [(probe, model) for _neg, _used, _idx, _mid, probe, model in found]

    def serve(self, client_api, body):
        """Answer one client request: (status, content_type, body str)."""
        with self.lock:
            return self._serve_locked(client_api, body)

    def _serve_locked(self, client_api, body):
        manifest = self.brief.manifest
        entry = manifest[self.manifest_index] if self.manifest_index < len(manifest) else None
        self.manifest_index += 1
        requires = entry["requires"] if entry is not None else requirements_from_body(body)
        stream = body.get("stream") is True
        json_mode = body_json_mode(client_api, body)
        options = self.candidates(requires, client_api, json_mode)
        if not options:
            err_type = "overloaded_error" if client_api == "anthropic" else "no_free_model"
            return 503, "application/json", json.dumps(error_body(
                client_api, 503, err_type, "no free model on the directory meets this request"))
        attempts = 0
        last_status = 502
        # First pass: providers that have not refused twice in a row. Second
        # pass: the skipped ones, when nothing else is left.
        for skip_dead in (True, False):
            for probe, model in options:
                provider = probe["provider"]
                if skip_dead and self.refusals.get(provider.id, 0) >= 2:
                    continue
                if attempts >= MAX_ATTEMPTS:
                    break
                attempts += 1
                status, answer = call_provider(provider, model.id, client_api, body,
                                               stream, json_mode or bool(requires.get("json")))
                if answer is not None:
                    self.refusals[provider.id] = 0
                    self.served_count[provider.id] = \
                        self.served_count.get(provider.id, 0) + 1
                    text, usage_in, usage_out, stop = answer
                    content_type, rendered = render_answer(
                        client_api, stream, model.id, text, usage_in, usage_out,
                        stop_family(stop))
                    return 200, content_type, rendered
                last_status = status
                self.refusals[provider.id] = self.refusals.get(provider.id, 0) + 1
                log(f"{provider.id} refused with {status}; trying the next candidate")
            if attempts >= MAX_ATTEMPTS:
                break
        err_type = "api_error" if client_api == "anthropic" else "upstream_error"
        return 502, "application/json", json.dumps(error_body(
            client_api, last_status, err_type,
            "every free model that meets this request refused or was unreachable"))


AGG = None  # the Aggregator, set in main()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, content_type, text):
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return None
        if length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return value if is_record(value) else None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            models = AGG.served_models()
            if self.headers.get("x-api-key") is not None:
                self._send(200, "application/json", json.dumps(anthropic_model_list(models)))
            else:
                self._send(200, "application/json", json.dumps(openai_model_list(models)))
            return
        self._send(404, "application/json", json.dumps(
            error_body("openai", 404, "not_found", f"no route GET {path}")))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send(404, "application/json", json.dumps(
                error_body("openai", 404, "not_found", f"no route POST {path}")))
            return
        body = self._read_body()
        if body is None:
            err_type = "invalid_request_error"
            self._send(400, "application/json", json.dumps(error_body(
                client_api, 400, err_type, "the request body is not one JSON object")))
            return
        try:
            status, content_type, text = AGG.serve(client_api, body)
        except Exception as err:  # never leave a request hanging
            log(f"serve failed: {err!r}")
            status, content_type, text = 500, "application/json", json.dumps(error_body(
                client_api, 500, "internal_error", "the server failed to answer"))
        self._send(status, content_type, text)


def load():
    brief = read_brief(PROMPT_FILE)
    if brief.error:
        log(brief.error)
    probes = [probe_provider(ENV_URL, provider) for provider in brief.providers]
    for probe in probes:
        if not probe["ok"] and probe["error"]:
            log(probe["error"])
    catalog = build_catalog(probes, brief)
    return brief, probes, catalog


def register(port, catalog):
    _status, _hdrs, raw = http_request(
        "POST", f"{ENV_URL}/register",
        body={"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
        timeout=PROBE_TIMEOUT_S)
    reply = _json_or_none(raw)
    if is_record(reply) and isinstance(reply.get("requests"), int):
        log(f"registered: the workload holds {reply['requests']} requests")


def poll_work(server):
    while True:
        try:
            status, _hdrs, raw = http_request("GET", f"{ENV_URL}/work", timeout=2)
            if status == 200:
                work = _json_or_none(raw)
                if is_record(work) and work.get("done") is True:
                    server.shutdown()
                    return
        except Exception:
            pass
        threading.Event().wait(0.1)


def main():
    global AGG
    brief, probes, catalog = load()
    AGG = Aggregator(brief, probes, catalog)
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = None
    for _attempt in range(5):
        candidate = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        if candidate.server_address[1] != 8080:
            server = candidate
            break
        candidate.server_close()
    if server is None:
        log("the OS kept picking 8080, which the env owns on one layout")
        return 1
    server.daemon_threads = True
    port = server.server_address[1]
    log(f"listening on 0.0.0.0:{port} with {len(catalog['providers'])} answered "
        f"providers and {len(AGG.served_models())} free models")
    try:
        register(port, catalog)
    except Exception as err:
        log(f"register failed: {err}")
        return 1
    threading.Thread(target=poll_work, args=(server,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    sys.exit(main())
