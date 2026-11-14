#!/usr/bin/env python3
"""Aggregator entry for free-model-aggregator-01.

One local HTTP server fronts the free tiers of every provider named in the
trial brief. The brief (MAKEFUN_PROMPT_FILE) carries three JSON documents,
found by their `schema` fields: a provider directory, a sealed quality
table, and a workload manifest. The server probes each provider's model
listing in that provider's own wire format, folds the listings and the
quality table into a single catalog, registers the catalog and its port
with the trial env (MAKEFUN_ENV_URL), and then answers client requests on
the two public wires -- OpenAI Chat Completions and Anthropic Messages,
buffered or streamed -- always with the highest-rated free model that
satisfies the request and is still answering, translating the request and
the reply across wires as needed. It leaves when the env reports the
workload complete.

Determinism is a hard requirement: a replay must reproduce a live run byte
for byte, so nothing here reads a clock or a random source; every rendered
timestamp is a fixed constant. Standard library only.
"""

import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

SCHEMA_PROVIDERS = "free-model-aggregator-01/providers/v1"
SCHEMA_RATINGS = "free-model-aggregator-01/ratings/v1"
SCHEMA_WORKLOAD = "free-model-aggregator-01/workload/v1"

PROTOCOL_VERSION = 1
ANTHROPIC_VERSION = "2023-06-01"

# Fixed render constants: a moving `created` would break replay equality.
CREATED_EPOCH = 1767225600
CREATED_ISO = "2026-01-01T00:00:00Z"

PROBE_DEADLINE = 5.0
UPSTREAM_DEADLINE = 8.0
REGISTER_DEADLINE = 5.0
WORK_DEADLINE = 2.0
REGISTER_TRIES = 30
POLL_PAUSE = 0.1
BODY_LIMIT = 32 * 1024 * 1024
STREAM_SLICE = 24

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_PATH = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# The providers live behind the env itself; a proxy must never sit in
# between, so every call goes through an opener with an empty proxy map.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def note(text):
    sys.stderr.write("aggregator: %s\n" % text)
    sys.stderr.flush()


def is_map(value):
    return isinstance(value, dict)


def dump_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def text_or_empty(value):
    return value if isinstance(value, str) else ""


def number_or_none(value):
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Reading the brief.
# ---------------------------------------------------------------------------

def scan_documents(text):
    """All JSON objects embedded in *text* that carry a string `schema`."""
    decoder = json.JSONDecoder()
    found = []
    pos = text.find("{")
    while pos != -1:
        try:
            value, _end = decoder.raw_decode(text, pos)
        except ValueError:
            value = None
        if is_map(value) and isinstance(value.get("schema"), str):
            found.append(value)
        pos = text.find("{", pos + 1)
    return found


def clean_requirement(raw):
    if not is_map(raw):
        return {"tools": False, "vision": False, "json": False, "min_context": None}
    return {
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
        "min_context": number_or_none(raw.get("min_context")),
    }


def clean_provider(raw):
    if not is_map(raw):
        return None
    pid = text_or_empty(raw.get("id"))
    wire = raw.get("wire")
    if not pid or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = raw.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if text_or_empty(raw.get(key)):
            base = raw[key]
            break
    return {
        "id": pid,
        "name": text_or_empty(raw.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": text_or_empty(raw.get("key_env")),
        "key": text_or_empty(raw.get("key")),
    }


class Brief:
    """The three brief documents, normalized."""

    def __init__(self):
        self.providers = []
        self.quality = {}   # (provider id, model id) -> quality row
        self.manifest = []  # workload rows in sending order
        self.problem = None


def load_brief(path):
    brief = Brief()
    if not path:
        brief.problem = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.problem = "cannot read the brief at %s: %s" % (path, err)
        return brief
    directory = table = workload = None
    for doc in scan_documents(text):
        if doc.get("schema") == SCHEMA_PROVIDERS and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == SCHEMA_RATINGS and isinstance(doc.get("ratings"), list):
            table = doc
        elif doc.get("schema") == SCHEMA_WORKLOAD and isinstance(doc.get("manifest"), list):
            workload = doc
    if directory is None:
        brief.problem = "the brief holds no %s document" % SCHEMA_PROVIDERS
        return brief
    for raw in directory["providers"]:
        provider = clean_provider(raw)
        if provider is not None:
            brief.providers.append(provider)
    if table is not None:
        for raw in table["ratings"]:
            if not is_map(raw):
                continue
            pid = text_or_empty(raw.get("provider"))
            mid = text_or_empty(raw.get("model"))
            if not pid or not mid:
                continue
            brief.quality[(pid, mid)] = {
                "rating_milli": number_or_none(raw.get("rating_milli")) or 0,
                "context_length": number_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            }
    if workload is not None:
        for raw in workload["manifest"]:
            if not is_map(raw) or not text_or_empty(raw.get("id")):
                continue
            brief.manifest.append({
                "id": raw["id"],
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": clean_requirement(raw.get("requires")),
            })
    return brief


# ---------------------------------------------------------------------------
# Talking to the env and the providers (always direct, never proxied).
# ---------------------------------------------------------------------------

def call_http(method, url, headers=None, payload=None, timeout=UPSTREAM_DEADLINE):
    """A single HTTP exchange; returns (status, body). Transport trouble
    raises OSError so the caller can treat it as a refusal."""
    data = None
    merged = dict(headers or {})
    if payload is not None:
        data = dump_bytes(payload)
        merged.setdefault("content-type", "application/json")
    request = urllib.request.Request(url, data=data, headers=merged, method=method)
    try:
        with DIRECT.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def provider_root(env_url, base):
    """Join a directory base onto the env URL; an absolute base stands."""
    text = (base or "").strip()
    if re.match(r"^https?://", text, re.IGNORECASE):
        return text.rstrip("/")
    if not text:
        return env_url.rstrip("/")
    root = env_url.rstrip("/")
    return root + ("" if text.startswith("/") else "/") + text.rstrip("/")


def key_headers(provider):
    key = provider["key"]
    if not key or provider["auth"] == "none":
        return {}
    if provider["auth"] == "bearer":
        return {"authorization": "Bearer " + key}
    if provider["auth"] == "x-api-key":
        return {"x-api-key": key}
    if provider["auth"] == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# Probing the providers' listings, each in its own wire.
# ---------------------------------------------------------------------------

class Listing:
    """The outcome of probing one provider."""

    def __init__(self, provider):
        self.provider = provider
        self.ok = False
        self.models = []
        self.problem = None


def clean_listing_model(wire, raw):
    if not is_map(raw):
        return None
    if wire == "gemini":
        mid = text_or_empty(raw.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
        display = text_or_empty(raw.get("displayName")) or text_or_empty(raw.get("display_name"))
        context = number_or_none(raw.get("context_length"))
        if context is None:
            context = number_or_none(raw.get("inputTokenLimit"))
    else:
        mid = text_or_empty(raw.get("id"))
        display = text_or_empty(raw.get("display_name")) or text_or_empty(raw.get("displayName"))
        context = number_or_none(raw.get("context_length"))
        if context is None:
            context = number_or_none(raw.get("context_window"))
    if not mid:
        return None
    free = raw.get("free")
    return {
        "id": mid,
        "display_name": display or mid,
        "free": free if isinstance(free, bool) else None,
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_one(env_url, provider):
    listing = Listing(provider)
    root = provider_root(env_url, provider["base"])
    url = root + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(key_headers(provider))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, raw = call_http("GET", url, headers=headers, timeout=PROBE_DEADLINE)
    except (OSError, ValueError) as err:
        listing.problem = "GET %s failed: %s" % (url, err)
        return listing
    if status < 200 or status >= 300:
        listing.problem = "GET %s answered HTTP %d" % (url, status)
        return listing
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        listing.problem = "GET %s answered a body that is not JSON" % url
        return listing
    if not is_map(body):
        return listing
    rows = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if isinstance(rows, list):
        for raw_model in rows:
            model = clean_listing_model(provider["wire"], raw_model)
            if model is not None:
                listing.models.append(model)
    listing.ok = True
    return listing


def probe_everything(env_url, providers):
    """Probe concurrently; the result keeps the directory's order."""
    results = [None] * len(providers)

    def work(index, provider):
        results[index] = probe_one(env_url, provider)

    threads = []
    for index, provider in enumerate(providers):
        thread = threading.Thread(target=work, args=(index, provider), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(PROBE_DEADLINE + 2.0)
    for index, provider in enumerate(providers):
        if results[index] is None:
            stuck = Listing(provider)
            stuck.problem = "the probe did not finish"
            results[index] = stuck
    return results


# ---------------------------------------------------------------------------
# The catalog: the one document --list prints and /register posts.
# ---------------------------------------------------------------------------

def quality_row(brief, provider_id, model_id):
    return brief.quality.get((provider_id, model_id))


def is_free(brief, model, provider_id):
    """The quality table's word wins; then the listing's; default paid."""
    row = quality_row(brief, provider_id, model["id"])
    if row is not None:
        return row["free"]
    return model["free"] is True


def context_of(brief, model, provider_id):
    row = quality_row(brief, provider_id, model["id"])
    if row is not None and row["context_length"] is not None:
        return row["context_length"]
    return model["context_length"]


def make_catalog(brief, listings):
    providers = []
    for listing in listings:
        if not listing.ok:
            continue
        models = []
        seen = set()
        for model in sorted(listing.models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            models.append({
                "id": model["id"],
                "free": is_free(brief, model, listing.provider["id"]),
                "context_length": context_of(brief, model, listing.provider["id"]),
            })
        providers.append({
            "id": listing.provider["id"],
            "base_url": listing.provider["base"],
            "key_env": listing.provider["key_env"],
            "wire": listing.provider["wire"],
            "models": models,
        })
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Wire translation.
#
# A provider parses an incoming request its own way: the system texts joined
# with "\n", the final user message's text, the tool names in order,
# max_tokens, the stop list, the final image's hash, and whether JSON mode
# was requested. A translation is faithful when the provider's parser reads
# the translated request exactly as the client's own parser read the
# original, so each converter below is written against those readings.
# ---------------------------------------------------------------------------

def content_text(content):
    """A message content the way the parsers read it: a plain string as it
    is, otherwise its text blocks joined with newlines."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"] for block in content
        if is_map(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def system_text_openai(body):
    parts = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_map(message) and message.get("role") == "system":
                text = content_text(message.get("content"))
                if text:
                    parts.append(text)
    return "\n".join(parts)


def system_text_anthropic(body):
    return content_text(body.get("system"))


def as_stop_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def token_cap(body, *names):
    for name in names:
        value = number_or_none(body.get(name))
        if value is not None:
            return value
    return None


DATA_URL = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


def split_data_url(url):
    """(media type, base64 payload) of a base64 data URL, else None."""
    if not isinstance(url, str):
        return None
    match = DATA_URL.match(url)
    if match is None or match.group(2) is None:
        return None
    return (match.group(1) or "image/png"), match.group(3)


def arguments_object(value):
    """Tool-call arguments as an object: parse the JSON string, else {}."""
    if is_map(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if is_map(parsed):
            return parsed
    return {}


def tools_openai(body):
    """OpenAI tool definitions normalized to name/description/parameters."""
    found = []
    raw = body.get("tools")
    if isinstance(raw, list):
        for item in raw:
            if not is_map(item) or not is_map(item.get("function")):
                continue
            fn = item["function"]
            if not isinstance(fn.get("name"), str):
                continue
            found.append({
                "name": fn["name"],
                "description": text_or_empty(fn.get("description")),
                "parameters": fn.get("parameters") if is_map(fn.get("parameters"))
                else {"type": "object", "properties": {}},
            })
    return found


def tools_anthropic(body):
    """Anthropic tool definitions normalized to the same triple."""
    found = []
    raw = body.get("tools")
    if isinstance(raw, list):
        for item in raw:
            if not is_map(item) or not isinstance(item.get("name"), str):
                continue
            found.append({
                "name": item["name"],
                "description": text_or_empty(item.get("description")),
                "parameters": item.get("input_schema") if is_map(item.get("input_schema"))
                else {"type": "object", "properties": {}},
            })
    return found


# --- client OpenAI -> provider Anthropic ---------------------------------

def image_block_from_openai(url):
    """An OpenAI image_url part as an Anthropic image block carrying the
    same bytes, so both wires hash the image alike."""
    data_url = split_data_url(url)
    if data_url is not None:
        media, data = data_url
        return {"type": "image",
                "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def request_openai_to_anthropic(body, model_id):
    out = {"model": model_id}
    system = system_text_openai(body)
    if system:
        out["system"] = system
    messages = []
    for message in body.get("messages") or []:
        if not is_map(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": text_or_empty(message.get("tool_call_id")),
                "content": content_text(content) if isinstance(content, list)
                else text_or_empty(content),
            }
            if messages and messages[-1].get("role") == "user":
                last = messages[-1]
                if isinstance(last.get("content"), list):
                    last["content"].append(block)
                else:
                    previous = last.get("content")
                    last["content"] = ([{"type": "text", "text": previous}, block]
                                        if isinstance(previous, str) and previous else [block])
            else:
                messages.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            blocks = []
            if isinstance(content, list):
                for part in content:
                    if is_map(part) and part.get("type") == "text" \
                            and isinstance(part.get("text"), str):
                        blocks.append({"type": "text", "text": part["text"]})
            for call in message.get("tool_calls") or []:
                if not is_map(call) or not is_map(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": text_or_empty(call.get("id")),
                    "name": text_or_empty(call["function"].get("name")),
                    "input": arguments_object(call["function"].get("arguments")),
                })
            messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        if isinstance(content, str) or content is None:
            messages.append({"role": "user", "content": text_or_empty(content)})
            continue
        blocks = []
        for part in content:
            if not is_map(part):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url" and is_map(part.get("image_url")):
                blocks.append(image_block_from_openai(part["image_url"].get("url")))
        messages.append({"role": "user", "content": blocks})
    out["messages"] = messages
    cap = token_cap(body, "max_tokens", "max_completion_tokens")
    if cap is not None:
        out["max_tokens"] = cap
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = as_stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = tools_openai(body)
    if tools:
        out["tools"] = [{"name": t["name"], "description": t["description"],
                         "input_schema": t["parameters"]} for t in tools]
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_map(choice) and is_map(choice.get("function")):
        out["tool_choice"] = {"type": "tool",
                              "name": text_or_empty(choice["function"].get("name"))}
    return out


# --- client Anthropic -> provider OpenAI ---------------------------------

def request_anthropic_to_openai(body, model_id):
    out = {"model": model_id}
    messages = []
    system = system_text_anthropic(body)
    if system:
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not is_map(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            text = content_text(content)
            calls = []
            if isinstance(content, list):
                for block in content:
                    if is_map(block) and block.get("type") == "tool_use":
                        calls.append({
                            "id": text_or_empty(block.get("id")),
                            "type": "function",
                            "function": {
                                "name": text_or_empty(block.get("name")),
                                "arguments": json.dumps(
                                    block.get("input") if is_map(block.get("input")) else {}),
                            },
                        })
            converted = {"role": "assistant", "content": text if text else None}
            if calls:
                converted["tool_calls"] = calls
            messages.append(converted)
            continue
        # user: text and images keep their order as parts; each tool_result
        # becomes its own tool message (the OpenAI parser reads only user
        # messages for echoed text, so a result can never clobber it).
        if isinstance(content, str) or content is None:
            messages.append({"role": "user", "content": text_or_empty(content)})
            continue
        parts = []
        for block in content:
            if not is_map(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_map(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = text_or_empty(source.get("media_type")) or "image/png"
                    parts.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url",
                                  "image_url": {"url": source["url"]}})
            elif block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": text_or_empty(block.get("tool_use_id")),
                    "content": content_text(block.get("content"))
                    if isinstance(block.get("content"), list)
                    else text_or_empty(block.get("content")),
                })
        if parts:
            messages.append({"role": "user", "content": parts})
    out["messages"] = messages
    cap = token_cap(body, "max_tokens")
    if cap is not None:
        out["max_tokens"] = cap
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = as_stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = tools_anthropic(body)
    if tools:
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in tools]
    choice = body.get("tool_choice")
    if is_map(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool":
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": text_or_empty(choice.get("name"))}}
    return out


# --- either client -> provider Gemini -------------------------------------

def gemini_part_for_image(url):
    data_url = split_data_url(url)
    if data_url is not None:
        media, data = data_url
        return {"inlineData": {"mimeType": media, "data": data}}
    return {"fileData": {"fileUri": url}}


def request_to_gemini(client_wire, body):
    """An OpenAI or Anthropic client body as a Gemini generateContent body."""
    out = {}
    if client_wire == "openai":
        system = system_text_openai(body)
        tools = tools_openai(body)
        cap = token_cap(body, "max_tokens", "max_completion_tokens")
        stops = as_stop_list(body.get("stop"))
    else:
        system = system_text_anthropic(body)
        tools = tools_anthropic(body)
        cap = token_cap(body, "max_tokens")
        stops = as_stop_list(body.get("stop_sequences"))
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    # Gemini answers a function call by name, not by id; remember which name
    # each call id carried so later results line up.
    names_by_id = {}
    contents = []
    for message in body.get("messages") or []:
        if not is_map(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if client_wire == "openai" and role == "tool":
            part = {"functionResponse": {
                "name": names_by_id.get(text_or_empty(message.get("tool_call_id")), ""),
                "response": {"result": content_text(content) if isinstance(content, list)
                             else text_or_empty(content)},
            }}
            if contents and contents[-1].get("role") == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_map(block):
                    continue
                kind = block.get("type")
                if kind == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif client_wire == "openai" and kind == "image_url" \
                        and is_map(block.get("image_url")):
                    parts.append(gemini_part_for_image(block["image_url"].get("url")))
                elif client_wire == "anthropic" and kind == "image" \
                        and is_map(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" \
                            and isinstance(source.get("data"), str):
                        media = text_or_empty(source.get("media_type")) or "image/png"
                        parts.append({"inlineData": {"mimeType": media,
                                                     "data": source["data"]}})
                    elif source.get("type") == "url" \
                            and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif client_wire == "anthropic" and kind == "tool_use":
                    names_by_id[text_or_empty(block.get("id"))] = \
                        text_or_empty(block.get("name"))
                    parts.append({"functionCall": {
                        "name": text_or_empty(block.get("name")),
                        "args": block.get("input") if is_map(block.get("input")) else {},
                    }})
                elif client_wire == "anthropic" and kind == "tool_result":
                    parts.append({"functionResponse": {
                        "name": names_by_id.get(text_or_empty(block.get("tool_use_id")), ""),
                        "response": {"result": content_text(block.get("content"))
                                     if isinstance(block.get("content"), list)
                                     else text_or_empty(block.get("content"))},
                    }})
        if client_wire == "openai" and role == "assistant":
            for call in message.get("tool_calls") or []:
                if not is_map(call) or not is_map(call.get("function")):
                    continue
                names_by_id[text_or_empty(call.get("id"))] = \
                    text_or_empty(call["function"].get("name"))
                parts.append({"functionCall": {
                    "name": text_or_empty(call["function"].get("name")),
                    "args": arguments_object(call["function"].get("arguments")),
                }})
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": parts})
    out["contents"] = contents
    config = {}
    if cap is not None:
        config["maxOutputTokens"] = cap
    if number_or_none(body.get("temperature")) is not None:
        config["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        config["topP"] = body["top_p"]
    if stops:
        config["stopSequences"] = stops
    if client_wire == "openai":
        fmt = body.get("response_format")
        if is_map(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    if tools:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in tools]}]
    return out


# ---------------------------------------------------------------------------
# Upstream calls and their answers.
# ---------------------------------------------------------------------------

def upstream_call(client_wire, body, provider, model_id):
    """(url, headers, payload) for one provider call, in its own wire."""
    root = provider_root(ENV_URL, provider["base"])
    wire = provider["wire"]
    headers = {"accept": "application/json"}
    headers.update(key_headers(provider))
    if wire == "openai":
        if client_wire == "anthropic":
            payload = request_anthropic_to_openai(body, model_id)
        else:
            payload = dict(body)
            payload["model"] = model_id
            payload.pop("stream", None)
            payload.pop("stream_options", None)
        return root + "/v1/chat/completions", headers, payload
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_wire == "anthropic":
            payload = dict(body)
            payload["model"] = model_id
            payload.pop("stream", None)
        else:
            payload = request_openai_to_anthropic(body, model_id)
        return root + "/v1/messages", headers, payload
    payload = request_to_gemini(client_wire, body)
    url = "%s/v1beta/models/%s:generateContent" % (root, quote(model_id, safe=""))
    return url, headers, payload


def read_answer(wire, raw):
    """A provider's buffered answer as (text, tokens in, tokens out)."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not is_map(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_map(choices[0]):
            return None
        message = choices[0].get("message")
        if not is_map(message):
            return None
        content = message.get("content")
        text = content if isinstance(content, str) else content_text(content)
        usage = body.get("usage") if is_map(body.get("usage")) else {}
        return text, number_or_none(usage.get("prompt_tokens")) or 0, \
            number_or_none(usage.get("completion_tokens")) or 0
    if wire == "anthropic":
        # The client's own reader concatenates text blocks with no
        # separator; read the provider's answer the same way it is read back.
        text = ""
        content = body.get("content")
        if isinstance(content, list):
            text = "".join(
                block["text"] for block in content
                if is_map(block) and block.get("type") == "text"
                and isinstance(block.get("text"), str))
        usage = body.get("usage") if is_map(body.get("usage")) else {}
        return text, number_or_none(usage.get("input_tokens")) or 0, \
            number_or_none(usage.get("output_tokens")) or 0
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_map(candidates[0]):
        return None
    content = candidates[0].get("content")
    if not is_map(content):
        return None
    text = ""
    parts = content.get("parts")
    if isinstance(parts, list):
        text = "".join(part["text"] for part in parts
                       if is_map(part) and isinstance(part.get("text"), str))
    usage = body.get("usageMetadata") if is_map(body.get("usageMetadata")) else {}
    return text, number_or_none(usage.get("promptTokenCount")) or 0, \
        number_or_none(usage.get("candidatesTokenCount")) or 0


# ---------------------------------------------------------------------------
# Routing: the best free model that meets the request, with failover.
# ---------------------------------------------------------------------------

BRIEF = load_brief(PROMPT_PATH)
LISTINGS = []
CATALOG = {"providers": []}
REFUSAL_STREAK = {}  # provider id -> refusals in a row (a success resets it)
ANSWERED = {}        # provider id -> requests it has answered
ARRIVAL = [0]        # client requests seen; the manifest is consumed in order
GATE = threading.Lock()


def prepare():
    global LISTINGS, CATALOG
    if BRIEF.problem is not None:
        note(BRIEF.problem)
    LISTINGS = probe_everything(ENV_URL, BRIEF.providers)
    for listing in LISTINGS:
        if not listing.ok and listing.problem is not None:
            note(listing.problem)
        REFUSAL_STREAK.setdefault(listing.provider["id"], 0)
        ANSWERED.setdefault(listing.provider["id"], 0)
    CATALOG = make_catalog(BRIEF, LISTINGS)


def listed_free_models():
    """The free models on offer: unique ids, in catalog order."""
    out = []
    seen = set()
    for listing in LISTINGS:
        if not listing.ok:
            continue
        for model in sorted(listing.models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            if not is_free(BRIEF, model, listing.provider["id"]):
                continue
            seen.add(model["id"])
            out.append({"id": model["id"], "provider": listing.provider["id"],
                        "display_name": model["display_name"]})
    return out


def capabilities_of(provider_id, model):
    row = quality_row(BRIEF, provider_id, model["id"])
    if row is not None:
        return {
            "context_length": row["context_length"]
            if row["context_length"] is not None else model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def satisfies(requires, caps):
    if requires["tools"] and caps["tools"] is not True:
        return False
    if requires["vision"] and caps["vision"] is not True:
        return False
    if requires["json"] and caps["json"] is not True:
        return False
    if requires["min_context"] is not None:
        if caps["context_length"] is None or caps["context_length"] < requires["min_context"]:
            return False
    return True


def requirements_from_body(client_wire, body):
    vision = False
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not is_map(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_map(block) and block.get("type") in ("image", "image_url"):
                    vision = True
    wants_json = False
    if client_wire == "openai":
        fmt = body.get("response_format")
        wants_json = is_map(fmt) and fmt.get("type") in ("json_object", "json_schema")
    tools = tools_anthropic(body) if client_wire == "anthropic" else tools_openai(body)
    return {"tools": len(tools) > 0, "vision": vision, "json": wants_json,
            "min_context": None}


def requirements_for(client_wire, body):
    """The arriving request's requirements: the manifest's row for this
    arrival (the client works through the manifest in order), or the body's
    own once the manifest is exhausted."""
    index = ARRIVAL[0]
    ARRIVAL[0] = index + 1
    if index < len(BRIEF.manifest):
        return BRIEF.manifest[index]["requires"]
    return requirements_from_body(client_wire, body)


def body_json_mode(client_wire, body):
    if client_wire != "openai":
        return False
    fmt = body.get("response_format")
    return is_map(fmt) and fmt.get("type") in ("json_object", "json_schema")


def route(requires, client_wire, body):
    """One candidate per provider -- its best free model that meets the
    request -- best rating first. A refusal is a property of the provider
    (its budget, its failure schedule, its outage), never of the model, so
    a second model on a refusing provider is never worth a call within the
    same request."""
    json_mode = body_json_mode(client_wire, body)
    found = []
    for index, listing in enumerate(LISTINGS):
        if not listing.ok:
            continue
        provider = listing.provider
        # An Anthropic-wire provider parses JSON mode as absent, so it could
        # never echo json_mode back for an OpenAI JSON-mode request: it is
        # not a candidate for one at all.
        if json_mode and provider["wire"] == "anthropic":
            continue
        best = None
        for model in listing.models:
            if not is_free(BRIEF, model, provider["id"]):
                continue
            if not satisfies(requires, capabilities_of(provider["id"], model)):
                continue
            row = quality_row(BRIEF, provider["id"], model["id"])
            rating = row["rating_milli"] if row is not None else -1
            key = (-rating, model["id"])
            if best is None or key < best[0]:
                best = (key, model)
        if best is not None:
            found.append((-best[0][0], index, provider, best[1]))
    # Rating decides first: the highest rating goes first. On a tie -- the
    # same model often sits on two providers at one rating -- the provider
    # that has answered fewer requests so far goes first, so one twin's
    # small budget is not drained while the other idles; directory order
    # breaks whatever is left.
    found.sort(key=lambda item: (-item[0], ANSWERED.get(item[2]["id"], 0), item[1]))
    return [(provider, model) for _neg, _idx, provider, model in found]


def try_provider(provider, model, client_wire, body):
    """One upstream call, buffered; the parsed answer or None."""
    url, headers, payload = upstream_call(client_wire, body, provider, model["id"])
    try:
        status, raw = call_http("POST", url, headers=headers, payload=payload,
                                timeout=UPSTREAM_DEADLINE)
    except (OSError, ValueError) as err:
        note("call to %s failed: %s" % (provider["id"], err))
        return None
    if status < 200 or status >= 300:
        return None
    return read_answer(provider["wire"], raw)


def answer_request(client_wire, body):
    """(model id, text, tokens in, tokens out) for one client request, or
    None when no candidate answers."""
    options = route(requirements_for(client_wire, body), client_wire, body)
    # The first pass skips a provider that has refused twice in a row while
    # any alternative remains -- it is most likely out of budget or inside
    # its outage, and every wasted call counts against the trial. The
    # second pass takes whatever is left, in the same order.
    for second_pass in (False, True):
        for provider, model in options:
            pid = provider["id"]
            if not second_pass and len(options) > 1 and REFUSAL_STREAK.get(pid, 0) >= 2:
                continue
            answer = try_provider(provider, model, client_wire, body)
            if answer is not None:
                REFUSAL_STREAK[pid] = 0
                ANSWERED[pid] = ANSWERED.get(pid, 0) + 1
                text, tokens_in, tokens_out = answer
                return model["id"], text, tokens_in, tokens_out
            REFUSAL_STREAK[pid] = REFUSAL_STREAK.get(pid, 0) + 1
    return None


# ---------------------------------------------------------------------------
# Rendering a provider's answer on the client's wire.
# ---------------------------------------------------------------------------

def slices_of(text):
    pieces = [text[i:i + STREAM_SLICE] for i in range(0, len(text), STREAM_SLICE)]
    return pieces if pieces else [""]


def models_document_openai(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": CREATED_EPOCH,
                  "owned_by": m["provider"]} for m in models],
    }


def models_document_anthropic(models):
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                  "created_at": CREATED_ISO} for m in models],
        "has_more": False,
    }


def error_openai(message, kind):
    return {"error": {"message": message, "type": kind, "code": None}}


def error_anthropic(message, kind):
    return {"type": "error", "error": {"type": kind, "message": message}}


def render_openai(model_id, text, tokens_in, tokens_out, stream):
    usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
             "total_tokens": tokens_in + tokens_out}
    if not stream:
        return "application/json", dump_bytes({
            "id": "chatcmpl-agg",
            "object": "chat.completion",
            "created": CREATED_EPOCH,
            "model": model_id,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage,
        })
    frames = []

    def chunk(delta, finish, with_usage=False):
        frame = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                 "created": CREATED_EPOCH, "model": model_id,
                 "choices": [] if with_usage else
                 [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if with_usage:
            frame["usage"] = usage
        return "data: %s\n\n" % json.dumps(frame, ensure_ascii=False)

    frames.append(chunk({"role": "assistant", "content": ""}, None))
    for piece in slices_of(text):
        frames.append(chunk({"content": piece}, None))
    frames.append(chunk({}, "stop"))
    frames.append(chunk({}, None, with_usage=True))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_anthropic(model_id, text, tokens_in, tokens_out, stream):
    if not stream:
        return "application/json", dump_bytes({
            "id": "msg_agg",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        })
    frames = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, ensure_ascii=False))

    frames.append(event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": tokens_in, "output_tokens": 0}},
    }))
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in slices_of(text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(event("content_block_stop",
                        {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": tokens_out}}))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")


# ---------------------------------------------------------------------------
# The HTTP server the client talks to.
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, value):
        self._send(status, "application/json", dump_bytes(value))

    def _fail(self, wire, status, kind, message):
        if wire == "anthropic":
            self._send_json(status, error_anthropic(message, kind))
        else:
            self._send_json(status, error_openai(message, kind))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            models = listed_free_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, models_document_anthropic(models))
            else:
                self._send_json(200, models_document_openai(models))
            return
        self._fail("openai", 404, "not_found", "no route GET %s" % path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            self._fail("openai", 404, "not_found", "no route POST %s" % path)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._fail(wire, 400, "invalid_request_error",
                       "the Content-Length header is not a number")
            return
        if length > BODY_LIMIT:
            self._fail(wire, 413, "invalid_request_error",
                       "the request body is too large")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = None
        if not is_map(body):
            self._fail(wire, 400, "invalid_request_error",
                       "the request body is not one JSON object")
            return
        with GATE:
            answered = answer_request(wire, body)
        if answered is None:
            self._fail(wire, 503,
                       "overloaded_error" if wire == "anthropic" else "no_free_model",
                       "no free model on the directory could answer this request")
            return
        model_id, text, tokens_in, tokens_out = answered
        stream = body.get("stream") is True
        if wire == "openai":
            content_type, payload = render_openai(model_id, text, tokens_in,
                                                  tokens_out, stream)
        else:
            content_type, payload = render_anthropic(model_id, text, tokens_in,
                                                     tokens_out, stream)
        self._send(200, content_type, payload)


# ---------------------------------------------------------------------------
# Register with the env, poll for the end of the workload, and main.
# ---------------------------------------------------------------------------

def register(port):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG}
    last = None
    for _try in range(REGISTER_TRIES):
        try:
            status, raw = call_http("POST", ENV_URL + "/register", payload=payload,
                                    timeout=REGISTER_DEADLINE)
            if status == 200:
                try:
                    reply = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    reply = {}
                if is_map(reply) and isinstance(reply.get("requests"), int):
                    note("registered: the workload holds %d requests" % reply["requests"])
                return
            last = "HTTP %d" % status
        except (OSError, ValueError) as err:
            last = str(err)
        time.sleep(0.2)
    raise RuntimeError("POST %s/register never answered 200 (%s)" % (ENV_URL, last))


def wait_until_done():
    while True:
        try:
            status, raw = call_http("GET", ENV_URL + "/work", timeout=WORK_DEADLINE)
            if status == 200:
                work = json.loads(raw.decode("utf-8"))
                if is_map(work) and work.get("done") is True:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(POLL_PAUSE)


def main(argv):
    prepare()
    if "--list" in argv:
        # The same document the register body carries: print it and stop
        # before any server starts.
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"),
                                    ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = None
    port = None
    for _try in range(5):
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
        if port != 8080:
            break
        server.server_close()
        server = None
    if server is None:
        raise RuntimeError("the OS kept picking 8080, which the env owns on one layout")
    note("listening on 0.0.0.0:%d with %d answered providers and %d free models"
         % (port, len(CATALOG["providers"]), len(listed_free_models())))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port)
    wait_until_done()
    server.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as err:  # noqa: BLE001 - a failed start must exit nonzero
        note("fatal: %r" % (err,))
        sys.exit(1)
