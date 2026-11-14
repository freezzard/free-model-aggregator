#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

The trial hands this program a brief (the path in MAKEFUN_PROMPT_FILE) holding
three JSON documents -- a directory of ten simulated free-tier providers, the
sealed quality table with every model's rating and capabilities, and the
workload manifest -- and an env at MAKEFUN_ENV_URL that hosts the providers
under /p/<id>. The program probes each provider's catalog route in the
provider's own wire, builds one catalog document, registers its port and that
catalog with the env, then answers the client's twenty requests, each with the
highest-rated free model that meets the request's requirements and still
answers, translating between the client's wire (OpenAI Chat Completions or
Anthropic Messages, JSON or server-sent events) and the provider's wire
(OpenAI, Anthropic or Gemini). When the env's GET /work says the workload is
done, the program exits 0.

Everything an answer is made of derives from the brief, the request and the
provider's answer: timestamps in rendered documents are fixed constants, no
clock is read and nothing is random, so a replayed trial reproduces the live
one byte for byte. Standard library only.
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

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 8.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
MAX_BODY_BYTES = 32 * 1024 * 1024
STREAM_PIECE_CHARS = 24

# Fixed instants for every rendered document: a `created` that moved between
# runs would make a replay differ from its live pass.
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# One opener with an empty proxy map: the providers live on the env's own
# loopback (or pod) address, and no proxy may sit between this server and it.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    decoder = json.JSONDecoder()
    found = []
    at = text.find("{")
    while at != -1:
        try:
            value, _end = decoder.raw_decode(text, at)
        except ValueError:
            value = None
        if is_record(value) and isinstance(value.get("schema"), str):
            found.append(value)
        at = text.find("{", at + 1)
    return found


def _string(value):
    return value if isinstance(value, str) else ""


def _number_or_none(value):
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _provider_of(value):
    if not is_record(value):
        return None
    pid = _string(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _string(value.get(key)) != "":
            base = _string(value.get(key))
            break
    return {
        "id": pid,
        "name": _string(value.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string(value.get("key_env")),
        "key": _string(value.get("key")),
    }


def _requirement_of(value):
    if not is_record(value):
        return {"tools": False, "vision": False, "json": False, "min_context": None}
    return {
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "min_context": _number_or_none(value.get("min_context")),
    }


class Brief:
    def __init__(self):
        self.providers = []          # directory rows, in the brief's order
        self.ratings = {}            # (provider, model) -> quality-table row
        self.best = {}               # request id -> best rating_milli
        self.manifest = []           # workload rows, in sending order
        self.error = None


def read_brief(path):
    brief = Brief()
    if path == "":
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
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
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        provider = _provider_of(raw)
        if provider is not None:
            brief.providers.append(provider)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = _string(raw.get("model"))
            provider = _string(raw.get("provider"))
            if model == "" or provider == "":
                continue
            brief.ratings[(provider, model)] = {
                "rating_milli": _number_or_none(raw.get("rating_milli")) or 0,
                "context_length": _number_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            }
        best = ratings_doc.get("best")
        if isinstance(best, list):
            for raw in best:
                if is_record(raw) and _string(raw.get("id")) != "":
                    brief.best[_string(raw.get("id"))] = _number_or_none(raw.get("rating_milli")) or 0
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw) or _string(raw.get("id")) == "":
                continue
            brief.manifest.append({
                "id": _string(raw.get("id")),
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": _requirement_of(raw.get("requires")),
            })
    return brief


# --------------------------------------------------------------------------
# HTTP plumbing (no proxies anywhere; the env is the only peer).
# --------------------------------------------------------------------------

def http_request(method, url, headers=None, payload=None, timeout=CALL_TIMEOUT_S):
    """One HTTP call; returns (status, body bytes). Raises OSError on transport failure."""
    data = None
    sent_headers = dict(headers or {})
    if payload is not None:
        data = json_bytes(payload)
        sent_headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(url, data=data, headers=sent_headers, method=method)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def join_base(env_url, base):
    """A directory base joined to the env's URL; an absolute base stands as it is."""
    text = (base or "").strip()
    if re.match(r"^https?://", text, re.IGNORECASE):
        return text.rstrip("/")
    prefix = env_url.rstrip("/")
    if text == "":
        return prefix
    return prefix + ("" if text.startswith("/") else "/") + text.rstrip("/")


def auth_headers(provider):
    """The headers a provider wants its key in; `none` takes no key."""
    key = provider["key"]
    if key == "" or provider["auth"] == "none":
        return {}
    if provider["auth"] == "bearer":
        return {"authorization": "Bearer " + key}
    if provider["auth"] == "x-api-key":
        return {"x-api-key": key}
    if provider["auth"] == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# --------------------------------------------------------------------------
# The catalog probe: each provider's listing, in its own wire.
# --------------------------------------------------------------------------

class Probe:
    def __init__(self, provider):
        self.provider = provider
        self.ok = False
        self.status = 0
        self.models = []             # listing rows, normalized
        self.error = None


def _listing_model(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        model_id = _string(raw.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        display = _string(raw.get("displayName")) or _string(raw.get("display_name"))
        context = _number_or_none(raw.get("context_length"))
        if context is None:
            context = _number_or_none(raw.get("inputTokenLimit"))
    else:
        model_id = _string(raw.get("id"))
        display = _string(raw.get("display_name")) or _string(raw.get("displayName"))
        context = _number_or_none(raw.get("context_length"))
        if context is None:
            context = _number_or_none(raw.get("context_window"))
    if model_id == "":
        return None
    free = raw.get("free")
    if not isinstance(free, bool):
        free = None
    return {
        "id": model_id,
        "display_name": display or model_id,
        "free": free,
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_provider(env_url, provider):
    probe = Probe(provider)
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, raw = http_request("GET", url, headers=headers, timeout=PROBE_TIMEOUT_S)
    except (OSError, ValueError) as err:
        probe.error = "GET %s failed: %s" % (url, err)
        return probe
    probe.status = status
    if status < 200 or status >= 300:
        probe.error = "GET %s answered HTTP %d" % (url, status)
        return probe
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        probe.error = "GET %s answered a body that is not JSON" % url
        return probe
    if not is_record(body):
        return probe
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if isinstance(listing, list):
        for raw_model in listing:
            model = _listing_model(provider["wire"], raw_model)
            if model is not None:
                probe.models.append(model)
    probe.ok = True
    return probe


def probe_all(env_url, providers):
    """Probe every provider at once; the result list keeps the directory order."""
    probes = [None] * len(providers)

    def run(index, provider):
        probes[index] = probe_provider(env_url, provider)

    threads = []
    for index, provider in enumerate(providers):
        thread = threading.Thread(target=run, args=(index, provider), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(PROBE_TIMEOUT_S + 2.0)
    for index, probe in enumerate(probes):
        if probe is None:
            failed = Probe(providers[index])
            failed.error = "the probe did not finish"
            probes[index] = failed
    return probes


# --------------------------------------------------------------------------
# The catalog document: the same object --list prints and /register posts.
# --------------------------------------------------------------------------

def rating_row(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


def model_free(brief, probe_model, provider_id):
    """Free per the quality table where it has a row, else the listing's word, else paid."""
    row = rating_row(brief, provider_id, probe_model["id"])
    if row is not None:
        return row["free"]
    return probe_model["free"] is True


def model_context(brief, probe_model, provider_id):
    row = rating_row(brief, provider_id, probe_model["id"])
    if row is not None and row["context_length"] is not None:
        return row["context_length"]
    return probe_model["context_length"]


def build_catalog(brief, probes):
    providers = []
    for probe in probes:
        if not probe.ok:
            continue
        seen = set()
        models = []
        for model in sorted(probe.models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            models.append({
                "id": model["id"],
                "free": model_free(brief, model, probe.provider["id"]),
                "context_length": model_context(brief, model, probe.provider["id"]),
            })
        providers.append({
            "id": probe.provider["id"],
            "base_url": probe.provider["base"],
            "key_env": probe.provider["key_env"],
            "wire": probe.provider["wire"],
            "models": models,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire translations.
#
# A provider reads a request with its own wire's parser: the system texts
# joined with "\n", the last user message's text, the tool names in order,
# max_tokens, the stop list, the last image's hash, and whether JSON mode was
# asked for. A translation is faithful when the provider's parser reads the
# translated request exactly as the client's own wire parser reads the
# original, so every converter below is written against those readings.
# --------------------------------------------------------------------------

def blocks_text(content):
    """A content value as the provider parsers read it: a string as it is, or
    its text blocks joined with newlines."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def openai_system_text(body):
    parts = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_record(message) and message.get("role") == "system":
                text = blocks_text(message.get("content"))
                if text != "":
                    parts.append(text)
    return "\n".join(parts)


def anthropic_system_text(body):
    return blocks_text(body.get("system"))


def stop_list(value):
    """A stop field (one string or a list) as the list the parsers echo."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def max_tokens_of(body, *names):
    for name in names:
        value = _number_or_none(body.get(name))
        if value is not None:
            return value
    return None


DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


def split_data_url(url):
    """(media type, base64 data) for a base64 data URL, else None."""
    if not isinstance(url, str):
        return None
    match = DATA_URL_RE.match(url)
    if match is None or match.group(2) is None:
        return None
    return (match.group(1) or "image/png"), match.group(3)


def parse_arguments(value):
    """A tool call's arguments as an object: the JSON string parsed, else {}."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if is_record(parsed):
            return parsed
    return {}


# --- OpenAI client shapes -------------------------------------------------

def openai_image_block(url):
    """An OpenAI image_url part as an Anthropic image block (same bytes, so
    the same image hash on both wires)."""
    data_url = split_data_url(url)
    if data_url is not None:
        media, data = data_url
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def openai_to_anthropic(body, model_id):
    out = {"model": model_id}
    system = openai_system_text(body)
    if system != "":
        out["system"] = system
    messages = []
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": _string(message.get("tool_call_id")),
                "content": blocks_text(content) if isinstance(content, list) else _string(content),
            }
            if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), list):
                messages[-1]["content"].append(block)
            elif messages and messages[-1].get("role") == "user":
                previous = messages[-1]["content"]
                messages[-1]["content"] = [{"type": "text", "text": previous}, block] \
                    if isinstance(previous, str) and previous != "" else [block]
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
                    if is_record(part) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        blocks.append({"type": "text", "text": part["text"]})
            for call in message.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": _string(call.get("id")),
                    "name": _string(call["function"].get("name")),
                    "input": parse_arguments(call["function"].get("arguments")),
                })
            messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        if isinstance(content, str) or content is None:
            messages.append({"role": "user", "content": _string(content)})
            continue
        blocks = []
        for part in content:
            if not is_record(part):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                blocks.append(openai_image_block(part["image_url"].get("url")))
        messages.append({"role": "user", "content": blocks})
    out["messages"] = messages
    cap = max_tokens_of(body, "max_tokens", "max_completion_tokens")
    if cap is not None:
        out["max_tokens"] = cap
    if _number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if _number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = openai_tools(body)
    if tools:
        out["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                        for t in tools]
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": _string(choice["function"].get("name"))}
    return out


def openai_tools(body):
    """The body's tool definitions, normalized to name/description/parameters."""
    tools = []
    raw = body.get("tools")
    if isinstance(raw, list):
        for item in raw:
            if not is_record(item) or not is_record(item.get("function")):
                continue
            fn = item["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": _string(fn.get("description")),
                "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                else {"type": "object", "properties": {}},
            })
    return tools


def anthropic_tools(body):
    """The body's tool definitions, normalized the same way from the other spelling."""
    tools = []
    raw = body.get("tools")
    if isinstance(raw, list):
        for item in raw:
            if not is_record(item) or not isinstance(item.get("name"), str):
                continue
            tools.append({
                "name": item["name"],
                "description": _string(item.get("description")),
                "parameters": item.get("input_schema") if is_record(item.get("input_schema"))
                else {"type": "object", "properties": {}},
            })
    return tools


def anthropic_to_openai(body, model_id):
    out = {"model": model_id}
    messages = []
    system = anthropic_system_text(body)
    if system != "":
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            text = blocks_text(content)
            calls = []
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "tool_use":
                        calls.append({
                            "id": _string(block.get("id")),
                            "type": "function",
                            "function": {
                                "name": _string(block.get("name")),
                                "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                            },
                        })
            converted = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                converted["tool_calls"] = calls
            messages.append(converted)
            continue
        # user: text and images keep their order; each tool_result becomes a
        # tool message (the OpenAI parser reads only user messages for the
        # echoed text, so a tool result can never clobber it).
        if isinstance(content, str) or content is None:
            messages.append({"role": "user", "content": _string(content)})
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = _string(source.get("media_type")) or "image/png"
                    parts.append({"type": "image_url",
                                  "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
            elif block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": _string(block.get("tool_use_id")),
                    "content": blocks_text(block.get("content")) if isinstance(block.get("content"), list)
                    else _string(block.get("content")),
                })
        if parts:
            messages.append({"role": "user", "content": parts})
    out["messages"] = messages
    cap = max_tokens_of(body, "max_tokens")
    if cap is not None:
        out["max_tokens"] = cap
    if _number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if _number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = anthropic_tools(body)
    if tools:
        out["tools"] = [{"type": "function",
                         "function": {"name": t["name"], "description": t["description"],
                                      "parameters": t["parameters"]}} for t in tools]
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": _string(choice.get("name"))}}
    return out


# --- Gemini upstream shapes -------------------------------------------------

def gemini_image_part(url):
    data_url = split_data_url(url)
    if data_url is not None:
        media, data = data_url
        return {"inlineData": {"mimeType": media, "data": data}}
    return {"fileData": {"fileUri": url}}


def to_gemini(client_wire, body):
    """An OpenAI or Anthropic client body as a Gemini generateContent body."""
    out = {}
    if client_wire == "openai":
        system = openai_system_text(body)
        raw_messages = body.get("messages") or []
        tools = openai_tools(body)
        cap = max_tokens_of(body, "max_tokens", "max_completion_tokens")
        stops = stop_list(body.get("stop"))
    else:
        system = anthropic_system_text(body)
        raw_messages = body.get("messages") or []
        tools = anthropic_tools(body)
        cap = max_tokens_of(body, "max_tokens")
        stops = stop_list(body.get("stop_sequences"))
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    # Tool calls are named in the assistant turn and answered by id later;
    # Gemini answers by name, so remember which name each id carried.
    call_names = {}
    contents = []
    for message in raw_messages:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if client_wire == "openai" and role == "tool":
            part = {"functionResponse": {
                "name": call_names.get(_string(message.get("tool_call_id")), ""),
                "response": {"result": blocks_text(content) if isinstance(content, list) else _string(content)},
            }}
            if contents and contents[-1].get("role") == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue
        gemini_role = "model" if role == "assistant" else "user"
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif client_wire == "openai" and btype == "image_url" and is_record(block.get("image_url")):
                    parts.append(gemini_image_part(block["image_url"].get("url")))
                elif client_wire == "anthropic" and btype == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        media = _string(source.get("media_type")) or "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif client_wire == "anthropic" and btype == "tool_use":
                    call_names[_string(block.get("id"))] = _string(block.get("name"))
                    parts.append({"functionCall": {
                        "name": _string(block.get("name")),
                        "args": block.get("input") if is_record(block.get("input")) else {},
                    }})
                elif client_wire == "anthropic" and btype == "tool_result":
                    parts.append({"functionResponse": {
                        "name": call_names.get(_string(block.get("tool_use_id")), ""),
                        "response": {"result": blocks_text(block.get("content"))
                                     if isinstance(block.get("content"), list)
                                     else _string(block.get("content"))},
                    }})
        if client_wire == "openai" and role == "assistant":
            for call in message.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                call_names[_string(call.get("id"))] = _string(call["function"].get("name"))
                parts.append({"functionCall": {
                    "name": _string(call["function"].get("name")),
                    "args": parse_arguments(call["function"].get("arguments")),
                }})
        contents.append({"role": gemini_role, "parts": parts})
    out["contents"] = contents
    config = {}
    if cap is not None:
        config["maxOutputTokens"] = cap
    if _number_or_none(body.get("temperature")) is not None:
        config["temperature"] = body["temperature"]
    if _number_or_none(body.get("top_p")) is not None:
        config["topP"] = body["top_p"]
    if stops:
        config["stopSequences"] = stops
    if client_wire == "openai":
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    if tools:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in tools]}]
    return out


def build_upstream(client_wire, body, provider, model_id):
    """(url, headers, payload) for one provider call, in the provider's wire."""
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if wire == "openai":
        payload = dict(body)
        payload["model"] = model_id
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        if client_wire == "anthropic":
            payload = anthropic_to_openai(body, model_id)
        return base + "/v1/chat/completions", headers, payload
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_wire == "anthropic":
            payload = dict(body)
            payload["model"] = model_id
            payload.pop("stream", None)
        else:
            payload = openai_to_anthropic(body, model_id)
        return base + "/v1/messages", headers, payload
    # gemini
    payload = to_gemini(client_wire, body)
    url = "%s/v1beta/models/%s:generateContent" % (base, quote(model_id, safe=""))
    return url, headers, payload


def parse_answer(wire, raw):
    """A provider's buffered answer as (text, input tokens, output tokens), or None."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        message = choices[0].get("message")
        if not is_record(message):
            return None
        content = message.get("content")
        text = content if isinstance(content, str) else blocks_text(content)
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return text, _number_or_none(usage.get("prompt_tokens")) or 0, \
            _number_or_none(usage.get("completion_tokens")) or 0
    if wire == "anthropic":
        # The client's own checker concatenates text blocks with no
        # separator; read the provider the same way it will be read back.
        content = body.get("content")
        text = ""
        if isinstance(content, list):
            text = "".join(
                block["text"] for block in content
                if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str))
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return text, _number_or_none(usage.get("input_tokens")) or 0, \
            _number_or_none(usage.get("output_tokens")) or 0
    # gemini
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content")
    if not is_record(content):
        return None
    parts = content.get("parts")
    text = ""
    if isinstance(parts, list):
        text = "".join(part["text"] for part in parts
                       if is_record(part) and isinstance(part.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return text, _number_or_none(usage.get("promptTokenCount")) or 0, \
        _number_or_none(usage.get("candidatesTokenCount")) or 0


# --------------------------------------------------------------------------
# Routing: the best free model that meets the request, with failover.
# --------------------------------------------------------------------------

BRIEF = read_brief(PROMPT_FILE)
PROBES = []
CATALOG = {"providers": []}
REFUSALS = {}      # provider id -> refusals in a row (reset by a success)
SERVED = {}        # provider id -> requests it has answered
MANIFEST_INDEX = [0]
SERVE_LOCK = threading.Lock()


def load():
    global PROBES, CATALOG
    if BRIEF.error is not None:
        log(BRIEF.error)
    PROBES = probe_all(ENV_URL, BRIEF.providers)
    for probe in PROBES:
        if not probe.ok and probe.error is not None:
            log(probe.error)
        REFUSALS.setdefault(probe.provider["id"], 0)
        SERVED.setdefault(probe.provider["id"], 0)
    CATALOG = build_catalog(BRIEF, PROBES)


def served_models():
    """The free models this server answers with: unique ids, in catalog order."""
    out = []
    seen = set()
    for probe in PROBES:
        if not probe.ok:
            continue
        for model in sorted(probe.models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            if not model_free(BRIEF, model, probe.provider["id"]):
                continue
            seen.add(model["id"])
            out.append({"id": model["id"], "provider": probe.provider["id"],
                        "display_name": model["display_name"]})
    return out


def capabilities(provider_id, probe_model):
    row = rating_row(BRIEF, provider_id, probe_model["id"])
    if row is not None:
        return {
            "context_length": row["context_length"] if row["context_length"] is not None
            else probe_model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": probe_model["context_length"],
        "tools": probe_model["tools"],
        "vision": probe_model["vision"],
        "json": probe_model["json"],
    }


def meets(requires, caps):
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
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not is_record(block):
                    continue
                if block.get("type") in ("image", "image_url"):
                    vision = True
    is_json = False
    if client_wire == "openai":
        fmt = body.get("response_format")
        is_json = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    tools = anthropic_tools(body) if client_wire == "anthropic" else openai_tools(body)
    return {"tools": len(tools) > 0, "vision": vision, "json": is_json, "min_context": None}


def next_requirements(client_wire, body):
    """The arriving request's requirements: the manifest's row by arrival
    order (the client sends the manifest in order, one at a time), else the
    body's own."""
    index = MANIFEST_INDEX[0]
    MANIFEST_INDEX[0] = index + 1
    if index < len(BRIEF.manifest):
        return BRIEF.manifest[index]["requires"]
    return requirements_from_body(client_wire, body)


def client_json_mode(client_wire, body):
    if client_wire != "openai":
        return False
    fmt = body.get("response_format")
    return is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")


def candidates(requires, client_wire, body):
    """One candidate per provider -- its best free model meeting the request --
    best rating first. A refusal is a property of the provider (its budget,
    its failure schedule, its outage), never of the model, so a second model
    on a refusing provider is never worth a call in the same request."""
    json_mode = client_json_mode(client_wire, body)
    found = []
    for index, probe in enumerate(PROBES):
        if not probe.ok:
            continue
        provider = probe.provider
        # An Anthropic-wire provider parses JSON mode as absent, so an
        # OpenAI JSON-mode request it answered could never echo json_mode
        # back; it is not a candidate for such a request at all.
        if json_mode and provider["wire"] == "anthropic":
            continue
        best = None
        for model in probe.models:
            if not model_free(BRIEF, model, provider["id"]):
                continue
            if not meets(requires, capabilities(provider["id"], model)):
                continue
            row = rating_row(BRIEF, provider["id"], model["id"])
            rating = row["rating_milli"] if row is not None else -1
            key = (-rating, model["id"])
            if best is None or key < best[0]:
                best = (key, model)
        if best is not None:
            rating = -best[0][0]
            found.append((-rating, index, provider, best[1]))
    # Rating decides first. On a tie -- the same model often sits on two
    # providers at one rating (glm-4.7-flash on zai and cloudflare,
    # qwen3.8-27b on groq and cerebras) -- the provider that has answered
    # fewer requests so far goes first, so one twin's small budget is not
    # drained while the other idles; the directory order breaks what is left.
    found.sort(key=lambda item: (item[0], SERVED.get(item[2]["id"], 0), item[1]))
    return [(provider, model) for _neg, _idx, provider, model in found]


def call_provider(provider, model, client_wire, body):
    """One upstream call, buffered: (text, in, out) on a 2xx answer, else None."""
    url, headers, payload = build_upstream(client_wire, body, provider, model["id"])
    try:
        status, raw = http_request("POST", url, headers=headers, payload=payload,
                                   timeout=CALL_TIMEOUT_S)
    except (OSError, ValueError) as err:
        log("call to %s failed: %s" % (provider["id"], err))
        return None
    if status < 200 or status >= 300:
        return None
    return parse_answer(provider["wire"], raw)


def serve(client_wire, body):
    """Answer one client request: (model id, text, usage in, usage out), or
    None when no candidate answers."""
    requires = next_requirements(client_wire, body)
    options = candidates(requires, client_wire, body)
    # First pass skips a provider that has refused twice in a row while any
    # alternative remains -- it is most likely out of budget or in its outage,
    # and every wasted call counts against the trial's tie-break. The second
    # pass takes whatever is left, in the same order.
    for second_pass in (False, True):
        for provider, model in options:
            pid = provider["id"]
            if not second_pass and len(options) > 1 and REFUSALS.get(pid, 0) >= 2:
                continue
            answer = call_provider(provider, model, client_wire, body)
            if answer is not None:
                REFUSALS[pid] = 0
                SERVED[pid] = SERVED.get(pid, 0) + 1
                text, usage_in, usage_out = answer
                return model["id"], text, usage_in, usage_out
            REFUSALS[pid] = REFUSALS.get(pid, 0) + 1
    return None


# --------------------------------------------------------------------------
# Rendering the provider's answer in the client's wire.
# --------------------------------------------------------------------------

def text_pieces(text):
    pieces = [text[i:i + STREAM_PIECE_CHARS] for i in range(0, len(text), STREAM_PIECE_CHARS)]
    return pieces if pieces else [""]


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                  "owned_by": m["provider"]} for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                  "created_at": FIXED_CREATED_AT} for m in models],
        "has_more": False,
    }


def openai_error(message, error_type):
    return {"error": {"message": message, "type": error_type, "code": None}}


def anthropic_error(message, error_type):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def render_openai(model_id, text, usage_in, usage_out, stream):
    usage = {"prompt_tokens": usage_in, "completion_tokens": usage_out,
             "total_tokens": usage_in + usage_out}
    if not stream:
        return "application/json", json_bytes({
            "id": "chatcmpl-agg",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model_id,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage,
        })
    frames = []

    def chunk(delta, finish, with_usage=False):
        frame = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                 "created": FIXED_CREATED, "model": model_id,
                 "choices": [] if with_usage else
                 [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if with_usage:
            frame["usage"] = usage
        return "data: %s\n\n" % json.dumps(frame, ensure_ascii=False)

    frames.append(chunk({"role": "assistant", "content": ""}, None))
    for piece in text_pieces(text):
        frames.append(chunk({"content": piece}, None))
    frames.append(chunk({}, "stop"))
    frames.append(chunk({}, None, with_usage=True))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_anthropic(model_id, text, usage_in, usage_out, stream):
    if not stream:
        return "application/json", json_bytes({
            "id": "msg_agg",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
        })
    frames = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, ensure_ascii=False))

    frames.append(event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0}},
    }))
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in text_pieces(text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")


# --------------------------------------------------------------------------
# The HTTP server the client talks to.
# --------------------------------------------------------------------------

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
        self._send(status, "application/json", json_bytes(value))

    def _send_error(self, wire, status, error_type, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error(message, error_type))
        else:
            self._send_json(status, openai_error(message, error_type))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            models = served_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            self._send_error("anthropic" if path == "/v1/messages" else "openai",
                             404, "not_found", "no route POST %s" % path)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._send_error(wire, 413, "invalid_request_error", "the request body is too large")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = None
        if not is_record(body):
            self._send_error(wire, 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        with SERVE_LOCK:
            answered = serve(wire, body)
        if answered is None:
            self._send_error(wire, 503, "overloaded_error" if wire == "anthropic" else "no_free_model",
                             "no free model on the directory could answer this request")
            return
        model_id, text, usage_in, usage_out = answered
        stream = body.get("stream") is True
        if wire == "openai":
            content_type, payload = render_openai(model_id, text, usage_in, usage_out, stream)
        else:
            content_type, payload = render_anthropic(model_id, text, usage_in, usage_out, stream)
        self._send(200, content_type, payload)


# --------------------------------------------------------------------------
# Register, poll, and the main entry.
# --------------------------------------------------------------------------

def register(port):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG}
    last_error = None
    for _attempt in range(30):
        try:
            status, raw = http_request("POST", ENV_URL + "/register", payload=payload,
                                       timeout=REGISTER_TIMEOUT_S)
            if status == 200:
                try:
                    reply = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    reply = {}
                if is_record(reply) and isinstance(reply.get("requests"), int):
                    log("registered: the workload holds %d requests" % reply["requests"])
                return
            last_error = "HTTP %d" % status
        except (OSError, ValueError) as err:
            last_error = str(err)
        time.sleep(0.2)
    raise RuntimeError("POST %s/register never answered 200 (%s)" % (ENV_URL, last_error))


def poll_until_done():
    while True:
        try:
            status, raw = http_request("GET", ENV_URL + "/work", timeout=WORK_TIMEOUT_S)
            if status == 200:
                work = json.loads(raw.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(POLL_INTERVAL_S)


def main(argv):
    load()
    if "--list" in argv:
        # The same document the register body carries: print the one built
        # object and stop before any server starts.
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = None
    port = None
    for _attempt in range(5):
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
        if port != 8080:
            break
        server.server_close()
        server = None
    if server is None:
        raise RuntimeError("the OS kept picking 8080, which the env owns on one layout")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(CATALOG["providers"]), len(served_models())))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port)
    poll_until_done()
    server.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as err:  # noqa: BLE001 - a failed start must exit nonzero
        log("fatal: %r" % (err,))
        sys.exit(1)
