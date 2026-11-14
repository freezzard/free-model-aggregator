# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/freezzard/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/freezzard/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that aggregates the free tiers of ten model providers behind the
two client APIs everyone already speaks: OpenAI Chat Completions and
Anthropic Messages, in JSON and in server-sent events.

## What it does

- Reads its brief: the provider directory, the public quality table and the
  workload manifest (three JSON documents identified by their `schema`).
- Probes every provider's catalog route **in the provider's own wire**
  (OpenAI `GET /v1/models`, Anthropic `GET /v1/models`, Gemini
  `GET /v1beta/models`) and builds one catalog from the listings, with the
  quality table's free flag and context length where it has a row.
- Registers the catalog with the trial env, then serves:
  - `POST /v1/chat/completions` — OpenAI wire, JSON or SSE
  - `POST /v1/messages` — Anthropic wire, JSON or SSE
  - `GET /v1/models` — OpenAI list shape for an `Authorization: Bearer`
    request, Anthropic list shape for an `x-api-key` request, listing exactly
    the free model ids it registered
- Routes each request to the **highest-rated free model that meets the
  request's requirements** (tools, vision, JSON mode, minimum context),
  translating the request into the provider's wire and the answer back into
  the client's wire — system prompt, images, tools, stop sequences,
  `max_tokens`, JSON mode, usage counts and stop reasons all preserved.
- **Fails over** when a provider refuses (402, 429 or 5xx): the request moves
  to the next candidate; a provider that refuses twice in a row is skipped
  while any other candidate remains. A paid model is never a legal pick.
- Polls the env's `/work` and exits 0 when the workload is done. Every output
  is a pure function of the brief, the requests and the providers' answers —
  fixed timestamps, no clock, no randomness — so a replayed trial reproduces
  the same bytes.

## Run it

Inside a make.fun trial the env provides `MAKEFUN_ENV_URL`,
`MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`:

```sh
python3 main.py          # serve, register, answer, exit 0 when done
python3 main.py --list   # print the catalog document and exit 0
```

Python 3.13 standard library only; nothing to install.

## Layout

- `main.py` — the whole entry: brief parsing, catalog probe, routing,
  the three provider wires, both client wires, the register and the poll.
