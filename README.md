# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/freezzard/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/freezzard/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A single local server that gathers the free tiers of ten model providers
behind the two client APIs everyone already speaks: OpenAI Chat Completions
and Anthropic Messages, in JSON and in server-sent events.

## What it does

- Reads its brief: the provider directory, the sealed quality table and
  the workload manifest (three JSON documents told apart by their
  `schema`).
- Probes every provider's catalog route **in the provider's own wire**
  (OpenAI `GET /v1/models`, Anthropic `GET /v1/models`, Gemini
  `GET /v1beta/models`) and folds the listings into one catalog, with the
  quality table's free flag and context length wherever it has a row.
- Registers the catalog with the trial env, then serves:
  - `POST /v1/chat/completions` — OpenAI wire, JSON or SSE
  - `POST /v1/messages` — Anthropic wire, JSON or SSE
  - `GET /v1/models` — the OpenAI list shape for an `Authorization: Bearer`
    request, the Anthropic list shape for an `x-api-key` request, naming
    exactly the free model ids it registered
- Routes each request to the **highest-rated free model that meets the
  request's requirements** (tools, vision, JSON mode, minimum context),
  translating the request into the provider's wire and the answer back
  into the client's wire — system prompt, images, tools, stop sequences,
  `max_tokens`, JSON mode, usage counts and stop reasons all carried over.
- **Fails over** when a provider refuses (402, 429 or 5xx): the request
  moves to the next candidate, and a provider that refuses twice in a row
  is skipped while any other candidate remains. A paid model is never a
  legal pick.
- Polls the env's `/work` and exits 0 when the workload is done. Every
  output is a pure function of the brief, the requests and the providers'
  answers — fixed timestamps, no clock, no randomness — so a replayed
  trial reproduces the same bytes.

## Run it

Inside a make.fun trial the env provides `MAKEFUN_ENV_URL`,
`MAKEFUN_PROMPT_FILE` and the model gateway; the entry needs nothing else
(Python 3 standard library only).

```sh
python3 main.py            # serve one trial, then exit 0
python3 main.py --list     # print the catalog document and stop
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=2k68scpr&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
