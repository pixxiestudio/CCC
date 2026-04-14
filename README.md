# CCC — Compact Claude Continuity

A minimal Anthropic API proxy (~740 lines) that automatically switches between provider credentials when one fails. Built as a lightweight replacement for LiteLLM proxy — same `config.yaml` format, none of the bloat.

## How it works

```
Claude Code client
  │  POST /v1/messages
  ▼
CCC Proxy (port 4000)
  ├── Sub A OAuth token   ← try first
  ├── Sub B OAuth token   ← failover
  ├── Paid API key        ← emergency fallback
  └── Local Ollama        ← last resort (free, always on)
```

Each "model group" (e.g. `it-opus`) can have multiple deployments ordered by priority. CCC tries them in order, applying per-error retry/cooldown policies, then falls back to the next configured group.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/pixxiestudio/CCC
cd CCC
pip install -e .

# 2. Configure (copy samples and fill in your keys)
cp config.sample.yaml config.yaml
cp .env.sample .env

# 3. Run
uvicorn ccc.server:app --host 0.0.0.0 --port 4000
```

Point Claude Code at the proxy:

```bash
# In ~/.claude settings or per-project
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=<your LITELLM_MASTER_KEY from .env>
```

---

## File structure

```
CCC/
├── config.yaml                  # model list + router + retry settings
├── .env                         # secrets (API keys, Redis creds)
├── config.sample.yaml           # template — copy to config.yaml
├── .env.sample                  # template — copy to .env
├── ccc/
│   ├── config_loader.py         # parse config.yaml, resolve os.environ/VAR
│   ├── exceptions.py            # typed error hierarchy
│   ├── models.py                # Anthropic request/response models
│   ├── providers/
│   │   ├── anthropic_provider.py  # httpx passthrough to api.anthropic.com
│   │   └── ollama_provider.py     # Anthropic↔OpenAI translation for Ollama
│   ├── health.py                # failure counts + cooldown (Redis or in-memory)
│   ├── router.py                # retry-then-fallback core logic
│   └── server.py                # FastAPI app
└── tests/
    ├── test_config_loader.py
    └── test_router.py
```

---

## Configuration

### Model groups

Entries with the same `model_name` form a group. `order` controls priority — lowest tried first.

```yaml
model_list:
  - model_name: it-opus          # group name clients use as "model"
    litellm_params:
      model: anthropic/claude-opus-4-6
      api_key: os.environ/SUB_A_OAUTH_TOKEN
    model_info:
      order: 1                   # primary

  - model_name: it-opus
    litellm_params:
      model: anthropic/claude-opus-4-6
      api_key: os.environ/SUB_B_OAUTH_TOKEN
    model_info:
      order: 2                   # failover
```

### Fallback chains

After all deployments in a group fail, CCC tries the listed fallback groups in order:

```yaml
router_settings:
  fallbacks:
    - it-opus: ["api-fallback", "local-fallback"]
  context_window_fallbacks:       # only triggered on context-length errors
    - it-opus: ["api-fallback"]
```

### Per-error retry policy

```yaml
litellm_settings:
  retry_policy:
    RateLimitErrorRetries: 0      # don't retry — switch immediately
    AuthenticationErrorRetries: 0
    TimeoutErrorRetries: 2
    InternalServerErrorRetries: 3

  allowed_fails_policy:
    RateLimitErrorAllowedFails: 1  # 1 rate-limit hit → cooldown
    TimeoutErrorAllowedFails: 3
    InternalServerErrorAllowedFails: 5
```

---

## Supported providers

| Provider | Config prefix | Notes |
|----------|--------------|-------|
| Anthropic API | `anthropic/` | Direct API key or OAuth token |
| Ollama (local) | `ollama/` | Auto-translates Anthropic↔OpenAI format |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `CCC_CONFIG` | Path to config file (default: `config.yaml`) |
| `CCC_ENV` | Path to .env file (default: `.env`) |
| `CCC_HOST` | Bind address (default: `0.0.0.0`) |
| `CCC_PORT` | Port (default: `4000`) |
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING` (default: `INFO`) |

---

## Auth

Set `master_key` in `general_settings` (or `LITELLM_MASTER_KEY` env var). Clients must send it as:

```
x-api-key: <master_key>
# or
Authorization: Bearer <master_key>
```

Set `disable_key_check: true` to skip auth (development only).

---

## Redis (optional)

Redis enables shared cooldown state across multiple proxy instances:

```yaml
router_settings:
  redis_host: os.environ/REDIS_HOST
  redis_port: 6379
  redis_password: os.environ/REDIS_PASSWORD
```

Without Redis, health state is in-memory per process.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Verification checklist

1. `curl http://localhost:4000/health` → `{"status":"ok"}`
2. Normal request → served by deployment 1 (Sub A)
3. Break Sub A key → auto-switches to Sub B
4. Break both Sub keys → switches to `api-fallback`
5. Disable Anthropic entirely → switches to `local-fallback` (Ollama)
