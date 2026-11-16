# Local and self-hosted models

How to run WICA on a model you host, on your own machine or on your own server. WICA integrates no inference engine: the model runs in a separate server process that speaks OpenAI's Chat Completions API, and WICA is its client through the `openai-chat-completions-compatible` provider. Every field used here is described in [../reference/configuration.md](../reference/configuration.md#local-and-self-hosted-models).

On this page: [Why a server](#why-a-server) · [Start a server](#start-a-server) · [Point WICA at it](#point-wica-at-it) · [Which model size works](#which-model-size-works) · [What to tune](#what-to-tune) · [Cancellation](#cancellation) · [On a cloud server](#on-a-cloud-server).

## Why a server

An engine running inside the WICA process (llama.cpp's bindings, mlx-lm, `transformers`) generates in a blocking call: cancelling a reaction would free the Agent while the generation kept the GPU. With the engine in its own process, cancelling the call closes the connection and the server stops. Chat templates, tool-call parsing, forced tool choice, token usage and prompt caching are the server's job, and the engine becomes a deployment choice: the same config covers a laptop, a GPU box and a managed endpoint.

| Engine | Server | Runs on | Best for |
|---|---|---|---|
| llama.cpp | `llama-server` (also inside Ollama and LM Studio) | macOS, Linux, Windows; CPU, Metal, CUDA, Vulkan | The portable default |
| mlx-lm | `mlx_lm.server` | Apple silicon only | Speed on a Mac |
| `transformers` and others | vLLM, TGI | In practice CUDA | GPU servers, many concurrent clients |

## Start a server

Two servers were tried with WICA, with Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct on an Apple M5 Pro:

- **`llama-server`, recommended.** Forcing a tool call and cancellation both work. Start it with `--jinja`, which enables tool calling through the model's chat template:

  ```
  llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --jinja -np 1 -c 8192 --port 8080
  ```

  `-c` is the context size, `-np` the number of requests served at once.

- **`mlx_lm.server`, works with one limit.** It ignores `tool_choice`, so forcing a tool call has no effect. It downloads and loads the model on the first request that names it:

  ```
  mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
  ```

Ollama (on its `/v1` address), LM Studio and vLLM implement the same API but were not tried: check forcing and cancellation on them before relying on either.

## Point WICA at it

Install the extra, then name the server's address:

```
uv add "wica[openai-compatible] @ git+https://github.com/funwithagents/wica-framework"
```

```json
{
  "agent": {
    "provider": "openai-chat-completions-compatible",
    "model": "local",
    "base_url": "http://127.0.0.1:8080/v1",
    "system_prompt_file": "prompts/robot.md",
    "context_window": 8192,
    "model_timeout": 30,
    "model_kwargs": { "max_tokens": 512 }
  }
}
```

- `model` is sent as the request's model name: `llama-server` serves its one model under any name; `mlx_lm.server` loads the repo id you give (`mlx-community/Qwen2.5-7B-Instruct-4bit`).
- No key is needed. A server behind an authenticating gateway takes `api_key_env` as any provider does. WICA never sends your `OPENAI_API_KEY` to this provider.
- The config carries no secret, so it can be committed.

## Which model size works

The model matters more than the server. In a session with a speech Input, a `say` output Command and a `set_light` Command:

| Model | Result |
|---|---|
| Qwen2.5-7B-Instruct | Every reaction right: `say` to greet, `set_light("red")` on request, `noop` after each. About 0.5 s per reaction (1.1 s for the first), on both servers |
| Qwen2.5-1.5B-Instruct, no output Command | Sensible one-line replies in 0.07 to 0.25 s, but it said it set the light without calling `set_light` |
| Qwen2.5-1.5B-Instruct, output Command | Forced: every call ran to the deadline, writing text without ever calling a tool. Unforced: plain text every time, which nobody hears |

Expect a 7B-class model to drive Commands, and a model under about 3B not to. The floor between them was not measured.

## What to tune

| Field | Why |
|---|---|
| `context_window` | A local model has no LangChain profile, so `history_budget` needs it. Match the server's context size (`-c`) |
| `model_timeout` | The default 10 s fails a cold start (48 s on `mlx_lm.server` with a download) or a long prompt on a small machine. Raise it, or warm the server with one request before `start()` |
| `model_kwargs.max_tokens` | An output cap. A model too small for forced tool calls writes until the deadline on every reaction; a cap bounds the damage |
| `force_tool_call` | Read only with an output Command. `llama-server` honours `tool_choice: "required"`, `mlx_lm.server` ignores it; set `false` for a server that rejects it (every call then fails `model_error`) |

Token usage and prompt-prefix reuse are reported by both servers tried: `history_budget` works, and `cache_read_tokens` in the reaction record shows the reused prefix ([latency-and-traces.md](latency-and-traces.md)).

## Cancellation

When a reaction is abandoned (a `model_timeout`, `stop()`, `close()`), WICA closes the connection. The model call is streamed, so the server's next write fails and it stops generating within about a token: measured on `llama-server` and `mlx_lm.server`, and the next call is served at once.

A proxy or load balancer between WICA and the server may not pass the disconnect through promptly; check on the real path, not only on localhost.

## On a cloud server

The server is a network address, so the same config covers three shapes:

1. **Sidecar:** WICA and the model server on one GPU machine or in one pod. Lowest latency.
2. **A shared GPU service:** one server for many WICA instances. Each WICA has one model call in flight, so N instances need N slots: `-np` on `llama-server`; vLLM batches concurrent requests by design.
3. **A managed endpoint** that speaks Chat Completions: the same provider, with its key in `api_key_env`.
