# Local and self-hosted models

How to run WICA on a model you host, on your own machine or on your own server. WICA integrates no inference engine: the model runs in a separate server process that speaks OpenAI's Chat Completions API, and WICA is its client through the `openai-chat-completions-compatible` provider. Every field used here is described in [../reference/configuration.md](../reference/configuration.md#local-and-self-hosted-models).

On this page: [Why a server](#why-a-server) · [The solutions](#the-solutions) · [What every server has in common](#what-every-server-has-in-common) · [llama-server](#llama-server-llamacpp) · [mlx_lm.server](#mlx_lmserver-mlx-lm) · [Ollama](#ollama) · [Other compatible servers](#other-compatible-servers) · [Which model to use](#which-model-to-use) · [Cancellation](#cancellation) · [On a cloud server](#on-a-cloud-server).

## Why a server

An engine running inside the WICA process (llama.cpp's bindings, mlx-lm, `transformers`) generates in a blocking call: cancelling a reaction would free the Agent while the generation kept the GPU. With the engine in its own process, cancelling the call closes the connection and the server stops. Chat templates, tool-call parsing, token usage and prompt caching are the server's job, and the engine becomes a deployment choice: the same config covers a laptop, a GPU box and a managed endpoint.

## The solutions

The provider is a protocol: it works with any server that implements Chat Completions, on the same machine or another one. The servers below are the ones WICA was run against, not the only ones that can work.

| Server | Engine | Runs on | Models | Status |
|---|---|---|---|---|
| [`llama-server`](#llama-server-llamacpp) | llama.cpp | macOS, Linux, Windows; Metal, CUDA, Vulkan or CPU | Any GGUF model | Run. The portable default |
| [`mlx_lm.server`](#mlx_lmserver-mlx-lm) | mlx-lm | Apple silicon | MLX models from the Hugging Face Hub | Run |
| [Ollama](#ollama) | llama.cpp | macOS, Linux, Windows | Ollama's library | Run, with three settings to know |
| [LM Studio, vLLM](#other-compatible-servers) | llama.cpp or MLX; vLLM's own | Depends on the server | Depends on the server | Not run |

"Run" means WICA's integration was checked through the server on 2026-11-17, on an Apple M5 Pro, with Qwen2.5-7B-Instruct at 4 bits on all three: speech streamed to the output Command, a Command called beside speech, token usage and prompt-prefix reuse reported, an output cap applied (on Ollama through `extra_body`, see below), and a deadline or `close()` stopping the generation at once. It says the integration works, not how well a model drives your agent: see [Which model to use](#which-model-to-use).

## What every server has in common

Install the extra:

```
uv add "wica[openai-compatible] @ git+https://github.com/funwithagents/wica-framework"
```

Then point the config at the server:

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

- **`base_url`** is the server's address with `/v1`. **`model`** is the name the server expects; each section below gives it.
- **No key.** Leave `api_key` and `api_key_env` out; WICA sends a placeholder and never your `OPENAI_API_KEY`. A server behind an authenticating gateway takes `api_key_env` as any provider does. The config carries no secret, so it can be committed.
- **`context_window`** is the server's own context size. A local model has no profile WICA can read it from, and `history_budget` needs it to cut history in time.
- **`model_timeout` bounds the whole reply, not only the wait.** The deadline covers the model call from the request to its last token, and the reply is streamed to the output Command as it is written: a reply longer than the deadline stops mid-sentence (what was already streamed is heard). At about 50 tokens a second, the default 10 s is about 500 tokens. Raise it to cover your longest reply.
- **The first request can be slow.** A server that loads the model on its first request takes seconds, longer with a download; send it one request before `start()`, or raise `model_timeout`.
- **`model_kwargs.max_tokens` caps the output** of a model that writes without end. OpenAI's client sends it as `max_completion_tokens`, which `llama-server` and `mlx_lm.server` read and Ollama ignores ([Ollama](#ollama) gives the form that works there). With none set, the server's own limit applies.
- **Speaking needs no tool calling.** The model speaks by writing the output Command's tag in its text ([commands.md](commands.md#make-your-agent-speak)), and WICA sends no `tool_choice`, so every server speaks alike. `set_output_command(say, as_tool=True)` is the exception: it makes speaking a tool call and depends on `tool_choice` to force it, which `mlx_lm.server` and Ollama ignore, so keep the tag on them ([tag-or-tool-call.md](tag-or-tool-call.md)). The other Commands are tool calls: they need the server's tool calling, which each section below turns on.
- **Token usage and prefix reuse** are reported by all three servers, so `history_budget` works and `cache_read_tokens` in the reaction record shows the reused prefix ([latency-and-traces.md](latency-and-traces.md)).

## `llama-server` (llama.cpp)

- **Runs on:** macOS, Linux and Windows; Metal on Apple silicon, CUDA or Vulkan where available, the CPU otherwise.
- **Models:** any GGUF file; the server holds one, chosen when it starts.
- **Best use:** the default on any platform, and the only one of the three that refuses a prompt longer than its context instead of cutting it.

```
brew install llama.cpp        # or a prebuilt release from github.com/ggml-org/llama.cpp
curl -L -O https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --jinja -np 1 -c 8192 --port 8080
```

```json
{
  "agent": {
    "provider": "openai-chat-completions-compatible",
    "model": "local",
    "base_url": "http://127.0.0.1:8080/v1",
    "context_window": 8192
  }
}
```

- `--jinja` turns on tool calling through the model's chat template: without it the model can speak but never calls a Command.
- `-c` is the context size: give `context_window` the same value. `-np` is the number of requests served at once; one WICA needs one.
- `model` can be any name: the server serves the one model it loaded.
- Measured: the first words reach the output Command 0.26 s after the call starts; loading the 7B model takes 1.3 to 3.5 s. A prompt longer than `-c` is refused, and the reaction ends `model_error`.

## `mlx_lm.server` (mlx-lm)

- **Runs on:** Apple silicon only.
- **Models:** MLX models from the Hugging Face Hub, chosen per request by `model` and downloaded and loaded on first use.
- **Best use:** speed on a Mac with MLX models.

```
uv tool install mlx-lm        # or: pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8081
```

```json
{
  "agent": {
    "provider": "openai-chat-completions-compatible",
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "base_url": "http://127.0.0.1:8081/v1",
    "context_window": 8192
  }
}
```

- `model` is the Hub repo id; the server loads the model a request names.
- Tool calling needs nothing to turn on.
- Measured: the first words reach the output Command 0.47 s after the call starts. The very first request took 48 s with the download: warm the server before `start()`.

## Ollama

- **Runs on:** macOS, Linux and Windows.
- **Models:** Ollama's library, pulled by tag.
- **Best use:** the simplest install, with models managed for you. Two of its defaults need setting for an agent, and the output cap takes another form.

```
brew install ollama           # or the installer from ollama.com
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=-1 ollama serve
ollama pull qwen2.5:7b
```

```json
{
  "agent": {
    "provider": "openai-chat-completions-compatible",
    "model": "qwen2.5:7b",
    "base_url": "http://127.0.0.1:11434/v1",
    "context_window": 8192,
    "model_kwargs": { "extra_body": { "max_tokens": 512 } }
  }
}
```

- **The output cap goes in `extra_body`.** Ollama reads `max_tokens` only, and OpenAI's client renames `model_kwargs.max_tokens` to `max_completion_tokens`, so that cap is dropped and the model writes on (649 tokens under a cap of 12). Under `extra_body` the field is sent as written.
- **`OLLAMA_CONTEXT_LENGTH`**: Ollama's context is 4096 tokens by default, whatever the model supports, and **a longer prompt is cut from the start without an error**: the system prompt goes first, and the persona is lost silently. Set it, and give `context_window` the same value.
- **`OLLAMA_KEEP_ALIVE`**: Ollama unloads a model after five idle minutes, so a reaction after a quiet spell pays the load again (5.7 s for the 7B). `-1` keeps it loaded.
- `model` is the Ollama tag. Ollama's own API (`provider: "ollama"` through `langchain-ollama`) is not needed: its `/v1` address is enough.
- Measured: the first words reach the output Command 0.36 s after the call starts.

## Other compatible servers

Not run with WICA. Each documents the Chat Completions API, so the provider should reach it with its `base_url` and `model`; check tool calling and cancellation on it before relying on it.

- **LM Studio** serves GGUF and MLX models on `http://127.0.0.1:1234/v1`.
- **vLLM** serves models on an NVIDIA GPU and batches concurrent requests, which suits one server shared by many WICA instances. Its tool calling is turned on with `--enable-auto-tool-choice` and a `--tool-call-parser` matching the model.

A report of how one of them behaves is welcome at the issue tracker.

## Which model to use

The server decides whether WICA works; the model decides how well your agent behaves. A model too small for the job still talks, since speaking is text with a tag, but it misses Commands, speaks when it should stay silent, or keeps going when it has nothing to do.

Measure the models you consider with a benchmark over your own scenarios ([benchmarking.md](benchmarking.md)). The repo's own benchmark has opt-in rows for Qwen2.5-7B-Instruct on each of the three servers (`openai-chat-completions-compatible/qwen2.5-7b-instruct@llama-server`, `@mlx`, `@ollama`): start the server as above, then run one from a checkout of the repo ([benchmarks/README.md](../../benchmarks/README.md)):

```
uv run --group demo python -m benchmarks.bench --only 'openai-chat-completions-compatible/*@llama-server' --include-disabled --k 5
```

Add a row for another model the same way.

## Cancellation

When a reaction is abandoned (a `model_timeout`, `stop()`, `close()`), WICA closes the connection. The model call is streamed, so the server's next write fails and it stops generating. Measured on all three servers above: `close()` returned within 5 ms, and the next request was served as fast as on an idle server.

A proxy or load balancer between WICA and the server may not pass the disconnect through promptly; check on the real path, not only on localhost.

## On a cloud server

The server is a network address, so the same config covers three shapes:

1. **Sidecar:** WICA and the model server on one GPU machine or in one pod. Lowest latency.
2. **A shared GPU service:** one server for many WICA instances. Each WICA has one model call in flight, so N instances need N slots: `-np` on `llama-server`; vLLM batches concurrent requests by design.
3. **A managed endpoint** that speaks Chat Completions: the same provider, with its key in `api_key_env`.
