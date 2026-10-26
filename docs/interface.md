# GPU-Orc — Interface

[Deutsch](interface.de.md)

As of version 0.6.0. The suites' common release API, through which GPU-Orc talks to the STT-Suite and
the TTS-Suite, is described in [release-api.md](release-api.md).

GPU-Orc is a small service per machine. It frees GPUs on request, tells who holds what right now and
decides on which GPU a side model (an engine or worker of a suite) starts. It loads nothing itself and
replaces no model server. It only talks to the interfaces of the services that hold GPUs and never ends a
process from outside. The hardest step is a service's own `force`, and only where the config allows it.

## Principles

- **Address:** `http://<machine>:5070` (port in the config). User unit `gpu-orc`, system unit
  `gpu-orc@<user>`.
- **Authentication:** Over loopback without a token. From other machines only with
  `Authorization: Bearer <token>` (missing: `401`, wrong: `403`). The token is in a file on the
  machine, named in the config. If GPU-Orc listens on more than loopback and there is no token, it
  does not start.
- **GPUs:** Given as an index (nvidia-smi order, i.e. PCI bus) or as a UUID, also as a unique prefix
  of a UUID. Answers always carry index and UUID; internally GPU-Orc works with the UUID.
- **Order:** GPU-Orc handles `/free` requests one after another. Within one request it asks all
  affected consumers *at the same time*; they are independent of each other. `/status`, `/health` and
  `/place` never wait in that queue.
  - **Exception for the `model` form:** Before the request is queued, GPU-Orc asks the model server
    whether the model already runs (loaded or loading). If so, it answers `"already_running": true`
    at once, even while another request is running. This check changes nothing, so a caller that asks
    on every prompt never waits behind a long release.
  - Only if the model does not run does the request join the queue. There GPU-Orc checks again before
    freeing, as a request ahead of it may have loaded the model meanwhile.
- **Time limit:** Every `/free` request ends at the latest `free_timeout_s` after it arrived (config,
  default 180 s), waiting in the queue included.
  - A consumer's wait for "busy" (`busy_wait_s`) and the verification lie *within* this limit. For
    `gpu-release` the wait ends at the latest 5 s before the limit, so the forcing request still fits.
  - Callers set their HTTP timeout to `caller_timeout_s` from `GET /health` (`free_timeout_s` + 30 s),
    so they do not keep the value themselves.
- **Consumers:** They are in the config, not in the code, each with a name, a type and rules. There is
  one implementation per protocol, not per machine: `gpu-release` (STT-Suite and TTS-Suite, the common
  release API) and `llama-swap`.
- **Who holds what:**
  - Each consumer reports through its own service's interface what it holds and on which GPU: the
    suites through `GET /gpu/holders`, llama-swap through `GET /running` plus the GPU mapping from its
    config.
  - nvidia-smi only gives total and used memory per GPU; this also works under WSL2, where nvidia-smi
    names no processes.
  - Nothing is mapped through PIDs, neither in `/status` nor in `/free`.

- **Claims:** What an item may take on a GPU, whether it holds that much yet or not.
  - A model of the model server: the claim of its profile (`GET /profiles`), the whole GPU while that is
    unknown.
  - A side model: the larger of the memory its suite reports and the need it reports (`need_mib` in
    `/gpu/holders`); the memory still grows while it starts and peaks while it computes. The whole GPU while
    the need is unknown.
  - Promised room (a reservation, see `/place` and `/free`) that no holder accounts for yet: its need. A
    holder of the same item on the same GPU counts once, with the larger of the two.
  - Memory in use that no holder accounts for (another program) counts on top. What is left is a GPU's
    `available_mib`.

## Endpoints

Errors always come as `{"error": "<text>"}`. An unknown path gives `404`.

### `GET /health`

```json
{"status": "ok", "version": "0.6.0", "free_timeout_s": 180.0, "caller_timeout_s": 210.0}
```

### `GET /status` — who holds what

```json
{
  "queue": {"running": true, "waiting": 1, "running_since_s": 42.0},
  "gpus": [
    {
      "index": 0,
      "uuid": "GPU-…",
      "name": "Quadro RTX 8000",
      "compute_cap": 7.5,
      "memory_total_mib": 49152,
      "memory_used_mib": 31200,
      "holders": [
        {"consumer": "llama-swap", "item": "<model>", "memory_mib": null, "need_mib": null, "claim_mib": 29491, "busy": null, "persistent": false},
        {"consumer": "stt-suite", "item": "gpu-worker", "memory_mib": 3100, "need_mib": 3712, "claim_mib": 3712, "busy": false, "persistent": null}
      ],
      "reservations": [{"consumer": "tts-suite", "item": "<engine>", "need_mib": 6024, "left_s": 57.2}],
      "unattributed_mib": null,
      "available_mib": 9925
    }
  ],
  "errors": []
}
```

- `queue` shows whether a `/free` request is running, for how many seconds (`null` if none), and how
  many wait.
- `holders`: A holder that sees several GPUs is listed with each of them.
  - `memory_mib` is only there where the service reports its memory itself: the TTS-Suite per engine,
    the STT-Suite for the GPU worker (not for diarization). llama-swap reports only the GPUs, so `null`.
  - `need_mib` is reported only by the suites, `busy` too (a transcription, a reading, a start);
    `persistent` only by the model server (member of a group with `persistent: true`). The other fields
    are `null`.
  - `claim_mib` is the item's claim on this GPU (see Principles); `null` while a consumer is in `errors`.
- `reservations` is the room promised on the GPU and how many seconds the promise has left.
- `unattributed_mib` is the used memory minus what the holders report. It is a number only when every
  holder on the GPU reports its memory, otherwise `null`. On a GPU without holders it is all of the
  used memory. GPU-Orc never touches what nobody reports.
- `available_mib` is what another program may still take on the GPU: its memory minus the claims, the
  reservations and the memory nobody accounts for (see Principles). A vLLM model allocates part of its
  share only while it computes; nvidia-smi shows it as free in between, and whoever takes it leaves the
  model without memory at its next step (2026-10-25). `null` while a consumer is in `errors`: then nobody
  knows what is free.
- `errors` names consumers that could not say what they hold (`[{"consumer": …, "detail": …}]`);
  their holders are then missing from `holders`.
- `500` when nvidia-smi fails.

### `GET /profiles` — the model server's profiles

```json
{
  "profiles": [
    {
      "name": "<model>-tts-qwen3local", "model_server": "llama-swap", "backend": "llama.cpp",
      "cards": ["GPU-…"], "persistent": false, "context": 131072, "memory_fraction": null, "kv_blocks": null,
      "variant": {"base": "<model>", "tts": "qwen3local", "vlm": null, "speed": false, "describer": false, "describer_gpu": null},
      "claim_mib": {"GPU-…": 41200}, "claim_source": "measured", "measured_at": "2026-07-12T23:02:40",
      "measured_as": "<model>", "problem": null
    }
  ],
  "errors": []
}
```

- Every entry of llama-swap's config is a profile. `cards` are the GPUs of `CUDA_VISIBLE_DEVICES` in
  nvidia-smi's order; `backend` comes from `cmd` (`vllm`, `llama.cpp`, else `other`), and so do `context`
  (`--max-model-len` or `-c`), `kv_blocks` (`--num-gpu-blocks-override`) and `memory_fraction`.
- `variant` follows the calibration's names: `<base>[-tts-<engine>][-vlm-<key>][-speed]` for the main
  models, `<base>-visiond[-gpu<N>]` for the vision describers. It says which side models a profile leaves
  room for.
- `claim_mib` is what the profile takes per GPU:
  - `measured`: from a calibration that recorded what stayed free after loading (GPU minus free, so with
    the GPU's base use; rather too much); for a vision describer (`-visiond`) the burn-in peak of its base
    model. Either counts only when it was measured at the profile's context. A variant whose entry is
    the same as that of a measured profile (the calibration writes a variant whose room lies on a GPU
    outside the model as a copy of its base) takes that measurement; `measured_as` names the profile,
    else it is `null`;
  - `fraction`: for vLLM without a measurement, its share of the GPU;
  - `no_cards`: `{}`, the entry sees no GPU (`CUDA_VISIBLE_DEVICES=`) and runs on the CPU;
  - `unknown`: `null`, for example a variant with settings of its own and no measurement.
- `problem` says why the GPUs of an entry could not be read; `errors` names model servers whose config
  could not be read.

### `GET /needs` — what the side models need

```json
{"needs": [
  {"consumer": "tts-suite", "item": "qwen3local", "need_mib": 6024, "source": "estimate", "default": false},
  {"consumer": "stt-suite", "item": "parakeet-fp32", "need_mib": 3712, "source": "measured", "default": true},
  {"consumer": "llama-swap", "item": "Qwen3VL-4B-Instruct-Q8_0", "need_mib": 7396, "source": "measured", "default": false}
], "errors": []}
```

- The suites report their models themselves (`GET /gpu/needs`, see [release-api.md](release-api.md));
  `default` is the model a suite loads onto the GPU when a request names none.
- A model server's vision models are listed in the config under `vision_models`; GPU-Orc measures their
  need itself (until then taken over with `gpu-orc import-aifred`, else `null` with `unknown`).
- `source`: `measured` (burn-in), `configured`, `estimate`, `unknown`. A value without margin: GPU-Orc adds
  the residual per GPU itself.

### `POST /place` — where a side model may start

A suite asks before it starts an engine or a worker on a GPU:

```json
{"consumer": "tts-suite", "item": "qwen3local", "need_mib": 6024, "reason": "reading for AIfred"}
```

| Field | Required | Meaning |
|---|---|---|
| `consumer` | yes | The suite (a `gpu-release` consumer of the config) |
| `item` | yes | The item as the suite names it in `/gpu/holders` (`gpu-worker`, `<engine>`) |
| `need_mib` | yes | What it needs on the GPU, without margin (GPU-Orc adds `headroom_mib`) |
| `reason` | yes | Written to GPU-Orc's log; when side models have to give way, also to theirs |
| `card` | no | Only this GPU (index or UUID), for example for a burn-in on a given GPU |
| `min_compute_cap` | no | Only GPUs with at least this CUDA compute capability (diarization: 7.5) |

GPU-Orc refuses other fields with `400`.

- **Fit:** A GPU fits where `available_mib` minus `headroom_mib` (config, default 512) covers `need_mib`.
  Of those, a GPU without claims and without memory in use comes first (side models spread over the free
  GPUs), then the one with the most room, then the lowest index.
- **The item itself:** A holder and a promise of the same consumer and item do not count, and neither does
  the memory that holder has in use: the item is being replaced (the STT worker switching models).
- **Giving way:** Where no GPU fits, idle side models of a suite after the asking one in `priority` (config)
  may have to let go: the ones on the GPU where the fewest must go, the lowest priority first, without force
  and without waiting. One that is busy stays. Then GPU-Orc measures again until `verify_timeout_s`; the
  whole answer stays within `free_timeout_s`, as for `/free` (a caller waits `caller_timeout_s` of `/health`).
- **Promise:** The answer's GPU keeps `need_mib` for the item for `reserve_s` (config, default 60 s), so
  that a second request does not get the same room before the item shows in `/gpu/holders`; after that its
  own claim counts. A promise runs out by itself, also when the suite never starts the item.
- GPU-Orc only decides here and asks nobody but for `/gpu/holders` (and for giving way); a lock in the
  service covers only the decision itself.

Answer `200`:

```json
{"card": {"index": 2, "uuid": "GPU-…"}, "why": null, "reserved_s": 60.0, "actions": [], "duration_s": 0.4}
```
```json
{"card": null, "why": "needs 6024 MiB and 512 MiB headroom, the most room is 1650 MiB on GPU 4",
 "reserved_s": null, "actions": [], "duration_s": 0.3}
```

- `card: null` with `why`: no GPU fits; no GPU meets `card` or `min_compute_cap`; a consumer could not say
  what it holds (`cannot say what is free: …`); a side model did not let go.
- `actions` are the side models that had to give way, with the outcomes of `/free` (`released`,
  `refused_busy`, `not_freed`, `error`).
- `400`: wrong fields or an unknown `card`. `404`: `consumer` is no suite of the config. `500`: nvidia-smi
  failed.

### The page under `/ui/`

A web page like the suites': cards (`/status`), profiles (`/profiles`), needs (`/needs`). `/` leads there.
The page's files are served without a token, they hold no data; every call the page makes needs the token
like any other call from another machine (button “Token”, kept in the browser).

### `POST /free` — free GPUs

Exactly one of the two forms:

```json
{"gpus": [0, 2], "keep": ["tts-suite"], "reason": "another machine needs two GPUs", "dry_run": false}
```
```json
{"model": "<model>", "model_server": "llama-swap", "reason": "Agent-Orc: agent in <folder>", "dry_run": false}
```

| Field | Required | Meaning |
|---|---|---|
| `gpus` | one of the two | List of indices or UUIDs, or `"all"` |
| `model` | one of the two | The model about to be loaded; GPU-Orc resolves its GPUs through the model server's config |
| `model_server` | only with `model`, and only when more than one model server is configured | Name of the model-server consumer; not allowed with the `gpus` form |
| `keep` | no | What stays: a consumer name (`"tts-suite"`) or `consumer:item` (`"llama-swap:<model>"`) |
| `reason` | yes | Written to GPU-Orc's log and the log of the services asked; on a forced abort also into the message to their clients |
| `dry_run` | no | `true`: only tell what would happen, touch nothing |

GPU-Orc refuses other fields with `400`.

**`gpus` form:** Every consumer on these GPUs is to let go, the model server's main models included.
- **Careful:** llama-swap unloads a model even in the middle of an answer; whoever uses it is cut off.
  This is hard on purpose.
- `keep` protects, for example `"llama-swap:<model>"`, or `"llama-swap"` for all of its models.

**`model` form:**
- If the model already runs, nothing happens (`"already_running": true`). The answer then has no
  actions, no `memory`, `free: true` and `status: null`; the consumers are not even asked.
- Otherwise, on the model's GPUs these let go:
  - services like the STT-Suite and the TTS-Suite,
  - the model server's *persistent* models. For llama-swap these are the members of the groups with
    `persistent: true` in its own config.
- The model server swaps its other models out itself when loading; GPU-Orc does not touch them. This
  form therefore never cuts off a main model in use.
- From its turn in the queue until `reserve_s` after the answer, the model's GPUs are promised to it with
  the claim of its profile (the whole GPU where that is unknown): a side model placed meanwhile would take
  the room it is about to load into. Not with `dry_run`.
- A model that is persistent itself, or matches a pattern in `no_eviction_models` (today the `-vlm-`
  profiles, which leave room for the persistent models by calibration), unloads no persistent models.
- **GPU mapping:**
  - If the model has no `CUDA_VISIBLE_DEVICES` in the model server's config, it sees every GPU. Then
    every GPU is affected, as with CUDA itself.
  - An entry that maps to no GPU (an index without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, an unknown UUID) is
    an error in the model server's config: answer `400` naming it, no silent clearing.

Answer `200`:

```json
{
  "gpus": [{"index": 0, "uuid": "GPU-…"}, {"index": 2, "uuid": "GPU-…"}],
  "already_running": false,
  "duration_s": 13.1,
  "actions": [
    {"consumer": "stt-suite", "item": "gpu-worker", "outcome": "released", "waited_s": 12.4},
    {"consumer": "tts-suite", "item": "<engine>", "outcome": "refused_busy", "detail": "still busy", "waited_s": 5.0},
    {"consumer": "llama-swap", "item": "<embed-model>", "outcome": "released", "waited_s": 0.8}
  ],
  "memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}],
  "free": false,
  "status": {"gpus": ["… as GET /status, only for the affected GPUs …"], "errors": []}
}
```

`actions` are in the order of the consumers in the config; `detail` and `waited_s` (seconds since the
first request to that consumer) only where there is something to say.

| `outcome` | Meaning |
|---|---|
| `released` | The service let go (or nothing ran any more), and the verification confirmed it |
| `released_forced` | Let go only with `force` after `busy_wait_s`, where `force_after_wait` allows it; verified |
| `not_freed` | The service says "free", but still reports the item or the memory did not go down; or the service itself reports `failed`. `detail` names the findings |
| `refused_busy` | The item was still busy after `busy_wait_s`: `force_after_wait` is off, or the forcing request also came back "busy" |
| `timeout` | The overall limit was reached before the item was free: the limit cut the wait short, or the request never got its turn in the queue |
| `kept` | Was in `keep` |
| `error` | Service unreachable or an unexpected answer; llama-swap: unloading failed or the model is still loaded afterwards. A consumer that could not say what it holds appears as `error` with an empty `item` |
| `would_release` | Only with `dry_run` |

`memory` is the used memory per affected GPU before the release and after the verification
(`after_mib` is `null` with `dry_run` and when the request never got its turn in the queue).

`free` is `true` when, after the actions, no item the request was to free is still reported, no
action is `not_freed` or `error`, and every consumer could say what it holds. The exception is a
consumer that `keep` protects as a whole: if it cannot report, that is no error of the request
(calibration stops llama-swap and then has the rest cleared); `status.errors` still names it. Memory
nobody reports does not make `free` false; it is in `status`. Whether a caller loads anyway on
`free: false` is the caller's decision.

Errors:
- `400`: no JSON object, unknown fields, both or neither of the forms, `reason` missing, `keep` names
  an unknown consumer, unknown GPU, unknown model, `model_server` missing with several model servers,
  `model_server` with the `gpus` form, or a `CUDA_VISIBLE_DEVICES` that maps to no GPU.
- `401` / `403`: token missing or wrong.
- `404`: unknown `model_server`.
- `502`: In the `model` form the model server does not answer whether the model runs.
- `500`: nvidia-smi failed.

An unreachable consumer is no HTTP error but an action with `outcome: "error"`.

## Config

Described in [deploy/config.example.yaml](../deploy/config.example.yaml); the file lives in
`~/.config/gpu-orc/config.yaml`. A sketch, no real addresses:

```yaml
listen: {host: 127.0.0.1, port: 5070}
token_file: ~/.config/gpu-orc/token        # required as soon as host is not only loopback
free_timeout_s: 180                         # longest a /free request may take from its arrival
verify_timeout_s: 30                        # measure this long after a release (within free_timeout_s)
min_freed_mib: 256                          # must go down by this much where the service reports no memory
                                            # (at most by what was in use before)
consumers:
  - name: stt-suite
    type: gpu-release
    url: http://127.0.0.1:5080
    busy_wait_s: 30                         # wait this long for a running transcription
    force_after_wait: true                  # then have it aborted (device=auto goes on on the CPU)
  - name: tts-suite
    type: gpu-release
    url: http://127.0.0.1:5090
    busy_wait_s: 5                          # wait this long for a reading or a start
    force_after_wait: true                  # then have it aborted (reading_aborted), container stopped
  - name: llama-swap
    type: llama-swap
    url: http://127.0.0.1:11435
    config_file: ~/.config/llama-swap/config.yaml   # GPUs per model, persistent groups
    no_eviction_models: ["*-vlm-*"]         # evict no persistent models when they load
state_dir: ~/.local/state/gpu-orc           # measurements (measurements.json)
vision_models:                              # vision models per model server; GPU-Orc measures their need
  llama-swap: [Qwen3VL-4B-Instruct-Q8_0]
headroom_mib: 512                           # stays free on a GPU on top of a side model's need (/place)
reserve_s: 60                               # how long promised room counts without a holder for it
priority: [tts-suite, stt-suite]            # suites whose idle side models give way, the first last
```

Without a value, `busy_wait_s: 0` and `force_after_wait: false` apply. Unknown keys of a consumer and
duplicate consumer names keep GPU-Orc from starting. GPU-Orc reads llama-swap's config on every
request; it may change while running.

## Command line

- `gpu-orc serve [--config PATH]` runs the service.
- `gpu-orc import-aifred --data <AIfred's data folder> [--config PATH]` takes over the measurements of
  AIfred's calibration once into `<state_dir>/measurements.json`: per llama.cpp profile the newest
  calibration that recorded the free memory per GPU and its context (`model_vram_cache.json`), and the
  burn-in peak of each vision model (`vlm_vram_cache.json`). An imported value replaces the one of the same
  name, all others stay. The running service reads the file on every request, no restart is needed.

## Callers

- **STT-Suite and TTS-Suite** call `POST /place` before they start a GPU worker or a GPU engine, and start
  it on the GPU of the answer.
- **AIfred** calls `POST /free {"model": …}` before loading a model and a describer. Calibration and
  switching the TTS profile call `{"gpus": "all", "keep": ["llama-swap"]}`. The describer's room check
  reads `GET /status`. AIfred loads even on `free: false` and reports it.
- **Agent-Orc** calls GPU-Orc for backends with local models (the backend's `gpu_orc` field with
  `port`, `model_server` if needed and `token_file` for remote machines). GPU-Orc runs on the model
  server's machine, at its address with the port from `gpu_orc`. Agent-Orc asks when Claude Code starts
  on the backend and in the hook `agent-orc agent-free-cards` before every prompt, with
  `POST /free {"model": …, "reason": "Agent-Orc: agent in <folder>"}`. It shows what GPU-Orc did and
  what went wrong; the start or the prompt goes on regardless.
- If GPU-Orc does not run, the callers say so loudly. There is no other way to free the GPUs.

Settled: `already_running` is enough for the hook; there is no "report only" switch (what must load
must be freed for first). `dry_run` is the preview; there is no separate query for it.
