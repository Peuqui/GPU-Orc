# GPU-Orc — Interface

[Deutsch](interface.de.md)

As of version 0.6.0. The suites' common release API, through which GPU-Orc talks to the STT-Suite and
the TTS-Suite, is described in [release-api.md](release-api.md).

GPU-Orc is a small service per machine. It frees GPUs on request and tells who holds what right now.
It loads nothing, plans nothing and replaces no model server. It only talks to the interfaces of the
services that hold GPUs and never ends a process from outside. The hardest step is a service's own
`force`, and only where the config allows it.

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
  affected consumers *at the same time*; they are independent of each other. `/status` and `/health`
  always answer at once.
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

## Endpoints

Errors always come as `{"error": "<text>"}`. An unknown path gives `404`.

### `GET /health`

```json
{"status": "ok", "version": "0.8.0", "free_timeout_s": 180.0, "caller_timeout_s": 210.0}
```

### `GET /loading` — the traffic light

```json
{"loading": ["DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm"], "held": null, "hold_requested": null}
```

Which big models are loading right now, and whether the cards are held (`held`, see `POST /hold`, only for an
ACTIVE hold, so that everybody who reads the light keeps its meaning); `hold_requested` names a hold that waits
for the work on the cards to end (callers that start new work, like AIfred's turns, wait on it, or a stream of new
work would postpone the hold for ever; it does not make the light red). Empty and `null` = green. While a model loads, its cards look emptier than
they are about to be. Whoever wants to put something on a card (STT-Suite, TTS-Suite, AIfred's image
describer) asks here first and starts nothing on the GPU while there is an entry. The answer comes at once,
without nvidia-smi and without asking the consumers.

A model counts as loading from the moment a `/free` in the `model` form clears for it, before the model
server knows of it, and for `load_grace_s` (config, default 15 s) after that `/free` ended; from then on the
model server says so itself (`starting`). Its persistent helpers do not count, they load into their own
room, not even when something is cleared for them. If the model server does not answer: `502`. That is no green light.

### `POST /hold`, `DELETE /hold` — keeping the cards for somebody

```json
{"owner": "vllm-research", "reason": "gaming"}
```

Somebody keeps this machine's cards (a game, other work). The hold has an `owner` and exists once at a time.
GPU-Orc stores it in `hold_file` (config, default `~/.local/state/gpu-orc/hold.json`; it outlasts a restart of
GPU-Orc and of the machine). A hold file of the old format (no owner) keeps GPU-Orc from starting. A body that is
not exactly `owner` and `reason` (non-empty strings): `400`.

- A new hold is REQUESTED (`"state": "requested"`, `"free": null`), and the answer comes at once; nothing is
  touched yet:

  ```json
  {"hold": {"owner": "vllm-research", "reason": "gaming", "since": "2026-11-05T09:12:00+01:00",
            "state": "requested", "free": null}}
  ```

- One thread in GPU-Orc, running as long as the service, makes every change of the hold: at the first moment no
  work of others (`POST /work`) is left, it clears every card like `POST /free` with `"gpus": "all"` and makes the
  hold ACTIVE with that clearing in `free` (`{"...": "as POST /free"}`). Without work of others this follows right
  after the answer. The work of the hold's own owner does not count: use the same owner name for `/work` and
  `/hold`. `POST /hold` and `DELETE /work/<owner>` wake the thread, and a lease that runs out (the reporter
  crashed) is found at the moment it expires. Only the very hold the cards were cleared for turns active: released
  (and maybe requested anew) meanwhile, the clearing counts for nothing and the new hold waits for its own.
  Nothing clears the cards inside a `GET`. After a start of GPU-Orc a hold that is still requested waits
  `work_lease_s` (config, default 60 s) before it may activate, so that the renewals of running work come back
  first. Callers wait for ACTIVE with `hold_and_wait` (clients). Should the thread itself fail outside the
  clearing (the hold file cannot be written, a bug), GPU-Orc ends with the traceback on stderr instead of
  answering on with holds that never turn active; the unit's `Restart=on-failure` starts it again.
- A hold only REQUESTED refuses nobody by itself: the work that keeps it back (a turn of AIfred) goes on, and
  what serves that work (reading aloud, dictation) may still load. Whoever reports no work and would load
  something big for itself (an Agent-Orc session) treats `hold_requested` like `held`, or the clearing takes it
  away in the middle. What starts in the seconds between the end of the last work and ACTIVE, while the cards
  are cleared, stays on them: a known gap, short by nature.
- When the clearing ends with a problem (a consumer busy, memory left, or an error while clearing, then one
  action `gpu-orc` / `all cards` / `error` with the detail), the hold is active anyway and keeps that `free` (also
  in the hold file and in `GET /status`); nobody retries behind the owner's back.
- **Asking again** as the same owner changes nothing and returns the current hold. **Another owner** while any hold
  exists (requested or active): `409`, `{"error": "...", "owner": "<holder>", "state": "...", "since": "..."}`.

`GET /loading` and `GET /status` carry `"held"` (the hold, only while ACTIVE) and `"hold_requested"` (the hold,
only while REQUESTED). Whoever would put something new on a card (a describer of another machine, a TTS
container started from there) leaves the cards alone while `held` is set. GPU-Orc ends nothing a caller loads
without asking.

`DELETE /hold` with `{"owner": "...", "force": false}` lets go (a requested hold is withdrawn); answer
`{"hold": null}`. Only the owner may: another owner gets `409` (same body as above). `"force": true` lets go of
the hold of anybody, for the person when a program died holding the cards; GPU-Orc logs it on stderr with both
owners. `owner` is required also with `force`. Without a hold the answer is `{"hold": null}`.

### `POST /work`, `GET /work`, `DELETE /work/<owner>` — a job runs here

```json
{"owner": "aifred", "reason": "AIfred: 2 running turn(s)", "ttl_s": 60}
```

A caller that runs a job on this machine reports it, so that whoever would restart the machine (the weekly
reboot) can ask first. A report is a lease: it ends with `DELETE /work/<owner>` (owner URL-encoded; answer
`{"ended": true, "work": [...]}`, `ended: false` when it was gone already) or by itself when `ttl_s` seconds pass
without the same owner reporting again. Reporting again renews the lease, keeps the start and takes the new
reason. Nothing is cleared, nothing is held: the cards stay as they are. The reports live in memory only.
`GET /work` answers `{"work": [{"owner", "reason", "running_for_s", "expires_in_s"}]}` at once (no nvidia-smi);
`GET /status` carries the same list as `work`. A body that is not exactly `owner`, `reason` (non-empty strings)
and `ttl_s` (a positive number): `400`.

`gpu-orc run-if-idle [--config PATH] -- <command>` asks `GET /work` on this machine and runs the command only
when the list is empty (its exit code is the command's). With work running it skips the command and exits 0:
no waiting, no deadline, the next date of the cron job tries again. When GPU-Orc does not answer it skips the
command too and exits 3 with a message on stderr. The weekly reboot is
`30 5 * * 1,3,5 root /home/mp/.local/share/gpu-orc/venv/bin/gpu-orc run-if-idle -- /sbin/reboot`.

### `GET /status` — who holds what

```json
{
  "queue": {"running": true, "waiting": 1, "running_since_s": 42.0},
  "held": null,
  "work": [],
  "gpus": [
    {
      "index": 0,
      "uuid": "GPU-…",
      "name": "Quadro RTX 8000",
      "memory_total_mib": 49152,
      "memory_used_mib": 31200,
      "holders": [
        {"consumer": "llama-swap", "item": "<model>", "memory_mib": null, "busy": null, "persistent": false},
        {"consumer": "stt-suite", "item": "gpu-worker", "memory_mib": 3100, "busy": false, "persistent": null},
        {"consumer": "tts-suite", "item": "<engine>", "memory_mib": 6024, "busy": false, "persistent": null}
      ],
      "unattributed_mib": null
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
  - `busy` is reported only by the suites (a transcription, a reading, a start), `persistent` only by
    the model server (member of a group with `persistent: true`). The other field is `null`.
- `unattributed_mib` is the used memory minus what the holders report. It is a number only when every
  holder on the GPU reports its memory, otherwise `null`. On a GPU without holders it is all of the
  used memory. GPU-Orc never touches what nobody reports.
- `errors` names consumers that could not say what they hold (`[{"consumer": …, "detail": …}]`);
  their holders are then missing from `holders`.
- `500` when nvidia-smi fails.

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
| `error` | Service unreachable or an unexpected answer; llama-swap: unloading failed or the model is still loaded afterwards. A consumer that could not say what it holds appears as `error` with an empty `item`, but only where it could hold something (see `free`) |
| `would_release` | Only with `dry_run` |

`memory` is the used memory per affected GPU before the release and after the verification
(`after_mib` is `null` with `dry_run` and when the request never got its turn in the queue).

`free` is `true` when, after the actions, no item the request was to free is still reported, no
action is `not_freed` or `error`, and no consumer that could not say what it holds could sit on the
cards. It could only where a target card holds memory that no reporting consumer accounts for (at
least `min_freed_mib`, or any memory beside a holder that reports none): on cards that are empty, or
whose memory the reporting consumers explain, a stopped service holds nothing, and the request is
not held up by it. A consumer that `keep` protects as a whole never counts either (calibration stops
llama-swap and then has the rest cleared). `status.errors` names every consumer that could not report. Memory
nobody reports does not make `free` false by itself, as long as every consumer could report; it is in
`status`. Whether a caller loads anyway on
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
                                            # (at most by what was in use before); also the memory a card
                                            # may hold unexplained before an unreachable consumer counts
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
  - name: aifred
    type: gpu-release
    url: http://127.0.0.1:8002/api          # its face detection (face-worker)
    token_file: ~/.config/gpu-orc/aifred.token   # AIfred's GPU_RELEASE_API_TOKEN
    busy_wait_s: 5                          # a detection takes a fraction of a second
  - name: llama-swap
    type: llama-swap
    url: http://127.0.0.1:11435
    config_file: ~/.config/llama-swap/config.yaml   # GPUs per model, persistent groups
    no_eviction_models: ["*-vlm-*"]         # evict no persistent models when they load
```

Without a value, `busy_wait_s: 0` and `force_after_wait: false` apply. With `token_file`, GPU-Orc sends a
`gpu-release` service `Authorization: Bearer <token>` with every request (AIfred requires it, the suites do not). Unknown keys of a consumer and
duplicate consumer names keep GPU-Orc from starting. GPU-Orc reads llama-swap's config on every
request; it may change while running.

## Callers

- **AIfred** calls GPU-Orc before loading any model since 30 October 2026 (`POST /free {"model": …}`, before a
  calibration the `gpus` form with `keep`), asks the traffic light (`GET /loading`) before its
  image describer or its face detection puts anything on a card, and holds the cards through `POST /hold` /
  `DELETE /hold` (with its own owner name) from its status bar. If GPU-Orc does not answer, AIfred loads anyway and says so loudly. With
  the Vue rebuild AIfred is also a consumer (`gpu-release`, item `face-worker`, with a token).
- **Agent-Orc** calls GPU-Orc for backends with local models (the backend's `gpu_orc` field with
  `port`, `model_server` if needed and `token_file` for remote machines). GPU-Orc runs on the model
  server's machine, at its address with the port from `gpu_orc`. Agent-Orc asks when Claude Code starts
  on the backend and in the hook `agent-orc agent-free-cards` before every prompt, with
  `POST /free {"model": …, "reason": "Agent-Orc: agent in <folder>"}`. It shows what GPU-Orc did and
  what went wrong; the start or the prompt goes on regardless.
- If GPU-Orc does not run, the callers say so loudly. There is no other way to free the GPUs.

Settled: `already_running` is enough for the hook; there is no "report only" switch (what must load
must be freed for first). `dry_run` is the preview; there is no separate query for it.
