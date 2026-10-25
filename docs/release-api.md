# The suites' common release API

[Deutsch](release-api.de.md)

As of GPU-Orc 0.2.0, with the STT-Suite and the TTS-Suite on their current main branch.

The STT-Suite and the TTS-Suite free their GPUs the same way: the same endpoints, the same status
codes, the same `force`, the same result fields. GPU-Orc needs one consumer type for them
(`gpu-release`) and one procedure: wait up to `busy_wait_s`, then `force` if the config allows it for
this consumer.

**"Released" means the same for both:** the process with the CUDA context has ended. For the STT-Suite
that is the child process of the GPU worker (and of diarization), for the TTS-Suite the engine's
container. An `/unload` *inside* the process is not enough: PyTorch keeps its memory pool, and the GPU
never gets fully empty.

## `GET /gpu/holders`

What the service holds on this machine's GPUs; engines on other machines are not listed.

```json
{"holders": [
  {"item": "gpu-worker", "gpu_uuids": ["GPU-…"], "memory_mib": {"GPU-…": 3100}, "busy": true}
]}
```

- `item`: for the STT-Suite `gpu-worker` or `diarize-worker`, for the TTS-Suite the engine's name.
- `gpu_uuids`: the GPUs the process can see, from its configuration rather than from a measurement, so
  always known. STT: the GPU the worker was started on. TTS: the GPUs from the container's
  configuration (`DeviceRequests` or `NVIDIA_VISIBLE_DEVICES`, as `docker inspect` shows them). Only a
  container that really sees every GPU is listed on all of them.
- `memory_mib`: what the service measured itself, per GPU; `null` where there is no measurement (STT:
  for diarization). The GPU does not depend on it.
- `busy`: STT a running transcription, TTS a running reading that holds the engine, or a start of the
  engine under way.

## `POST /gpu/release`

```json
{"items": ["gpu-worker"], "force": false, "reason": "GPU-Orc: model X is about to load"}
```

`items` is a list or `"all"` (everything `/gpu/holders` names). `reason` is required; it goes into the
service's log and, with `force`, into the message to the clients that were cut off. An unknown or
repeated item, an empty list or an empty reason is refused.

Answer, always with one result per item:

```json
{"results": [
  {"item": "gpu-worker", "outcome": "released", "detail": "worker PID 4711 ended"},
  {"item": "<engine>", "outcome": "busy", "detail": "a reading holds it"}
]}
```

| `outcome` | Meaning |
|---|---|
| `released` | The process with the CUDA context has provably ended: STT the worker process is no longer alive, TTS Docker reports the container gone or exited |
| `not_running` | Nothing ran, nothing to do |
| `busy` | The item is working and was left alone (without `force`; for the TTS-Suite also with `force` while the engine is starting, see below) |
| `failed` | Ending was tried, but the proof is missing (`detail` says what still runs) |

Status code: `200` when every item is `released` or `not_running`; `409` when one is `busy` and none
`failed`; `500` when one is `failed`. Items that are not busy are released even with a `409`.

**`force: true`**, the same for both:
1. Abort the item's work, naming the reason.
2. End the process or the container. If it does not end in time, the suite ends it hard itself: STT
   `kill`, TTS `docker kill` after the stop timeout. GPU-Orc never ends anything itself.
3. Report `released` only after the proof described above.

### Particulars

- **TTS abort:** Every reading that holds the engine ends with
  `{"error": "reading_aborted", "detail": "…<reason>"}`: as an error frame if audio already came,
  otherwise as an answer with HTTP 503. This holds for `POST /speak` and `/speak/stream`. The reading
  does not move on to another engine; clients show the abort and read again on request, the text is
  theirs.
- **TTS start:** An engine that is starting stays `busy`, also with `force`; its start cannot be called
  back. If a release falls into an engine's cold start (XTTS takes about 29 s to load), GPU-Orc reports
  `refused_busy` after the wait, and the GPU stays in use.
- **STT abort:** After `force`, a running transcription counts as "GPU unavailable".
  - A request with `device=auto` goes on on the CPU; the dictation is kept, only slower.
  - A request with `device=cuda` gets `503` with the reason; the client can repeat it on the CPU.
  - The suite does not keep the recording; the client that uploaded it has it.
- **One way to release a GPU:** Local GPUs are released only through `/gpu/release`.
  - STT `/unload` only unloads the CPU models (`device=cuda` gives `400`). The STT-Suite's web page
    calls `/gpu/release` for the GPU.
  - TTS `/release` only controls engines on *other* machines. Local engines go through `/gpu/release`.

## What GPU-Orc makes of it: checked, not believed

After every release GPU-Orc checks the result itself instead of believing the answer:

1. **Status:** The service no longer names the item (`/gpu/holders`, for llama-swap `/running`).
2. **Process or container gone:** For the suites this is part of `released` (see above); GPU-Orc sees
   it through the status.
3. **Memory:** nvidia-smi per GPU, before and after. The driver often returns memory only 1–2 s after
   the process ended; GPU-Orc measures again every 0.5 s until `verify_timeout_s` (config, default 30 s).
   - If an item reported its memory, the memory of its GPU must go down by at least half of that.
   - Otherwise it must go down by at least `min_freed_mib` over its GPUs together (config, default
     256; a CUDA context alone takes more).
   - This also works under WSL2.
   - The verification lies *within* `free_timeout_s`, not on top of it. It ends at `verify_timeout_s`
     or at the overall limit, whichever comes first; `caller_timeout_s` from `/health` stays right.

`released` and `released_forced` exist only after a passed check; otherwise the outcome is
`not_freed`. Its `detail` names the status finding apart from the memory finding, for example
`status: no longer reported; memory: GPU-… 31200 → 31150 MiB, expected −1550`. If the suite itself
reports `failed`, that becomes `not_freed` too (`detail`: `service: …`).

**Known limit:** If something else loads onto the same GPU during the verification (a model starting
in parallel, a measurement run), the memory does not go down although the release worked. The result
is then `not_freed` with "no longer reported, memory did not go down". This is on purpose: loud rather
than silent.

The answer of `/free` carries the measurements per affected GPU:

```json
"memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}]
```

## Waiting times in the config

Peuqui sets them per machine, as values in the config and not in the code. Today on the Mini:

- **TTS:** wait 5 s, then `force`. An abort costs little, the client has the text; a cold start would
  only finish loading to be aborted afterwards.
- **STT:** wait 30 s, then `force`. A dictation is done on the GPU in seconds; a long recording with
  `device=auto` is transcribed again from the start on the CPU with the same file, nothing is lost.

## Tests

A consumer that says "ok" but frees nothing (the status still reports it, or the memory does not go
down) is caught as `not_freed`. The same holds for the suites: a worker or container that survives the
stop gives `failed`, never `released`.
