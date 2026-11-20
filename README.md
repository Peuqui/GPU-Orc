# GPU-Orc

[Deutsch](README.de.md)

A small service per machine that frees GPUs on request and tells who holds what. Before a model
loads, its caller asks GPU-Orc to clear the cards the model will use; GPU-Orc asks the services that
sit there to let go, through their own interfaces. It never kills a process from outside: the hardest
step is a service's own `force`, and only where the config allows it.

It loads nothing, plans nothing and is no model server. One place per machine knows how to free its
cards, instead of every caller carrying its own copy of that logic.

## How it works

- **Consumers** are the services that hold GPUs. Each has a type, one implementation per protocol:
  - `gpu-release`: a service with the common release API of the STT-Suite and the TTS-Suite
    (`GET /gpu/holders`, `POST /gpu/release`, also AIfred's face detection; with `token_file` as a bearer token). While an item is busy (a transcription, a reading) the
    service answers "busy"; GPU-Orc waits up to `busy_wait_s`, then has the service force it if
    `force_after_wait` is set. The service ends the process or container itself and says `released`
    only once it is gone.
  - `llama-swap`: a model server. GPU-Orc reads which cards each model uses (`CUDA_VISIBLE_DEVICES`
    in the model's `env`) and which models are persistent (groups with `persistent: true`) from
    llama-swap's own config.
  - `tts-suite-remote`: the TTS containers that the TTS-Suite of another machine runs on this one
    (its `hosts`). That suite reports them (`GET /engines`) and stops them (`POST /release`); GPU-Orc
    sends the suite's token, waits up to `busy_wait_s` for a reading that holds a container, and counts
    a running container as holding every card (the suite does not say which).
- **Holding the cards:** `POST /hold {"owner", "reason", "model"}` keeps this machine's cards for somebody (a game,
  other work); `model` is the model of the model server the owner needs (it stays loaded) or `null`. One hold at a
  time, only its owner lets go (`DELETE /hold {"owner"}`; `"force": true` for the person, logged). A new hold is
  requested first and the answer comes at once; one thread in GPU-Orc clears the cards and makes it active as soon
  as no main model is loaded or loading on a model server any more (at once when they are free, else it looks
  again every `hold_check_interval_s`). A person can grant a waiting hold (`POST /hold/grant {"owner", "by"}`):
  it clears although a main model is loaded, but only once no work of others is reported (`/work`), so a running
  answer is never broken off; the grant lives in the hold file. Callers wait for it with `hold_and_wait`. The
  work of the owner itself does not count, so use **the same owner name for `/work` and `/hold`**. While the hold is active, `GET /loading` (`held`) and `GET /status` say
  so, and callers that would start something put nothing new on the cards; a requested hold shows as
  `hold_requested`. The hold lives in `hold_file` and outlasts a restart; GPU-Orc ends nothing a caller loads
  without asking.
- **Work in progress:** a caller that runs a job here (AIfred, for every turn it runs) reports it with
  `POST /work` as a lease that ends by itself without renewal; `DELETE /work/<owner>` ends it. `GET /work`
  and `GET /status` list what runs. `gpu-orc run-if-idle -- /sbin/reboot` runs a command only when nothing is
  reported: the weekly reboot leaves a running job alone (no waiting, the next date tries again).
- **Checked, not believed:** after a release GPU-Orc asks the service again and measures the card
  memory with nvidia-smi, until `verify_timeout_s`. An item still reported, or memory that did not
  go down, is `not_freed`.
- **Who holds what** comes from the services themselves. nvidia-smi only gives total and used memory
  per card, which also works under WSL2, where it lists no processes. Memory nobody reports is shown,
  never touched.
- **Requests** to free are handled one after another; within one request all consumers are asked at
  the same time. Every request ends at the latest `free_timeout_s` after it arrived, waiting in the
  queue included.

## Interface

| Endpoint | Purpose |
|---|---|
| `GET /health` | version, `free_timeout_s` and the HTTP timeout callers should use |
| `GET /loading` | which big models are loading right now, whether the cards are held (`held`) or a hold waits (`hold_requested`), for those who want to start something on a card |
| `GET /status` | per GPU: memory, holders, unattributed memory; the queue; consumers that could not report |
| `POST /free` | free GPUs: `{"gpus": [0, "GPU-…"] \| "all", ...}` or `{"model": "<name>", ...}` |
| `POST /hold` | hold the cards for somebody: `{"owner": "...", "reason": "...", "model": "..." or null}`; answers at once with the requested hold, which turns active once no main model is loaded or loading (the requested model may run); another owner: 409 |
| `POST /hold/grant` | a person grants a waiting hold: `{"owner": "...", "by": "..."}`; it clears once no work of others runs; no hold: 404, another owner: 409 |
| `DELETE /hold` | the owner lets go: `{"owner": "...", "force": false}`; another owner: 409, `"force": true` overrides |
| `GET /work` | the jobs reported right now (answers at once) |
| `POST /work` | report a job, or renew its lease: `{"owner": "...", "reason": "...", "ttl_s": 60}` |
| `DELETE /work/<owner>` | the job is over |

`POST /free` takes exactly one of two forms, plus `reason` (required), `keep` and `dry_run`:

- **`gpus`:** everything on these cards goes, main models of a model server included, even in the
  middle of an answer. `keep` protects (`"llama-swap"` or `"llama-swap:<model>"`).
- **`model`:** the model about to be loaded.
  - If it already runs, GPU-Orc answers `already_running` at once, without queueing.
  - Otherwise services on its cards and the server's persistent models there go. The server swaps
    its other models itself.

The full description is in [docs/interface.md](docs/interface.md); the suites' common release API in
[docs/release-api.md](docs/release-api.md).

Requests over loopback need no token. From other machines they need `Authorization: Bearer <token>`;
GPU-Orc refuses to start on a non-loopback address without a token file.

## Clients for callers

The protocol is written down once per language, in `clients/`, and tested against the real server (`tests/test_client_python.py`, `tests/test_client_js.py`):

- `clients/python/gpu_orc_client.py`: standard library only (`GpuOrc`, `FreeResult.problems()`, `TrafficLight.red`, `hold`, `grant_hold`, `hold_and_wait` and work calls).
- `clients/js/gpu-orc-client.js`: no dependencies (`freeForModel`, `loading`), Node 18+.

The programs hold byte-identical copies. `clients/consumers.json` lists them; `python scripts/sync_clients.py` writes them (`--check` changes nothing and exits 1 for a stale or missing copy, `--only NAME` for one program, `--repos DIR` for the folder holding the repos, default: the parent of this one, so a sync can go into worktrees first). Edit the clients only here, never in a copy.

A caller turns GPU-Orc off by leaving its address empty. That decision is made in the caller before it calls the client; the client has no "off" state.

## Install

Needs Python 3.12 with venv (Debian/Ubuntu: `python3-venv`) and `nvidia-smi`.

```bash
deploy/install.sh                       # venv in ~/.local/share/gpu-orc, user unit, first config
$EDITOR ~/.config/gpu-orc/config.yaml   # the consumers of this machine
systemctl --user enable --now gpu-orc
# or, to run at boot without a login:
sudo deploy/switch-to-system-unit.sh
```

The system unit comes with a polkit rule: its user restarts it without a password
(`systemctl restart gpu-orc@<user>`), for instance after changing the config. It needs polkit
(Debian/Ubuntu: `polkitd`); without it the script stops before changing anything.

The config is described in [deploy/config.example.yaml](deploy/config.example.yaml).

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/mypy src tests
```

## Star History

![Star History](https://raw.githubusercontent.com/Peuqui/Traffic/main/data/GPU-Orc/star-history.svg)

<sub>Recorded daily in [Peuqui/Traffic](https://github.com/Peuqui/Traffic), together with views and clones. GitHub restricted the stargazer API to repo admins on 2026-06-30, so external chart services now need a token with write access.</sub>

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to download and use for any noncommercial purpose; commercial use needs the author's permission.
