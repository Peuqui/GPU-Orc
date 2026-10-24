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
    (`GET /gpu/holders`, `POST /gpu/release`). While an item is busy (a transcription, a reading) the
    service answers "busy"; GPU-Orc waits up to `busy_wait_s`, then has the service force it if
    `force_after_wait` is set. The service ends the process or container itself and says `released`
    only once it is gone.
  - `llama-swap`: a model server. GPU-Orc reads which cards each model uses (`CUDA_VISIBLE_DEVICES`
    in the model's `env`) and which models are persistent (groups with `persistent: true`) from
    llama-swap's own config.
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
| `GET /status` | per GPU: memory, holders, unattributed memory; the queue; consumers that could not report |
| `POST /free` | free GPUs: `{"gpus": [0, "GPU-…"] \| "all", ...}` or `{"model": "<name>", ...}` |

`POST /free` takes exactly one of two forms, plus `reason` (required), `keep` and `dry_run`:

- **`gpus`:** everything on these cards goes, main models of a model server included, even in the
  middle of an answer. `keep` protects (`"llama-swap"` or `"llama-swap:<model>"`).
- **`model`:** the model about to be loaded.
  - If it already runs, GPU-Orc answers `already_running` at once, without queueing.
  - Otherwise services on its cards and the server's persistent models there go. The server swaps
    its other models itself.

The full description (German) is in [docs/schnittstelle.md](docs/schnittstelle.md).

Requests over loopback need no token. From other machines they need `Authorization: Bearer <token>`;
GPU-Orc refuses to start on a non-loopback address without a token file.

## Install

Needs Python 3.12 and `nvidia-smi`.

```bash
deploy/install.sh                       # venv in ~/.local/share/gpu-orc, user unit, first config
$EDITOR ~/.config/gpu-orc/config.yaml   # the consumers of this machine
systemctl --user enable --now gpu-orc
# or, to run at boot without a login:
sudo deploy/switch-to-system-unit.sh
```

The config is described in [deploy/config.example.yaml](deploy/config.example.yaml).

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/mypy src tests
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to download and use for any noncommercial purpose; commercial use needs the author's permission.
