# GPU-Orc — Schnittstelle

Stand 24.10.2026, umgesetzt in Version 0.2.0 (abgestimmt mit Agent-Orc). Die gemeinsame Freigabe-Schnittstelle
der Suiten, die GPU-Orc dafür nutzt, steht in [freigabe-api.md](freigabe-api.md).

GPU-Orc ist ein kleiner Dienst je Rechner. Er räumt auf Anfrage Grafikkarten frei und sagt, wer gerade
was belegt. Er lädt nichts, plant nichts und ersetzt keinen Modellserver. Er spricht nur die
Schnittstellen der Dienste an, die Karten belegen, und beendet nie Prozesse von außen. Das härteste
Mittel ist das `force` eines Dienstes, und das nur, wo die Config es erlaubt.

Aufrufer sind AIfred (vor dem Laden eines Modells), Agent-Orc (`before_start` und der Hook vor jeder
Eingabe von claude-local) und entfernte Rechner. Läuft GPU-Orc nicht, melden die Aufrufer das laut.
Einen eigenen Ersatzweg gibt es nicht.

## Grundsätze

- **Adresse:** `http://<rechner>:5070` (Port in der Config), systemd-Unit `gpu-orc`.
- **Anmeldung:** Über Loopback ohne Token. Von anderen Rechnern nur mit
  `Authorization: Bearer <token>`. Das Token steht in einer Datei auf dem Rechner, die in der Config
  genannt ist. Lauscht GPU-Orc nicht nur auf Loopback und es gibt kein Token, startet er nicht.
- **Karten:** Angaben als Index (Reihenfolge von nvidia-smi, also PCI-Bus) oder als UUID. In
  Antworten stehen immer beide, gerechnet wird intern mit der UUID.
- **Reihenfolge:** `/free`-Anfragen arbeitet GPU-Orc nacheinander ab. Innerhalb einer Anfrage bittet
  er alle betroffenen Verbraucher *gleichzeitig*, sie sind unabhängig voneinander. `/status` und
  `/health` antworten immer sofort.
  - **Ausnahme für die Form `model`:** Bevor die Anfrage eingereiht wird, fragt GPU-Orc den
    Modellserver, ob das Modell schon läuft. Wenn ja, antwortet er sofort mit
    `"already_running": true`, auch wenn gerade eine andere Anfrage läuft.
  - Diese Prüfung ändert nichts am Zustand und braucht deshalb keine Reihenfolge. So hält der Hook,
    der bei jeder Eingabe fragt, nie hinter einer langen Freigabe an.
  - Nur wenn das Modell nicht läuft, kommt die Anfrage in die Schlange. Dort prüft GPU-Orc vor dem
    Freiräumen noch einmal, denn eine vorausgehende Anfrage kann es inzwischen geladen haben.
- **Höchstdauer:** Jede `/free`-Anfrage endet spätestens `free_timeout_s` nach ihrem Eingang (Config,
  Standard 660 s), die Wartezeit in der Schlange eingeschlossen.
  - Die Wartezeit eines Verbrauchers auf „beschäftigt“ (`busy_wait_s`) endet spätestens mit dem
    Gesamtlimit. Danach folgt `force`, wo die Config es erlaubt, sonst `timeout`.
  - Aufrufer setzen ihr HTTP-Timeout auf mindestens `free_timeout_s + 30 s`. Beide Werte meldet
    `GET /health`, damit Aufrufer sie nicht doppelt pflegen.
- **Verbraucher:** Sie stehen in der Config, nicht im Code, je mit Name, Typ und Regeln. Für den Typ
  gibt es eine Implementierung je Protokoll, nicht je Rechner: `gpu-release` (STT- und TTS-Suite,
  gemeinsame Freigabe-Schnittstelle), `llama-swap` (später vLLM direkt oder Ollama).
- **Wer hält was:**
  - Jeder Verbraucher meldet über die Schnittstelle seines Dienstes selbst, was er hält und auf
    welcher Karte: bei den Suiten `/gpu/holders`, bei llama-swap `/running` plus die Kartenzuordnung
    aus dessen Config.
  - nvidia-smi liefert nur Gesamt- und belegten Speicher je Karte; das geht auch unter WSL2, wo
    nvidia-smi keine Prozesse nennt.
  - Es gibt keine Zuordnung über PIDs, weder in `/status` noch in `/free`.

## Endpunkte

### `GET /health`

```json
{"status": "ok", "version": "0.2.0", "free_timeout_s": 660, "caller_timeout_s": 690}
```

### `GET /status` — wer hält was

```json
{
  "queue": {"running": true, "waiting": 1, "running_since_s": 42.0},
  "gpus": [
    {
      "index": 0,
      "uuid": "GPU-…",
      "name": "Quadro RTX 8000",
      "memory_total_mib": 49152,
      "memory_used_mib": 31200,
      "holders": [
        {"consumer": "llama-swap", "item": "<modell>", "memory_mib": null, "persistent": false},
        {"consumer": "stt-suite", "item": "gpu-worker", "memory_mib": 3100, "busy": false},
        {"consumer": "tts-suite", "item": "<engine>", "memory_mib": 6024, "busy": false}
      ],
      "unattributed_mib": null
    }
  ]
}
```

- `queue` zeigt, ob gerade eine `/free`-Anfrage läuft, seit wann, und wie viele warten.
- `errors` nennt Verbraucher, die nicht sagen konnten, was sie halten
  (`[{"consumer": …, "detail": …}]`); ihre Halter fehlen dann in `holders`.
- `memory_mib` steht nur da, wo der Dienst seinen Speicher selbst meldet: die TTS-Suite je Engine,
  die STT-Suite für den GPU-Worker (nicht für die Diarisierung); llama-swap meldet nur die Karte.
- `unattributed_mib` ist der belegte Speicher minus dem, was die Halter melden. Er ist nur dann eine
  Zahl, wenn jeder Halter auf der Karte seinen Speicher meldet, sonst `null`. Auf einer Karte ohne
  Halter ist es der ganze belegte Speicher. GPU-Orc fasst Unbekanntes nie an.

### `POST /free` — Karten freiräumen

Genau eine der beiden Formen:

```json
{"gpus": [0, 2], "keep": ["tts-suite"], "reason": "ein anderer Rechner braucht zwei Karten", "dry_run": false}
```
```json
{"model": "<modell>", "model_server": "llama-swap", "reason": "Agent-Orc: claude-local", "dry_run": false}
```

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `gpus` | eins von beiden | Liste von Indizes oder UUIDs, oder `"all"` |
| `model` | eins von beiden | Das Modell, das gleich geladen wird; GPU-Orc löst seine Karten über die Config des Modellservers auf |
| `model_server` | nur bei `model`, wenn mehr als einer konfiguriert ist | Name des Verbrauchers vom Typ Modellserver |
| `keep` | nein | Was stehen bleibt: Verbrauchername (`"tts-suite"`) oder `verbraucher:element` (`"llama-swap:<modell>"`) |
| `reason` | ja | Steht im Log von GPU-Orc und im Log der angesprochenen Dienste, soweit sie einen Grund annehmen |
| `dry_run` | nein | `true`: nur sagen, was passieren würde, nichts anfassen |

**Form `gpus`:** Jeder Verbraucher auf diesen Karten soll weichen, auch die Hauptmodelle des
Modellservers.
- **Achtung:** llama-swap entlädt ein Modell auch mitten in einer Antwort; wer es gerade benutzt,
  bricht ab. Das ist gewollt hart.
- Schutz ist `keep`, zum Beispiel `"llama-swap:<modell>"` oder `"llama-swap"` für alle seine Modelle.

**Form `model`:** wie heute in AIfred.
- Läuft das Modell schon, passiert nichts (`"already_running": true`). Das ist der schnelle Weg für
  den Hook vor jeder Eingabe. Die Antwort hat dann keine Aktionen, `free: true` und `status: null`;
  die Verbraucher werden gar nicht erst gefragt.
- Sonst weichen auf den Karten des Modells:
  - Dienste wie STT und TTS
  - die *dauerhaften* Modelle des Modellservers. Bei llama-swap sind das die Mitglieder der Gruppen
    mit `persistent: true` aus dessen eigener Config.
- Die übrigen Modelle tauscht der Modellserver beim Laden selbst aus; GPU-Orc fasst sie nicht an.
  Gerade benutzte Hauptmodelle bricht diese Form also nicht ab.
- Modelle, die laut Config nichts verdrängen (heute die `-vlm-`-Profile), lösen keine Entladung
  dauerhafter Modelle aus.
- **Kartenzuordnung:**
  - Hat das Modell in der Config des Modellservers kein `CUDA_VISIBLE_DEVICES`, sieht es alle Karten.
    Dann sind auch alle Karten betroffen, wie bei CUDA selbst und wie heute in `lclaude-free-gpus`.
  - Ein Eintrag, der sich keiner Karte zuordnen lässt (ein Index ohne
    `CUDA_DEVICE_ORDER=PCI_BUS_ID`, eine unbekannte UUID), ist ein Fehler in der Config des
    Modellservers: Antwort `400` mit der Stelle, kein stilles Abräumen.

Antwort `200`:

```json
{
  "gpus": [{"index": 0, "uuid": "GPU-…"}, {"index": 2, "uuid": "GPU-…"}],
  "already_running": false,
  "duration_s": 13.1,
  "actions": [
    {"consumer": "stt-suite", "item": "gpu-worker", "outcome": "released", "waited_s": 12.4},
    {"consumer": "tts-suite", "item": "<engine>", "outcome": "refused_busy", "detail": "Vorlesung läuft"},
    {"consumer": "llama-swap", "item": "<embed-modell>", "outcome": "released"}
  ],
  "memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}],
  "free": false,
  "status": {"gpus": ["… wie GET /status, nur für die betroffenen Karten …"]}
}
```

`outcome` je Aktion:

| Wert | Bedeutung |
|---|---|
| `released` | Der Dienst hat freigegeben, und die Nachprüfung hat es bestätigt |
| `released_forced` | Erst nach Ablauf der Wartezeit mit `force`, wo die Config das erlaubt; nachgeprüft |
| `not_freed` | Der Dienst sagt „frei“ (oder `failed`), aber er meldet das Element weiter oder der Speicher ist nicht gesunken; `detail` nennt beide Befunde |
| `refused_busy` | Der Dienst arbeitet und darf nicht gezwungen werden (`force_after_wait` aus) |
| `timeout` | Das Gesamtlimit war erreicht, bevor der Dienst freigeben konnte, und `force` ist nicht erlaubt |
| `kept` | Stand in `keep` |
| `error` | Dienst nicht erreichbar oder unerwartete Antwort; `detail` sagt was |
| `would_release` | Nur bei `dry_run` |

`memory` ist der belegte Speicher je betroffener Karte vor der Freigabe und nach der Nachprüfung
(`after_mib` ist `null` bei `dry_run`).

`free` ist true, wenn nach den Aktionen kein bekannter Verbraucher mehr auf den Karten sitzt, keine
Aktion `not_freed` oder `error` ist und jeder Verbraucher sagen konnte, was er hält. Ausgenommen ist ein
Verbraucher, den `keep` als Ganzes schützt: Kann er nichts melden, ist das kein Fehler der Anfrage (die
Kalibrierung hält llama-swap an und lässt dann den Rest freiräumen); `status.errors` nennt ihn trotzdem.
Unbekannter Speicher macht `free` nicht falsch, er steht im `status`. Ob der Aufrufer bei
`free: false` trotzdem lädt, entscheidet er. AIfred und Agent-Orc laden trotzdem und melden es laut.

Fehler:
- `400`: beide oder keine der Formen angegeben, unbekannte Karte, unbekanntes Modell, oder ein
  `CUDA_VISIBLE_DEVICES`, das sich keiner Karte zuordnen lässt.
- `401` / `403`: Token fehlt oder ist falsch.
- `404`: unbekannter `model_server`.
- `502`: Bei der Form `model` antwortet der Modellserver nicht auf die Frage, ob das Modell läuft.
- `500`: nvidia-smi ist gescheitert.

Ein nicht erreichbarer Verbraucher ist kein HTTP-Fehler, sondern eine Aktion mit `outcome: "error"`.

## Config (Skizze, keine echten Adressen)

```yaml
listen: {host: 127.0.0.1, port: 5070}
token_file: ~/.config/gpu-orc/token        # Pflicht, sobald host nicht nur Loopback ist
free_timeout_s: 660                         # Höchstdauer einer /free-Anfrage ab Eingang
verify_timeout_s: 30                        # so lange nach einer Freigabe nachmessen (innerhalb free_timeout_s)
min_freed_mib: 256                          # so viel muss sinken, wo der Dienst keinen Speicher meldet
consumers:
  - name: stt-suite
    type: gpu-release
    url: http://127.0.0.1:5080
    busy_wait_s: 600                        # so lange auf eine laufende Transkription warten
    force_after_wait: true                  # danach abbrechen lassen (device=auto läuft auf der CPU weiter)
  - name: tts-suite
    type: gpu-release
    url: http://127.0.0.1:5090
    busy_wait_s: 20                         # so lange auf eine Vorlesung warten
    force_after_wait: true                  # danach abbrechen lassen (reading_aborted), Container stoppen
  - name: llama-swap
    type: llama-swap
    url: http://127.0.0.1:11435
    config_file: ~/.config/llama-swap/config.yaml   # Karten je Modell, persistent-Gruppen
    no_eviction_models: ["*-vlm-*"]         # verdrängen beim Laden keine dauerhaften Modelle
```

## Was sich bei den anderen ändert

- **STT-Suite und TTS-Suite:** gemeinsame Freigabe-Schnittstelle `GET /gpu/holders` und
  `POST /gpu/release` ([freigabe-api.md](freigabe-api.md)); STT `/unload` entlädt nur noch die
  CPU-Modelle, TTS `/release` gilt nur noch für Engines auf anderen Rechnern.
- **AIfred:**
  - Vor dem Laden eines Modells und eines Describers ruft AIfred nur noch `POST /free {"model": …}` auf;
    Kalibrierung und TTS-Profilwechsel rufen `{"gpus": "all", "keep": ["llama-swap"]}`.
  - `release_whisper_gpu`, die TTS-Freigabe vor dem Laden und die Sidecar-Logik sind entfallen; die
    Platzprüfung des Describers liest `GET /status`.
  - Die TTS-Freigabe fehlte in `lclaude-free-gpus`; die Kopien waren also schon auseinandergelaufen.
- **Agent-Orc:**
  - Je Backend gibt es einen Block `gpu_orc` mit `url`, `model_server` (falls nötig) und
    `token_file` für entfernte Rechner.
  - `before_start` und der Hook rufen `POST /free {model, reason: "Agent-Orc: <agent>"}` auf.
  - Ist GPU-Orc nicht erreichbar, kommt eine laute Meldung, der Start bricht nicht ab.
  - `lclaude-free-gpus` entfällt.

## Abgestimmt (24.10.2026)

- `already_running` reicht für den Hook; kein Schalter „nur melden“: Muss geladen werden, muss vorher
  freigeräumt werden.
- Port 5070, die Unit heißt `gpu-orc` (System-Unit `gpu-orc@<user>`).
- `dry_run` reicht als Vorschau, keine eigene Abfrage.
