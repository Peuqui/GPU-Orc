# GPU-Orc — Schnittstelle

[English](interface.md)

Stand: Version 0.6.0. Die gemeinsame Freigabe-Schnittstelle der Suiten, über die GPU-Orc STT-Suite
und TTS-Suite anspricht, steht in [release-api.de.md](release-api.de.md).

GPU-Orc ist ein kleiner Dienst je Rechner. Er räumt auf Anfrage Grafikkarten frei, sagt, wer gerade was
belegt, und entscheidet, auf welcher Karte ein Zusatzmodell (Engine oder Worker einer Suite) startet. Er lädt
selbst nichts und ersetzt keinen Modellserver. Er spricht nur die Schnittstellen der Dienste an, die Karten
belegen, und beendet nie Prozesse von außen. Das härteste Mittel ist das `force` eines Dienstes, und das nur,
wo die Config es erlaubt.

## Grundsätze

- **Adresse:** `http://<rechner>:5070` (Port in der Config). User-Unit `gpu-orc`, System-Unit
  `gpu-orc@<benutzer>`.
- **Anmeldung:** Über Loopback ohne Token. Von anderen Rechnern nur mit
  `Authorization: Bearer <token>` (fehlt es: `401`, falsch: `403`). Das Token steht in einer Datei auf
  dem Rechner, die in der Config genannt ist. Lauscht GPU-Orc nicht nur auf Loopback und es gibt kein
  Token, startet er nicht.
- **Karten:** Angaben als Index (Reihenfolge von nvidia-smi, also PCI-Bus) oder als UUID, auch als
  eindeutiger Anfang einer UUID. In Antworten stehen immer Index und UUID, gerechnet wird intern mit
  der UUID.
- **Reihenfolge:** `/free`-Anfragen arbeitet GPU-Orc nacheinander ab. Innerhalb einer Anfrage fragt
  und bittet er alle betroffenen Verbraucher *gleichzeitig*, sie sind unabhängig voneinander.
  `/status`, `/health` und `/place` warten nie in dieser Schlange.
  - **Ausnahme für die Form `model`:** Bevor die Anfrage eingereiht wird, fragt GPU-Orc den
    Modellserver, ob das Modell schon läuft (geladen oder am Laden). Wenn ja, antwortet er sofort mit
    `"already_running": true`, auch wenn gerade eine andere Anfrage läuft. Diese Prüfung ändert nichts
    am Zustand; so hält ein Aufrufer, der bei jeder Eingabe fragt, nie hinter einer langen Freigabe an.
  - Nur wenn das Modell nicht läuft, kommt die Anfrage in die Schlange. Dort prüft GPU-Orc vor dem
    Freiräumen noch einmal, denn eine vorausgehende Anfrage kann es inzwischen geladen haben.
- **Höchstdauer:** Jede `/free`-Anfrage endet spätestens `free_timeout_s` nach ihrem Eingang (Config,
  Standard 180 s), die Wartezeit in der Schlange eingeschlossen.
  - Die Wartezeit eines Verbrauchers auf „beschäftigt“ (`busy_wait_s`) und die Nachmessung liegen
    *innerhalb* dieser Frist. Bei `gpu-release` endet die Wartezeit spätestens 5 s vor der Frist, damit
    der erzwingende Aufruf noch hineinpasst.
  - Aufrufer setzen ihr HTTP-Timeout auf `caller_timeout_s` aus `GET /health`
    (`free_timeout_s` + 30 s), damit sie den Wert nicht selbst pflegen.
- **Verbraucher:** Sie stehen in der Config, nicht im Code, je mit Name, Typ und Regeln. Für den Typ
  gibt es eine Implementierung je Protokoll, nicht je Rechner: `gpu-release` (STT- und TTS-Suite,
  gemeinsame Freigabe-Schnittstelle) und `llama-swap`.
- **Wer hält was:**
  - Jeder Verbraucher meldet über die Schnittstelle seines Dienstes selbst, was er hält und auf
    welcher Karte: bei den Suiten `GET /gpu/holders`, bei llama-swap `GET /running` plus die
    Kartenzuordnung aus dessen Config.
  - nvidia-smi liefert nur den gesamten und den belegten Speicher je Karte; das geht auch unter WSL2,
    wo nvidia-smi keine Prozesse nennt.
  - Es gibt keine Zuordnung über PIDs, weder in `/status` noch in `/free`.
- **Ansprüche:** Was ein Element auf einer Karte belegen darf, ob es das schon tut oder nicht.
  - Ein Modell des Modellservers: der Anspruch seines Profils (`GET /profiles`), die ganze Karte, solange
    der unbekannt ist.
  - Ein Zusatzmodell: der größere Wert von gemeldetem Speicher und gemeldetem Bedarf (`need_mib` in
    `/gpu/holders`); der Speicher wächst beim Start noch und hat beim Rechnen Spitzen. Die ganze Karte,
    solange der Bedarf unbekannt ist.
  - Zugesagter Platz (eine Reservierung, siehe `/place` und `/free`), den noch kein Halter abdeckt: sein
    Bedarf. Ein Halter desselben Elements auf derselben Karte zählt einmal, mit dem größeren der beiden Werte.
  - Speicher, den kein Halter erklärt (ein anderes Programm), zählt obendrauf. Was übrig bleibt, ist das
    `available_mib` einer Karte.

## Endpunkte

Fehler kommen immer als `{"error": "<text>"}`. Ein unbekannter Pfad ergibt `404`.

### `GET /health`

```json
{"status": "ok", "version": "0.6.0", "free_timeout_s": 180.0, "caller_timeout_s": 210.0}
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
      "compute_cap": 7.5,
      "memory_total_mib": 49152,
      "memory_used_mib": 31200,
      "holders": [
        {"consumer": "llama-swap", "item": "<modell>", "memory_mib": null, "need_mib": null, "claim_mib": 29491, "busy": null, "persistent": false},
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

- `queue` zeigt, ob gerade eine `/free`-Anfrage läuft, seit wie vielen Sekunden (`null`, wenn keine),
  und wie viele warten.
- `holders`: Ein Halter, der mehrere Karten sieht, steht bei jeder davon.
  - `memory_mib` steht nur da, wo der Dienst seinen Speicher selbst meldet: die TTS-Suite je Engine,
    die STT-Suite für den GPU-Worker (nicht für die Diarisierung). llama-swap meldet nur die Karten, also `null`.
  - `need_mib` melden nur die Suiten, ebenso `busy` (Transkription, Vorlesung, Start); `persistent` nur der
    Modellserver (Mitglied einer Gruppe mit `persistent: true`). Die jeweils anderen Felder sind `null`.
  - `claim_mib` ist der Anspruch des Elements auf dieser Karte (siehe Grundsätze); `null`, solange ein
    Verbraucher in `errors` steht.
- `reservations` ist der zugesagte Platz auf der Karte und wie viele Sekunden die Zusage noch gilt.
- `unattributed_mib` ist der belegte Speicher minus dem, was die Halter melden. Er ist nur dann eine
  Zahl, wenn jeder Halter auf der Karte seinen Speicher meldet, sonst `null`. Auf einer Karte ohne
  Halter ist es der ganze belegte Speicher. GPU-Orc fasst Unbekanntes nie an.
- `available_mib` ist, was ein weiteres Programm auf der Karte noch belegen darf: Kartenspeicher minus
  Ansprüchen, Reservierungen und Speicher, den niemand erklärt (siehe Grundsätze). Ein vLLM-Modell belegt
  einen Teil seines Anteils nur, während es rechnet; nvidia-smi zeigt ihn dazwischen als frei, und wer ihn
  nimmt, lässt das Modell beim nächsten Rechnen ohne Speicher (25.10.2026). `null`, solange ein Verbraucher
  in `errors` steht: Dann weiß niemand, was frei ist.
- `errors` nennt Verbraucher, die nicht sagen konnten, was sie halten
  (`[{"consumer": …, "detail": …}]`); ihre Halter fehlen dann in `holders`.
- `500`, wenn nvidia-smi scheitert.

### `GET /profiles` — die Profile des Modellservers

```json
{
  "profiles": [
    {
      "name": "<modell>-tts-qwen3local", "model_server": "llama-swap", "backend": "llama.cpp",
      "cards": ["GPU-…"], "persistent": false, "context": 131072, "memory_fraction": null, "kv_blocks": null,
      "variant": {"base": "<modell>", "tts": "qwen3local", "vlm": null, "speed": false, "describer": false, "describer_gpu": null},
      "claim_mib": {"GPU-…": 41200}, "claim_source": "measured", "measured_at": "2026-07-12T23:02:40",
      "measured_as": "<modell>", "problem": null
    }
  ],
  "errors": []
}
```

- Jeder Eintrag der Config von llama-swap ist ein Profil. `cards` sind die Karten aus `CUDA_VISIBLE_DEVICES`
  in der Reihenfolge von nvidia-smi; `backend` kommt aus `cmd` (`vllm`, `llama.cpp`, sonst `other`), ebenso
  `context` (`--max-model-len` bzw. `-c`), `kv_blocks` (`--num-gpu-blocks-override`) und `memory_fraction`.
- `variant` folgt den Namen der Kalibrierung: `<basis>[-tts-<engine>][-vlm-<schlüssel>][-speed]` für die
  Hauptmodelle, `<basis>-visiond[-gpu<N>]` für die Bildbeschreiber. Sie sagt, für welche Zusatzmodelle ein
  Profil Platz lässt.
- `claim_mib` ist, was das Profil je Karte nimmt:
  - `measured`: aus einer Kalibrierung, die festgehalten hat, was nach dem Laden frei blieb (Karte minus
    frei, daher mit dem Grundverbrauch der Karte; also eher zu viel); bei einem Bildbeschreiber
    (`-visiond`) der Burn-in-Spitzenwert seines Basismodells. Beides zählt nur, wenn es beim Kontext des
    Profils gemessen wurde. Eine Variante, deren Eintrag dem eines gemessenen Profils gleicht (die
    Kalibrierung schreibt eine Variante, deren Platz auf einer Karte außerhalb des Modells liegt, als Kopie
    ihrer Basis), übernimmt dessen Messung; `measured_as` nennt das Profil, sonst ist es `null`;
  - `fraction`: bei vLLM ohne Messung sein Anteil der Karte;
  - `no_cards`: `{}`, der Eintrag sieht keine Karte (`CUDA_VISIBLE_DEVICES=`) und läuft auf der CPU;
  - `unknown`: `null`, zum Beispiel bei einer Variante mit eigenen Einstellungen und ohne Messung.
- `problem` nennt, warum die Karten eines Eintrags nicht lesbar waren; `errors` Modellserver, deren Config
  nicht lesbar war.

### `GET /needs` — was die Zusatzmodelle brauchen

```json
{"needs": [
  {"consumer": "tts-suite", "item": "qwen3local", "need_mib": 6024, "source": "estimate", "default": false},
  {"consumer": "stt-suite", "item": "parakeet-fp32", "need_mib": 3712, "source": "measured", "default": true},
  {"consumer": "llama-swap", "item": "Qwen3VL-4B-Instruct-Q8_0", "need_mib": 7396, "source": "measured", "default": false}
], "errors": []}
```

- Die Suiten melden ihre Modelle selbst (`GET /gpu/needs`, siehe [release-api.de.md](release-api.de.md)),
  `default` ist das Modell, das eine Suite ohne Angabe in der Anfrage auf die GPU lädt.
- Die Bildmodelle eines Modellservers stehen in der Config unter `vision_models`; ihren Bedarf misst
  GPU-Orc selbst (bis dahin übernommen mit `gpu-orc import-aifred`, sonst `null` mit `unknown`).
- `source`: `measured` (Burn-in), `configured`, `estimate`, `unknown`. Ein Wert ohne Puffer: Den Restpuffer
  je Karte rechnet GPU-Orc dazu.

### `POST /place` — wo ein Zusatzmodell starten darf

Eine Suite fragt, bevor sie eine Engine oder einen Worker auf einer Karte startet:

```json
{"consumer": "tts-suite", "item": "qwen3local", "need_mib": 6024, "reason": "Vorlesung für AIfred"}
```

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `consumer` | ja | Die Suite (ein `gpu-release`-Verbraucher der Config) |
| `item` | ja | Das Element, wie die Suite es in `/gpu/holders` nennt (`gpu-worker`, `<engine>`) |
| `need_mib` | ja | Was es auf der Karte braucht, ohne Puffer (den rechnet GPU-Orc mit `headroom_mib` dazu) |
| `reason` | ja | Steht im Log von GPU-Orc; müssen Zusatzmodelle weichen, auch in deren Log |
| `card` | nein | Nur diese Karte (Index oder UUID), etwa für einen Burn-in auf einer vorgegebenen Karte |
| `min_compute_cap` | nein | Nur Karten mit mindestens dieser CUDA-Compute-Capability (Diarisierung: 7.5) |

Andere Felder weist GPU-Orc mit `400` ab.

- **Passt:** Eine Karte passt, wo `available_mib` minus `headroom_mib` (Config, Standard 512) den `need_mib`
  deckt. Davon kommt zuerst eine Karte ohne Ansprüche und ohne belegten Speicher (Zusatzmodelle verteilen sich
  auf die freien Karten), dann die mit dem meisten Platz, dann der niedrigste Index.
- **Das Element selbst:** Ein Halter und eine Zusage desselben Verbrauchers und Elements zählen nicht, ebenso
  wenig der Speicher, den dieser Halter belegt: Das Element wird ersetzt (der STT-Worker wechselt das Modell).
- **Weichen:** Passt keine Karte, müssen untätige Zusatzmodelle einer Suite, die in `priority` (Config) nach
  der fragenden steht, freigeben: die auf der Karte, wo am wenigsten weichen müssen, die mit dem niedrigsten
  Vorrang zuerst, ohne Zwang und ohne Warten. Ein beschäftigtes bleibt. Danach misst GPU-Orc bis
  `verify_timeout_s` nach.
- **Zusage:** Die Karte der Antwort hält `need_mib` für das Element `reserve_s` lang fest (Config, Standard
  60 s), damit eine zweite Anfrage nicht denselben Platz bekommt, bevor das Element in `/gpu/holders`
  erscheint; danach zählt sein eigener Anspruch. Eine Zusage läuft von selbst ab, auch wenn die Suite das
  Element nie startet.
- GPU-Orc entscheidet hier nur und fragt niemanden außer `/gpu/holders` (und beim Weichen); ein Lock im
  Dienst umfasst nur die Entscheidung selbst.

Antwort `200`:

```json
{"card": {"index": 2, "uuid": "GPU-…"}, "why": null, "reserved_s": 60.0, "actions": [], "duration_s": 0.4}
```
```json
{"card": null, "why": "needs 6024 MiB and 512 MiB headroom, the most room is 1650 MiB on GPU 4",
 "reserved_s": null, "actions": [], "duration_s": 0.3}
```

- `card: null` mit `why`: Keine Karte passt; keine Karte erfüllt `card` oder `min_compute_cap`; ein
  Verbraucher konnte nicht sagen, was er hält (`cannot say what is free: …`); ein Zusatzmodell hat nicht
  freigegeben.
- `actions` sind die Zusatzmodelle, die weichen mussten, mit den Ergebnissen von `/free` (`released`,
  `refused_busy`, `not_freed`, `error`).
- `400`: falsche Felder oder unbekannte `card`. `404`: `consumer` ist keine Suite der Config. `500`:
  nvidia-smi gescheitert.

### Die Seite unter `/ui/`

Eine Weboberfläche wie die der Suiten: Karten (`/status`), Profile (`/profiles`), Bedarfe (`/needs`).
`/` leitet dorthin weiter. Die Dateien der Seite gibt es ohne Token, sie enthalten keine Daten; jede Abfrage,
die die Seite macht, braucht das Token wie jeder andere Aufruf von einem anderen Rechner (Knopf „Token“, im
Browser gespeichert).

### `POST /free` — Karten freiräumen

Genau eine der beiden Formen:

```json
{"gpus": [0, 2], "keep": ["tts-suite"], "reason": "ein anderer Rechner braucht zwei Karten", "dry_run": false}
```
```json
{"model": "<modell>", "model_server": "llama-swap", "reason": "Agent-Orc: agent in <ordner>", "dry_run": false}
```

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `gpus` | eins von beiden | Liste von Indizes oder UUIDs, oder `"all"` |
| `model` | eins von beiden | Das Modell, das gleich geladen wird; GPU-Orc löst seine Karten über die Config des Modellservers auf |
| `model_server` | nur bei `model`, und nur, wenn mehr als ein Modellserver konfiguriert ist | Name des Verbrauchers vom Typ Modellserver; bei der Form `gpus` nicht erlaubt |
| `keep` | nein | Was stehen bleibt: Verbrauchername (`"tts-suite"`) oder `verbraucher:element` (`"llama-swap:<modell>"`) |
| `reason` | ja | Steht im Log von GPU-Orc und im Log der angesprochenen Dienste; bei einem erzwungenen Abbruch auch in der Meldung an deren Clients |
| `dry_run` | nein | `true`: nur sagen, was passieren würde, nichts anfassen |

Andere Felder lehnt GPU-Orc mit `400` ab.

**Form `gpus`:** Jeder Verbraucher auf diesen Karten soll weichen, auch die Hauptmodelle des
Modellservers.
- **Achtung:** llama-swap entlädt ein Modell auch mitten in einer Antwort; wer es gerade benutzt,
  bricht ab. Das ist gewollt hart.
- Schutz ist `keep`, zum Beispiel `"llama-swap:<modell>"` oder `"llama-swap"` für alle seine Modelle.

**Form `model`:**
- Läuft das Modell schon, passiert nichts (`"already_running": true`). Die Antwort hat dann keine
  Aktionen, kein `memory`, `free: true` und `status: null`; die Verbraucher werden gar nicht erst gefragt.
- Sonst weichen auf den Karten des Modells:
  - Dienste wie STT- und TTS-Suite,
  - die *dauerhaften* Modelle des Modellservers. Bei llama-swap sind das die Mitglieder der Gruppen
    mit `persistent: true` aus dessen eigener Config.
- Die übrigen Modelle tauscht der Modellserver beim Laden selbst aus; GPU-Orc fasst sie nicht an.
  Gerade benutzte Hauptmodelle bricht diese Form also nicht ab.
- Ab ihrer Reihe in der Schlange bis `reserve_s` nach der Antwort sind die Karten des Modells ihm mit dem
  Anspruch seines Profils zugesagt (die ganze Karte, wo der unbekannt ist): Ein Zusatzmodell, das in dieser
  Zeit platziert würde, nähme den Platz, in den es gleich lädt. Nicht bei `dry_run`.
- Keine dauerhaften Modelle entladen lässt ein Modell, das selbst dauerhaft ist oder auf ein Muster
  aus `no_eviction_models` passt (heute die `-vlm-`-Profile, die per Kalibrierung Platz für die
  dauerhaften Modelle lassen).
- **Kartenzuordnung:**
  - Hat das Modell in der Config des Modellservers kein `CUDA_VISIBLE_DEVICES`, sieht es alle Karten.
    Dann sind auch alle Karten betroffen, wie bei CUDA selbst.
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
    {"consumer": "tts-suite", "item": "<engine>", "outcome": "refused_busy", "detail": "still busy", "waited_s": 5.0},
    {"consumer": "llama-swap", "item": "<embed-modell>", "outcome": "released", "waited_s": 0.8}
  ],
  "memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}],
  "free": false,
  "status": {"gpus": ["… wie GET /status, nur für die betroffenen Karten …"], "errors": []}
}
```

`actions` stehen in der Reihenfolge der Verbraucher in der Config; `detail` und `waited_s` (Sekunden
seit der ersten Bitte an diesen Verbraucher) nur, wo es etwas zu sagen gibt.

| `outcome` | Bedeutung |
|---|---|
| `released` | Der Dienst hat freigegeben (oder es lief schon nichts mehr), und die Nachprüfung hat es bestätigt |
| `released_forced` | Erst nach Ablauf von `busy_wait_s` mit `force` freigegeben, wo `force_after_wait` das erlaubt; nachgeprüft |
| `not_freed` | Der Dienst sagt „frei“, aber er meldet das Element weiter oder der Speicher ist nicht gesunken; oder der Dienst meldet selbst `failed`. `detail` nennt die Befunde |
| `refused_busy` | Das Element war nach `busy_wait_s` noch beschäftigt: `force_after_wait` ist aus, oder auch der erzwingende Aufruf kam mit „beschäftigt“ zurück |
| `timeout` | Das Gesamtlimit war erreicht, bevor das Element frei war: die Wartezeit wurde von der Frist abgeschnitten, oder die Anfrage kam aus der Schlange nicht mehr dran |
| `kept` | Stand in `keep` |
| `error` | Dienst nicht erreichbar oder unerwartete Antwort; llama-swap: Entladen gescheitert oder Modell danach noch geladen. Ein Verbraucher, der nicht sagen konnte, was er hält, steht als `error` mit leerem `item` |
| `would_release` | Nur bei `dry_run` |

`memory` ist der belegte Speicher je betroffener Karte vor der Freigabe und nach der Nachprüfung
(`after_mib` ist `null` bei `dry_run` und wenn die Anfrage aus der Schlange nicht mehr drankam).

`free` ist `true`, wenn nach den Aktionen kein Element, das die Anfrage freiräumen sollte, mehr
gemeldet wird, keine Aktion `not_freed` oder `error` ist und jeder Verbraucher sagen konnte, was er
hält. Ausgenommen ist ein Verbraucher, den `keep` als Ganzes schützt: Kann er nichts melden, ist das
kein Fehler der Anfrage (die Kalibrierung hält llama-swap an und lässt dann den Rest freiräumen);
`status.errors` nennt ihn trotzdem. Unbekannter Speicher macht `free` nicht falsch, er steht im
`status`. Ob ein Aufrufer bei `free: false` trotzdem lädt, entscheidet er.

Fehler:
- `400`: kein JSON-Objekt, unbekannte Felder, beide oder keine der Formen, `reason` fehlt, `keep` nennt
  einen unbekannten Verbraucher, unbekannte Karte, unbekanntes Modell, `model_server` fehlt bei mehreren
  Modellservern, `model_server` bei der Form `gpus`, oder ein `CUDA_VISIBLE_DEVICES`, das sich keiner
  Karte zuordnen lässt.
- `401` / `403`: Token fehlt oder ist falsch.
- `404`: unbekannter `model_server`.
- `502`: Bei der Form `model` antwortet der Modellserver nicht auf die Frage, ob das Modell läuft.
- `500`: nvidia-smi ist gescheitert.

Ein nicht erreichbarer Verbraucher ist kein HTTP-Fehler, sondern eine Aktion mit `outcome: "error"`.

## Config

Beschrieben in [deploy/config.example.yaml](../deploy/config.example.yaml); die Datei liegt in
`~/.config/gpu-orc/config.yaml`. Skizze, keine echten Adressen:

```yaml
listen: {host: 127.0.0.1, port: 5070}
token_file: ~/.config/gpu-orc/token        # Pflicht, sobald host nicht nur Loopback ist
free_timeout_s: 180                         # Höchstdauer einer /free-Anfrage ab Eingang
verify_timeout_s: 30                        # so lange nach einer Freigabe nachmessen (innerhalb free_timeout_s)
min_freed_mib: 256                          # so viel muss sinken, wo der Dienst keinen Speicher meldet
                                            # (höchstens so viel, wie vorher belegt war)
consumers:
  - name: stt-suite
    type: gpu-release
    url: http://127.0.0.1:5080
    busy_wait_s: 30                         # so lange auf eine laufende Transkription warten
    force_after_wait: true                  # danach abbrechen lassen (device=auto läuft auf der CPU weiter)
  - name: tts-suite
    type: gpu-release
    url: http://127.0.0.1:5090
    busy_wait_s: 5                          # so lange auf eine Vorlesung oder einen Start warten
    force_after_wait: true                  # danach abbrechen lassen (reading_aborted), Container stoppen
  - name: llama-swap
    type: llama-swap
    url: http://127.0.0.1:11435
    config_file: ~/.config/llama-swap/config.yaml   # Karten je Modell, persistent-Gruppen
    no_eviction_models: ["*-vlm-*"]         # verdrängen beim Laden keine dauerhaften Modelle
state_dir: ~/.local/state/gpu-orc           # Messwerte (measurements.json)
vision_models:                              # Bildmodelle je Modellserver, ihren Bedarf misst GPU-Orc
  llama-swap: [Qwen3VL-4B-Instruct-Q8_0]
headroom_mib: 512                           # bleibt auf einer Karte über den Bedarf eines Zusatzmodells frei (/place)
reserve_s: 60                               # so lange zählt zugesagter Platz ohne Halter
priority: [tts-suite, stt-suite]            # Suiten, deren untätige Zusatzmodelle weichen, die erste zuletzt
```

Ohne Angabe gelten `busy_wait_s: 0` und `force_after_wait: false`. Unbekannte Schlüssel eines Verbrauchers
und doppelte Verbrauchernamen lassen GPU-Orc nicht starten. Die Config von llama-swap liest GPU-Orc bei jeder
Anfrage neu, sie darf sich im Betrieb ändern.

## Kommandozeile

- `gpu-orc serve [--config PATH]` startet den Dienst.
- `gpu-orc import-aifred --data <AIfreds data-Ordner> [--config PATH]` übernimmt einmalig die Messwerte von
  AIfreds Kalibrierung in `<state_dir>/measurements.json`: je llama.cpp-Profil die neueste Kalibrierung, die
  den freien Speicher je Karte und ihren Kontext festgehalten hat (`model_vram_cache.json`), und den
  Burn-in-Spitzenwert jedes Bildmodells (`vlm_vram_cache.json`). Ein übernommener Wert ersetzt den
  gleichnamigen, alle anderen bleiben. Der laufende Dienst liest die Datei bei jeder Anfrage, ein Neustart ist
  nicht nötig.

## Aufrufer

- **STT-Suite und TTS-Suite** rufen `POST /place`, bevor sie einen GPU-Worker oder eine GPU-Engine starten,
  und starten auf der Karte der Antwort.
- **AIfred** ruft vor dem Laden eines Modells und eines Describers `POST /free {"model": …}` auf.
  Kalibrierung und Wechsel des TTS-Profils rufen `{"gpus": "all", "keep": ["llama-swap"]}`. Die
  Platzprüfung des Describers liest `GET /status`. AIfred lädt auch bei `free: false` und meldet es.
- **Agent-Orc** ruft GPU-Orc für Backends mit lokalen Modellen auf (Feld `gpu_orc` des Backends mit
  `port`, bei Bedarf `model_server` und für entfernte Rechner `token_file`). GPU-Orc läuft dabei auf
  dem Rechner des Modellservers, unter dessen Adresse mit dem Port aus `gpu_orc`. Agent-Orc fragt beim
  Start von Claude Code auf dem Backend und im Hook `agent-orc agent-free-cards` vor jeder Eingabe mit
  `POST /free {"model": …, "reason": "Agent-Orc: agent in <ordner>"}`. Was GPU-Orc getan hat und was
  schiefging, zeigt Agent-Orc an; der Start oder die Eingabe geht trotzdem weiter.
- Läuft GPU-Orc nicht, melden die Aufrufer das laut. Einen eigenen Ersatzweg gibt es nicht.

Festgelegt: Für den Hook reicht `already_running`, einen Schalter „nur melden“ gibt es nicht (muss
geladen werden, muss vorher freigeräumt werden). `dry_run` ist die Vorschau, eine eigene Abfrage dafür
gibt es nicht.
