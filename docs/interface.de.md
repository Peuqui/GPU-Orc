# GPU-Orc — Schnittstelle

[English](interface.md)

Stand: Version 0.6.0. Die gemeinsame Freigabe-Schnittstelle der Suiten, über die GPU-Orc STT-Suite
und TTS-Suite anspricht, steht in [release-api.de.md](release-api.de.md).

GPU-Orc ist ein kleiner Dienst je Rechner. Er räumt auf Anfrage Grafikkarten frei und sagt, wer gerade
was belegt. Er lädt nichts, plant nichts und ersetzt keinen Modellserver. Er spricht nur die
Schnittstellen der Dienste an, die Karten belegen, und beendet nie Prozesse von außen. Das härteste
Mittel ist das `force` eines Dienstes, und das nur, wo die Config es erlaubt.

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
  `/status` und `/health` antworten immer sofort.
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

## Endpunkte

Fehler kommen immer als `{"error": "<text>"}`. Ein unbekannter Pfad ergibt `404`.

### `GET /health`

```json
{"status": "ok", "version": "0.8.0", "free_timeout_s": 180.0, "caller_timeout_s": 210.0}
```

### `GET /loading` — die Ampel

```json
{"loading": ["DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm"], "held": null, "hold_requested": null}
```

Welche großen Modelle gerade laden und ob die Karten freigehalten werden (`held`, siehe `POST /hold`, nur für einen
AKTIVEN Halt); `hold_requested` nennt einen Halt, der darauf wartet, dass die Karten frei werden (wer selbst ein
Hauptmodell laden will, behandelt ihn wie `held`; er macht die Ampel nicht rot); leer und `null` = grün. Während ein Modell lädt, sehen seine Karten leerer aus,
als sie gleich sind. Wer etwas auf eine Karte legen will (STT-Suite, TTS-Suite, AIfreds Bildbeschreiber),
fragt hier vorher und startet bei einem Eintrag nichts auf der GPU. Die Antwort kommt sofort, ohne
nvidia-smi und ohne die Verbraucher zu fragen.

Ein Modell zählt als ladend, sobald ein `/free` in der Form `model` dafür räumt, also bevor der
Modellserver davon weiß, und noch `load_grace_s` (Config, Standard 15 s) nach dem Ende dieses `/free`;
danach sagt es der Modellserver selbst (`starting`). Seine persistenten Hilfsmodelle zählen nicht, die
laden auf ihren eigenen Platz, auch nicht, wenn für sie geräumt wird. Antwortet der Modellserver nicht: `502`. Das ist keine grüne Ampel.

### `POST /hold`, `POST /hold/grant`, `DELETE /hold` — Karten für jemanden freihalten

```json
{"owner": "vllm-research", "reason": "Spielen", "model": null}
```

Jemand hält die Karten dieses Rechners für sich frei (ein Spiel, andere Arbeit). Der Halt hat einen `owner`, und es
gibt immer nur einen. GPU-Orc legt ihn in `hold_file` ab (Config, Standard `~/.local/state/gpu-orc/hold.json`; er
übersteht einen Neustart von GPU-Orc und des Rechners). Eine Hold-Datei älteren Formats (ohne `model`,
`model_server`, `granted_by`) verhindert den Start von GPU-Orc; es gibt keine Migration, die Datei wird entfernt.

Body: genau `owner` und `reason` (nicht leere Strings) und `model`, als Schlüssel Pflicht. `model` ist ein Modell
des Modellservers (llama-swap), das der Eigentümer braucht und das geladen bleiben muss, oder ausdrücklich `null`
(er braucht keins, z. B. ein Spiel). Das Modell wird beim Anfordern wie bei `POST /free` geprüft: unbekannt `400`.
`model_server` (optional, nur mit einem Modell) nennt den Modellserver nach derselben Regel wie bei `POST /free`:
nur nötig, wenn mehrere Modellserver konfiguriert sind. Der Halt speichert `model` und den aufgelösten
`model_server` (`null` ohne Modell). Jeder andere Body: `400`.

- Ein neuer Halt ist ANGEFORDERT (`"state": "requested"`, `"free": null`, `"granted_by": null`), die Antwort kommt
  sofort und zeigt den Zustand zu diesem Zeitpunkt; angefasst wird noch nichts:

  ```json
  {"hold": {"owner": "vllm-research", "reason": "Spielen", "model": null, "model_server": null,
            "since": "2026-11-05T09:12:00+01:00", "state": "requested", "free": null, "granted_by": null}}
  ```

- **Die Karten sind frei**, wenn auf keinem Modellserver ein Hauptmodell geladen ist oder lädt. Ein Hauptmodell ist
  ein Modell des Modellservers, das nicht persistent ist, außer dem Modell, das der Halt selbst anfordert (gleicher
  Server und Name; als ladend zählt nur ein anderes Modell). Persistente Hilfsmodelle und alle anderen Verbraucher
  (STT-Suite, TTS-Suite, AIfred) zählen nicht, das Räumen nimmt sie mit. Antwortet ein Modellserver (oder nvidia-smi)
  nicht, gilt das als nicht frei; GPU-Orc meldet es je Wechsel einmal auf stderr, nicht bei jedem Blick.
- Ein Thread in GPU-Orc, der so lange läuft wie der Dienst, macht jede Änderung des Halts. Bei einem angeforderten
  Halt prüft er, ob die Karten frei sind: sofort nach dem Anfordern (ein neuer Halt auf freien Karten wird gleich
  nach der Antwort wirksam) und dann alle `hold_check_interval_s` (Config, Standard 30 s). `POST /hold`, `POST
  /hold/grant` und `DELETE /work/<owner>` wecken ihn. Sind die Karten frei, räumt er alle Karten wie `POST /free`
  mit `"gpus": "all"`, `keep` auf das angeforderte Modell (ein laufendes angefordertes Modell bleibt, im Ergebnis
  `kept`) und dem Grund `hold: <owner>: <reason>` und macht den Halt AKTIV, mit dem Räumen in `free` (`{"...": "wie
  POST /free"}`). Nur genau der Halt, für den geräumt wurde, wird aktiv: wurde er unterdessen aufgehoben (und
  vielleicht neu angefordert), zählt das Räumen für nichts, und der neue Halt wartet auf sein eigenes. In einem
  `GET` räumt nichts. Gemeldete Arbeit (`POST /work`) beeinflusst einen nicht freigegebenen Halt nicht. Aufrufer
  warten auf AKTIV mit `hold_and_wait` (Clients). Scheitert der Thread selbst außerhalb des Räumens (die Hold-Datei
  lässt sich nicht schreiben, ein Fehler), beendet sich GPU-Orc mit dem Traceback auf stderr, statt weiter zu
  antworten, während Halte nie aktiv werden; `Restart=on-failure` der Unit startet ihn neu.
- Ein nur ANGEFORDERTER Halt verweigert für sich niemanden: das Hauptmodell, das ihn aufhält, läuft weiter, und was
  ihm dient, darf noch laden. Wer für sich etwas Großes laden würde (eine Agent-Orc-Sitzung), behandelt
  `hold_requested` wie `held`. Was in den Sekunden zwischen „Karten frei“ und AKTIV startet, während geräumt wird,
  bleibt auf den Karten: eine bekannte Lücke, kurz von Natur aus.
- Endet das Räumen mit einem Problem (ein Verbraucher beschäftigt, Speicher übrig oder ein Fehler beim Räumen, dann
  eine Aktion `gpu-orc` / `all cards` / `error` mit dem Detail), ist der Halt trotzdem aktiv und behält dieses
  `free` (auch in der Hold-Datei und in `GET /status`); hinter dem Rücken des Eigentümers versucht niemand es neu.
- **Erneutes Anfordern** durch denselben Eigentümer ändert nichts und gibt den bestehenden Halt zurück, auch wenn
  `reason` oder `model` abweichen. **Ein anderer Eigentümer**, solange irgendein Halt besteht (angefordert oder
  aktiv): `409`, `{"error": "...", "owner": "<Halter>", "state": "...", "since": "..."}`.

`POST /hold/grant` mit `{"owner": "<Eigentümer des Halts>", "by": "<wer freigibt>"}` (beides nicht leere Strings) ist
der Klick des Menschen, der die Karten trotz geladenem Hauptmodell räumt. Kein Halt: `404`. Halt eines anderen
Eigentümers: `409` (gleicher Rumpf wie oben). Ein Halt, der schon aktiv ist: `200`, der Halt unverändert. Sonst
gilt genau dieser Halt als freigegeben: `"granted_by": "<by>"`, in der Hold-Datei (die Freigabe übersteht einen
Neustart; ein aufgehobener und neu angeforderter Halt beginnt mit `granted_by` `null`), der Thread wird geweckt,
GPU-Orc schreibt `[hold] GRANTED by <by> for <owner> (...)` auf stderr, und die Antwort ist `{"hold": {...}}`
(noch angefordert). Der Thread räumt einen freigegebenen Halt, ohne zu prüfen, ob die Karten frei sind, aber erst,
wenn keine Arbeit eines anderen als des Eigentümers gemeldet ist (`POST /work`): eine laufende Antwort von AIfred
wird nie abgebrochen, auch nicht durch einen Klick; `DELETE /work/<owner>` weckt den Thread, sonst schaut er alle
`hold_check_interval_s` nach. Die Arbeit des Eigentümers selbst zählt nicht, deshalb für `/work` und `/hold`
**denselben Eigentümernamen** benutzen.

`GET /loading` und `GET /status` tragen `"held"` (der Halt, nur solange AKTIV) und `"hold_requested"` (der Halt, nur
solange ANGEFORDERT, mit `granted_by`). Wer etwas Neues auf eine Karte legen würde (ein Bildbeschreiber eines anderen
Rechners, ein von dort gestarteter TTS-Container), lässt sie in Ruhe, solange `held` gesetzt ist. Was ein Aufrufer
lädt, ohne zu fragen, beendet GPU-Orc nicht.

`DELETE /hold` mit `{"owner": "...", "force": false}` gibt frei (ein angeforderter Halt wird zurückgezogen);
Antwort `{"hold": null}`. Nur der Eigentümer darf: ein anderer bekommt `409` (gleicher Rumpf wie oben).
`"force": true` hebt den Halt von irgendwem auf, für den Menschen, wenn ein Programm die Karten haltend gestorben
ist; GPU-Orc protokolliert es auf stderr mit beiden Eigentümern. `owner` ist auch mit `force` Pflicht. Ohne Halt
ist die Antwort `{"hold": null}`.

### `POST /work`, `GET /work`, `DELETE /work/<owner>` — hier läuft ein Job

```json
{"owner": "aifred", "reason": "AIfred: 2 running turn(s)", "ttl_s": 60}
```

Wer auf diesem Rechner einen Job laufen lässt, meldet ihn, damit der, der den Rechner neu starten würde (der
wöchentliche Neustart), vorher fragen kann. Eine Meldung ist eine Lease: sie endet mit `DELETE /work/<owner>`
(Besitzer URL-kodiert; Antwort `{"ended": true, "work": [...]}`, `ended: false`, wenn sie schon weg war) oder von
selbst, wenn `ttl_s` Sekunden ohne erneute Meldung desselben Besitzers vergehen. Wer erneut meldet, erneuert die
Lease, behält den Start und übernimmt den neuen Grund. Nichts wird geräumt oder gehalten: die Karten bleiben,
wie sie sind. Die Meldungen liegen nur im Speicher. `GET /work` antwortet sofort (ohne nvidia-smi) mit
`{"work": [{"owner", "reason", "running_for_s", "expires_in_s"}]}`; `GET /status` führt dieselbe Liste als `work`.
Ein Body, der nicht genau `owner`, `reason` (nicht-leere Texte) und `ttl_s` (positive Zahl) enthält: `400`.

`gpu-orc run-if-idle [--config PFAD] -- <Befehl>` fragt `GET /work` auf diesem Rechner und führt den Befehl nur
aus, wenn die Liste leer ist (sein Exit-Code ist der des Befehls). Läuft Arbeit, überspringt es den Befehl und
endet mit 0: kein Warten, keine Obergrenze, der nächste Termin des Cron-Jobs versucht es wieder. Antwortet
GPU-Orc nicht, überspringt es den Befehl ebenfalls und endet mit 3 und einer Meldung auf stderr. Der wöchentliche
Neustart lautet `30 5 * * 1,3,5 root /home/mp/.local/share/gpu-orc/venv/bin/gpu-orc run-if-idle -- /sbin/reboot`.

### `GET /status` — wer hält was

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
        {"consumer": "llama-swap", "item": "<modell>", "memory_mib": null, "busy": null, "persistent": false},
        {"consumer": "stt-suite", "item": "gpu-worker", "memory_mib": 3100, "busy": false, "persistent": null},
        {"consumer": "tts-suite", "item": "<engine>", "memory_mib": 6024, "busy": false, "persistent": null}
      ],
      "unattributed_mib": null
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
  - `busy` melden nur die Suiten (Transkription, Vorlesung, Start), `persistent` nur der Modellserver
    (Mitglied einer Gruppe mit `persistent: true`). Das jeweils andere Feld ist `null`.
- `unattributed_mib` ist der belegte Speicher minus dem, was die Halter melden. Er ist nur dann eine
  Zahl, wenn jeder Halter auf der Karte seinen Speicher meldet, sonst `null`. Auf einer Karte ohne
  Halter ist es der ganze belegte Speicher. GPU-Orc fasst Unbekanntes nie an.
- `errors` nennt Verbraucher, die nicht sagen konnten, was sie halten
  (`[{"consumer": …, "detail": …}]`); ihre Halter fehlen dann in `holders`.
- `500`, wenn nvidia-smi scheitert.

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
| `error` | Dienst nicht erreichbar oder unerwartete Antwort; llama-swap: Entladen gescheitert oder Modell danach noch geladen. Ein Verbraucher, der nicht sagen konnte, was er hält, steht als `error` mit leerem `item`, aber nur, wo er etwas halten könnte (siehe `free`) |
| `would_release` | Nur bei `dry_run` |

`memory` ist der belegte Speicher je betroffener Karte vor der Freigabe und nach der Nachprüfung
(`after_mib` ist `null` bei `dry_run` und wenn die Anfrage aus der Schlange nicht mehr drankam).

`free` ist `true`, wenn nach den Aktionen kein Element, das die Anfrage freiräumen sollte, mehr
gemeldet wird, keine Aktion `not_freed` oder `error` ist und kein Verbraucher, der nicht sagen konnte,
was er hält, auf den Karten sitzen könnte. Das kann er nur, wo eine Zielkarte Speicher belegt, den kein
meldender Verbraucher erklärt (mindestens `min_freed_mib`, oder irgendein Speicher neben einem Halter,
der keinen meldet): Auf leeren Karten oder solchen, deren Speicher die meldenden Verbraucher erklären,
hält ein gestoppter Dienst nichts, und die Anfrage hängt nicht an ihm. Ein Verbraucher, den `keep` als
Ganzes schützt, zählt nie (die Kalibrierung hält llama-swap an und lässt dann den Rest freiräumen).
`status.errors` nennt jeden Verbraucher, der nichts melden konnte. Unbekannter Speicher allein macht `free` nicht falsch, solange jeder Verbraucher melden konnte; er steht im
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
                                            # (höchstens so viel, wie vorher belegt war); auch so viel
                                            # unerklärter Speicher, bevor ein unerreichbarer Dienst zählt
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
  - name: aifred
    type: gpu-release
    url: http://127.0.0.1:8002/api          # die Gesichtserkennung (face-worker)
    token_file: ~/.config/gpu-orc/aifred.token   # AIfreds GPU_RELEASE_API_TOKEN
    busy_wait_s: 5                          # eine Erkennung dauert Sekundenbruchteile
  - name: llama-swap
    type: llama-swap
    url: http://127.0.0.1:11435
    config_file: ~/.config/llama-swap/config.yaml   # Karten je Modell, persistent-Gruppen
    no_eviction_models: ["*-vlm-*"]         # verdrängen beim Laden keine dauerhaften Modelle
```

Ohne Angabe gelten `busy_wait_s: 0` und `force_after_wait: false`. Mit `token_file` schickt GPU-Orc einem
`gpu-release`-Dienst bei jeder Anfrage `Authorization: Bearer <token>` (AIfred verlangt das, die Suiten nicht). Unbekannte Schlüssel eines Verbrauchers
und doppelte Verbrauchernamen lassen GPU-Orc nicht starten. Die Config von llama-swap liest GPU-Orc bei jeder
Anfrage neu, sie darf sich im Betrieb ändern.

## Aufrufer

- **AIfred** ruft GPU-Orc seit dem 30.10.2026 vor jedem Laden eines Modells auf (`POST /free {"model": …}`,
  vor einer Kalibrierung die `gpus`-Form mit `keep`), fragt die Ampel (`GET /loading`), bevor
  sein Bildbeschreiber oder seine Gesichtserkennung etwas auf eine Karte legt, und hält die Karten über
  `POST /hold` / `DELETE /hold` aus seiner Statusleiste frei. Antwortet GPU-Orc nicht, lädt AIfred trotzdem und
  meldet es laut. Mit dem Vue-Umbau ist AIfred auch Verbraucher (`gpu-release`, Element `face-worker`, mit Token).
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
