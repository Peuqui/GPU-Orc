# Gemeinsame Freigabe-Schnittstelle der Suiten (GPU-Orc 0.2.0)

Stand 24.10.2026, mit Agent-Orc abgestimmt und in GPU-Orc 0.2.0, der STT-Suite und der TTS-Suite umgesetzt.

STT-Suite und TTS-Suite geben ihre Grafikkarten auf dieselbe Weise frei: gleiche Endpunkte, gleiche
Antwortcodes, gleiches `force`, gleiche Ergebnisfelder. GPU-Orc braucht dafür einen einzigen
Verbraucher-Typ (`gpu-release`) und einen Ablauf: warten bis `busy_wait_s`, dann `force`, wenn die
Config es für diesen Verbraucher erlaubt.

**„Freigegeben“ heißt bei beiden:** Der Prozess mit dem CUDA-Kontext ist beendet. Bei der STT-Suite ist
das der Kindprozess des GPU-Workers (und der Diarisierung), bei der TTS-Suite der Container der
Engine. Ein `/unload` *im* Container reicht nicht: PyTorch behält seinen Speicherpool, die Karte wird
nie ganz leer.

## `GET /gpu/holders`

Was der Dienst auf den Karten dieses Rechners hält; Engines auf anderen Rechnern stehen nicht darin.

```json
{"holders": [
  {"item": "gpu-worker", "gpu_uuids": ["GPU-…"], "memory_mib": {"GPU-…": 3100}, "busy": true}
]}
```

- `item`: STT `gpu-worker` / `diarize-worker`, TTS der Name der Engine.
- `gpu_uuids`: die Karten, die der Prozess sehen kann, aus seiner Konfiguration statt aus einer
  Messung, also immer bekannt. STT: die Karte, auf der der Worker gestartet wurde. TTS: die Karte aus
  `NVIDIA_VISIBLE_DEVICES` des Containers (`docker inspect`; die Suite startet jeden Engine-Container mit
  `NVIDIA_VISIBLE_DEVICES=<uuid>`). Nur ein Container, der wirklich alle Karten sieht, steht auf allen.
- `memory_mib`: was der Dienst selbst gemessen hat, je Karte; `null`, wenn die Messung fehlt. Die Karte
  hängt nicht davon ab.
- `busy`: STT eine laufende Transkription, TTS eine laufende Vorlesung, die die Engine hält.

## `POST /gpu/release`

```json
{"items": ["gpu-worker"], "force": false, "reason": "GPU-Orc: Modell X wird geladen"}
```

`items` ist eine Liste oder `"all"` (alles, was `/gpu/holders` nennt). `reason` ist Pflicht und steht
im Log des Dienstes und, bei `force`, in der Meldung an die abgebrochenen Clients.

Antwort, immer mit einem Ergebnis je Element:

```json
{"results": [
  {"item": "gpu-worker", "outcome": "released", "detail": "worker PID 4711 ended"},
  {"item": "qwen3local", "outcome": "busy", "detail": "a reading holds it"}
]}
```

| `outcome` | Bedeutung |
|---|---|
| `released` | Der Prozess mit dem CUDA-Kontext ist nachweislich beendet: STT `is_alive()` falsch, TTS Container nicht mehr `running` laut `docker inspect` |
| `not_running` | Es lief nichts, nichts zu tun |
| `busy` | Ohne `force`: Das Element arbeitet und wurde in Ruhe gelassen |
| `failed` | Beenden versucht, aber der Nachweis fehlt (`detail` sagt, was noch läuft) |

HTTP-Code: `200`, wenn alle Elemente `released` oder `not_running` sind; `409`, wenn eines `busy` ist
und keines `failed`; `500`, wenn eines `failed` ist. Elemente, die nicht beschäftigt sind, gibt der
Dienst auch bei einem `409` frei.

**`force: true`**, gleich bei beiden:
1. Die Arbeit des Elements mit Grund abbrechen.
2. Den Prozess bzw. den Container beenden. Hält er die Frist nicht ein, folgt das harte Ende in der
   Suite selbst: STT `kill`, TTS `docker kill` nach dem Stopp-Timeout. GPU-Orc beendet nie selbst etwas.
3. Erst nach dem Nachweis wie oben `released` melden.

### Besonderheiten, benannt statt still

- **TTS-Abbruch:** Jede Vorlesung, die die Engine hält, endet mit dem Fehler-Frame
  `{"error": "reading_aborted", "detail": "Vorlesung abgebrochen, die Karte wird gebraucht: <reason>"}`.
  Das gilt für `POST /speak` und `/speak/stream`. Clients zeigen das so an und lesen auf Wunsch neu
  vor; der Text ist bei ihnen. Neu für die Clients ist nur der Code `reading_aborted`; AIfred setze
  ich um, Agent-Orc seine Seite.
- **STT-Abbruch:** Heute geht eine laufende Transkription bei `force` verloren (Fehler „GPU worker
  was unloaded mid-transcription“).
  - Neu gilt der Abbruch als „GPU nicht verfügbar“. Eine Anfrage mit `device=auto` läuft dann auf der
    CPU weiter, das Diktat bleibt erhalten, nur langsamer.
  - Eine Anfrage mit `device=cuda` bekommt `503` mit Grund, der Client kann auf der CPU wiederholen.
  - Die Suite behält die Aufnahme nicht; sie liegt beim Client, der sie hochgeladen hat.
- **Ein Weg für die GPU-Freigabe:** Lokale Karten gibt es nur noch über `/gpu/release` frei.
  - STT `/unload` behält nur, was `/gpu/release` nicht kann: die CPU-Modelle entladen
    (`device=cuda` und der GPU-Teil von `device=all` entfallen). Die Oberfläche der STT-Suite ruft für
    die GPU `/gpu/release`.
  - TTS `/release` steuert nur noch Engines auf *anderen* Rechnern. Lokale Engines, auch für AIfreds
    TTS-Plan und den Burn-in, laufen über `/gpu/release`.
  - Die Clients (AIfred, die STT-Oberfläche) stelle ich um; `lclaude-free-gpus` entfällt mit der
    Umstellung von Agent-Orc.

## GPU-Orc 0.2.0: Nachprüfen statt glauben

Nach jeder Freigabe prüft GPU-Orc das Ergebnis selbst, statt der Antwort zu glauben:

1. **Status:** Der Dienst nennt das Element nicht mehr: `/gpu/holders`, bei llama-swap `/running`.
2. **Prozess bzw. Container weg:** Das ist bei den Suiten Teil von `released` (siehe oben); GPU-Orc
   sieht es über den Status.
3. **Speicher:** nvidia-smi je Karte, vorher und nachher, mit Frist nachgemessen. Der Treiber gibt den
   Speicher oft erst 1–2 s nach Prozessende zurück; GPU-Orc misst bis `verify_timeout_s`
   (Config, Standard 30 s) alle 0,5 s nach.
   - Hat ein Element seinen Speicher gemeldet, muss der Speicher seiner Karte um mindestens die
     Hälfte davon sinken.
   - Sonst muss er auf seinen Karten zusammen um mindestens `min_freed_mib` sinken (Config,
     Standard 256; ein CUDA-Kontext allein belegt mehr).
   - Das geht auch unter WSL2.
   - **Zeitrahmen:** Die Nachmessung liegt *innerhalb* von `free_timeout_s`, nicht obendrauf. Sie endet
     bei `verify_timeout_s` oder beim Gesamtlimit, je nachdem, was früher kommt. `caller_timeout_s` aus
     `/health` bleibt damit richtig.

`released` und `released_forced` gibt es nur nach bestandener Prüfung. Sonst kommt der neue Ausgang
**`not_freed`**. Sein `detail` nennt den Befund aus Status und Prozess getrennt vom Speicherbefund,
zum Beispiel „Status: nicht mehr gemeldet; Speicher GPU-…: 31200 → 31150 MiB, erwartet −1550“.

**Bekannte Grenze:** Lädt während der Nachmessung etwas anderes auf dieselbe Karte (ein Modell, das
parallel startet, ein Messlauf), sinkt der Speicher nicht, obwohl freigegeben wurde. Dann steht
`not_freed` mit dem Befund „Prozess weg, Speicher nicht gesunken“. Das ist gewollt: lieber laut als
still.

Die Antwort von `/free` bekommt die Messwerte je betroffener Karte:

```json
"memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}]
```

`free` beruht auf dieser Nachmessung. Es ist true, wenn kein Element, das die Anfrage freiräumen
sollte, noch gemeldet wird, keine Aktion `not_freed` oder `error` ist und alle Speicherprüfungen
bestanden sind.

**Verbraucher-Typen in 0.2.0:**
- `gpu-release` für STT- und TTS-Suite: `busy_wait_s`, `force_after_wait`.
- `llama-swap` wie bisher.
- `stt-suite` und `tts-suite` entfallen.

**Vorgabe für die Config, als Wert und nicht im Code** (Peuqui legt sie je Rechner fest):
- TTS: 5 s warten, dann `force`. Ein Abbruch kostet wenig, der Text ist da; ein Kaltstart würde nur zu
  Ende laden, um danach abgebrochen zu werden.
- STT: 30 s warten, dann `force`. Ein Diktat ist auf der GPU in Sekunden fertig; eine lange Aufnahme mit
  `device=auto` transkribiert die Suite danach mit derselben Datei von vorn auf der CPU, es geht nichts
  verloren (Peuqui 25.10.: keine zehn Minuten warten, bei der TTS sowieso nicht).

**Tests:** Ein Verbraucher, der „ok“ sagt, aber nichts freigibt (Status meldet ihn weiter, oder der
Speicher sinkt nicht), wird als `not_freed` erkannt. Ebenso die Suiten: Ein Worker oder Container,
der den Stopp überlebt, ergibt `failed`, nie `released`.
