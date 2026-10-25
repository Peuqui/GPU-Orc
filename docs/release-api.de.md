# Gemeinsame Freigabe-Schnittstelle der Suiten

[English](release-api.md)

Stand: GPU-Orc 0.3.0, STT-Suite und TTS-Suite auf ihrem aktuellen Hauptzweig.

STT-Suite und TTS-Suite geben ihre Grafikkarten auf dieselbe Weise frei: gleiche Endpunkte, gleiche
Antwortcodes, gleiches `force`, gleiche Ergebnisfelder. GPU-Orc braucht dafür einen einzigen
Verbraucher-Typ (`gpu-release`) und einen Ablauf: warten bis `busy_wait_s`, dann `force`, wenn die
Config es für diesen Verbraucher erlaubt.

**„Freigegeben“ heißt bei beiden:** Der Prozess mit dem CUDA-Kontext ist beendet. Bei der STT-Suite ist
das der Kindprozess des GPU-Workers (und der Diarisierung), bei der TTS-Suite der Container der
Engine. Ein `/unload` *im* Prozess reicht nicht: PyTorch behält seinen Speicherpool, die Karte wird
nie ganz leer.

## `GET /gpu/holders`

Was der Dienst auf den Karten dieses Rechners hält; Engines auf anderen Rechnern stehen nicht darin.

```json
{"holders": [
  {"item": "gpu-worker", "gpu_uuids": ["GPU-…"], "memory_mib": {"GPU-…": 3100}, "busy": true}
]}
```

- `item`: bei der STT-Suite `gpu-worker` oder `diarize-worker`, bei der TTS-Suite der Name der Engine.
- `gpu_uuids`: die Karten, die der Prozess sehen kann, aus seiner Konfiguration statt aus einer
  Messung, also immer bekannt. STT: die Karte, auf der der Worker gestartet wurde. TTS: die Karten aus
  der Konfiguration des Containers (`DeviceRequests` oder `NVIDIA_VISIBLE_DEVICES`, laut
  `docker inspect`). Nur ein Container, der wirklich alle Karten sieht, steht auf allen.
- `memory_mib`: was der Dienst selbst gemessen hat, je Karte; `null`, wo die Messung fehlt (STT: für
  die Diarisierung). Die Karte hängt nicht davon ab.
- `busy`: STT eine laufende Transkription, TTS eine laufende Vorlesung, die die Engine hält, oder ein
  Start der Engine, der gerade läuft.

## `POST /gpu/release`

```json
{"items": ["gpu-worker"], "force": false, "reason": "GPU-Orc: Modell X wird geladen"}
```

`items` ist eine Liste oder `"all"` (alles, was `/gpu/holders` nennt). `reason` ist Pflicht und steht
im Log des Dienstes und, bei `force`, in der Meldung an die abgebrochenen Clients. Ein unbekanntes
oder doppeltes Element, eine leere Liste oder ein leerer Grund werden abgelehnt.

Antwort, immer mit einem Ergebnis je Element:

```json
{"results": [
  {"item": "gpu-worker", "outcome": "released", "detail": "worker PID 4711 ended"},
  {"item": "<engine>", "outcome": "busy", "detail": "a reading holds it"}
]}
```

| `outcome` | Bedeutung |
|---|---|
| `released` | Der Prozess mit dem CUDA-Kontext ist nachweislich beendet: STT der Worker-Prozess lebt nicht mehr, TTS der Container ist laut Docker weg oder beendet |
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

### Besonderheiten

- **TTS-Abbruch:** Jede Vorlesung, die die Engine hält, endet mit
  `{"error": "reading_aborted", "detail": "…<reason>"}`: als Fehler-Frame, wenn schon Audio kam, sonst
  als Antwort mit HTTP 503. Das gilt für `POST /speak` und `/speak/stream`. Die Vorlesung geht nicht
  auf eine andere Engine über; Clients zeigen den Abbruch an und lesen auf Wunsch neu vor, der Text ist
  bei ihnen.
- **TTS-Start:** Mit `force` wird ein laufender Start der Engine zurückgezogen: Er gibt vor
  `docker compose up` oder bei seiner nächsten Prüfung während des Ladens auf, danach wird der
  Container abgeschossen. Wer auf diesen Start wartet, bekommt `reading_aborted`, vor dem ersten Audio
  als HTTP 503. Ein ausdrücklicher Start über `/engines/<engine>/start` scheitert mit „not started
  (withdrawn)“. Gemessen auf dem Mini mit `busy_wait_s` 5: ein Kaltstart in 8,2 s freigegeben
  (`released_forced`), eine warme Vorlesung in 10,3 s.
- **STT-Abbruch:** Eine laufende Transkription gilt nach `force` als „GPU nicht verfügbar“.
  - Eine Anfrage mit `device=auto` läuft auf der CPU weiter; das Diktat bleibt erhalten, nur langsamer.
  - Eine Anfrage mit `device=cuda` bekommt `503` mit dem Grund; der Client kann auf der CPU wiederholen.
  - Die Suite behält die Aufnahme nicht; sie liegt beim Client, der sie hochgeladen hat.
- **Ein Weg für die GPU-Freigabe:** Lokale Karten gibt es nur über `/gpu/release` frei.
  - STT `/unload` entlädt nur noch die CPU-Modelle (`device=cuda` gibt `400`). Die Oberfläche der
    STT-Suite ruft für die GPU `/gpu/release`.
  - TTS `/release` steuert nur Engines auf *anderen* Rechnern. Lokale Engines laufen über
    `/gpu/release`.

## Was GPU-Orc daraus macht: nachprüfen statt glauben

Nach jeder Freigabe prüft GPU-Orc das Ergebnis selbst, statt der Antwort zu glauben:

1. **Status:** Der Dienst nennt das Element nicht mehr (`/gpu/holders`, bei llama-swap `/running`).
2. **Prozess bzw. Container weg:** Das ist bei den Suiten Teil von `released` (siehe oben); GPU-Orc
   sieht es über den Status.
3. **Speicher:** nvidia-smi je Karte, vorher und nachher. Der Treiber gibt den Speicher oft erst 1–2 s
   nach Prozessende zurück; GPU-Orc misst bis `verify_timeout_s` (Config, Standard 30 s) alle 0,5 s nach.
   - Hat ein Element seinen Speicher gemeldet, muss der Speicher seiner Karte um mindestens die
     Hälfte davon sinken.
   - Sonst muss er auf seinen Karten zusammen um mindestens `min_freed_mib` sinken (Config,
     Standard 256; ein CUDA-Kontext allein belegt mehr), aber nie um mehr, als vorher darauf belegt
     war: Eine Karte, die vorher und nachher leer war (ein Start, der zurückgezogen wurde, bevor er
     etwas geladen hatte), besteht.
   - Das geht auch unter WSL2.
   - Die Nachmessung liegt *innerhalb* von `free_timeout_s`, nicht obendrauf. Sie endet bei
     `verify_timeout_s` oder beim Gesamtlimit, je nachdem, was früher kommt; `caller_timeout_s` aus
     `/health` bleibt damit richtig.

`released` und `released_forced` gibt es nur nach bestandener Prüfung, sonst `not_freed`. Sein
`detail` nennt den Befund des Status getrennt vom Speicherbefund, zum Beispiel
`status: no longer reported; memory: GPU-… 31200 → 31150 MiB, expected −1550`. Meldet die Suite selbst
`failed`, wird daraus ebenfalls `not_freed` (`detail`: `service: …`).

**Bekannte Grenze:** Lädt während der Nachmessung etwas anderes auf dieselbe Karte (ein Modell, das
parallel startet, ein Messlauf), sinkt der Speicher nicht, obwohl freigegeben wurde. Dann steht
`not_freed` mit dem Befund „nicht mehr gemeldet, Speicher nicht gesunken“. Das ist gewollt: lieber
laut als still.

Die Antwort von `/free` enthält die Messwerte je betroffener Karte:

```json
"memory": [{"index": 0, "uuid": "GPU-…", "before_mib": 31200, "after_mib": 1100}]
```

## Wartezeiten in der Config

Peuqui legt sie je Rechner fest, als Wert in der Config und nicht im Code. Heute auf dem Mini:

- **TTS:** 5 s warten, dann `force`. Ein Abbruch kostet wenig, der Text ist beim Client; ein
  Kaltstart würde nur zu Ende laden, um danach abgebrochen zu werden.
- **STT:** 30 s warten, dann `force`. Ein Diktat ist auf der GPU in Sekunden fertig; eine lange
  Aufnahme mit `device=auto` transkribiert die Suite danach mit derselben Datei von vorn auf der CPU,
  es geht nichts verloren.

## Tests

Ein Verbraucher, der „ok“ sagt, aber nichts freigibt (der Status meldet ihn weiter, oder der Speicher
sinkt nicht), wird als `not_freed` erkannt. Ebenso die Suiten: Ein Worker oder Container, der den
Stopp überlebt, ergibt `failed`, nie `released`.
