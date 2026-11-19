# Plan: ein gemeinsamer Client für GPU-Orc

Stand 18.11.2026: **umgesetzt und auf dem Mini ausgerollt**. Alle sechs Programme nutzen die Kopie
(`clients/consumers.json`), kein Abbruchkriterium trat ein. Offen: Agent-Orc auf Aragon nachziehen (Aragon war aus).
Abweichungen vom Entwurf: `model_server` kam als Protokollfeld in den Client (Agent-Orc, AIfred); `/status` hat eine
eigene Zeitgrenze (25 s, per Test an GPU-Orcs innere Grenzen gekoppelt), `/loading` bleibt bewusst bei 5 s (langsam
= rot); die STT-Suite beachtet jetzt wie die TTS-Suite auch gehaltene Karten.

## Ziel und Grenzen

Heute spricht jedes Programm das Protokoll von GPU-Orc mit eigenem Code. Was „die Karten sind frei“ heißt, welche
Ergebnisse in Ordnung sind, woher die Zeitgrenze für `/free` kommt und wie ein Problem lautet, steht dadurch an fünf
Stellen in leicht verschiedener Form. Jede Änderung an GPU-Orc (zuletzt 0.8.0 Token, 0.8.1 gestoppter Dienst, 0.9.0
Arbeitsmeldung) muss überall einzeln nachgezogen werden.

Ziel: **einmal** Code für das Protokoll, je Sprache eine Datei, gegen den echten GPU-Orc-Server getestet, per Skript an
die Programme verteilt.

Ausdrücklich **nicht**:
- Keine neue Aufgabe für GPU-Orc. Er bleibt der Räumer und die Ampel, der Server ändert sich nicht. Keine Platzwahl,
  keine Profilwahl, keine Modellnamen (das war der am 26.10. verworfene Orchestrator).
- Keine Suche nach laufenden Profil-Varianten im Client. Das ist llama-swap-Sache, die Regeln sind je Programm
  verschieden (PharmaLens/HaemoTrace: Variante mit `--mmproj`; AIfred: eigene Namensregeln in `vision_routing`;
  Agent-Orc: `model_runs`), sie bleiben, wo sie sind.
- Kein Zwang für fremden Code (Stable Diffusion) und keine Shell-Aufrufe (vllm-research benutzt `curl`).
- AI-ATC bekommt GPU-Orc nicht nebenbei; ob es vor dem Laden räumen soll, ist eine eigene Entscheidung.

## GPU-Orc abschaltbar (Peuqui 18.11.2026)

Jedes Programm muss ohne GPU-Orc laufen können, wenn es woanders installiert ist, wo es keinen gibt oder keiner gewollt
ist. Der Schalter ist überall derselbe: **die Adresse von GPU-Orc. Leer oder nicht gesetzt heißt aus.** Ein Modellaufrufer
lädt dann direkt, eine Suite wählt ihre Karte ohne Ampel, also so, wie es vor GPU-Orc war. Ein zweiter Ein/Aus-Schalter
daneben kommt nicht dazu (er könnte der Adresse widersprechen).

- Bewusst aus (Adresse leer): kein Aufruf, keine Warnung.
- Eingetragen, aber nicht erreichbar: laute Meldung wie heute, das Verhalten danach wie heute je Programm.
- Die Entscheidung trifft der Anschluss im Programm, bevor er den Client ruft. Der Client selbst kennt keinen
  Aus-Zustand.

Heute schon so: STT-Suite (`STT_GPU_ORC_URL` leer), TTS-Suite (`gpu_orc_url` nicht gesetzt), Agent-Orc (`gpu_orc` je
Backend weggelassen), PharmaLens (`gpuOrc` weggelassen). Neu nötig: **AIfred** (`GPU_ORC_URL` hat einen festen
Standardwert) und **HaemoTrace** (`gpu_orc_url` immer gesetzt). Je Programm ein Test: Adresse leer, kein Aufruf an
GPU-Orc, das Modell wird trotzdem angefragt.

## Bestand

| Programm | Sprache | nutzt | Datei | Zeilen |
|---|---|---|---|---|
| AIfred | Python, httpx | free (Modell und „alles außer Modellen“), loading, hold, work, status | `aifred/lib/gpu_orc_client.py` (+ `gpu_orc_work.py`) | 230 (+74) |
| Agent-Orc | Python, urllib | free, loading/hold | `src/agent_orc/gpuorc.py` (+ `card_gate.py` als Proxy, bleibt) | 107 |
| TTS-Suite | Python, urllib | loading (mit Token für Aragon) | `src/tts_suite/placement.py` (Teil) | ~30 |
| STT-Suite | Python, urllib, im Container | loading | `server.py` (Teil) | ~25 |
| PharmaLens | JavaScript, axios | free | `src/backend/services/gpu-orc.js` | 40 |
| HaemoTrace | Python, httpx async | free | `backend/services/gpu_orc.py` | 54 |

Gemeinsam heißt dabei nur der Protokollteil. Was jedes Programm daraus macht (AIfred loggt mit Kategorie und meldet der
Statusleiste, Agent-Orc weist Sitzungen ab, die Suiten lehnen einen Start ab), bleibt in seinem eigenen Code.

## Zuschnitt

**Python: `clients/python/gpu_orc_client.py`** (Ziel unter 200 Zeilen, nur Standardbibliothek: `urllib`, `json`,
`dataclasses`). So läuft dieselbe Datei im STT-Container, in den Suiten, in Agent-Orc, AIfred und HaemoTrace, ohne neue
Abhängigkeit. Asynchrone Aufrufer (HaemoTrace, AIfred) rufen sie über `asyncio.to_thread`.

- `GpuOrc(url, token=None)`: der Rechner und sein Token.
- `free_for_model(model, reason, keep=())` und `free_gpus(gpus, reason, keep=())`: `/health` liefert die Zeitgrenze
  (`caller_timeout_s`), dann `/free`; Rückgabe `FreeResult` (`free`, `already_running`, `actions`) mit `problems()`:
  die Liste der Texte, die heute jeder selbst baut (Ergebnisse außer `released`, `released_forced`, `kept`; „nicht
  frei“).
- `loading()` gibt `TrafficLight` (`loading`, `held`, `red`).
- `hold(owner, reason)`, `hold_and_wait(owner, reason, poll_s, timeout_s)`, `release_hold(owner, force)` (seit 0.10.0, siehe docs/interface.md), `work_begin(owner, reason, ttl_s)`, `work_end(owner)`, `status()`.
- Ein Fehler (nicht erreichbar, falsche Antwort) ist `GpuOrcError`. Ob der Aufrufer dann trotzdem lädt (AIfred,
  PharmaLens, HaemoTrace) oder ablehnt (Suiten), entscheidet er selbst, wie heute.

**JavaScript: `clients/js/gpu-orc-client.js`**, nur was PharmaLens braucht (`freeForModel`, `loading`), mit `fetch`
statt axios (Node 18+), damit die Datei keine Abhängigkeit mitbringt.

**Verteilung: `scripts/sync_clients.py` plus `clients/consumers.json`**, nach dem Muster von
`STT-Suite/scripts/sync_ui.py`, aber einfacher: Die Datei wird byte-genau kopiert, mit einer Kopfzeile „erzeugt aus
GPU-Orc/clients/…, nicht von Hand ändern“. `--check` meldet veraltete Kopien (Exit 1). Die Liste nennt je Programm
Sprache und Zielpfad (z. B. AIfred `aifred/lib/gpu_orc_protocol.py`). Kein Umformen, keine Vorlagen.

**Tests im GPU-Orc-Repo:** Der Python-Client wird gegen den echten Server getestet (`GpuOrcServer` mit den Fake-Diensten
aus `tests/fakes.py`, in einem Thread auf einem freien Port): Freigeben, schon laufend, abgelehnt, nicht frei, Ampel,
Freihalten, Arbeitsmeldung, Token. Ändert sich der Server, schlägt der Client-Test im selben Repo fehl. Damit ist
ausgeschlossen, was heute passieren kann: dass ein Client unbemerkt vom Protokoll abweicht. Der JS-Client bekommt
einen kleinen `node --test` gegen Antworten, die der Python-Test aus dem echten Server mitschreibt (eine JSON-Datei im
Repo, vom Test erneuert).

## Schritte

Jeder Schritt ist ein eigener Commit im jeweiligen Repo und lässt alles lauffähig zurück.

1. **GPU-Orc: Python-Client und Tests gegen den Server.** Nichts wird verteilt.
2. **GPU-Orc: Sync-Skript und Verbraucherliste**, dazu der JS-Client und sein Test.
3. **HaemoTrace und PharmaLens** umstellen (die beiden kleinsten, eben erst gebaut): eigener Code raus, Kopie rein,
   ihre Tests grün, je ein echter Aufruf gegen GPU-Orc. HaemoTrace bekommt dabei den Aus-Schalter (leere Adresse).
   **→ Prüfpunkt A (siehe unten).** Hier entscheiden wir, ob es weitergeht.
4. **TTS-Suite und STT-Suite** (nur Ampel). Die STT-Suite braucht die Kopie im Bauordner des Containers und einen
   Neubau. Das Diktat fällt dafür kurz aus, Zeitpunkt mit Peuqui.
5. **Agent-Orc**: `gpuorc.py` wird ein dünner Anschluss an die Kopie, `card_gate.py` bleibt. Danach auf Mini und Aragon
   ausrollen.
6. **AIfred** zuletzt (der größte Client): `gpu_orc_client.py` behält Konfiguration, Logging mit Kategorie und
   `notify_changed`, das Protokoll kommt aus der Kopie. httpx fällt dort weg. `GPU_ORC_URL` leer schaltet GPU-Orc ab
   (heute fester Standardwert). Danach Neustart.

## Wann wir aufhören

Bei **Prüfpunkt A** und danach bei jedem weiteren Programm gilt: Wir hören auf und lassen den Rest beim eigenen Code,
sobald eines davon eintritt:
- Der Anschluss im Programm ist länger als der Code, den er ersetzt.
- Der gemeinsame Client braucht einen Schalter oder einen Sonderfall für ein einzelnes Programm.
- Ein Programm braucht ein anderes Verhalten als das, was der Client bietet. Dann behält genau dieses Programm seinen
  eigenen Code, die anderen nicht.
- Das Verteilen wird mehr als Kopieren (Umformen, Vorlagen, Bauschritte).
- Bei Prüfpunkt A ist HaemoTrace oder PharmaLens danach nicht spürbar einfacher.

Abbrechen kostet wenig: Schritte 1 und 2 sind neue Dateien in GPU-Orc, Schritt 3 betrifft zwei kleine Dateien.

## Einschätzung

Der Nutzen ist mäßig, aber echt. Gespart werden einige hundert Zeilen, wichtiger sind eine Bedeutung für „frei“ und
ein Test gegen den echten Server. Die Gefahr eines „Tanzes“ sehe ich gering, solange die Grenzen oben gelten: Der
Client hat keine Logik außer dem Protokoll, GPU-Orc ändert sich nicht, und Kopieren bleibt Kopieren. Am ehesten kann
AIfred kippen (viele Aufrufer, Kategorien im Log, Statusleiste); dafür steht es am Ende, und notfalls behält es seinen
Client.

Empfehlung: Schritte 1 bis 3 umsetzen, bei Prüfpunkt A gemeinsam entscheiden.
