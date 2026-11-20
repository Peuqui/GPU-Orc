# GPU-Orc

[English](README.md)

Ein kleiner Dienst je Rechner, der auf Anfrage Grafikkarten freiräumt und sagt, wer was belegt.
Bevor ein Modell lädt, bittet sein Aufrufer GPU-Orc, die Karten des Modells freizumachen. GPU-Orc
bittet die Dienste, die dort sitzen, über deren eigene Schnittstellen, Platz zu machen. Er beendet nie
einen Prozess von außen. Das härteste Mittel ist das `force` eines Dienstes, und das nur, wo die Config
es erlaubt.

Er lädt nichts, plant nichts und ist kein Modellserver. Eine Stelle je Rechner weiß, wie man dessen
Karten freiräumt, statt dass jeder Aufrufer eine eigene Kopie dieser Logik mitbringt.

## Wie es funktioniert

- **Verbraucher** sind die Dienste, die Karten belegen. Jeder hat einen Typ, eine Implementierung je
  Protokoll:
  - `gpu-release`: ein Dienst mit der gemeinsamen Freigabe-Schnittstelle von STT-Suite und TTS-Suite
    (`GET /gpu/holders`, `POST /gpu/release`, auch AIfreds Gesichtserkennung; mit `token_file` als Bearer-Token). Solange ein Element arbeitet (Transkription, Vorlesung),
    antwortet der Dienst „beschäftigt“. GPU-Orc wartet bis zu `busy_wait_s` und lässt den Dienst
    danach erzwingen, wenn `force_after_wait` gesetzt ist. Der Dienst beendet Prozess oder Container
    selbst und meldet `released` erst, wenn er weg ist.
  - `llama-swap`: ein Modellserver. GPU-Orc liest aus dessen eigener Config, welche Karten ein Modell
    belegt (`CUDA_VISIBLE_DEVICES` im `env` des Modells) und welche Modelle dauerhaft geladen bleiben
    (Gruppen mit `persistent: true`).
  - `tts-suite-remote`: die TTS-Container, die die TTS-Suite eines anderen Rechners auf diesem betreibt
    (ihre `hosts`). Die Suite meldet sie (`GET /engines`) und stoppt sie (`POST /release`). GPU-Orc
    schickt das Token der Suite mit, wartet bis zu `busy_wait_s` auf eine Vorlesung, die einen Container
    hält, und zählt einen laufenden Container auf alle Karten (die Suite sagt nicht, auf welcher er liegt).
- **Karten freihalten:** `POST /hold {"owner", "reason", "model"}` hält die Karten dieses Rechners für jemanden
  frei (ein Spiel, andere Arbeit); `model` ist das Modell des Modellservers, das der Eigentümer braucht (es bleibt
  geladen), oder `null`. Es gibt immer nur einen Halt, nur sein Eigentümer gibt ihn frei (`DELETE /hold
  {"owner"}`; `"force": true` für den Menschen, wird laut protokolliert). Ein neuer Halt ist zuerst angefordert,
  die Antwort kommt sofort; ein Thread in GPU-Orc räumt die Karten und macht ihn aktiv, sobald auf keinem
  Modellserver mehr ein Hauptmodell geladen ist oder lädt (sind sie frei, gleich, sonst prüft er alle
  `hold_check_interval_s` erneut). Ein Mensch kann einen wartenden Halt freigeben (`POST /hold/grant {"owner",
  "by"}`): Er räumt dann trotz geladenem Hauptmodell, aber erst, wenn keine Arbeit anderer gemeldet ist (`/work`),
  eine laufende Antwort wird also nie abgebrochen; die Freigabe steht in der Hold-Datei und übersteht einen
  Neustart. Aufrufer warten darauf mit `hold_and_wait`. Die Arbeit des Eigentümers selbst zählt nicht: für
  `/work` und `/hold` **denselben Eigentümernamen** benutzen. Solange der Halt aktiv ist, melden `GET /loading` (`held`) und `GET /status` ihn,
  und Aufrufer, die etwas starten würden, legen nichts Neues auf die Karten; ein angeforderter Halt steht als
  `hold_requested` da. Der Halt liegt in `hold_file` und übersteht einen Neustart; was ein Aufrufer lädt, ohne
  zu fragen, beendet GPU-Orc nicht.
- **Arbeit im Gange:** Wer hier einen Job laufen lässt (AIfred für jeden Lauf), meldet ihn mit `POST /work` als
  Lease, die ohne Erneuerung von selbst endet; `DELETE /work/<owner>` beendet sie. `GET /work` und `GET /status`
  zeigen, was läuft. `gpu-orc run-if-idle -- /sbin/reboot` führt einen Befehl nur aus, wenn nichts gemeldet ist:
  der wöchentliche Neustart lässt einen laufenden Job in Ruhe (ohne Warten, der nächste Termin versucht es wieder).
- **Nachgeprüft, nicht geglaubt:** Nach einer Freigabe fragt GPU-Orc den Dienst erneut und misst den
  Speicher der Karten mit nvidia-smi nach, bis `verify_timeout_s`. Ein Element, das noch gemeldet
  wird, oder Speicher, der nicht sinkt, ergibt `not_freed`.
- **Wer was hält**, melden die Dienste selbst. nvidia-smi liefert nur den gesamten und den belegten
  Speicher je Karte; das geht auch unter WSL2, wo nvidia-smi keine Prozesse nennt. Speicher, den
  niemand meldet, wird angezeigt, aber nie angefasst.
- **Anfragen** zum Freiräumen werden nacheinander abgearbeitet; innerhalb einer Anfrage werden alle
  Verbraucher gleichzeitig gefragt. Jede Anfrage endet spätestens `free_timeout_s` nach ihrem Eingang,
  die Wartezeit in der Schlange eingeschlossen.

## Schnittstelle

| Endpunkt | Zweck |
|---|---|
| `GET /health` | Version, `free_timeout_s` und das HTTP-Timeout, das Aufrufer nehmen sollen |
| `GET /loading` | welche großen Modelle gerade laden, ob die Karten freigehalten werden (`held`) oder ein Halt wartet (`hold_requested`), für alle, die etwas auf einer Karte starten wollen |
| `GET /status` | je Karte: Speicher, Halter, nicht zugeordneter Speicher; die Schlange; Verbraucher, die nichts melden konnten |
| `POST /free` | Karten freiräumen: `{"gpus": [0, "GPU-…"] \| "all", ...}` oder `{"model": "<name>", ...}` |
| `POST /hold` | Karten für jemanden freihalten: `{"owner": "...", "reason": "...", "model": "..." oder null}`; antwortet sofort mit dem angeforderten Halt, der aktiv wird, sobald kein Hauptmodell mehr geladen ist oder lädt (das angeforderte Modell darf laufen); anderer Eigentümer: 409 |
| `POST /hold/grant` | ein Mensch gibt einen wartenden Halt frei: `{"owner": "...", "by": "..."}`; er räumt, sobald keine Arbeit anderer läuft; kein Halt: 404, anderer Eigentümer: 409 |
| `DELETE /hold` | der Eigentümer gibt frei: `{"owner": "...", "force": false}`; anderer Eigentümer: 409, `"force": true` übergeht das |
| `GET /work` | die Jobs, die gerade gemeldet sind (antwortet sofort) |
| `POST /work` | einen Job melden oder seine Lease erneuern: `{"owner": "...", "reason": "...", "ttl_s": 60}` |
| `DELETE /work/<owner>` | der Job ist zu Ende |

`POST /free` nimmt genau eine von zwei Formen, dazu `reason` (Pflicht), `keep` und `dry_run`:

- **`gpus`:** Alles auf diesen Karten weicht, auch Hauptmodelle eines Modellservers, notfalls mitten in
  einer Antwort. `keep` schützt (`"llama-swap"` oder `"llama-swap:<modell>"`).
- **`model`:** das Modell, das gleich geladen wird.
  - Läuft es schon, antwortet GPU-Orc sofort mit `already_running`, ohne sich einzureihen.
  - Sonst weichen die Dienste auf seinen Karten und die dauerhaften Modelle des Servers dort. Die
    übrigen Modelle tauscht der Server selbst aus.

Die vollständige Beschreibung steht in [docs/interface.de.md](docs/interface.de.md), die gemeinsame
Freigabe-Schnittstelle der Suiten in [docs/release-api.de.md](docs/release-api.de.md).

Anfragen über Loopback brauchen kein Token, von anderen Rechnern `Authorization: Bearer <token>`.
Auf einer Adresse außer Loopback startet GPU-Orc nur mit Token-Datei.

## Clients für Aufrufer

Das Protokoll steht je Sprache einmal in `clients/` und ist gegen den echten Server getestet (`tests/test_client_python.py`, `tests/test_client_js.py`):

- `clients/python/gpu_orc_client.py`: nur Standardbibliothek (`GpuOrc`, `FreeResult.problems()`, `TrafficLight.red`, `hold`, `grant_hold`, `hold_and_wait`, `release_hold`, Arbeitsmeldung).
- `clients/js/gpu-orc-client.js`: ohne Abhängigkeiten (`freeForModel`, `loading` mit `held` und `holdRequested`), Node 18+.

Die Programme tragen byte-genaue Kopien. `clients/consumers.json` listet sie, `python scripts/sync_clients.py` schreibt sie (`--check` ändert nichts und endet mit Exit 1 bei veralteter oder fehlender Kopie, `--only NAME` für ein Programm, `--repos DIR` für den Ordner mit den Repos, Standard: der Elternordner dieses Repos, damit zuerst in Worktrees verteilt werden kann). Die Clients werden nur hier geändert, nie in einer Kopie.

Ein Aufrufer schaltet GPU-Orc ab, indem er die Adresse leer lässt. Diese Entscheidung trifft der Aufrufer, bevor er den Client ruft; der Client kennt keinen Aus-Zustand.

## Installation

Braucht Python 3.12 mit venv (Debian/Ubuntu: `python3-venv`) und `nvidia-smi`.

```bash
deploy/install.sh                       # venv in ~/.local/share/gpu-orc, User-Unit, erste Config
$EDITOR ~/.config/gpu-orc/config.yaml   # die Verbraucher dieses Rechners
systemctl --user enable --now gpu-orc
# oder, damit er beim Hochfahren ohne Anmeldung läuft:
sudo deploy/switch-to-system-unit.sh
```

Zur System-Unit gehört eine Polkit-Regel: Ihr Benutzer startet sie ohne Passwort neu
(`systemctl restart gpu-orc@<benutzer>`), etwa nach einer Änderung der Config. Dafür braucht es
polkit (Debian/Ubuntu: `polkitd`); fehlt es, bricht das Skript ab, bevor es etwas ändert.

Die Config ist in [deploy/config.example.yaml](deploy/config.example.yaml) beschrieben.

## Entwicklung

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/mypy src tests
```

## Star History

![Star History](https://raw.githubusercontent.com/Peuqui/Traffic/main/data/GPU-Orc/star-history.svg)

<sub>Täglich erhoben in [Peuqui/Traffic](https://github.com/Peuqui/Traffic), zusammen mit Aufrufen und Klonen. GitHub hat die Stargazer-API am 30.06.2026 auf Repo-Admins beschränkt, deshalb brauchen externe Diagramm-Dienste jetzt ein Token mit Schreibrechten.</sub>

## Lizenz

[PolyForm Noncommercial 1.0.0](LICENSE): Herunterladen und Nutzen für jeden nichtkommerziellen Zweck frei; kommerzielle Nutzung nur mit Erlaubnis des Autors.
