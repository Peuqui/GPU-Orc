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
    (`GET /gpu/holders`, `POST /gpu/release`). Solange ein Element arbeitet (Transkription, Vorlesung),
    antwortet der Dienst „beschäftigt“. GPU-Orc wartet bis zu `busy_wait_s` und lässt den Dienst
    danach erzwingen, wenn `force_after_wait` gesetzt ist. Der Dienst beendet Prozess oder Container
    selbst und meldet `released` erst, wenn er weg ist.
  - `llama-swap`: ein Modellserver. GPU-Orc liest aus dessen eigener Config, welche Karten ein Modell
    belegt (`CUDA_VISIBLE_DEVICES` im `env` des Modells) und welche Modelle dauerhaft geladen bleiben
    (Gruppen mit `persistent: true`).
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
| `GET /status` | je Karte: Speicher, Halter, nicht zugeordneter Speicher, was andere Programme noch belegen dürfen (`available_mib`); die Schlange; Verbraucher, die nichts melden konnten |
| `GET /profiles` | die Profile des Modellservers: Karten, Backend, Kontext, KV-Blöcke, Variante (für welche Zusatzmodelle Platz bleibt) und was jedes je Karte nimmt (gemessen, Anteil oder unbekannt) |
| `GET /needs` | was die Zusatzmodelle auf einer Karte brauchen: die Engines der Suiten (deren `GET /gpu/needs`) und die Bildmodelle der Modellserver |
| `GET /ui/` | die Weboberfläche (Karten, Profile, Bedarfe), wie bei den Suiten |
| `POST /free` | Karten freiräumen: `{"gpus": [0, "GPU-…"] \| "all", ...}` oder `{"model": "<name>", ...}` |

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
