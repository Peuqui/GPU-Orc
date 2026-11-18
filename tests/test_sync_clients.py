import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "sync_clients.py"
spec = importlib.util.spec_from_file_location("sync_clients", SCRIPT)
assert spec is not None and spec.loader is not None
sync_clients = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_clients)

CONSUMERS = json.loads((SCRIPT.parent.parent / "clients" / "consumers.json").read_text())["consumers"]
TARGETS = [f"{consumer['repo']}/{consumer['target']}" for consumer in CONSUMERS]  # HaemoTrace, PharmaLens, ...


def make_repos(root: Path) -> None:
    for target in TARGETS:
        (root / target).parent.mkdir(parents=True)


def test_sync_writes_byte_identical_copies_and_check_passes(tmp_path: Path) -> None:
    make_repos(tmp_path)
    assert sync_clients.main(["--repos", str(tmp_path), "--check"]) == 1  # nothing there yet
    assert sync_clients.main(["--repos", str(tmp_path)]) == 0
    assert (tmp_path / TARGETS[0]).read_bytes() == sync_clients.SOURCES["python"].read_bytes()
    assert (tmp_path / TARGETS[1]).read_bytes() == sync_clients.SOURCES["js"].read_bytes()
    assert sync_clients.main(["--repos", str(tmp_path), "--check"]) == 0


def test_check_reports_a_changed_copy_and_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_repos(tmp_path)
    sync_clients.main(["--repos", str(tmp_path)])
    (tmp_path / TARGETS[0]).write_text("edited by hand")
    assert sync_clients.main(["--repos", str(tmp_path), "--check"]) == 1
    assert "stale: HaemoTrace" in capsys.readouterr().out
    assert (tmp_path / TARGETS[0]).read_text() == "edited by hand"
    sync_clients.main(["--repos", str(tmp_path)])
    assert sync_clients.main(["--repos", str(tmp_path), "--check"]) == 0


def test_check_reports_a_missing_copy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_repos(tmp_path)
    sync_clients.main(["--repos", str(tmp_path)])
    (tmp_path / TARGETS[1]).unlink()
    assert sync_clients.main(["--repos", str(tmp_path), "--check"]) == 1
    assert "missing: PharmaLens" in capsys.readouterr().out


def test_only_restricts_to_one_consumer(tmp_path: Path) -> None:
    make_repos(tmp_path)
    assert sync_clients.main(["--repos", str(tmp_path), "--only", "PharmaLens"]) == 0
    assert (tmp_path / TARGETS[1]).exists() and not (tmp_path / TARGETS[0]).exists()
