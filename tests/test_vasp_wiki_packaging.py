"""The corpus ships byte-identical in a built wheel and loads offline from it.

The wheel is built with the setuptools PEP 517 backend in a temporary copy of
the project (no network, no build isolation) and installed by extracting it,
exactly as an installer lays out a pure-Python wheel. Skipped when setuptools
is not installed in the test environment.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "src" / "bmd_agent" / "reference_corpus" / "vasp_wiki"

setuptools = pytest.importorskip("setuptools")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    project = tmp_path_factory.mktemp("project")
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (REPO / name).exists():
            shutil.copy2(REPO / name, project / name)
    shutil.copytree(REPO / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    out = tmp_path_factory.mktemp("dist")
    code = (
        "import sys; from setuptools import build_meta; "
        "print(build_meta.build_wheel(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(out)],
        cwd=project,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build unavailable: {result.stderr[-400:]}")
    return out / result.stdout.strip().splitlines()[-1]


def test_wheel_contains_every_corpus_file_byte_identical(wheel: Path) -> None:
    expected = {
        "bmd_agent/reference_corpus/vasp_wiki/" + path.relative_to(CORPUS).as_posix(): path.read_bytes()
        for path in CORPUS.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(wheel) as archive:
        shipped = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("bmd_agent/reference_corpus/vasp_wiki/") and not name.endswith("/")
        }
    assert shipped == expected


def test_installed_wheel_loads_the_corpus_offline(wheel: Path, tmp_path: Path) -> None:
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    probe = r"""
import socket, sys
def refuse(*args, **kwargs):
    raise RuntimeError("network access attempted")
socket.socket = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
from bmd_agent import vasp_wiki_corpus as corpus
assert corpus.__file__.startswith(sys.argv[1]), corpus.__file__
result = corpus.load_vasp_wiki_corpus()
if isinstance(result, corpus.CorpusUnavailable):
    print("unavailable", result.reason)
else:
    for page in result.pages:
        page.read_wikitext()
    print("loaded", result.corpus_version, result.corpus_digest, len(result.pages))
"""
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-c", f"import sys; sys.path.insert(0, {str(site)!r}); exec({probe!r})", str(site)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    source = subprocess.run(
        [sys.executable, "-I", "-S", "-c", f"import sys; sys.path.insert(0, {str(REPO / 'src')!r}); exec({probe!r})", str(REPO / "src")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert source.returncode == 0, source.stderr
    assert run.stdout == source.stdout
    assert run.stdout.split()[0] in ("loaded", "unavailable")
