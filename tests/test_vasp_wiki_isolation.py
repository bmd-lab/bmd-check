"""The VASP Wiki corpus is inert, offline, isolated, packaged and third-party-labelled.

R1 adds no consumer of the corpus: these tests prove BMD Check's existing
modules do not import it and that loading it cannot reach the network,
subprocesses or dynamic code execution.
"""

from __future__ import annotations

import ast
import fnmatch
import inspect
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

import pytest

from bmd_agent import vasp_wiki_corpus as corpus
from vasp_wiki_support import SYNTHETIC_POLICY

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "bmd_agent"
RUNTIME_MODULE = SRC / "vasp_wiki_corpus.py"
PACKAGED = SRC / "reference_corpus" / "vasp_wiki"
FIXTURE = Path(__file__).parent / "fixtures" / "vasp_wiki_synthetic"

ALLOWED_IMPORTS = {
    "__future__",
    "collections.abc",
    "dataclasses",
    "hashlib",
    "importlib",
    "importlib.resources.abc",
    "json",
    "pathlib",
    "re",
    "typing",
    "urllib.parse",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
    return modules


def test_runtime_loader_imports_only_pure_standard_library_modules() -> None:
    tree = _tree(RUNTIME_MODULE)
    assert _imported_modules(tree) <= ALLOWED_IMPORTS
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            assert [alias.name for alias in node.names] == ["resources"]
        if isinstance(node, ast.ImportFrom) and node.module == "urllib.parse":
            assert {alias.name for alias in node.names} <= {"parse_qs", "urlsplit"}


def test_runtime_loader_has_no_dynamic_execution_output_or_process_calls() -> None:
    forbidden_names = {"eval", "exec", "compile", "__import__", "breakpoint", "input", "print", "open", "getattr"}
    forbidden_attributes = {"import_module", "system", "popen", "Popen", "run", "check_output", "urlopen",
                            "connect", "loads_unsafe", "load_module", "exec_module"}
    for node in ast.walk(_tree(RUNTIME_MODULE)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names, node.func.id
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes, node.func.attr


def test_reference_corpus_package_contains_no_code() -> None:
    tree = _tree(SRC / "reference_corpus" / "__init__.py")
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert sorted(p.name for p in (SRC / "reference_corpus").glob("*.py")) == ["__init__.py"]
    assert not list(PACKAGED.rglob("*.py"))


def test_no_existing_bmd_check_module_consumes_the_corpus_in_r1() -> None:
    for path in SRC.rglob("*.py"):
        if path in (RUNTIME_MODULE, SRC / "reference_corpus" / "__init__.py"):
            continue
        source = path.read_text(encoding="utf-8")
        modules = _imported_modules(_tree(path))
        assert not any("vasp_wiki_corpus" in module or "reference_corpus" in module for module in modules), path
        assert "vasp_wiki_corpus" not in source and "reference_corpus" not in source, path
        assert "fetch_corpus" not in source and "tools.vasp_reference" not in source, path


def test_importing_and_running_bmd_check_does_not_load_the_corpus() -> None:
    code = (
        "import sys\n"
        "import bmd_agent.cli\n"
        "from bmd_agent.cli import main\n"
        "main(['--help'])\n"
        "assert 'bmd_agent.vasp_wiki_corpus' not in sys.modules, 'corpus loaded'\n"
        "assert 'bmd_agent.reference_corpus' not in sys.modules, 'corpus package loaded'\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, env=environment, timeout=120
    )
    assert completed.returncode == 0, completed.stderr


def test_loading_imports_no_network_or_process_modules() -> None:
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "from bmd_agent import vasp_wiki_corpus as c\n"
        "result = c.load_vasp_wiki_corpus()\n"
        "assert result.reason == 'unpopulated', result\n"
        "new = set(sys.modules) - before\n"
        "bad = {m for m in new if m.split('.')[0] in {'socket', 'ssl', 'http', 'subprocess', 'urllib3',"
        " 'requests', 'pickle', 'yaml', 'ctypes', 'multiprocessing'} or m in {'urllib.request', 'urllib.error'}}\n"
        "assert not bad, bad\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", f"import sys; sys.path.insert(0, {str(REPO / 'src')!r})\n" + code],
        capture_output=True, text=True, env=environment, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_loading_works_with_network_and_subprocesses_disabled(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("runtime corpus loading must not use the network or subprocesses")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(os, "system", refuse)
    assert corpus.load_vasp_wiki_corpus().reason == corpus.UNAVAILABLE_UNPOPULATED
    synthetic = corpus.load_vasp_wiki_corpus(FIXTURE, policy=SYNTHETIC_POLICY)
    assert isinstance(synthetic, corpus.VaspWikiCorpus)
    assert all(page.read_wikitext() for page in synthetic.pages)


def test_default_policy_is_the_authoritative_policy() -> None:
    default = inspect.signature(corpus.load_vasp_wiki_corpus).parameters["policy"].default
    assert default is corpus.AUTHORITATIVE_POLICY


def test_packaged_corpus_is_reached_through_package_resources() -> None:
    root = corpus.packaged_corpus_root()
    assert root.joinpath("manifest.json").is_file()
    assert Path(str(root)).resolve() == PACKAGED.resolve()


# ---------------------------------------------------------------------------
# Packaging and third-party separation
# ---------------------------------------------------------------------------


def _package_data_globs() -> list[str]:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["setuptools"]["package-data"]["bmd_agent"]


def test_package_data_covers_every_corpus_file_including_future_pages() -> None:
    globs = _package_data_globs()
    candidates = [p.relative_to(SRC).as_posix() for p in PACKAGED.rglob("*") if p.is_file()]
    candidates += [
        "reference_corpus/vasp_wiki/COPYING.GFDL-1.2.txt",
        "reference_corpus/vasp_wiki/pages/nelm.r123.wiki",
    ]
    for candidate in candidates:
        assert any(fnmatch.fnmatchcase(candidate, pattern) for pattern in globs), candidate


def test_maintainer_tool_is_outside_the_installed_package() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert (REPO / "tools" / "vasp_reference" / "fetch_corpus.py").is_file()
    assert not (REPO / "tools" / "__init__.py").exists()
    assert not list(SRC.rglob("fetch_corpus*"))
    assert "fetch_corpus" not in RUNTIME_MODULE.read_text(encoding="utf-8").split('"""', 2)[2]


def test_corpus_directory_holds_only_corpus_files() -> None:
    allowed = {"README.md", "NOTICE", "manifest.json", "RELEASES.json", "SHA256SUMS", corpus.AUTHORITATIVE_POLICY.license_text_file}
    for path in PACKAGED.rglob("*"):
        relative = path.relative_to(PACKAGED).as_posix()
        if path.is_dir():
            assert relative == "pages"
        elif relative.startswith("pages/"):
            assert relative.endswith(".wiki")
        else:
            assert relative in allowed, relative


def test_third_party_material_is_labelled_and_mit_license_is_unchanged() -> None:
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
    notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "src/bmd_agent/reference_corpus/vasp_wiki/" in notices and "not under the MIT License" in notices
    readme = (PACKAGED / "README.md").read_text(encoding="utf-8")
    assert "third-party material" in readme and "not covered by" in readme
    assert "THIRD_PARTY_NOTICES.md" in (REPO / "README.md").read_text(encoding="utf-8")


def test_corpus_readme_lists_the_ten_intended_titles() -> None:
    readme = (PACKAGED / "README.md").read_text(encoding="utf-8")
    for title in ("NELM", "EDIFF", "ALGO", "EDIFFG", "NSW", "IBRION", "ISIF", "Not enough memory",
                  "Difficult_to_converge_systems", "Memory"):
        assert f"`{title}`" in readme


@pytest.mark.parametrize("path", [RUNTIME_MODULE, REPO / "tools" / "vasp_reference" / "fetch_corpus.py"])
def test_corpus_code_compiles(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
