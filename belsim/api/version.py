"""Build information for `/api/version` and the methodology page.

    uv run python -m belsim.api.version            # print it
    uv run python -m belsim.api.version --write    # write belsim/api/build_info.json

In a checkout the commit, its date and the number of commits come from git and the number of
tests from `pytest --collect-only` (once, in a background thread). An image built without `.git`
and without `tests/` carries none of that: write `build_info.json` before building it (the file
takes precedence). Parameter and assumption counts are always read from the parameter files.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_FILE = Path(__file__).with_name("build_info.json")

_lock = threading.Lock()
_static: dict | None = None
_tests: dict = {"count": None, "started": False}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def modules() -> list[str]:
    """The packages of the model (directories of `belsim/` with an `__init__.py`)."""
    pkg = ROOT / "belsim"
    return sorted(p.name for p in pkg.iterdir() if p.is_dir() and (p / "__init__.py").is_file())


def test_functions() -> int | None:
    """Number of `def test_` in tests/ (a floor: parametrised tests count once)."""
    d = ROOT / "tests"
    if not d.is_dir():
        return None
    return sum(len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8", errors="replace"), flags=re.M))
               for p in d.glob("test_*.py"))


def collect_tests() -> int | None:
    """Number of tests pytest collects (about two seconds)."""
    if not (ROOT / "tests").is_dir():
        return None
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+) tests? collected", out.stdout)
    return int(m.group(1)) if m else None


def _count_tests_in_background() -> None:
    with _lock:
        if _tests["started"]:
            return
        _tests["started"] = True

    def work():
        _tests["count"] = collect_tests()
    threading.Thread(target=work, name="belsim-test-count", daemon=True).start()


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "commit_short": None, "commit_date": None, "commits": None, "dirty": None}
    count = _git("rev-list", "--count", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "commit_short": commit[:7],
            "commit_date": _git("log", "-1", "--format=%cI"),
            "commits": int(count) if count and count.isdigit() else None,
            "first_commit_date": (_git("log", "--reverse", "--format=%cI") or "").split("\n")[0] or None,
            "dirty": bool(status) if status is not None else None}


def _static_info() -> dict:
    global _static
    with _lock:
        if _static is None:
            info: dict = {}
            if BUILD_FILE.is_file():
                try:
                    info = json.loads(BUILD_FILE.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    info = {}
            if not info.get("commit"):
                info = {**info, **git_info()}
            for env, key in (("BELSIM_COMMIT", "commit"), ("BELSIM_BUILD_DATE", "commit_date")):
                if os.environ.get(env):
                    info[key] = os.environ[env]
                    if key == "commit":
                        info["commit_short"] = os.environ[env][:7]
            info.setdefault("modules", modules())
            info.setdefault("test_functions", test_functions())
            _static = info
        return dict(_static)


def param_counts() -> dict:
    from . import service as svc
    specs = svc.load_params().with_scenario(None).describe()
    files: dict[str, dict] = {}
    for d in specs:
        f = files.setdefault(d.get("file") or "?", {"file": d.get("file") or "?", "params": 0, "assumptions": 0})
        f["params"] += 1
        f["assumptions"] += int("ASSUMPTION" in str(d.get("source", "")))
    return {"params": len(specs), "assumptions": sum(f["assumptions"] for f in files.values()),
            "param_files": sorted(files.values(), key=lambda f: f["file"])}


def build_info(count_tests: bool = True) -> dict:
    info = _static_info()
    if info.get("tests") is None:
        if _tests["count"] is None and count_tests:
            _count_tests_in_background()
        info["tests"] = _tests["count"]
    try:
        info.update(param_counts())
    except Exception:  # parameters unreadable: the rest is still worth showing
        info.update({"params": None, "assumptions": None, "param_files": []})
    return info


def main() -> None:
    info = {**git_info(), "modules": modules(), "test_functions": test_functions(), "tests": collect_tests()}
    if "--write" in sys.argv[1:]:
        BUILD_FILE.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BUILD_FILE.relative_to(ROOT)}")
    print(json.dumps({**info, **param_counts()}, indent=1))


if __name__ == "__main__":
    main()
