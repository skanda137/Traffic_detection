"""Pre-submission checks. Exit code 1 with the full list of failures, 0 when everything passes."""
import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"
REQUIREMENTS = ROOT / "requirements.txt"
PLACEHOLDER = "<Member A to fill>"
# These files contain the placeholder string only to detect it
PLACEHOLDER_CHECKERS = {"dashboard/app.py", "scripts/preflight.py"}
MAX_BLOB_BYTES = 50 * 1024 * 1024
IMPORT_NAMES = {"scikit-learn": "sklearn"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def check_placeholder(tracked):
    failures, skipped = [], []
    for rel in tracked:
        if rel in PLACEHOLDER_CHECKERS:
            skipped.append(rel)
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary or deleted in the working tree (the git status check reports the latter)
        for n, line in enumerate(text.splitlines(), 1):
            if PLACEHOLDER in line:
                failures.append(f"placeholder {PLACEHOLDER!r} present: {rel}:{n}: {line.strip()}")
    print(f"  placeholder scan: {len(tracked)} tracked files, skipped (they test for it): {sorted(skipped)}")
    return failures


def dashboard_files():
    """Every ROOT / "..." / "..." path with a file suffix in dashboard/app.py."""
    paths = []
    for match in re.finditer(r'ROOT((?:\s*/\s*"[^"]+")+)', APP.read_text(encoding="utf-8")):
        parts = re.findall(r'"([^"]+)"', match.group(1))
        if Path(parts[-1]).suffix:
            paths.append(Path(*parts))
    if not paths:
        raise RuntimeError(f"No data paths found in {APP}")
    return paths


def check_dashboard_files():
    paths = dashboard_files()
    print(f"  dashboard reads: {[p.as_posix() for p in paths]}")
    return [f"dashboard input missing: {p.as_posix()}" for p in paths if not (ROOT / p).is_file()]


def check_blob_sizes():
    entries = [line.split(None, 3) for line in git("ls-files", "-s").splitlines()]
    sizes = subprocess.run(["git", "cat-file", "--batch-check=%(objectsize)"], cwd=ROOT, check=True,
                           input="\n".join(e[1] for e in entries) + "\n", capture_output=True, text=True).stdout.split()
    largest = max(zip(map(int, sizes), (e[3] for e in entries)))
    print(f"  tracked blobs: {len(entries)}, largest {largest[1]} ({largest[0] / 1024 / 1024:.2f} MB)")
    return [f"tracked blob over 50 MB: {path} ({int(size) / 1024 / 1024:.1f} MB)"
            for size, (_, _, _, path) in zip(sizes, entries) if int(size) > MAX_BLOB_BYTES]


def check_git_clean():
    status = git("status", "--porcelain")
    return [f"git status not clean: {line}" for line in status.splitlines()]


def check_apptest():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    labels = [t.label for t in at.tabs]
    print(f"  AppTest tabs: {labels} | exceptions: {len(at.exception)}")
    failures = [f"AppTest exception: {e.message}" for e in at.exception]
    if labels != ["Plan ahead", "Live traffic", "About"]:
        failures.append(f"AppTest tabs are {labels}")
    return failures


def check_requirements():
    failures, names = [], []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        package = re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0]
        module = IMPORT_NAMES.get(package, package.replace("-", "_"))
        names.append(module)
        try:
            importlib.import_module(module)
        except Exception as e:  # any import failure counts
            failures.append(f"requirements import failed: {package} (import {module}): {type(e).__name__}: {e}")
    print(f"  requirements imports checked: {names}")
    return failures


def main():
    tracked = git("ls-files").splitlines()
    checks = [("placeholder", lambda: check_placeholder(tracked)), ("dashboard files", check_dashboard_files),
              ("blob sizes", check_blob_sizes), ("git status", check_git_clean),
              ("AppTest", check_apptest), ("requirements", check_requirements)]
    failures = []
    for name, check in checks:
        print(f"[{name}]")
        try:
            result = check()
        except Exception as e:
            result = [f"{name} check crashed: {type(e).__name__}: {e}"]
        print(f"  -> {'FAIL' if result else 'ok'}")
        failures += result
    print("\nPREFLIGHT", "FAILED" if failures else "PASSED")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
