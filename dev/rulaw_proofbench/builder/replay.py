"""Replay frozen code in a temporary copy; original evidence is never rewritten."""

import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from unittest.mock import patch

from .common import digest, read_json, require, write_json


def replay(root):
    root = Path(root).resolve()
    manifest = read_json(root / "build-input.json")
    require(read_json(root / "report.json")["status"] in ("complete", "shortfall", "prepared"), "Replay requires a finished run")
    for name, expected in manifest["implementation"].items():
        require(digest((root / "implementation" / name).read_bytes()) == expected, "Frozen code changed: " + name)
    watched = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".bin", ".txt")
               and p.name not in ("progress.json",)]
    before = {str(p.relative_to(root)): digest(p.read_bytes()) for p in watched}
    with tempfile.TemporaryDirectory(prefix="rulaw-replay-") as temp:
        copy = Path(temp) / "run"
        shutil.copytree(root, copy)
        config = Path(temp) / "config.json"
        write_json(config, manifest["config"])
        package = copy / "implementation"
        name = "rulaw_frozen"
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        pipeline = importlib.import_module(name + (".preparation" if manifest.get("phase") == "prepare" else ".pipeline"))
        env = dict(os.environ)
        env.pop(manifest["config"]["api"].get("api_key_env", "OPENROUTER_API_KEY"), None)
        with patch.dict(os.environ, env, clear=True), \
                patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden in replay")), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in replay")):
            report = pipeline.run(config, copy, offline=True)
        changed = [name for name, expected in before.items() if digest((copy / name).read_bytes()) != expected]
        require(not changed, "Replay changed artifacts: " + ", ".join(changed))
    return dict(status="replay_passed", actual=report["actual"], unchanged_files=len(before), network="disabled")
