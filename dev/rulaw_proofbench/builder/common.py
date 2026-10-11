"""Small JSON/provenance helpers shared by the standalone builder."""

import hashlib
import json
from pathlib import Path


class Rejected(ValueError):
    """A candidate failed a documented construction gate."""


class InputGap(Rejected):
    """The accepted input cannot support this task; return to preparation."""


def accepted(value):
    if value.get("reject"):
        error = InputGap if value.get("kind") == "input_gap" else Rejected
        raise error(value["reject"])
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    data = value if isinstance(value, bytes) else canonical(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise Rejected(message)


def same(left, right):
    return type(left) is type(right) and left == right


def freeze_run(output, config, phase, allow_transport_update=False, **inputs):
    """Freeze the implementation and explicit inputs before any model request."""
    package = Path(__file__).parent
    code = {str(p.relative_to(package)): digest(p.read_bytes()) for p in sorted(package.rglob("*"))
            if p.suffix in (".py", ".yaml")}
    identity = dict(config=config, implementation=code, phase=phase, **inputs)
    manifest = output / "build-input.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous != identity and allow_transport_update:
            migrate_transport(output, previous, identity, package)
        require(read_json(manifest) == identity, "Run inputs/code changed. Use a new output directory")
        require(digest((output / "prompts.yaml").read_bytes()) == code["prompts.yaml"], "Frozen prompts changed")
    else:
        write_json(manifest, identity)
        for name in code:
            target = output / "implementation" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / name).read_bytes())
        (output / "prompts.yaml").write_bytes((package / "prompts.yaml").read_bytes())
    return identity


def migrate_transport(output, previous, current, package):
    """Apply known transport/scheduling hotfixes while retaining prior provenance."""
    compatible = {
        "client.py": {"125d442545dce82a1413c5c837797ac14b774d2ce81884e0a9fc5445f6114ca8",
                      "ca81927ebe79618cff837f953fc1861bc879b46a4905fdab0866b3af8e3c3d07",
                      "bb1d8ff859b07b6cf8a6a6fc72028265c24603a3009597212362d22f9a657360",
                      "06c42882d676698e55c67b529cf9fef104d23408edfab43f01a598a223ca457d"},
        "common.py": {"bb52da2aac75e3a2baea262158618708818795bcc10c6b65b5f415f74a77a86b",
                      "67ed43df2c885128403bc473dc3108266c67c922db90af2e9dd2f9233a1e2d12",
                      "1417e4605abb49acc272eeb5e4319f6feb257f9d78b47542f1aa9b398f723fb3",
                      "90e11046d4cd65342a2aecb788a01c5e1a9ff3d5f016ea74ce4ac215f11d4fe4",
                      "76ba950ef2e0045bbdfc7bb197841dcf49d4f61422b36023b2771fd1b8c2ad95"},
        "audit.py": {"a8f483822442d91018c2774c8ef83e5ffd27ea83b327431da194813c782dd979",
                     "cc98270ed2189f1fc2b79afe99427ae0a8e0a1ea7271efdad9996581f071aa8d"},
        "__main__.py": {"bdc8b0bff3587a9f3b570ebd62adca6976546cc5ea76b2e12c161860dd4a6b04",
                        "3a56e7211bb42d53555211e1b8fb5950d76dea4c6d1cb4347dd0ee021194b2a9"},
        "preparation.py": {"54284e212a5ec334b297c4574b2d541540ff8f30ebf6d6dfa90ac341f8dd3d74"},
        "pipeline.py": {"cb8410aabf9b1606f882b61bc6e4d9ddccaa965822b43d1577258e3fbba513d4",
                        "2a36b147ff3a485fa735ff823b54ff42b8ea1be85a4fb4f3c669a61610ce061d",
                        "8fd90e6c41807e2b0471bd3529bc9e6149709911fe7937d8d8ded5bef59fbd81"},
    }
    require({k: v for k, v in previous.items() if k != "implementation"} ==
            {k: v for k, v in current.items() if k != "implementation"},
            "Retry cannot change run inputs/configuration")
    old, new = previous["implementation"], current["implementation"]
    require(set(old) == set(new), "Retry cannot change implementation files")
    changed = [name for name in old if old[name] != new[name]]
    require(all(old[name] in compatible.get(name, set()) for name in changed),
            "Retry upgrade supports only known transport/scheduling fixes; other code/prompts changed")
    require(not (previous["phase"] == "prepare" and (output / "input-package.json").exists()),
            "Preparation already emitted an accepted package; do not migrate its provenance")
    require(digest((output / "prompts.yaml").read_bytes()) == old["prompts.yaml"], "Frozen prompts changed")
    archive = output / "implementation-history" / digest(previous)
    # Validate every snapshot before making any change. Keep the old version for audit.
    for name, expected in old.items():
        require(digest((output / "implementation" / name).read_bytes()) == expected,
                "Frozen code changed: " + name)
    write_json(archive / "build-input.json", previous)
    write_json(archive / "upgrade.json", dict(reason="Explicit --retry-failed compatible hotfix",
                                              changed_files=changed, implementation=new))
    for name in old:
        target = archive / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((output / "implementation" / name).read_bytes())
    for name in changed:
        target = output / "implementation" / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes((package / name).read_bytes())
        temporary.replace(target)
    write_json(output / "build-input.json", current)
