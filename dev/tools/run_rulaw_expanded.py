"""Run RuLaw preparation/build through the local subscription adapter.

The fixed module entry point permits a narrowly scoped persistent shell approval.
Credentials are read in-process, never placed in shell arguments or config files.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dev.rulaw_proofbench.builder.client import Client
from dev.rulaw_proofbench.builder.common import read_json, require
from dev.rulaw_proofbench.builder.inputs import check_inputs
from dev.rulaw_proofbench.builder.pipeline import run as build
from dev.rulaw_proofbench.builder.preparation import run as prepare


def configure_adapter():
    root = Path(__file__).resolve().parents[2]
    adapter = root.parent.parent / "chatgpt_adapter"
    endpoint = os.environ.get("RULAW_API_BASE", "http://127.0.0.1:8080/v1")
    parsed = urlsplit(endpoint)
    require(parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "This launcher sends the adapter key only to a local HTTP endpoint")
    key = os.environ.get("RULAW_API_KEY")
    if not key:
        key = read_json(adapter / ".local" / "adapter-key.json")["key"]
    require(isinstance(key, str) and bool(key), "Adapter API key is missing")
    os.environ.update(RULAW_API_BASE=endpoint, RULAW_API_PROFILE="subscription", RULAW_API_KEY=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-api", "prepare", "build"))
    parser.add_argument("--config", type=Path, default=Path("dev/rulaw_proofbench/configs/expanded.json"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--source-proxy", help="Proxy for official source downloads; adapter calls stay local")
    args = parser.parse_args()
    if args.command != "check-api" and args.output is None:
        parser.error("--output is required")
    if args.command == "build" and args.inputs is None:
        parser.error("--inputs is required for build")
    configure_adapter()
    if args.source_proxy:
        os.environ.update(HTTP_PROXY=args.source_proxy, HTTPS_PROXY=args.source_proxy,
                          http_proxy=args.source_proxy, https_proxy=args.source_proxy)
    Client.retry_failed = args.retry_failed
    if args.command == "check-api":
        client = Client(Path("."), read_json(args.config)["api"])
        report = client.check_api()
    elif args.command == "prepare":
        check_inputs(args.config)
        report = prepare(args.config, args.output, workers=args.workers)
    else:
        report = build(args.config, args.output, workers=args.workers, input_path=args.inputs)
    keys = ("status", "actual", "selection", "unique_requests", "unknown_cost_requests")
    print(json.dumps({key: report[key] for key in keys if key in report}, ensure_ascii=False))
    return {"shortfall": 2, "failed": 1}.get(report.get("status"), 0)


if __name__ == "__main__":
    raise SystemExit(main())
