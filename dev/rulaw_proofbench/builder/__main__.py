"""python -m dev.rulaw_proofbench.builder build --config ... --output ..."""

import argparse
import json
from pathlib import Path

from .pipeline import run
from .inputs import check_inputs
from .audit import verify
from .client import Client, PROMPTS, usage_summary
from .replay import replay
from .preparation import run as prepare
from .packet import load_packet, packet_summary, seal_packet
from .common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "seal-inputs", "check-inputs", "check-package", "check-api", "build", "replay", "verify", "cost"))
    parser.add_argument("--config", type=Path, help="Preparation or build config; default: corresponding release config")
    parser.add_argument("--inputs", type=Path, help="Accepted package for build/check-package; unsealed draft for seal-inputs")
    parser.add_argument("--output", type=Path, help="Run directory, or destination JSON file for seal-inputs")
    parser.add_argument("--workers", type=int, help="Concurrent articles (default: config workers)")
    parser.add_argument("--quiet", action="store_true", help="Hide progress; still print the final JSON")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Resume failed HTTP/transport requests in the same run; successful responses stay cached")
    parser.add_argument("--expected-bundle", type=Path, default=PROMPTS)
    args = parser.parse_args()
    if args.retry_failed and args.command not in ("prepare", "build"):
        parser.error("--retry-failed is only valid for prepare/build")
    Client.retry_failed = args.retry_failed
    if args.config is None:
        name = "prepare-release.json" if args.command in ("prepare", "check-inputs") else "release.json"
        args.config = Path(__file__).parents[1] / "configs" / name
    try:
        if args.command == "check-inputs":
            report = check_inputs(args.config)
        elif args.command == "check-package":
            if args.inputs is None:
                parser.error("--inputs is required for check-package")
            report = packet_summary(load_packet(args.inputs))
        elif args.command == "check-api":
            report = Client(Path("."), read_json(args.config)["api"]).check_api()
        else:
            if args.output is None:
                parser.error("--output is required for " + args.command)
            if args.command == "build":
                if args.inputs is None:
                    parser.error("--inputs is required: build accepts a prepared package, not raw documents")
                report = run(args.config, args.output, workers=args.workers, progress=not args.quiet, input_path=args.inputs)
            elif args.command == "prepare":
                check_inputs(args.config)
                report = prepare(args.config, args.output, workers=args.workers, progress=not args.quiet)
            elif args.command == "seal-inputs":
                if args.inputs is None:
                    parser.error("--inputs is required for seal-inputs")
                report = packet_summary(seal_packet(read_json(args.inputs), args.output))
            elif args.command == "replay":
                report = replay(args.output)
            elif args.command == "verify":
                report = verify(args.output, args.expected_bundle)
            else:
                report = usage_summary(args.output)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    keys = ("status", "actual", "cost_usd", "known_cost_usd", "unknown_cost_requests")
    print(json.dumps({key: report[key] for key in keys} if args.command in ("build", "prepare") else report, ensure_ascii=False))
    return {"shortfall": 2, "failed": 1}.get(report.get("status"), 0)


if __name__ == "__main__":
    raise SystemExit(main())
