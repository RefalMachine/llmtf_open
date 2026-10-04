"""Explicit maintainer release freeze; normal evaluation never updates hashes."""
import json

from .generate import ROOT, AS_OF, SEED, digest
from .validate import validate
from .mcq import validate_mcq
from .judge_calibration import build_calibration


def main():
    report=validate(require_freeze=False)
    if report['failures']:raise ValueError(report['failures'])
    mcq_report=validate_mcq()
    calibration=[json.loads(line) for line in (ROOT/'judge_calibration.jsonl').read_text().splitlines()]
    if calibration!=build_calibration():raise ValueError('Judge calibration does not match deterministic construction')
    excluded={'release_manifest.json','validation_report.json','runtime_validation.json','baselines.json','model_analysis.json'}
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.suffix in {'.py','.json','.jsonl','.csv','.txt','.html','.mht'} and p.name not in excluded]
    manifest=dict(version='rulaw_proofbench_v1',as_of=AS_OF,seed=SEED,
                  files={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)})
    (ROOT/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=validate()
    if report['failures']:raise ValueError(report['failures'])
    report['details']['mcq']=mcq_report
    (ROOT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Construction validated and frozen: {len(manifest["files"])} files. Measurement validation remains pending.')


if __name__=='__main__':main()
