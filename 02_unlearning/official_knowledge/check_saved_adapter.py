"""Check actual saved NPO adapters on an idle GPU before later pilot gates."""
import argparse
import json
from pathlib import Path

from official_knowledge.production_jobs import atomic_json
from official_knowledge.production_worker import FormalPilot, PROTOCOLS, activation_spec
from official_knowledge.run_pilot import Pilot, now


class ArtifactPilot(Pilot):
    activation = FormalPilot.activation

    def record(self, **values):
        self.state.update(values, updated_at_utc=now())
        atomic_json(self.root / 'SAVED_ADAPTER_CHECK_STATUS.json', self.state)

    def report(self):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', choices=('llama2-chat', 'llama3-instruct'), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    args = p.parse_args()
    suite = args.root / ('official-lora-pilot-v1-' + args.base)
    gate = json.loads((suite / 'gates/biographies-development_NPO-quality.json').read_text())
    assert gate['passed'] and gate['proof_kind'] == 'official_model_evaluation'
    pilot = ArtifactPilot(suite, args.upstream, 24)
    parent = next(x for x in json.loads((args.root / 'formal-lora-400-v1/parents.json').read_text())
                  if x['base'] == args.base and x['source'] == 'biographies')
    model = Path(pilot.manifest['jobs']['biographies_learn']['output']) / 'merged'
    adapter = Path(pilot.manifest['jobs']['biographies_development_NPO']['output'])
    results = {}
    for protocol in PROTOCOLS:
        rid = 'development-artifact-' + args.base + '-npo'
        spec = activation_spec(args.root, model, adapter, rid, suite / 'saved-adapter-check' / protocol, protocol)
        spec.update(base_model_id=parent['base_model_id'], tokenizer_source=parent['hdd_original'])
        results[protocol] = pilot.activation(spec)
    atomic_json(suite / 'SAVED_ADAPTER_CHECK.json', {'passed': True, 'at_utc': now(),
        'proof_kind': 'real_saved_adapter_activation_check', 'accepted_formal_models': 0, 'checks': results})
    pilot.record(phase='saved_adapter_verified', current_job=None)
    print(json.dumps({'passed': True, 'base': args.base, 'protocols': list(results)}), flush=True)


if __name__ == '__main__':
    main()
