"""Investigate a failed biography confirmation using official trainers."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from official_knowledge.run_pilot import Pilot, now
from official_knowledge.production_jobs import atomic_json


def validate_failure(method, report, policy):
    assert not report['passed'] and report['proof_kind'] == 'official_model_evaluation'
    if method in ('NPO', 'GradDiff', 'SimNPO'):
        assert report['target_previously_correct_fraction_remaining'] > policy['remaining_previously_correct_max']
    elif method == 'UNDIAL':
        assert report['target_previously_correct_fraction_remaining'] <= policy['remaining_previously_correct_max']
        assert report['retain_previously_correct_fraction_remaining'] < policy['retain_previously_correct_min']
    else:
        raise ValueError('Unsupported diagnostic method')


class DiagnosticPilot(Pilot):
    def record(self, **values):
        self.state.update(values, coordinator_pid=os.getpid(), updated_at_utc=now(),
                          scope=getattr(self, 'method', 'NPO').lower() + '_two_seen_targets_diagnostic_only', formal_batch_started=False)
        atomic_json(self.root / (getattr(self, 'tag', 'NPO_DIAGNOSTIC') + '_STATUS.json'), self.state)
        print(json.dumps({'at_utc': now(), **values}), flush=True)

    def report(self):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--method', choices=('NPO', 'UNDIAL', 'GradDiff', 'SimNPO'), default='NPO')
    p.add_argument('--failed-stage', help='Existing failed confirmation stage, including its selected suffix')
    p.add_argument('--suffixes', nargs='+', default=['g2', 'g4'])
    p.add_argument('--phases', nargs='+', choices=('development', 'confirmation'), default=['development', 'confirmation'])
    p.add_argument('--tag', choices=('NPO_DIAGNOSTIC', 'NPO_FACTSPAN_DIAGNOSTIC', 'UNDIAL_RETAIN_DIAGNOSTIC', 'GRADDIFF_DIAGNOSTIC', 'SIMNPO_DIAGNOSTIC'), default='NPO_DIAGNOSTIC')
    p.add_argument('--revision', type=int, default=1)
    args = p.parse_args()
    tags = {'NPO': {'NPO_DIAGNOSTIC', 'NPO_FACTSPAN_DIAGNOSTIC'},
            'UNDIAL': {'UNDIAL_RETAIN_DIAGNOSTIC'}, 'GradDiff': {'GRADDIFF_DIAGNOSTIC'},
            'SimNPO': {'SIMNPO_DIAGNOSTIC'}}
    if args.tag not in tags[args.method]:
        raise ValueError('Diagnostic method and receipt tag differ')
    if args.method == 'UNDIAL' and not set(args.suffixes) <= {'retain1', 'beta20retain1'}:
        raise ValueError('UNDIAL diagnostic requires the prepared retention candidates')
    if args.method == 'GradDiff' and not set(args.suffixes) <= {'factspan', 'g025', 'factspan_kl010', 'factspan_kl025', 'factspan_g005'}:
        raise ValueError('GradDiff diagnostic requires the prepared answer-loss or gamma=.25 candidate')
    if args.method == 'SimNPO' and args.suffixes != ['factspan']:
        raise ValueError('SimNPO diagnostic requires the prepared answer-loss candidate')
    if args.revision < 1:
        raise ValueError('Diagnostic revision must be positive')
    tag = args.tag + (f'_V{args.revision}' if args.revision > 1 else '')
    lock_name = 'npo-diagnostic.lock' if tag == 'NPO_DIAGNOSTIC' else tag + '.lock'
    with (args.suite / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot = DiagnosticPilot(args.suite, args.upstream, 48)
        pilot.tag, pilot.method = tag, args.method
        failed_stage = args.failed_stage or 'confirmation_' + args.method
        if not failed_stage.startswith('confirmation_' + args.method) or 'biographies_' + failed_stage not in pilot.manifest['jobs']:
            raise ValueError('Failed confirmation stage does not belong to this method and suite')
        original = json.loads((args.suite / ('gates/biographies-' + failed_stage + '-quality.json')).read_text())
        validate_failure(args.method, original, pilot.manifest['quality_policy'])
        result_file = args.suite / (tag + '_RESULT.json')
        previous = json.loads(result_file.read_text()) if result_file.exists() else {}
        if previous.get('both_seen_tuning_targets_passed'):
            raise ValueError('Preserve the passed diagnostic; use a new revision for another campaign')
        reports = previous.get('results', {})
        try:
            for suffix in args.suffixes:
                results = {}
                for phase in args.phases:
                    stage = phase + '_' + args.method + '_' + suffix
                    pilot.manifest = json.loads((args.suite / 'configs/jobs.json').read_text())
                    # Both targets have already been inspected. They are used
                    # only to diagnose robustness of a new loss coefficient.
                    pilot.run('biographies_' + stage)
                    pilot.evaluate('biographies', stage)
                    results[phase] = pilot.qualification('biographies', stage)
                    if not results[phase]['passed']:
                        break
                reports[suffix] = {phase: {k: value[k] for k in (
                    'passed', 'target_previously_correct_fraction_remaining',
                    'retain_previously_correct_fraction_remaining', 'mmlu_before', 'mmlu_after')}
                    for phase, value in results.items()}
                passed = len(results) == len(args.phases) and all(value['passed'] for value in results.values())
                receipt = {'at_utc': now(), 'proof_kind': 'official_model_evaluation', 'method': args.method,
                    'revision': args.revision,
                    'both_seen_tuning_targets_passed': passed and len(args.phases) == 2,
                    'requested_phase_checks_passed': passed, 'tested_phases': args.phases,
                    'selected_candidate': suffix if passed else None,
                    'fresh_independent_content_confirmation_required': True, 'formal_production_authorized': False,
                    'original_failed_confirmation_preserved': True, 'results': reports}
                atomic_json(result_file, receipt)
                if passed:
                    pilot.record(phase='needs_fresh_independent_confirmation', current_job=None, candidate=suffix)
                    return
            pilot.record(phase='needs_research', current_job=None)
        except Exception as error:
            pilot.record(phase='needs_repair', error=repr(error))
            raise


if __name__ == '__main__':
    main()
