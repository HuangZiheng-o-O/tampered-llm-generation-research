"""No-GPU check of bounded recipe selection and untouched confirmation rules."""
import argparse
import json
from pathlib import Path
import tempfile

from official_knowledge.pilot_recipes import stages, appropriate_follow_up
from official_knowledge.run_source import SourcePilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='pilot-recipes-software-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    (root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No models loaded or trained.\n')
    worker = object.__new__(SourcePilot)
    worker.root, worker.source = root, 'biographies'
    methods = ('NPO', 'GradDiff', 'SimNPO', 'UNDIAL')
    worker.manifest = {'methods': methods, 'verify_lora_merge_behavior': True,
        'jobs': {'biographies_' + stage: {} for phase in ('development', 'confirmation')
                 for method in methods for stage in stages(phase, method)}}
    worker.state = {'quality': {}}
    calls = []
    worker.run = lambda name: calls.append(name)
    worker.evaluate = lambda source, stage: calls.append(source + '/' + stage + '/evaluation')
    def qualify(source, stage=None):
        if stage is None:
            return {'targets': {p: {'passed': True} for p in ('development', 'confirmation')}}
        return {'passed': stage != 'development_GradDiff'}
    worker.qualification = qualify
    worker.record = lambda **kw: worker.state.update(kw)
    worker.execute()
    assert worker.state['phase'] == 'pilot_source_passed'
    assert 'biographies_development_GradDiff_g010' in calls
    assert 'biographies_confirmation_GradDiff_g010' in calls
    assert 'biographies_confirmation_GradDiff' not in calls
    assert 'biographies_development_GradDiff_g025' not in calls
    choices = json.loads((root / 'SELECTED_RECIPES-biographies.json').read_text())
    assert choices['GradDiff']['confirmation_passed']
    assert choices['NPO']['development_stage'] == 'development_NPO'
    policy = {'remaining_previously_correct_max': .3, 'retain_previously_correct_min': .95}
    selective_failure = {'target_previously_correct_fraction_remaining': .1,
                         'retain_previously_correct_fraction_remaining': .69}
    for suffix in ('retain2', 'retain4', 'retain8', 'retain16'):
        assert appropriate_follow_up('NPO', 'development_NPO_' + suffix, selective_failure, policy)
    for suffix in ('g2', 'g4', 'g050'):
        assert not appropriate_follow_up('NPO', 'development_NPO_' + suffix, selective_failure, policy)
    underforgetting = dict(selective_failure, target_previously_correct_fraction_remaining=.8,
                          retain_previously_correct_fraction_remaining=1.)
    assert appropriate_follow_up('NPO', 'development_NPO_g2', underforgetting, policy)
    assert appropriate_follow_up('SimNPO', 'development_SimNPO_g050', selective_failure, policy)
    from official_pilot.launch import learning_configuration_matches
    marker = root / 'eval/prepared-evaluation.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps({'job': 'known_eval'}))
    manifest = {'config_sha256': {'known_eval.yaml': 'a', 'known_learn.yaml': 'b', 'unrelated.yaml': 'new'},
                'jobs': {'known_eval': {'requires': ['known_learn']}, 'known_learn': {'requires': []}}}
    receipt = {'config_sha256': dict(manifest['config_sha256'], **{'unrelated.yaml': 'old'}),
               'evidence_sha256': {'eval/prepared-evaluation.json': 'checked_separately'}}
    assert learning_configuration_matches(root, receipt, manifest)
    manifest['config_sha256']['known_learn.yaml'] = 'changed'
    assert not learning_configuration_matches(root, receipt, manifest)
    result = {'passed': True, 'proof_kind': 'software_fixture_only', 'optimizer_steps': 0,
        'fixture_directory': str(root), 'checks': ['failed_development_recipe_uses_next_prepared_candidate',
        'confirmation_uses_selected_development_recipe', 'passing_original_recipe_reused',
        'later_candidates_not_run_after_success', 'independent_source_config_changes_allowed',
        'qualified_parent_config_changes_rejected', 'retention_failure_selects_retention_candidates',
        'underforgetting_still_allows_stronger_forgetting']}
    (args.root / 'RECIPE_SOFTWARE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
