"""Prepare bounded official-loss adjustments without overwriting prior attempts."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from official_knowledge.run_pilot import sha


REPAIR_DIAGNOSTICS = {'NPO': 'NPO_FACTSPAN_DIAGNOSTIC', 'GradDiff': 'GRADDIFF_DIAGNOSTIC',
                      'SimNPO': 'SIMNPO_DIAGNOSTIC', 'UNDIAL': 'UNDIAL_RETAIN_DIAGNOSTIC'}


def precommitment_name(method, revision=1):
    assert method in REPAIR_DIAGNOSTICS and revision >= 1
    name = ('INDEPENDENT_CONFIRMATION_TARGET_PRECOMMITTED.json' if method == 'NPO' else
            method.upper() + '_INDEPENDENT_CONFIRMATION_TARGET_PRECOMMITTED.json')
    return name if revision == 1 else name.removesuffix('.json') + f'_V{revision}.json'


def diagnostic_name(method, revision=1):
    assert method in REPAIR_DIAGNOSTICS and revision >= 1
    return REPAIR_DIAGNOSTICS[method] + (f'_V{revision}' if revision > 1 else '')


def selected_control(suite, source, phase):
    manifest = json.loads((suite / 'configs/jobs.json').read_text())
    stage = manifest.get('control_stages', {}).get(source, {}).get(phase, phase + '_control_1e-4')
    assert phase in ('development', 'confirmation') and stage.startswith(phase + '_control_')
    assert source + '_' + stage in manifest['jobs']
    return stage


def prepare_retention_controls(suite, source):
    """Match unlearning updates with official retain-only NLL training.

    GradDiff(gamma=0, alpha=1) uses the same paired dataset and optimizer
    schedule. The target forward pass contributes zero training gradient.
    Retention examples are sampled from the complete existing retention pool.
    """
    from official_knowledge.production_jobs import atomic_json
    config_dir = suite / 'configs'
    with (config_dir / 'recipe-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = config_dir / 'jobs.json'
        manifest = json.loads(path.read_text())
        assert source in manifest.get('active_sources', ('biographies', 'squad'))
        for phase in ('development', 'confirmation'):
            old_name = f'{source}_{phase}_GradDiff'
            stage = phase + '_control_retainpool'
            name = source + '_' + stage
            for tail in ('', '_knowledge', '_mmlu_validation'):
                new = name + tail
                if new in manifest['jobs']:
                    assert sha(config_dir / (new + '.yaml')) == manifest['config_sha256'][new + '.yaml']
                    continue
                cfg = OmegaConf.load(config_dir / (old_name + tail + '.yaml'))
                job = copy.deepcopy(manifest['jobs'][old_name + tail])
                output = suite / ('evals' if tail else 'runs') / new
                cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = new, str(output), str(output)
                job['output'] = str(output)
                job['requires'] = [n.replace(old_name, name) for n in job['requires']]
                job['control_training_protocol'] = 'retain_pool_nll_matched_updates_v1'
                if tail:
                    cfg.eval.lm_eval.output_dir = str(output)
                    cfg.model.lora.adapter_path = str(suite / 'runs' / name)
                    job['stage'] = stage
                else:
                    cfg.mode, cfg.trainer.handler = 'unlearn', 'GradDiff'
                    cfg.trainer.method_args = {'gamma': 0., 'alpha': 1., 'retain_loss_type': 'NLL'}
                    cfg.trainer.args.output_dir, cfg.trainer.args.logging_dir = str(output), str(output / 'logs')
                    cfg.trainer.args.save_strategy, cfg.trainer.args.save_only_model = 'no', True
                    cfg.checkpoint_first_step = cfg.native_checkpoint_io = False
                    job.pop('matching_control', None)
                    job['scope'] = 'ordinary_retain_only_control'
                file = config_dir / (new + '.yaml')
                OmegaConf.save(cfg, file)
                manifest['jobs'][new] = job
                manifest['config_sha256'][file.name] = sha(file)
            manifest.setdefault('control_stages', {}).setdefault(source, {})[phase] = stage
        atomic_json(path, manifest)
    return manifest['control_stages'][source]


# These change only public trainer options. All retain the same learning
# rate, epochs and matched normal-training control as the original recipe.
VARIANTS = {
    'GradDiff': [('g010', {'gamma': .1}), ('g025', {'gamma': .25}),
                 ('g020', {'gamma': .2}), ('g005', {'gamma': .05}),
                 ('kl025', {'gamma': .25, 'alpha': 1., 'retain_loss_type': 'KL'})],
    'NPO': [('retain2', {'alpha': 2.}), ('g050', {'gamma': .5}), ('retain4', {'alpha': 4.}), ('retain8', {'alpha': 8.}),
            ('retain16', {'alpha': 16.}),
            ('kl1', {'retain_loss_type': 'KL', 'alpha': 1.}),
            ('kl8', {'retain_loss_type': 'KL', 'alpha': 8.}),
            ('g2', {'gamma': 2.}), ('g4', {'gamma': 4.})],
    'SimNPO': [('g025', {'gamma': .25}), ('g050', {'gamma': .5}), ('g1', {'gamma': 1.}),
               ('beta01', {'beta': .1}), ('beta01g025', {'beta': .1, 'gamma': .25})],
    'UNDIAL': [('beta20', {'beta': 20.}), ('retain1', {'alpha': 1.}),
               ('beta20retain1', {'beta': 20., 'alpha': 1.})],
}


def prepare(suite):
    config_dir = suite / 'configs'
    with (config_dir / 'recipe-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = config_dir / 'jobs.json'
        manifest = json.loads(path.read_text())
        for source in manifest.get('active_sources', ('biographies', 'squad')):
            for method, variants in VARIANTS.items():
                for suffix, overrides in variants:
                    for phase in ('development', 'confirmation'):
                        old_stage, stage = phase + '_' + method, phase + '_' + method + '_' + suffix
                        old_name, name = source + '_' + old_stage, source + '_' + stage
                        new_output = suite / 'runs' / name
                        for tail in ('', '_knowledge', '_mmlu_validation'):
                            old, new = old_name + tail, name + tail
                            if new in manifest['jobs']:
                                continue
                            cfg = OmegaConf.load(config_dir / (old + '.yaml'))
                            job = copy.deepcopy(manifest['jobs'][old])
                            output = new_output if not tail else suite / 'evals' / new
                            cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = new, str(output), str(output)
                            job['output'] = str(output)
                            job['requires'] = [new_name.replace(old_name, name) for new_name in job['requires']]
                            if tail:
                                cfg.eval.lm_eval.output_dir = str(output)
                                cfg.model.lora.adapter_path = str(new_output)
                                job['stage'] = stage
                            else:
                                cfg.trainer.args.output_dir = str(output)
                                cfg.trainer.args.logging_dir = str(output / 'logs')
                                cfg.trainer.args.save_strategy = 'no'
                                cfg.trainer.args.save_only_model = True
                                cfg.checkpoint_first_step = False
                                cfg.native_checkpoint_io = False
                                cfg.trainer.method_args.update(overrides)
                            file = config_dir / (new + '.yaml')
                            OmegaConf.save(cfg, file)
                            manifest['jobs'][new] = job
                            manifest['config_sha256'][file.name] = sha(file)
        pending = path.with_suffix('.recipes.pending.json')
        pending.write_text(json.dumps(manifest, indent=2) + '\n')
        pending.replace(path)
    return len(manifest['jobs'])


def stages(phase, method):
    return ([phase + '_' + method] + [phase + '_' + method + '_' + suffix for suffix, _ in VARIANTS[method]]
            + [phase + '_' + method + '_factspan'])


def selected(suite, source):
    path = suite / ('SELECTED_RECIPES-' + source + '.json')
    return json.loads(path.read_text()) if path.exists() else {}


def appropriate_follow_up(method, stage, initial, policy):
    """A selective-forgetting failure needs stronger retention, not stronger erasure."""
    if method != 'NPO' or not initial:
        return True
    target_ok = initial['target_previously_correct_fraction_remaining'] <= policy['remaining_previously_correct_max']
    retain_bad = initial['retain_previously_correct_fraction_remaining'] < policy['retain_previously_correct_min']
    if target_ok and retain_bad:
        return stage.split('_NPO', 1)[1] in ('', '_retain2', '_retain4', '_retain8', '_retain16', '_kl1', '_kl8', '_factspan')
    return True


def validate_follow_up(suite, name, manifest):
    """Enforce current quality decisions before an old coordinator loads a model."""
    if '_performance_' in name:
        return
    for source in manifest.get('active_sources', ('biographies', 'squad')):
        prefix = source + '_development_NPO_'
        if not name.startswith(prefix):
            continue
        file = suite / 'gates' / (source + '-development_NPO-quality.json')
        if not file.exists():
            return
        report = json.loads(file.read_text())
        default = source + '_development_NPO.yaml'
        if report.get('proof_kind') != 'official_model_evaluation' or \
                report.get('data_readiness_sha256') != manifest['data_readiness_sha256'] or \
                report.get('config_sha256', {}).get(default) != manifest['config_sha256'][default]:
            raise ValueError('Quality decision needs a current model/data/configuration receipt')
        if not appropriate_follow_up('NPO', name[len(source) + 1:], report, manifest['quality_policy']):
            raise ValueError('Quality plan changed: refresh coordinator before training ' + name)


def repaired(suite, source, method):
    """Accept a repaired recipe only after real, fresh-content confirmation."""
    path = suite / ('REPAIR_RECIPES-' + source + '.json')
    if not path.exists():
        return None
    recipe = json.loads(path.read_text()).get(method)
    if recipe is None:
        return None
    assert source == 'biographies' and method in REPAIR_DIAGNOSTICS
    fresh = Path(recipe['fresh_confirmation_suite']).resolve()
    assert fresh != suite.resolve() and fresh.is_relative_to(suite.parent.resolve()) and fresh.stat().st_uid == os.getuid()
    prepared = json.loads((fresh / 'FRESH_PREPARED.json').read_text())
    result = json.loads((fresh / 'FRESH_CONFIRMATION_RESULT.json').read_text())
    assert result['proof_kind'] == 'official_model_evaluation' and result['passed']
    assert not result['parameter_search_on_this_target']
    assert result.get('method', 'NPO') == method and prepared['method'] == method
    revision = prepared.get('revision', 1)
    assert recipe.get('revision', 1) == revision
    precommit = suite / precommitment_name(method, revision)
    assert prepared['target_precommitment_sha256'] == sha(precommit)
    target = json.loads(precommit.read_text())
    assert result['fresh_target_id'] == target['unit_id'] and target['unit_id'] not in target['previous_unlearning_target_ids']
    diagnostic = suite / (diagnostic_name(method, revision) + '_RESULT.json')
    assert prepared['recipe_selection_evidence_sha256'] == sha(diagnostic)
    prior = json.loads(diagnostic.read_text())
    assert prior['both_seen_tuning_targets_passed'] and prior['selected_candidate']
    candidate = prior['selected_candidate']
    old_stage, new_stage = 'development_' + method + '_' + candidate, 'confirmation_' + method
    manifest = json.loads((suite / 'configs/jobs.json').read_text())
    assert 'biographies_' + old_stage in manifest['jobs']
    old_cfg = OmegaConf.load(suite / ('configs/biographies_' + old_stage + '.yaml'))
    new_cfg = OmegaConf.load(fresh / ('configs/biographies_' + new_stage + '.yaml'))
    assert OmegaConf.to_container(old_cfg.trainer.method_args) == OmegaConf.to_container(new_cfg.trainer.method_args)
    assert old_cfg.model.model_args.pretrained_model_name_or_path == new_cfg.model.model_args.pretrained_model_name_or_path
    for key in ('learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps'):
        assert old_cfg.trainer.args[key] == new_cfg.trainer.args[key]
    expected_protocol = manifest['jobs']['biographies_' + old_stage].get('forget_data_protocol', 'full_published_records_v1')
    assert prepared['forget_data_protocol'] == expected_protocol
    from official_knowledge.qualify import assess
    assert assess(fresh, source)['targets']['confirmation']['passed']
    assert assess(fresh, source, 'confirmation_control_1e-4')['passed']
    assert assess(fresh, source, new_stage)['passed']
    assert recipe['development_stage'] == old_stage
    assert recipe['confirmation_stage'] == old_stage.replace('development_', 'confirmation_', 1)
    return recipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--retention-controls-source', choices=('biographies', 'squad'))
    args = parser.parse_args()
    if args.retention_controls_source:
        print(json.dumps({'control_stages': prepare_retention_controls(args.suite, args.retention_controls_source),
                          'training_started': False}))
    else:
        print(json.dumps({'prepared_jobs': prepare(args.suite), 'training_started': False}))
