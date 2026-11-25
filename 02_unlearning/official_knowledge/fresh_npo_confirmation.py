"""Freeze a repaired official recipe and test a precommitted third biography.

Only configuration/data glue: all training and evaluation use the pinned
OpenUnlearning entry points. No parameter search on this fresh target.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from omegaconf import OmegaConf
from transformers import AutoTokenizer
import yaml

from official_knowledge.fact_values import PROTOCOL, validated_rows
from official_knowledge.prepare_data import biographies, matched_control, write
from official_knowledge.production_jobs import atomic_json, records
from official_knowledge.run_pilot import Pilot, now, sha
from official_knowledge.run_source import SourcePilot
from official_knowledge.pilot_recipes import REPAIR_DIAGNOSTICS, precommitment_name, diagnostic_name


def prepare(project, upstream, base, method='NPO', revision=1):
    original = project / ('official-lora-pilot-v1-' + base)
    result_file = original / (diagnostic_name(method, revision) + '_RESULT.json')
    result = json.loads(result_file.read_text())
    assert result['proof_kind'] == 'official_model_evaluation'
    assert result['both_seen_tuning_targets_passed'] and result['selected_candidate']
    candidate = result['selected_candidate']
    old_method = method + '_' + candidate
    target_file = original / precommitment_name(method, revision)
    target = json.loads(target_file.read_text())
    uid = target['unit_id']
    assert uid not in target['previous_unlearning_target_ids']
    if revision > 1:
        prior = json.loads((project / f'official-fresh-{method.lower()}-v{revision - 1}-{base}' / 'FRESH_CONFIRMATION_RESULT.json').read_text())
        assert prior['proof_kind'] == 'official_model_evaluation' and not prior['passed']
        assert uid != prior['fresh_target_id']
    suite = project / f'official-fresh-{method.lower()}-v{revision}-{base}'
    suite.mkdir(exist_ok=True)
    ready = suite / 'FRESH_PREPARED.json'
    if ready.exists():
        receipt = json.loads(ready.read_text())
        assert receipt['recipe_selection_evidence_sha256'] == sha(result_file)
        assert receipt['target_precommitment_sha256'] == sha(target_file)
        return suite
    with (original / 'configs/recipe-preparation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        old = json.loads((original / 'configs/jobs.json').read_text())
        names = ['biographies_learn']
        names += [f'biographies_{stage}_{a}' for stage in ('base', 'learn', 'learn_adapter')
                  for a in ('knowledge', 'mmlu_validation')]
        names += [f'biographies_confirmation_{kind}{tail}' for kind in ('control_1e-4', old_method)
                  for tail in ('', '_knowledge', '_mmlu_validation')]
        configs = {n: (original / 'configs' / (n + '.yaml')).read_text() for n in names}
    protocol = old['jobs']['biographies_confirmation_' + old_method].get('forget_data_protocol', 'full_published_records_v1')
    assert protocol in (PROTOCOL, 'full_published_records_v1')
    original_data = json.loads((original / 'data-readiness.json').read_text())
    learning = records(original / 'data/biographies/learn.jsonl')
    assert len(learning) == 1440 and len({r['unit_id'] for r in learning}) == 12
    selected = [r for r in learning if r['unit_id'] == uid]
    assert len(selected) == 120
    retained = [r for r in learning if r['unit_id'] != uid]
    units = {u['unit_id']: u for u in biographies(project / 'datasets/knowledge-20261101')[0]}
    parent = next(p for p in json.loads((project / 'formal-lora-400-v1/parents.json').read_text())
                  if p['base'] == base and p['source'] == 'biographies')
    tokenizer = AutoTokenizer.from_pretrained(parent['hdd_original'], local_files_only=True)
    from data.utils import preprocess_pretraining_instance
    converted = validated_rows(selected, units, tokenizer, preprocess_pretraining_instance) if protocol == PROTOCOL else selected
    data = suite / 'data/biographies'
    write(data / 'learn.jsonl', learning)
    assert sha(data / 'learn.jsonl') == sha(original / 'data/biographies/learn.jsonl')
    write(data / 'confirmation-forget.jsonl', converted)
    write(data / 'confirmation-retain.jsonl', retained)
    control = matched_control(selected, retained, uid)
    if protocol == PROTOCOL:
        control = validated_rows(control, units, tokenizer, preprocess_pretraining_instance)
    write(data / 'confirmation-control.jsonl', control)
    info = copy.deepcopy(original_data['sources']['biographies'])
    development = info['targets']['development']['unit_id']
    retention_ids = {r['unit_id'] for r in retained} - {development}
    assert len(retention_ids) == 10 and uid not in retention_ids
    evals = {'development': units[development]['probes'], 'confirmation': units[uid]['probes'],
             'retain': [p for other in sorted(retention_ids) for p in units[other]['probes']]}
    for phase, probes in evals.items():
        write(data / ('eval-' + phase + '.jsonl'), probes)
    info['targets']['confirmation'] = {
        'unit_id': uid, 'title': units[uid]['title'], 'forget_rows': 120,
        'retain_pool_rows': len(retained), 'control_rows': 120,
        'probe_rows': len(evals['confirmation']), 'forget_tokens': sum(r['tokens'] for r in converted),
        'control_tokens': sum(r['tokens'] for r in control), 'fresh_unlearning_target': True}
    info['eval_counts'] = {p: len(v) for p, v in evals.items()}
    readiness = copy.deepcopy(original_data)
    readiness.update(sources={'biographies': info}, scope='fresh_third_target_confirmation_only',
        original_suite=str(original),
        derived_sha256={str(p.relative_to(suite)): sha(p) for p in data.glob('*.jsonl')})
    atomic_json(suite / 'data-readiness.json', readiness)
    tasks = suite / 'tasks'
    tasks.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('knowledge_metrics.py'), tasks / 'knowledge_metrics.py')
    for phase in evals:
        name = 'knowledge_biographies_' + phase
        text = (original / 'tasks' / (name + '.yaml')).read_text()
        task = yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('process_results:')))
        task['dataset_kwargs']['data_files']['test'] = str(data / ('eval-' + phase + '.jsonl'))
        task['dataset_kwargs']['cache_dir'] = str(suite / 'cache/lm-eval')
        (tasks / (name + '.yaml')).write_text(yaml.safe_dump(task, sort_keys=False) +
            'process_results: !function knowledge_metrics.process_results\n')
    config_dir = suite / 'configs'
    config_dir.mkdir(exist_ok=True)
    manifest = copy.deepcopy(old)
    manifest.update(suite=suite.name, active_sources=['biographies'], methods=[method],
        scope='fresh_third_target_confirmation_only', jobs={}, config_sha256={}, task_sha256={},
        data_readiness_sha256=sha(suite / 'data-readiness.json'), formal_production_authorized=False)
    original_pilot = Pilot(original, upstream, 48)
    for old_name, text in configs.items():
        name = old_name.replace(old_method, method)
        job = copy.deepcopy(old['jobs'][old_name])
        cfg = OmegaConf.create(text)
        file = config_dir / (name + '.yaml')
        reuse_mmlu = job['kind'] == 'eval' and job['stage'] in ('base', 'learn', 'learn_adapter') and job['assessment'].startswith('mmlu_')
        if old_name == 'biographies_learn' or reuse_mmlu:
            if reuse_mmlu:
                assert original_pilot.completed(old_name)
                from official_knowledge.production_worker import assessment
                assessment(original_pilot, old_name)
                job['evaluation_alias'] = {'suite': str(original), 'source_config_sha256': sha(original / 'configs' / file.name), 'read_only': True}
            file.write_text(text)
        else:
            out = suite / ('runs' if job['kind'] == 'train' else 'evals') / name
            job['output'] = str(out)
            cfg.task_name, cfg.paths.root_dir, cfg.paths.data_dir = name, str(suite), str(suite / 'data') + '/'
            cfg.paths.output_dir = cfg.hydra.run.dir = str(out)
            if job['kind'] == 'train':
                cfg.trainer.args.output_dir, cfg.trainer.args.logging_dir = str(out), str(out / 'logs')
                cfg.trainer.args.save_strategy, cfg.trainer.args.save_only_model = 'no', True
                cfg.checkpoint_first_step = cfg.native_checkpoint_io = False
                job['scope'] = 'fresh_third_target_confirmation_only'
                if '_control_' in name:
                    cfg.data.train.published_records.args.hf_args.data_files = str(data / 'confirmation-control.jsonl')
                    job['training_data_protocol'] = protocol
                else:
                    for key in ('forget', 'retain'):
                        cfg.data[key].published_records.args.hf_args.data_files = str(data / ('confirmation-' + key + '.jsonl'))
                    job['forget_data_protocol'] = protocol
            else:
                cfg.eval.lm_eval.output_dir = str(out)
                if job['assessment'] == 'knowledge':
                    cfg.eval.lm_eval.include_path = str(tasks)
                if job['stage'] not in ('base', 'learn', 'learn_adapter'):
                    job['stage'] = job['stage'].replace(old_method, method)
                    cfg.model.lora.adapter_path = str(suite / 'runs' / ('biographies_' + job['stage']))
            job['requires'] = [n.replace(old_method, method) for n in job['requires']]
            OmegaConf.save(cfg, file)
        manifest['jobs'][name] = job
        manifest['config_sha256'][file.name] = sha(file)
    manifest['task_sha256'] = {str(p.relative_to(suite)): sha(p) for p in tasks.iterdir()}
    atomic_json(config_dir / 'jobs.json', manifest)
    assert Pilot(suite, upstream, 48).completed('biographies_learn')
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'data_configuration_only',
        'optimizer_steps': 0, 'formal_production_authorized': False,
        'original_suite': str(original), 'suite': str(suite), 'method': method, 'revision': revision,
        'forget_data_protocol': protocol, 'target_id': uid, 'target_title': units[uid]['title'],
        'learning_records': 1440, 'forget_records': 120, 'control_records': 120,
        'retention_training_records': len(retained), 'evaluation_counts': info['eval_counts'],
        'recipe_selection_evidence_sha256': sha(result_file),
        'target_precommitment_sha256': sha(target_file), 'learning_parent_retrained': False,
        'parameter_search_on_this_target': False, 'intermediate_checkpoints': 0}
    atomic_json(ready, receipt)
    return suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--base', choices=('llama2-chat', 'llama3-instruct'), default='llama3-instruct')
    p.add_argument('--method', choices=tuple(REPAIR_DIAGNOSTICS), default='NPO')
    p.add_argument('--revision', type=int, default=1)
    args = p.parse_args()
    original = args.root / ('official-lora-pilot-v1-' + args.base)
    state_prefix = '' if args.method == 'NPO' else args.method.upper() + '-'
    if args.revision < 1:
        raise ValueError('Fresh-confirmation revision must be positive')
    if args.revision > 1:
        state_prefix += f'v{args.revision}-'
    statefile = args.root / ('FRESH_CONFIRMATION-' + state_prefix + args.base + '.json')
    with (args.root / ('fresh-confirmation-' + state_prefix.lower() + args.base + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 48 * 3600
        while True:
            file = original / (diagnostic_name(args.method, args.revision) + '_RESULT.json')
            result = json.loads(file.read_text()) if file.exists() else {}
            if result.get('both_seen_tuning_targets_passed') and result.get('tested_phases') == ['development', 'confirmation']:
                break
            diagnostic_status = original / (diagnostic_name(args.method, args.revision) + '_STATUS.json')
            status = json.loads(diagnostic_status.read_text()) if diagnostic_status.exists() else {}
            if status.get('phase') == 'needs_research':
                atomic_json(statefile, {'at_utc': now(), 'pid': os.getpid(), 'phase': 'repair_still_failed', 'formal_production_authorized': False})
                return
            atomic_json(statefile, {'at_utc': now(), 'pid': os.getpid(), 'phase': 'waiting_for_two_diagnostic_targets', 'formal_production_authorized': False})
            if time.monotonic() > deadline:
                raise TimeoutError('Repair diagnostic waiting limit reached')
            time.sleep(60)
        try:
            suite = prepare(args.root, args.upstream, args.base, args.method, args.revision)
        except Exception as error:
            atomic_json(statefile, {'at_utc': now(), 'pid': os.getpid(), 'phase': 'needs_repair',
                'error': repr(error), 'formal_production_authorized': False})
            raise
        pilot = SourcePilot(suite, args.upstream, 48, 'biographies')
        try:
            for stage in ('base', 'learn', 'learn_adapter'):
                pilot.evaluate('biographies', stage)
            learning = pilot.qualification('biographies')
            if not learning['targets']['confirmation']['passed']:
                pilot.record(phase='fresh_target_learning_failed', current_job=None)
                atomic_json(statefile, {'at_utc': now(), 'pid': os.getpid(), 'phase': 'fresh_target_learning_failed',
                    'passed': False, 'formal_production_authorized': False})
                return
            passed = False
            for stage in ('confirmation_control_1e-4', 'confirmation_' + args.method):
                pilot.run('biographies_' + stage)
                pilot.evaluate('biographies', stage)
                outcome = pilot.qualification('biographies', stage)
                if not outcome['passed']:
                    break
                passed = stage == 'confirmation_' + args.method
            receipt = {'at_utc': now(), 'pid': os.getpid(), 'proof_kind': 'official_model_evaluation',
                'phase': 'fresh_confirmation_passed' if passed else 'fresh_confirmation_failed',
                'passed': passed, 'formal_production_authorized': False,
                'suite': str(suite), 'fresh_target_id': json.loads((suite / 'FRESH_PREPARED.json').read_text())['target_id'],
                'method': args.method, 'revision': args.revision, 'frozen_recipe': json.loads((suite / 'FRESH_PREPARED.json').read_text())['forget_data_protocol'],
                'parameter_search_on_this_target': False}
            atomic_json(suite / 'FRESH_CONFIRMATION_RESULT.json', receipt)
            atomic_json(statefile, receipt)
            if passed:
                with (original / 'repair-adoption.lock').open('a') as adopt:
                    fcntl.flock(adopt, fcntl.LOCK_EX)
                    path = original / 'REPAIR_RECIPES-biographies.json'
                    recipes = json.loads(path.read_text()) if path.exists() else {}
                    selected = json.loads((original / (diagnostic_name(args.method, args.revision) + '_RESULT.json')).read_text())['selected_candidate']
                    recipes[args.method] = {
                        'development_stage': 'development_' + args.method + '_' + selected,
                        'confirmation_stage': 'confirmation_' + args.method + '_' + selected,
                        'repaired_after_original_confirmation': True,
                        'fresh_confirmation_suite': str(suite), 'revision': args.revision}
                    atomic_json(path, recipes)
                from official_knowledge.pilot_recipes import repaired
                repaired(original, 'biographies', args.method)
            pilot.record(phase=receipt['phase'], current_job=None)
        except Exception as error:
            pilot.record(phase='needs_repair', error=repr(error))
            atomic_json(statefile, {'at_utc': now(), 'pid': os.getpid(), 'phase': 'needs_repair', 'error': repr(error), 'formal_production_authorized': False})
            raise


if __name__ == '__main__':
    main()
