"""Validate saved artifacts, then run a gated source/base cell of formal LoRA jobs."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

from official_knowledge.knowledge_metrics import process_results
from official_knowledge.production_jobs import ProductionJobs, records, atomic_json
from official_knowledge.resource_guard import monitor, scope_command
from official_knowledge.run_pilot import Pilot, now, sha


EXTRACTOR = Path('/local/rcs/zh2701/LLM-safety/code/LLM_ULP-BackdoorID-AcT')
EXTRACTOR_PYTHON = Path('/local/rcs/zh2701/LLM-safety/envs/backdoorid/bin/python')
PROTOCOLS = ('paper-t2', 't32-standard-chatfix')


def summarize(samples):
    groups = collections.defaultdict(list)
    units = collections.defaultdict(lambda: collections.defaultdict(list))
    f1 = []
    for sample in samples:
        doc = sample['doc']
        score = process_results(doc, [sample['filtered_resps'][0]])
        for key, value in score.items():
            assert abs(sample[key] - value) < 1e-8
        groups[doc['fact_id']].append(score['knowledge_accuracy'])
        units[doc['unit_id']][doc['fact_id']].append(score['knowledge_accuracy'])
        f1.append(score['answer_f1'])
    facts = {k: sum(v) / len(v) for k, v in groups.items()}
    by_unit = {uid: {key: sum(values) / len(values) for key, values in fs.items()} for uid, fs in units.items()}
    return {'questions': len(samples), 'knowledge_accuracy': sum(facts.values()) / len(facts),
            'answer_f1': sum(f1) / len(f1), 'per_fact': facts, 'per_unit': by_unit}


def kept(before, after):
    keys = [k for k, v in before.items() if v >= .5]
    return sum(after.get(k, 0) >= .5 for k in keys) / len(keys) if keys else 0.0


def matching_parent_mmlu(pilot, source, case_name):
    """Compare the parent and final adapter on the exact same saved questions."""
    parent_job = pilot.manifest['jobs'][source + '_learn_mmlu_validation']
    case_job = pilot.manifest['jobs'][case_name + '_mmlu_validation']
    parent = json.loads((Path(parent_job['output']) / 'LMEval_EVAL.json').read_text())['pilot_mmlu_validation']
    after = json.loads((Path(case_job['output']) / 'LMEval_EVAL.json').read_text())['pilot_mmlu_validation']
    correct, count = 0, 0
    for subject, samples in after.items():
        prior = {sample['doc_id']: sample for sample in parent[subject]}
        for sample in samples:
            before = prior[sample['doc_id']]
            assert sample['doc'] == before['doc']
            correct += int(before['acc'])
            count += 1
    assert count == case_job.get('expected_mmlu_count', 1531)
    return correct / count, count


def assessment(pilot, name):
    job = pilot.manifest['jobs'][name]
    directory = Path(job['output'])
    marker = directory / 'prepared-evaluation.json'
    identity = json.loads(marker.read_text())
    assert not (pilot.root / 'PROOF_KIND.txt').exists()
    assert identity.get('proof_kind') != 'software_fixture_only' and identity['model_files']
    assert identity['config_sha256'] == pilot.manifest['config_sha256'][name + '.yaml']
    assert identity['data_readiness_sha256'] == pilot.manifest['data_readiness_sha256']
    for path, expected in identity['model_files'].items():
        st = Path(path).stat()
        assert [st.st_size, st.st_mtime_ns] == expected
    logs_path, summary_path = directory / 'LMEval_EVAL.json', directory / 'LMEval_SUMMARY.json'
    logs, summary = json.loads(logs_path.read_text()), json.loads(summary_path.read_text())
    evidence = {str(p.relative_to(pilot.root)): sha(p) for p in (marker, logs_path, summary_path)}
    if job['assessment'] == 'mmlu_validation':
        group = logs['pilot_mmlu_validation']
        count = sum(len(x) for x in group.values())
        assert count == job.get('expected_mmlu_count', 1531)
        correct = 0
        for samples in group.values():
            assert len({x['doc_id'] for x in samples}) == len(samples)
            for sample in samples:
                predicted = max(range(len(sample['filtered_resps'])), key=lambda i: sample['filtered_resps'][i][0])
                truth = sample['doc']['answer']
                truth = 'ABCD'.index(truth) if isinstance(truth, str) and truth in 'ABCD' else int(truth)
                correct += int(predicted == truth)
                assert sample['acc'] == int(predicted == truth)
        result = summary['pilot_mmlu_validation/acc']
        assert 0 <= result <= 1 and abs(result - correct / count) < 1e-8
    else:
        result = {}
        for task, count in job['expected_task_counts'].items():
            samples = logs[task][task]
            assert len(samples) == count
            data_file = job.get('expected_data_paths', {}).get(task)
            if data_file is None:
                cfg = yaml.safe_load('\n'.join(line for line in (pilot.root / 'tasks' / (task + '.yaml')).read_text().splitlines()
                                               if not line.startswith('process_results:')))
                data_file = cfg['dataset_kwargs']['data_files']['test']
            expected = {q['probe_id']: q for q in records(Path(data_file))}
            assert len(expected) == count and len({s['doc']['probe_id'] for s in samples}) == count
            assert all(sample['doc'] == expected[sample['doc']['probe_id']] for sample in samples)
            result[task] = summarize(samples)
    return result, evidence, identity['model_files']


class FormalPilot(Pilot):
    def __init__(self, root, upstream, wait_hours):
        super().__init__(root, upstream, wait_hours)
        self.state.update(scope='formal_lora_400_gated_cell', primary_target=400)

    def report(self):
        return

    def record(self, **values):
        self.state.update(values, coordinator_pid=os.getpid(), updated_at_utc=now())
        atomic_json(self.root / 'PRODUCTION_WORKER.json', self.state)
        print(json.dumps({'at': now(), **values}), flush=True)

    def completed(self, name):
        job = self.manifest['jobs'][name]
        if job['kind'] != 'eval' or (job['assessment'] != 'knowledge' and 'expected_mmlu_count' not in job):
            return super().completed(name)
        try:
            assessment(self, name)
            if job.get('evaluation_execution_protocol') == 'joint-official-eval-v1' and job['assessment'] == 'knowledge':
                assessment(self, name.removesuffix('_knowledge') + '_mmlu_validation')
            return True
        except (FileNotFoundError, KeyError, AssertionError):
            return False

    def activation(self, spec):
        output = Path(spec['output'])
        if (output / 'ACTIVATION_RESULT.json').exists():
            result = json.loads((output / 'ACTIVATION_RESULT.json').read_text())
            assert result['adapter_sha256'] == sha(Path(spec['adapter']) / 'adapter_model.safetensors')
            return result
        output.mkdir(parents=True, exist_ok=True)
        spec_path = output / 'extraction-spec.json'
        atomic_json(spec_path, spec)
        name = next(n for n, j in self.manifest['jobs'].items() if j['kind'] == 'eval')
        while True:
            gpus = self.resources(name)
            unit = f'ulp-small-{os.getpid()}-activation-{time.time_ns()}.scope'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus),
                PYTHONPATH=str(self.support) + ':' + str(EXTRACTOR), CUDA_HOME='/usr/local/cuda',
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                HF_HOME=str(self.root / 'cache/hf'), ULP_SCOPE_UNIT=unit,
                ULP_HOST_MEMORY_MAX_BYTES=str(self.policy['evaluation_memory_max_gib'] * 1024**3),
                ULP_HOST_START_RESERVE_BYTES=str(self.policy['host_start_reserve_gib'] * 1024**3),
                ULP_CUDA_MEMORY_FRACTION=str(self.policy['cuda_allocator_fraction']))
            cmd = [str(EXTRACTOR_PYTHON), '-m', 'official_knowledge.activation_entry', '--spec', str(spec_path)]
            log = output / 'extract.log'
            self.record(phase='extracting_activations', current_job=spec['record_id'], protocol=spec['protocol'], gpus=gpus)
            with log.open('a') as stream:
                process = subprocess.Popen(scope_command(cmd, unit, self.policy, False), env=env,
                    cwd=self.support, stdout=stream, stderr=subprocess.STDOUT)
                reason = monitor(process, unit, gpus, self.policy, output / 'resources.json', log)
            if process.returncode == 0 and not reason:
                return json.loads((output / 'ACTIVATION_RESULT.json').read_text())
            if reason in ('another_process_appeared_on_selected_gpu', 'host_memory_pressure'):
                continue
            raise RuntimeError('Saved-adapter activation check failed: ' + str(log))


def activation_spec(project, parent, adapter, record_id, output, protocol, split='pilot_only'):
    return {'asset_root': str(project), 'adapter': str(adapter), 'parent': str(parent),
            'record_id': record_id, 'output': str(output), 'protocol': protocol, 'split': split,
            'base_model_id': None, 'extractor_code': str(EXTRACTOR),
            'extractor_sha256': sha(EXTRACTOR / 'tamperid/extract.py')}


def preflight(pilot, builder):
    parent = builder.parent
    path = builder.project / 'formal-lora-400-v1/preflight' / (builder.root.name + '.json')
    if path.exists():
        assert json.loads(path.read_text())['passed']
        return
    results = {}
    while not (builder.project / 'shared-bases/READY.json').exists():
        time.sleep(30)
    source = parent['source']
    small_learn = json.loads((builder.pilot / 'configs/jobs.json').read_text())['jobs'][source + '_learn']['output']
    small_parent = Path(small_learn) / 'merged'
    from official_knowledge.pilot_recipes import selected, selected_control
    recipes = selected(builder.pilot, source)
    requests = []
    for method in ('NPO', 'GradDiff', 'SimNPO', 'UNDIAL', 'control_1e-4'):
        stage = selected_control(builder.pilot, source, 'confirmation') if method == 'control_1e-4' else selected(builder.pilot, source)[method]['confirmation_stage']
        adapter = builder.pilot / 'runs' / (source + '_' + stage)
        requests.append((method, adapter, int(method != 'control_1e-4')))
    for method, recipe in recipes.items():
        if recipe.get('fresh_confirmation_suite') and pilot.manifest['quality_policy'].get('acceptance_mode') != 'pipeline_first':
            fresh = Path(recipe['fresh_confirmation_suite'])
            requests += [('fresh-' + method, fresh / 'runs' / (source + '_confirmation_' + method), 1),
                         ('fresh-control-' + method, fresh / 'runs' / (source + '_confirmation_control_1e-4'), 0)]
    for method, adapter, unlearning_label in requests:
        for protocol in PROTOCOLS:
            rid = 'pilot-' + builder.root.name + '-' + method.lower()
            output = builder.project / 'formal-lora-400-v1/preflight-activations' / protocol / rid
            spec = activation_spec(builder.project, small_parent, adapter, rid, output, protocol)
            spec['base_model_id'] = parent['base_model_id']
            spec['tokenizer_source'] = parent['hdd_original']
            spec['unlearning_label'] = unlearning_label
            results[method + '/' + protocol] = pilot.activation(spec)
    atomic_json(path, {'passed': True, 'at_utc': now(), 'checks': results})
    for handle in getattr(pilot, 'gpu_reservations', []):
        handle.close()
    pilot.gpu_reservations = []


def wait_preflight(project):
    root = project / 'formal-lora-400-v1'
    while True:
        paths = list((root / 'preflight').glob('*.json'))
        if len(paths) == 4 and all(json.loads(p.read_text()).get('passed') for p in paths):
            atomic_json(root / 'PREFLIGHT_ACCEPTED.json', {'passed': True, 'at_utc': now(),
                'source_receipts': {p.name: sha(p) for p in paths}})
            if (project / 'shared-bases/READY.json').exists():
                return
        time.sleep(30)


def parent_qualification(pilot, builder):
    source = builder.source
    evidence, model_files, scores = {}, {}, {}
    for stage in ('base', 'learn', 'learn_adapter'):
        for kind in ('knowledge', 'mmlu_validation'):
            name = f'{source}_{stage}_{kind}'
            if stage == 'learn':
                pilot.run(source + '_learn')
            pilot.run(name)
            value, files, models = assessment(pilot, name)
            scores[stage + '/' + kind] = value
            evidence.update(files)
            model_files.update(models)
    target_name, retain_name = f'formal_{source}_targets', f'formal_{source}_retain'
    learned, direct = scores['learn/knowledge'], scores['learn_adapter/knowledge']
    retained = learned[retain_name]
    mmlu_base, mmlu_learn, mmlu_adapter = (scores[k + '/mmlu_validation'] for k in ('base', 'learn', 'learn_adapter'))
    policy = pilot.manifest['quality_policy']
    global_ok = retained['knowledge_accuracy'] >= policy['learned_retain_accuracy_min'] and \
        mmlu_base - mmlu_learn <= policy['mmlu_drop_max'] and \
        mmlu_adapter - mmlu_learn <= policy['mmlu_drop_max'] and \
        kept(direct[retain_name]['per_fact'], retained['per_fact']) >= policy['retain_previously_correct_min']
    learning_effect_checks_passed = global_ok
    if policy.get('acceptance_mode') == 'pipeline_first':
        global_ok = kept(direct[retain_name]['per_fact'], retained['per_fact']) >= policy['retain_previously_correct_min'] and \
            mmlu_adapter - mmlu_learn <= policy['mmlu_drop_max']
    report = {'passed': global_ok, 'proof_kind': 'official_model_evaluation', 'at_utc': now(), 'parent': builder.parent,
              'scores': scores, 'evidence_sha256': evidence, 'model_files': model_files,
              'learning_records': builder.parent['learning_rows'], 'general_ability_questions': 1531,
              'learning_effect_checks_passed': learning_effect_checks_passed,
              'acceptance_scope': policy.get('acceptance_mode', 'effect_thresholds')}
    atomic_json(builder.root / 'PARENT_QUALIFIED.json', report)
    if not global_ok:
        raise RuntimeError('Formal learned parent needs parameter research; production remains gated')
    return report


def case_learning_gate(pilot, builder, task, parent_report, gate_source):
    assert parent_report['proof_kind'] == 'official_model_evaluation' and parent_report['evidence_sha256']
    source, uid = builder.source, task['unit_id']
    name = f'formal_{source}_targets'
    scores = parent_report['scores']
    before = scores['base/knowledge'][name]['per_unit'][uid]
    after = scores['learn/knowledge'][name]['per_unit'][uid]
    direct = scores['learn_adapter/knowledge'][name]['per_unit'][uid]
    accuracy, gain = sum(after.values()) / len(after), sum(after.values()) / len(after) - sum(before.values()) / len(before)
    policy = pilot.manifest['quality_policy']
    passed = parent_report['passed'] and accuracy >= policy['learned_knowledge_accuracy_min'] and \
        gain >= policy['learning_gain_min'] and kept(direct, after) >= policy['retain_previously_correct_min']
    if policy.get('acceptance_mode') == 'pipeline_first':
        passed = parent_report['passed'] and any(value >= .5 for value in after.values()) and \
            kept(direct, after) >= policy['retain_previously_correct_min']
    gate = {'proof_kind': 'official_model_evaluation', 'global_checks_passed': parent_report['passed'],
        'targets': {'development': {'passed': passed, 'learned_accuracy': accuracy, 'learning_gain': gain}},
        'config_sha256': pilot.manifest['config_sha256'], 'data_readiness_sha256': pilot.manifest['data_readiness_sha256'],
        'evidence_sha256': parent_report['evidence_sha256'], 'model_files': parent_report['model_files']}
    atomic_json(pilot.root / 'gates' / (gate_source + '-learning.json'), gate)
    return passed


def run_case(pilot, builder, task, parent_report):
    accepted_path = builder.root / 'accepted' / (task['task_id'] + '.json')
    if accepted_path.exists():
        return json.loads(accepted_path.read_text())
    name, gate_source = builder.case(task)
    pilot.manifest = builder.manifest
    if not case_learning_gate(pilot, builder, task, parent_report, gate_source):
        result = {'passed': False, 'task': task, 'reason': 'target_not_sufficiently_learned'}
        atomic_json(builder.root / 'rejected' / (task['task_id'] + '.json'), result)
        return result
    pilot.run(name)
    model = Path(pilot.manifest['jobs'][name]['output'])
    assert not list(model.glob('checkpoint-*')), 'A formal final-only task wrote an intermediate checkpoint'
    for kind in ('knowledge', 'mmlu_validation'):
        pilot.run(name + '_' + kind)
    after, evidence, models = assessment(pilot, name + '_knowledge')
    mmlu, em, mm = assessment(pilot, name + '_mmlu_validation')
    evidence.update(em)
    models.update(mm)
    target_name, retain_name = list(pilot.manifest['jobs'][name + '_knowledge']['expected_task_counts'])
    original = parent_report['scores']['learn/knowledge']
    prior_target = original[f'formal_{builder.source}_targets']['per_unit'][task['unit_id']]
    prior_retain = original[f'formal_{builder.source}_retain']['per_fact']
    remaining, retention = kept(prior_target, after[target_name]['per_fact']), kept(prior_retain, after[retain_name]['per_fact'])
    mmlu_before, mmlu_questions = matching_parent_mmlu(pilot, builder.source, name)
    policy = pilot.manifest['quality_policy']
    from official_knowledge.acceptance import effect_checks, unlearning_passed
    passed = unlearning_passed(policy, remaining, retention, mmlu_before, mmlu)
    train_cfg = pilot.configuration(name)
    assert train_cfg.data.anchor == 'forget'
    epochs = int(train_cfg.trainer.args.num_train_epochs)
    forget_rows = pilot.manifest['jobs'][name]['primary_rows']
    result = {'passed': passed, 'proof_kind': 'official_model_evaluation', 'at_utc': now(), 'task': task, 'adapter': str(model), 'adapter_sha256': sha(model / 'adapter_model.safetensors'),
        'quality_policy': dict(policy),
        'quality_policy_id': pilot.manifest.get('quality_policy_id', 'legacy-strict-30pct'),
        'acceptance_scope': policy.get('acceptance_mode', 'effect_thresholds'),
        'effect_checks': effect_checks(policy, remaining, retention, mmlu_before, mmlu),
        'target_previously_correct_fraction_remaining': remaining, 'retain_previously_correct_fraction_remaining': retention,
        'general_ability_before': mmlu_before, 'general_ability_after': mmlu, 'general_ability_questions': mmlu_questions,
        'general_ability_parent_full_validation': parent_report['scores']['learn/mmlu_validation'],
        'knowledge_results': after, 'evidence_sha256': evidence, 'model_files': models,
        'learning_parent_records': builder.parent['learning_rows'],
        'forget_training_records': forget_rows,
        'forget_record_presentations': forget_rows * epochs,
        'retain_pool_records': pilot.manifest['jobs'][name]['retain_rows'],
        'retain_record_draws': forget_rows * epochs,
        'retain_sampling': 'stateless_uniform_pairing_with_replacement; one retain draw per forget record',
        'forget_data_protocol': pilot.manifest['jobs'][name]['forget_data_protocol'],
        'evaluation_protocol': builder.manifest.get('evaluation_protocol', 'full_published_records_v1'),
        'target_eval_questions': after[target_name]['questions'], 'retain_eval_questions': after[retain_name]['questions']}
    if passed:
        activities = {}
        for protocol in PROTOCOLS:
            spec = activation_spec(builder.project, builder.learned_parent, model, task['task_id'],
                builder.root / 'activations' / protocol / task['task_id'], protocol, task['split'])
            spec['base_model_id'] = builder.parent['base_model_id']
            spec['tokenizer_source'] = builder.parent['hdd_original']
            activities[protocol] = pilot.activation(spec)
        result['activations'] = activities
    atomic_json((builder.root / ('accepted' if passed else 'rejected')) / (task['task_id'] + '.json'), result)
    return result


def run(project, upstream, key):
    from official_knowledge.production_dispatch import pilot_gate
    pilot_gate(project)
    formal = project / 'formal-lora-400-v1'
    parent = next(p for p in json.loads((formal / 'parents.json').read_text()) if p['key'] == key)
    builder = ProductionJobs(project, parent)
    with (builder.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot = FormalPilot(builder.root, upstream, 168)
        try:
            preflight(pilot, builder)
            pipeline_first = pilot.manifest['quality_policy'].get('acceptance_mode') == 'pipeline_first'
            if not pipeline_first:
                pilot.record(phase='waiting_for_all_activation_checks', current_job=None, gpus=[])
                wait_preflight(project)
            from official_knowledge.performance_check import choose_microbatch
            microbatch = choose_microbatch(pilot, builder)
            performance = json.loads((builder.root / 'PERFORMANCE_CHECK.json').read_text())
            builder.hardware(microbatch, performance.get('selected_eval_batch', 1))
            pilot.manifest = builder.manifest
            for handle in getattr(pilot, 'gpu_reservations', []):
                handle.close()
            pilot.gpu_reservations = []
            pilot.record(phase='waiting_for_all_hardware_checks', current_job=None, gpus=[])
            while not pipeline_first:
                profiles = list((formal / 'cells').glob('*/PERFORMANCE_CHECK.json'))
                if len(profiles) == 4 and all(json.loads(p.read_text()).get('passed') for p in profiles):
                    atomic_json(formal / 'HARDWARE_PRECHECK_ACCEPTED.json', {'passed': True, 'at_utc': now(),
                        'profiles': {p.parent.name: sha(p) for p in profiles}})
                    break
                time.sleep(30)
            pilot.record(phase='training_formal_learning_parent', current_job=builder.source + '_learn', formal_batch_started=True)
            learned = parent_qualification(pilot, builder)
            primary = [x for x in records(formal / 'inventory.jsonl') if x['parent_key'] == key]
            reserves = [x for x in records(formal / 'reserve-targets.jsonl') if x['parent_key'] == key]
            by_method = {m: [x for x in primary if x['method'] == m] for m in ('NPO', 'GradDiff', 'SimNPO', 'UNDIAL')}
            tasks = [by_method[m][i] for i in range(25) for m in by_method]
            existing = [json.loads(p.read_text()) for p in (builder.root / 'accepted').glob('*.json')]
            by_slot = {}
            for result in existing:
                slot = result['task'].get('primary_slot_id', result['task']['task_id'])
                if slot in by_slot:
                    raise RuntimeError('Multiple accepted artifacts claim one production slot: ' + slot)
                by_slot[slot] = result
            occupied = {x['task']['unit_id'] for x in existing}
            accepted_ids, failed_ids = set(), set()
            for index, primary_task in enumerate(tasks):
                task = dict(primary_task, primary_slot_id=primary_task['task_id'])
                result = by_slot.get(primary_task['task_id']) or run_case(pilot, builder, task, learned)
                attempted_training_failures = 0
                while not result['passed']:
                    failed_ids.add(result['task']['unit_id'])
                    attempted_training_failures += int(result.get('reason') != 'target_not_sufficiently_learned')
                    if attempted_training_failures >= 3:
                        raise RuntimeError('Three distinct targets failed quality; research this method before continuing: ' + task['method'])
                    spare = next((x for x in reserves if x['method'] == task['method'] and x['unit_id'] not in occupied | failed_ids), None)
                    if spare is None:
                        raise RuntimeError('Method exhausted its distinct learned reserve targets: ' + task['method'])
                    task = dict(spare, split=primary_task['split'], primary_slot_id=primary_task['task_id'],
                                replaces_primary_task=primary_task['task_id'])
                    result = run_case(pilot, builder, task, learned)
                accepted_ids.add(result['task']['unit_id'])
                occupied.add(result['task']['unit_id'])
                pilot.record(phase='formal_cases_running', current_job=task['task_id'], accepted_models=len(accepted_ids), rejected_attempts=len(failed_ids))
                if index == 3:
                    atomic_json(formal / 'rollout16' / (builder.root.name + '.json'), {'passed': True, 'at_utc': now(),
                        'accepted_first_case_per_method': True, 'count': 4})
                    for handle in getattr(pilot, 'gpu_reservations', []):
                        handle.close()
                    pilot.gpu_reservations = []
                    pilot.record(phase='waiting_for_first_16_qualified_models', current_job=None, gpus=[])
                    while not pipeline_first and len(list((formal / 'rollout16').glob('*.json'))) < 4:
                        time.sleep(30)
            assert len(accepted_ids) == 100
            pilot.record(phase='formal_cell_complete', current_job=None, accepted_models=100)
        except Exception as error:
            pilot.record(phase='needs_repair', error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.upstream.resolve(), args.cell)


if __name__ == '__main__':
    main()
