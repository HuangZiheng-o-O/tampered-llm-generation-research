"""Prepare an isolated QA-unlearning pilot using an existing learned parent."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import yaml

from official_knowledge.prepare_data import matched_control, write
from official_knowledge.production_jobs import atomic_json
from official_knowledge.run_pilot import Pilot, now, sha


def records(file):
    return [json.loads(line) for line in file.read_text().splitlines() if line]


def prepare(project, upstream, base):
    original = project / ('official-lora-pilot-v1-' + base)
    suite = project / ('official-qa-unlearning-v2-' + base)
    ready = suite / 'QA_V2_PREPARED.json'
    if ready.exists():
        assert json.loads(ready.read_text())['passed']
        return suite
    suite.mkdir(exist_ok=True)
    (suite / 'configs').mkdir(exist_ok=True)
    with (original / 'configs/recipe-preparation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        old_manifest = json.loads((original / 'configs/jobs.json').read_text())
        configs = {n: (original / 'configs' / (n + '.yaml')).read_text() for n in old_manifest['jobs']
                   if n.startswith('squad_') and n != 'squad_seen_training_qa_recall'}
    old_data = json.loads((original / 'data-readiness.json').read_text())
    rows = records(original / 'data/squad/learn.jsonl')
    qa = [r for r in rows if r['kind'] == 'published_qa']
    assert len(qa) == 280 and len(rows) == 667
    data = suite / 'data/squad'
    write(data / 'learn.jsonl', rows)
    assert sha(data / 'learn.jsonl') == sha(original / 'data/squad/learn.jsonl')
    # Keep the old, different-fact evaluation as an explicit separate artifact.
    for phase in ('development', 'confirmation', 'retain'):
        shutil.copy2(original / 'data/squad' / ('eval-' + phase + '.jsonl'),
                     data / ('different-facts-eval-' + phase + '.jsonl'))
    info = copy.deepcopy(old_data['sources']['squad'])
    roles = {p: {info['targets'][p]['unit_id']} for p in ('development', 'confirmation')}
    roles['retain'] = {q['unit_id'] for q in records(original / 'data/squad/eval-retain.jsonl')}
    raw_questions = {}
    for split in ('train', 'dev'):
        raw = json.loads((project / 'datasets/knowledge-20261101' / ('squad_' + split + '.json')).read_text())
        for article in raw['data']:
            for paragraph in article['paragraphs']:
                for question in paragraph['qas']:
                    raw_questions[question['id']] = (question, paragraph['context'], split)
    counts = {}
    for phase, ids in roles.items():
        selected = [r for r in qa if r['unit_id'] in ids]
        assert len(selected) == len(ids) * 40
        probes = []
        for row in selected:
            question, context, split = raw_questions[row['question_id']]
            answers = list(dict.fromkeys(a['text'] for a in question['answers']
                if context[a['answer_start']:a['answer_start'] + len(a['text'])] == a['text']))
            assert answers
            probes.append({'unit_id': row['unit_id'], 'fact_id': row['fact_id'], 'probe_id': row['question_id'],
                'prompt': row['prompt'], 'question': question['question'], 'answers': answers,
                'source': 'squad', 'source_split': split, 'evidence_sha256': row['evidence_sha256'],
                'evaluation_condition': 'published_question_used_during_learning'})
        write(data / ('eval-' + phase + '.jsonl'), probes)
        counts[phase] = len(probes)
        if phase != 'retain':
            retained = [r for r in qa if r['unit_id'] not in ids]
            controls = matched_control(selected, retained, next(iter(ids)))
            for suffix, values in [('forget', selected), ('retain', retained), ('control', controls)]:
                write(data / (phase + '-' + suffix + '.jsonl'), values)
            info['targets'][phase].update(forget_rows=40, retain_pool_rows=len(retained), control_rows=40,
                probe_rows=40, forget_tokens=sum(r['tokens'] for r in selected),
                control_tokens=sum(r['tokens'] for r in controls))
    info.update(eval_counts=counts, forgetting_data_protocol='published_qa_answers_only_v2',
                evaluation_condition='all_forty_learning_questions_per_article',
                old_different_fact_question_results_preserved=True)
    readiness = copy.deepcopy(old_data)
    readiness.update(sources={'squad': info}, scope='isolated_qa_unlearning_pilot_v2',
        original_suite=str(original), reused_learning_data_sha256=sha(data / 'learn.jsonl'),
        derived_sha256={str(p.relative_to(suite)): sha(p) for p in data.glob('*.jsonl')})
    atomic_json(suite / 'data-readiness.json', readiness)
    tasks = suite / 'tasks'
    tasks.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('knowledge_metrics.py'), tasks / 'knowledge_metrics.py')
    for phase in roles:
        name = 'knowledge_squad_' + phase
        text = (original / 'tasks' / (name + '.yaml')).read_text()
        task = yaml.safe_load('\n'.join(line for line in text.splitlines() if not line.startswith('process_results:')))
        task['dataset_kwargs']['data_files']['test'] = str(data / ('eval-' + phase + '.jsonl'))
        task['dataset_kwargs']['cache_dir'] = str(suite / 'cache/lm-eval')
        task['metadata']['evaluation_condition'] = 'all_published_learning_questions; no passage supplied'
        (tasks / (name + '.yaml')).write_text(yaml.safe_dump(task, sort_keys=False) +
            'process_results: !function knowledge_metrics.process_results\n')
    manifest = copy.deepcopy(old_manifest)
    manifest.update(suite=suite.name, active_sources=['squad'], scope='isolated_qa_unlearning_pilot_v2',
        jobs={}, config_sha256={}, task_sha256={}, training_enabled=True,
        data_readiness_sha256=sha(suite / 'data-readiness.json'), formal_production_authorized=False)
    for name, text in configs.items():
        job = copy.deepcopy(old_manifest['jobs'][name])
        file = suite / 'configs' / (name + '.yaml')
        if name == 'squad_learn':
            # Exact original config/identity; the completed adapter is read-only.
            file.write_text(text)
            job['reuses_completed_parent_from'] = str(original)
        else:
            cfg = OmegaConf.create(text)
            output = suite / ('runs' if job['kind'] == 'train' else 'evals') / name
            job['output'] = str(output)
            cfg.paths.root_dir, cfg.paths.data_dir = str(suite), str(suite / 'data') + '/'
            cfg.paths.output_dir = cfg.hydra.run.dir = str(output)
            if job['kind'] == 'train':
                cfg.trainer.args.output_dir, cfg.trainer.args.logging_dir = str(output), str(output / 'logs')
                cfg.trainer.args.save_strategy, cfg.trainer.args.save_only_model = 'no', True
                cfg.checkpoint_first_step = False
                job['primary_rows'], job['forget_data_protocol'] = 40, 'published_qa_answers_only_v2'
                phase = name.split('_')[1]
                if '_control_' in name:
                    cfg.data.train.published_records.args.hf_args.data_files = str(data / (phase + '-control.jsonl'))
                    cfg.data.train.published_records.args.hf_args.cache_dir = str(suite / 'cache/datasets')
                else:
                    for key in ('forget', 'retain'):
                        cfg.data[key].published_records.args.hf_args.data_files = str(data / (phase + '-' + key + '.jsonl'))
                        cfg.data[key].published_records.args.hf_args.cache_dir = str(suite / 'cache/datasets')
            else:
                cfg.eval.lm_eval.output_dir = str(output)
                if job['assessment'] == 'knowledge':
                    cfg.eval.lm_eval.include_path = str(tasks)
                stage = job['stage']
                if stage not in ('base', 'learn', 'learn_adapter'):
                    cfg.model.lora.adapter_path = str(suite / 'runs' / ('squad_' + stage))
            OmegaConf.save(cfg, file)
        manifest['jobs'][name] = job
        manifest['config_sha256'][file.name] = sha(file)
    manifest['task_sha256'] = {str(p.relative_to(suite)): sha(p) for p in tasks.iterdir()}
    original_pilot = Pilot(original, upstream, 48)
    for stage in ('base', 'learn', 'learn_adapter'):
        name = 'squad_' + stage + '_mmlu_validation'
        if original_pilot.completed(name):
            from official_knowledge.production_worker import assessment
            assessment(original_pilot, name)
            source_config = original / 'configs' / (name + '.yaml')
            shutil.copy2(source_config, suite / 'configs' / source_config.name)
            manifest['config_sha256'][source_config.name] = sha(source_config)
            manifest['jobs'][name]['output'] = old_manifest['jobs'][name]['output']
            manifest['jobs'][name]['evaluation_alias'] = {
                'suite': str(original), 'source_config_sha256': sha(source_config), 'read_only': True}
    atomic_json(suite / 'configs/jobs.json', manifest)
    checker = Pilot(suite, upstream, 48)
    assert checker.completed('squad_learn'), 'Original trained-parent identity did not match exactly'
    receipt = {'at_utc': now(), 'passed': True, 'scope': 'small_qa_unlearning_pilot_only',
        'formal_production_authorized': False, 'original_suite': str(original), 'suite': str(suite),
        'shared_learned_parent': old_manifest['jobs']['squad_learn']['output'],
        'learning_parent_retrained': False, 'complete_weights_copied': False,
        'original_data_and_evaluations_changed': False, 'learning_records': 667,
        'unlearning_records_per_target': 40, 'knowledge_evaluation_counts': counts,
        'evaluation_questions_seen_in_learning': True, 'old_different_fact_evaluation_preserved': True,
        'methods': manifest['methods'], 'intermediate_checkpoints_new_jobs': 0}
    atomic_json(ready, receipt)
    print(json.dumps(receipt), flush=True)
    return suite


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--base', choices=('llama2-chat', 'llama3-instruct'), required=True)
    a = p.parse_args()
    prepare(a.root, a.upstream, a.base)
