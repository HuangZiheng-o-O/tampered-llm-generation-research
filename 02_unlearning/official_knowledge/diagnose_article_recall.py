"""Compare recall of existing training QA with the separate held-out QA."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import yaml

from official_knowledge.prepare_data import write
from official_knowledge.production_jobs import atomic_json
from official_knowledge.run_pilot import Pilot, now, sha


class RecallPilot(Pilot):
    def record(self, **values):
        self.state.update(values, updated_at_utc=now(), coordinator_pid=os.getpid(),
                          scope='seen_training_qa_recall_diagnostic_only', formal_batch_started=False)
        atomic_json(self.root / 'ARTICLE_RECALL_DIAGNOSTIC_STATUS.json', self.state)
        print(json.dumps({'at_utc': now(), **values}), flush=True)

    def report(self):
        pass


def prepare(suite):
    directory = suite / 'data/seen-qa-diagnostic'
    tasks = directory / 'tasks'
    tasks.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (suite / 'data/squad/learn.jsonl').read_text().splitlines()]
    info = json.loads((suite / 'data-readiness.json').read_text())['sources']['squad']
    groups = {p: {info['targets'][p]['unit_id']} for p in ('development', 'confirmation')}
    groups['retain'] = {json.loads(x)['unit_id'] for x in (suite / 'data/squad/eval-retain.jsonl').read_text().splitlines()}
    counts, used = {}, set()
    for phase, ids in groups.items():
        probes = []
        for uid in sorted(ids):
            eligible = sorted([r for r in rows if r['kind'] == 'published_qa' and r['unit_id'] == uid], key=lambda r: r['question_id'])
            assert len(eligible) >= 20
            for row in eligible[:20]:
                answer = row['text']
                for eos in ('</s>', '<|eot_id|>'):
                    if answer.endswith(eos):
                        answer = answer[:-len(eos)]
                assert answer.strip() and row['question_id'] not in used
                used.add(row['question_id'])
                probes.append({'unit_id': uid, 'fact_id': row['fact_id'], 'probe_id': row['question_id'],
                    'prompt': row['prompt'], 'answers': [answer.strip()], 'source': 'squad',
                    'diagnostic_scope': 'question_and_answer_were_used_in_learning',
                    'source_training_record_id': row['record_id']})
        data = directory / (phase + '.jsonl')
        write(data, probes)
        name = 'knowledge_squad_' + phase
        original_task = suite / 'tasks' / (name + '.yaml')
        task = yaml.safe_load('\n'.join(line for line in original_task.read_text().splitlines()
                                         if not line.startswith('process_results:')))
        task['dataset_kwargs']['data_files']['test'] = str(data)
        (tasks / original_task.name).write_text(yaml.safe_dump(task, sort_keys=False) +
            'process_results: !function knowledge_metrics.process_results\n')
        counts[phase] = len(probes)
    assert counts == info['eval_counts']
    shutil.copy2(Path(__file__).with_name('knowledge_metrics.py'), tasks / 'knowledge_metrics.py')
    with (suite / 'configs/recipe-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = suite / 'configs/jobs.json'
        manifest = json.loads(manifest_path.read_text())
        name = 'squad_seen_training_qa_recall'
        if name not in manifest['jobs']:
            cfg = OmegaConf.load(suite / 'configs/squad_learn_knowledge.yaml')
            output = suite / 'evals' / name
            cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = name, str(output), str(output)
            cfg.eval.lm_eval.output_dir, cfg.eval.lm_eval.include_path = str(output), str(tasks)
            cfg.eval.lm_eval.tasks = ['knowledge_squad_' + p for p in groups]
            file = suite / 'configs' / (name + '.yaml')
            OmegaConf.save(cfg, file)
            job = copy.deepcopy(manifest['jobs']['squad_learn_knowledge'])
            job.update(output=str(output), stage='learn_seen_training_qa', scope='diagnostic_only_not_quality_gate')
            manifest['jobs'][name] = job
            manifest['config_sha256'][file.name] = sha(file)
        for file in directory.glob('*.jsonl'):
            manifest.setdefault('shared_files_sha256', {})[str(file)] = sha(file)
        for file in tasks.iterdir():
            manifest['task_sha256'][str(file.relative_to(suite))] = sha(file)
        atomic_json(manifest_path, manifest)
    atomic_json(suite / 'ARTICLE_RECALL_DIAGNOSTIC_PREPARED.json', {'at_utc': now(), 'passed': True,
        'proof_kind': 'diagnostic_data_preparation_only', 'questions_seen_during_learning': True,
        'original_heldout_evaluation_changed': False, 'formal_production_authorized': False, 'counts': counts})
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    with (args.suite / 'article-recall-diagnostic.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name = prepare(args.suite)
        if not args.execute:
            print(json.dumps({'prepared': name, 'model_loaded': False}), flush=True)
            return
        pilot = RecallPilot(args.suite, args.upstream, 48)
        try:
            pilot.run(name)
            output = Path(pilot.manifest['jobs'][name]['output'])
            seen = json.loads((output / 'LMEval_SUMMARY.json').read_text())
            heldout = Path(pilot.manifest['jobs']['squad_learn_knowledge']['output'])
            result = {'at_utc': now(), 'proof_kind': 'official_model_recall_diagnostic',
                'training_updates': 0, 'formal_production_authorized': False,
                'seen_training_questions': seen,
                'separate_heldout_questions': json.loads((heldout / 'LMEval_SUMMARY.json').read_text()),
                'seen_evaluation_sha256': sha(output / 'LMEval_EVAL.json'),
                'heldout_evaluation_sha256': sha(heldout / 'LMEval_EVAL.json')}
            atomic_json(args.suite / 'ARTICLE_RECALL_DIAGNOSTIC_RESULT.json', result)
            pilot.record(phase='recall_diagnostic_completed', current_job=None)
        except Exception as error:
            pilot.record(phase='needs_repair', error=repr(error))
            raise


if __name__ == '__main__':
    main()
