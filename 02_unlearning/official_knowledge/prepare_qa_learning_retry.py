"""Prepare a separate QA-only learning parent after mixed-text learning failed."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import yaml

from official_knowledge.prepare_data import write
from official_knowledge.production_jobs import atomic_json, records
from official_knowledge.run_pilot import now, sha


def prepare(project, base):
    old = project / ('official-qa-unlearning-v2-' + base)
    suite = project / ('official-qa-learning-v3-' + base)
    ready = suite / 'QA_ONLY_LEARNING_PREPARED.json'
    if ready.exists():
        assert json.loads(ready.read_text())['passed']
        return suite
    gate = json.loads((old / 'gates/squad-learning.json').read_text())
    assert gate['proof_kind'] == 'official_model_evaluation'
    assert not all(v['passed'] for v in gate['targets'].values())
    assert not list((old / 'runs').glob('squad_*NPO*/prepared-job.json'))
    manifest = copy.deepcopy(json.loads((old / 'configs/jobs.json').read_text()))
    readiness = copy.deepcopy(json.loads((old / 'data-readiness.json').read_text()))
    data = suite / 'data/squad'
    data.mkdir(parents=True, exist_ok=True)
    for file in (old / 'data/squad').glob('*.jsonl'):
        if file.name != 'learn.jsonl':
            shutil.copy2(file, data / file.name)
    qa = [r for r in records(old / 'data/squad/learn.jsonl') if r['kind'] == 'published_qa']
    assert len(qa) == 280 and len({r['unit_id'] for r in qa}) == 7
    write(data / 'learn.jsonl', qa)
    info = readiness['sources']['squad']
    info.update(learning_rows=280, learning_tokens=sum(r['tokens'] for r in qa),
                learning_data_protocol='published_qa_only_v1')
    readiness.update(scope='qa_only_learning_pilot_v3', original_suite=str(old),
        previous_learning_gate_sha256=sha(old / 'gates/squad-learning.json'),
        derived_sha256={str(p.relative_to(suite)): sha(p) for p in data.glob('*.jsonl')})
    atomic_json(suite / 'data-readiness.json', readiness)
    tasks, configs = suite / 'tasks', suite / 'configs'
    tasks.mkdir(exist_ok=True)
    configs.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('knowledge_metrics.py'), tasks / 'knowledge_metrics.py')
    for phase in ('development', 'confirmation', 'retain'):
        name = 'knowledge_squad_' + phase
        text = (old / 'tasks' / (name + '.yaml')).read_text()
        task = yaml.safe_load('\n'.join(s for s in text.splitlines() if not s.startswith('process_results:')))
        task['dataset_kwargs']['data_files']['test'] = str(data / ('eval-' + phase + '.jsonl'))
        task['dataset_kwargs']['cache_dir'] = str(suite / 'cache/lm-eval')
        (tasks / (name + '.yaml')).write_text(yaml.safe_dump(task, sort_keys=False) +
            'process_results: !function knowledge_metrics.process_results\n')
    parent = next(p for p in json.loads((project / 'formal-lora-400-v1/parents.json').read_text())
                  if p['base'] == base and p['source'] == 'squad')
    original_model = parent['hdd_original']
    learned_adapter = suite / 'runs/squad_learn_qa_10epochs'
    learned_parent = learned_adapter / 'merged'
    previous = copy.deepcopy(manifest)
    manifest.update(suite=suite.name, scope='qa_only_learning_pilot_v3',
        learning_data_protocol='published_qa_only_v1', active_sources=['squad'],
        data_readiness_sha256=sha(suite / 'data-readiness.json'), jobs={}, config_sha256={},
        task_sha256={}, formal_production_authorized=False)
    for name, old_job in previous['jobs'].items():
        job = copy.deepcopy(old_job)
        cfg = OmegaConf.load(old / 'configs' / (name + '.yaml'))
        output = learned_adapter if name == 'squad_learn' else suite / ('runs' if job['kind'] == 'train' else 'evals') / name
        job['output'] = str(output)
        job.pop('evaluation_alias', None)
        job.pop('reuses_completed_parent_from', None)
        cfg.paths.root_dir, cfg.paths.data_dir = str(suite), str(suite / 'data') + '/'
        cfg.paths.output_dir = cfg.hydra.run.dir = str(output)
        cfg.model.tokenizer_args.pretrained_model_name_or_path = original_model
        if job['kind'] == 'train':
            cfg.trainer.args.output_dir, cfg.trainer.args.logging_dir = str(output), str(output / 'logs')
            cfg.trainer.args.save_strategy, cfg.trainer.args.save_only_model = 'no', True
            cfg.checkpoint_first_step = cfg.native_checkpoint_io = False
            if name == 'squad_learn':
                cfg.model.model_args.pretrained_model_name_or_path = original_model
                cfg.data.train.published_records.args.hf_args.data_files = str(data / 'learn.jsonl')
                cfg.trainer.args.num_train_epochs = 10
                job['primary_rows'] = 280
                job['learning_data_protocol'] = 'published_qa_only_v1'
            else:
                cfg.model.model_args.pretrained_model_name_or_path = str(learned_parent)
                phase = name.split('_')[1]
                if '_control_' in name:
                    cfg.data.train.published_records.args.hf_args.data_files = str(data / (phase + '-control.jsonl'))
                else:
                    for key in ('forget', 'retain'):
                        cfg.data[key].published_records.args.hf_args.data_files = str(data / (phase + '-' + key + '.jsonl'))
        else:
            cfg.eval.lm_eval.output_dir = str(output)
            if job['assessment'] == 'knowledge':
                cfg.eval.lm_eval.include_path = str(tasks)
            stage = job['stage']
            cfg.model.model_args.pretrained_model_name_or_path = original_model if stage in ('base', 'learn_adapter') else str(learned_parent)
            if stage == 'learn_adapter':
                cfg.model.lora.adapter_path = str(learned_adapter)
            elif stage not in ('base', 'learn'):
                cfg.model.lora.adapter_path = str(suite / 'runs' / ('squad_' + stage))
            job['model'] = str(cfg.model.model_args.pretrained_model_name_or_path)
        file = configs / (name + '.yaml')
        if name == 'squad_base_mmlu_validation':
            # Reuse the exact, complete original-base assessment, including
            # its original data and configuration identity.
            source = project / ('official-lora-pilot-v1-' + base)
            prior = json.loads((source / 'configs/jobs.json').read_text())
            shutil.copy2(source / 'configs' / file.name, file)
            job = copy.deepcopy(prior['jobs'][name])
            job['evaluation_alias'] = {'suite': str(source), 'source_config_sha256': sha(file), 'read_only': True}
        else:
            OmegaConf.save(cfg, file)
        manifest['jobs'][name] = job
        manifest['config_sha256'][file.name] = sha(file)
    manifest['task_sha256'] = {str(p.relative_to(suite)): sha(p) for p in tasks.iterdir()}
    atomic_json(configs / 'jobs.json', manifest)
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'data_and_configuration_preparation',
        'formal_production_authorized': False, 'previous_suite': str(old), 'suite': str(suite),
        'learning_articles': 7, 'learning_qa': 280, 'learning_epochs': 10, 'expected_learning_updates': 90,
        'learning_rate': .0001, 'effective_batch': 32, 'forget_qa_per_target': 40, 'retain_training_qa': 240,
        'evaluation_questions_per_target': 40, 'evaluation_retention_questions': 200,
        'published_questions_or_answers_changed': False, 'acceptance_thresholds_changed': False,
        'official_trainer': 'FinetuneTrainer and original unlearning trainers', 'intermediate_checkpoints': 0}
    atomic_json(ready, receipt)
    candidate_file = project / 'QA_CANDIDATE_SUITES.json'
    candidates = json.loads(candidate_file.read_text()) if candidate_file.exists() else {}
    candidates[base] = str(suite)
    atomic_json(candidate_file, candidates)
    return suite


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', choices=('llama3-instruct',), default='llama3-instruct')
    a = p.parse_args()
    print(prepare(a.root, a.base), flush=True)
