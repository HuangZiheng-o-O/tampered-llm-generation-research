"""Frozen official-entry configurations for one formal source/base cell."""
import copy
import json
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import yaml

from official_knowledge.prepare_data import write
from official_knowledge.run_pilot import sha


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ProductionJobs:
    def __init__(self, project, parent):
        parent = copy.deepcopy(parent)
        self.project, self.parent = project, parent
        self.root = project / 'formal-lora-400-v1/cells' / (parent['base'] + '-' + parent['source'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = parent['source']
        from official_knowledge.pilot_registry import active
        self.pilot, entry = active(project, parent['base'], self.source)
        self.evaluation_protocol = entry['data_protocol'] if entry else 'full_published_records_v1'
        self.learning_protocol = entry.get('learning_data_protocol', 'mixed_body_qa_v1') if entry else 'mixed_body_qa_v1'
        if self.learning_protocol == 'published_qa_only_v1':
            parent['original_mixed_learning_rows'] = parent['learning_rows']
            parent['learning_rows'], parent['learning_tokens'] = entry['formal_learning_rows'], entry['formal_learning_tokens']
            parent['learning_data_protocol'] = self.learning_protocol
        self.configs, self.tasks = self.root / 'configs', self.root / 'tasks'
        self.configs.mkdir(exist_ok=True)
        self.tasks.mkdir(exist_ok=True)
        self.units = {u['unit_id']: u for u in records(Path(parent['data_dir']) / 'units.jsonl')}
        manifest_path = self.configs / 'jobs.json'
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            old_protocol = self.manifest.get('evaluation_protocol', 'full_published_records_v1')
            old_pilot = self.manifest.get('source_pilot_suite', str(project / ('official-lora-pilot-v1-' + parent['base'])))
            old_learning = self.manifest.get('learning_data_protocol', 'mixed_body_qa_v1')
            if old_protocol == self.evaluation_protocol and old_pilot == str(self.pilot) and old_learning == self.learning_protocol:
                pilot_manifest = json.loads((self.pilot / 'configs/jobs.json').read_text())
                support = pilot_manifest.get('support_sha256', {})
                if self.manifest.get('support_sha256', {}) != support:
                    self.manifest['support_sha256'] = copy.deepcopy(support)
                for job in self.manifest['jobs'].values():
                    if isinstance(job.get('evaluation_alias'), str):
                        assert job.get('evaluation_execution_protocol') == 'joint-official-eval-v1'
                        job['joint_evaluation_job'] = job.pop('evaluation_alias')
                self.save()
                self.refresh_learning_recipe()
                return
            self.preserve_unstarted_configuration()
        self.manifest = copy.deepcopy(json.loads((self.pilot / 'configs/jobs.json').read_text()))
        self.manifest.update(suite=self.root.name, jobs={}, config_sha256={}, task_sha256={},
                             training_enabled=True, scope='formal_lora_400_gated_cell',
                             source_pilot_suite=str(self.pilot), evaluation_protocol=self.evaluation_protocol,
                             learning_data_protocol=self.learning_protocol,
                             runtime_cache=str(project / 'cache/open-unlearning-official-v1'))
        if not (self.root / 'data').exists():
            (self.root / 'data').symlink_to(parent['data_dir'], target_is_directory=True)
        source_files = {str(p.relative_to(self.root)): sha(p) for p in (self.root / 'data').glob('*.jsonl')}
        if self.evaluation_protocol != 'full_published_records_v1':
            source_files.update({str(p.relative_to(self.root)): sha(p) for p in self.evaluation_file('targets').parent.glob('*.jsonl')})
        readiness = {'sources': {self.source: {'learning_rows': parent['learning_rows'],
                      'learning_pool_units': parent['learning_units']}}, 'derived_sha256': source_files,
                      'evaluation_protocol': self.evaluation_protocol}
        (self.root / 'data-readiness.json').write_text(json.dumps(readiness, indent=2) + '\n')
        self.manifest['data_readiness_sha256'] = sha(self.root / 'data-readiness.json')
        shutil.copy2(Path(__file__).with_name('knowledge_metrics.py'), self.tasks / 'knowledge_metrics.py')
        for stage in ('base', 'learn', 'learn_adapter'):
            for assessment in ('knowledge', 'mmlu_validation'):
                name = f'{self.source}_{stage}_{assessment}'
                cfg = self.template(name)
                out = self.root / 'evals' / name
                self.output(cfg, name, out, False)
                cfg.model.model_args.pretrained_model_name_or_path = str(
                    parent['hdd_original'] if stage in ('base', 'learn_adapter') else self.learned_parent)
                cfg.model.tokenizer_args.pretrained_model_name_or_path = parent['hdd_original']
                if stage == 'learn_adapter':
                    cfg.model.lora.adapter_path = str(self.learned_adapter)
                elif cfg.model.get('lora'):
                    del cfg.model.lora
                job = {'entry': 'src/eval.py', 'kind': 'eval', 'output': str(out),
                       'model': str(cfg.model.model_args.pretrained_model_name_or_path),
                       'source': self.source, 'stage': stage, 'assessment': assessment,
                       'requires': [] if stage == 'base' else [self.source + '_learn']}
                if assessment == 'knowledge':
                    names = [f'formal_{self.source}_targets', f'formal_{self.source}_retain']
                    files = [self.evaluation_file('targets'), self.evaluation_file('retain')]
                    cfg.eval.lm_eval.include_path = str(self.tasks)
                    cfg.eval.lm_eval.tasks = names
                    cfg.eval.lm_eval.simple_evaluate_args.num_fewshot = 0
                    job['expected_task_counts'] = {n: len(records(f)) for n, f in zip(names, files)}
                    job['expected_data_paths'] = {n: str(f) for n, f in zip(names, files)}
                    for n, f in zip(names, files):
                        self.task(n, f)
                self.register(name, cfg, job)
        name = self.source + '_learn'
        cfg = self.template(name)
        self.output(cfg, name, self.learned_adapter, True)
        cfg.model.model_args.pretrained_model_name_or_path = parent['hdd_original']
        cfg.model.tokenizer_args.pretrained_model_name_or_path = parent['hdd_original']
        cfg.data.train.published_records.args.hf_args.data_files = str(self.learning_file())
        self.final_only(cfg)
        self.register(name, cfg, {'entry': 'src/train.py', 'kind': 'train', 'output': str(self.learned_adapter),
            'requires': [self.source + '_base_knowledge', self.source + '_base_mmlu_validation'],
            'primary_rows': parent['learning_rows']})
        self.save()

    def learning_file(self):
        if getattr(self, 'learning_protocol', None) == 'published_qa_only_v1':
            return self.evaluation_file('targets').parent / 'learn.jsonl'
        return self.root / 'data/learn.jsonl'

    def preserve_unstarted_configuration(self):
        for job in self.manifest['jobs'].values():
            output = Path(job['output'])
            if (output / 'prepared-job.json').exists() or (output / 'prepared-evaluation.json').exists():
                raise ValueError('A started formal cell cannot adopt another pilot or evaluation protocol')
        archive = self.root / 'receipts' / ('before-protocol-' + sha(self.configs / 'jobs.json')[:12])
        shutil.copytree(self.configs, archive / 'configs', dirs_exist_ok=True)
        shutil.copytree(self.tasks, archive / 'tasks', dirs_exist_ok=True)
        if (self.root / 'data-readiness.json').exists():
            shutil.copy2(self.root / 'data-readiness.json', archive / 'data-readiness.json')

    def evaluation_file(self, role):
        folder = self.root / 'data'
        if getattr(self, 'evaluation_protocol', None) == 'published_qa_answers_only_v2':
            assert self.source == 'squad'
            folder = folder / 'protocols/published_qa_answers_only_v2'
        return folder / ('eval-' + role + '.jsonl')

    def evaluation_probes(self, unit):
        if getattr(self, 'evaluation_protocol', None) != 'published_qa_answers_only_v2':
            return unit['probes']
        if not hasattr(self, '_qa_probes'):
            self._qa_probes = {}
            for role in ('targets', 'retain'):
                for row in records(self.evaluation_file(role)):
                    self._qa_probes.setdefault(row['unit_id'], []).append(row)
        probes = self._qa_probes[unit['unit_id']]
        assert len(probes) == 40
        return probes

    @staticmethod
    def training_rows(unit, protocol):
        if protocol == 'published_qa_answers_only_v2':
            qa = [r for r in unit['learning'] if r['kind'] == 'published_qa']
            assert len(qa) == 40
            return qa
        return unit['learning']

    @property
    def learned_adapter(self):
        return self.root / 'runs' / (self.source + '_learn')

    def refresh_learning_recipe(self):
        name = self.source + '_learn'
        file = self.configs / (name + '.yaml')
        cfg, chosen = OmegaConf.load(file), self.template(name)
        values = {key: chosen.trainer.args[key] for key in ('num_train_epochs', 'learning_rate')}
        different = any(cfg.trainer.args[key] != value for key, value in values.items())
        if different:
            if (Path(self.manifest['jobs'][name]['output']) / 'prepared-job.json').exists():
                raise ValueError('Formal learning recipe is already locked by training')
            cfg.trainer.args.update(values)
            self.final_only(cfg)
            OmegaConf.save(cfg, file)
            self.manifest['config_sha256'][file.name] = sha(file)
            self.save()

    @property
    def learned_parent(self):
        return self.learned_adapter / 'merged'

    def template_name(self, name):
        from official_knowledge.pilot_recipes import selected
        for method, recipe in selected(self.pilot, self.source).items():
            prefix = self.source + '_development_' + method
            if name == prefix or name.startswith(prefix + '_'):
                name = self.source + '_' + recipe['development_stage'] + name[len(prefix):]
                break
        return name

    def template(self, name):
        name = self.template_name(name)
        text = (self.pilot / 'configs' / (name + '.yaml')).read_text()
        cfg = OmegaConf.create(text.replace(str(self.pilot), str(self.root)))
        cfg.paths.root_dir, cfg.paths.data_dir = str(self.root), str(self.root / 'data') + '/'
        return cfg

    def data_protocol(self, method):
        manifest = json.loads((self.pilot / 'configs/jobs.json').read_text())
        name = self.template_name(self.source + '_development_' + method)
        return manifest['jobs'][name].get('forget_data_protocol', 'full_published_records_v1')

    @staticmethod
    def output(cfg, name, out, training):
        cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = name, str(out), str(out)
        if training:
            cfg.trainer.args.output_dir, cfg.trainer.args.logging_dir = str(out), str(out / 'logs')
        else:
            cfg.eval.lm_eval.output_dir = str(out)

    @staticmethod
    def final_only(cfg):
        cfg.trainer.args.save_strategy = 'no'
        cfg.trainer.args.save_only_model = True
        cfg.checkpoint_first_step = False
        cfg.native_checkpoint_io = False

    def hardware(self, microbatch, eval_batch=1):
        assert microbatch in (1, 2, 4)
        for name, job in self.manifest['jobs'].items():
            if '_performance_' in name or '_case_' in name:
                continue
            path = self.configs / (name + '.yaml')
            cfg = OmegaConf.load(path)
            if job['kind'] == 'train':
                if (Path(job['output']) / 'prepared-job.json').exists():
                    assert cfg.trainer.args.per_device_train_batch_size == microbatch
                    continue
                cfg.trainer.args.per_device_train_batch_size = microbatch
                cfg.trainer.args.gradient_accumulation_steps = 32 // microbatch
            else:
                if (Path(job['output']) / 'prepared-evaluation.json').exists():
                    assert cfg.eval.lm_eval.simple_evaluate_args.batch_size == eval_batch
                    continue
                cfg.eval.lm_eval.simple_evaluate_args.batch_size = eval_batch
                cfg.eval.lm_eval.simple_evaluate_args.max_batch_size = 2
            OmegaConf.save(cfg, path)
            self.manifest['config_sha256'][path.name] = sha(path)
        self.save()

    def task(self, name, data):
        cfg = {'task': name, 'dataset_path': 'json', 'dataset_kwargs': {
            'data_files': {'test': str(data)}, 'cache_dir': str(self.root / 'cache/lm-eval')},
            'test_split': 'test', 'output_type': 'generate_until', 'doc_to_text': '{{prompt}}',
            'doc_to_target': '{{answers[0]}}', 'target_delimiter': '', 'num_fewshot': 0,
            'generation_kwargs': {'until': ['\n', '</s>', '<|eot_id|>'], 'max_gen_toks': 48, 'do_sample': False},
            'metric_list': [{'metric': m, 'aggregation': 'mean', 'higher_is_better': True}
                            for m in ('knowledge_accuracy', 'exact_match', 'answer_f1')]}
        (self.tasks / (name + '.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False) +
                                                 'process_results: !function knowledge_metrics.process_results\n')

    def register(self, name, cfg, job):
        path = self.configs / (name + '.yaml')
        if name in self.manifest['jobs']:
            assert sha(path) == self.manifest['config_sha256'][path.name]
            if OmegaConf.to_container(OmegaConf.load(path), resolve=True) == OmegaConf.to_container(cfg, resolve=True):
                existing_job = self.manifest['jobs'][name]
                for key, value in job.items():
                    existing_job.setdefault(key, value)
                return
            output = Path(job['output'])
            if '_case_' not in name or (output / 'prepared-job.json').exists() or (output / 'prepared-evaluation.json').exists():
                raise ValueError('An already prepared job cannot adopt a different recipe: ' + name)
        OmegaConf.save(cfg, path)
        self.manifest['jobs'][name] = job
        self.manifest['config_sha256'][path.name] = sha(path)

    def save(self):
        self.manifest['task_sha256'] = {str(p.relative_to(self.root)): sha(p) for p in self.tasks.iterdir() if p.is_file()}
        path = self.configs / 'jobs.json'
        pending = path.with_suffix('.pending.json')
        pending.write_text(json.dumps(self.manifest, indent=2) + '\n')
        pending.replace(path)

    def combine_case_evaluations(self, name, method=None):
        """Use the official evaluator twice with one loaded model and frozen tasks."""
        policy_path = self.project / 'THROUGHPUT_POLICY.json'
        if not policy_path.exists():
            return
        policy = json.loads(policy_path.read_text())
        if not policy.get('joint_evaluation_enabled'):
            return
        protocol = 'joint-official-eval-v1'
        names = [name + '_knowledge', name + '_mmlu_validation']
        jobs = [self.manifest['jobs'][n] for n in names]
        if all(job.get('evaluation_execution_protocol') == protocol for job in jobs):
            return
        if any((Path(job['output']) / 'prepared-evaluation.json').exists() for job in jobs):
            return
        benchmark = Path(policy['evaluation_benchmark'])
        if sha(benchmark) != policy['evaluation_benchmark_sha256']:
            raise ValueError('Evaluation throughput benchmark identity changed')
        proof = json.loads(benchmark.read_text())
        if not proof.get('passed') or proof['full_mmlu_questions'] != 1531:
            raise ValueError('Joint evaluation lacks a complete official benchmark')
        maximum_batch = policy.get('evaluation_max_batch_size', 2)
        assert maximum_batch in (2, 8) and proof.get('max_batch_size', 2) >= maximum_batch
        configs = [OmegaConf.load(self.configs / (n + '.yaml')) for n in names]
        if OmegaConf.to_container(configs[0].model, resolve=True) != OmegaConf.to_container(configs[1].model, resolve=True):
            raise ValueError('Joint evaluations must load the same parent and saved adapter')
        audited = any(json.loads(p.read_text())['task']['method'] == method
                      for p in (self.root / 'accepted').glob('*.json')) if method else False
        routine = audited and policy.get('routine_mmlu_questions_per_subject') == 3
        cfg = configs[0]
        output = self.root / 'evals' / (name + '_joint_evaluation')
        self.output(cfg, names[0], output, False)
        cfg.eval.lm_eval_mmlu = copy.deepcopy(configs[1].eval.lm_eval)
        for evaluator in cfg.eval.values():
            evaluator.output_dir = str(output)
            evaluator.simple_evaluate_args.batch_size = 'auto'
            evaluator.simple_evaluate_args.max_batch_size = maximum_batch
        if routine:
            cfg.eval.lm_eval_mmlu.simple_evaluate_args.limit = 3
        for n, job in zip(names, jobs):
            job = dict(job, output=str(output), evaluation_execution_protocol=protocol,
                       throughput_benchmark_sha256=sha(benchmark))
            if job['assessment'] == 'mmlu_validation':
                job.update(expected_mmlu_count=171 if routine else 1531, joint_evaluation_job=names[0],
                           mmlu_evaluation_scope='three_per_subject' if routine else 'full_validation')
            self.register(n, cfg, job)

    def case(self, task):
        name = self.source + '_case_' + task['task_id']
        out = self.root / 'runs' / name
        protocol = self.data_protocol(task['method'])
        if name in self.manifest['jobs']:
            old = self.manifest['jobs'][name].get('forget_data_protocol', 'full_published_records_v1')
            if old != protocol and (out / 'prepared-job.json').exists():
                raise ValueError('A started case cannot change its forgetting-data protocol')
        folder = self.root / 'case-data' / task['task_id'] if protocol == 'full_published_records_v1' else (
            self.root / 'case-data-protocols' / protocol / task['task_id'])
        unit = self.units[task['unit_id']]
        forget = self.training_rows(unit, protocol)
        retained = [r for uid, u in self.units.items() if uid != task['unit_id'] for r in self.training_rows(u, protocol)]
        ready = folder / 'DATA_READY.json'
        if not ready.exists():
            if protocol == 'published_fact_value_tokens_v1':
                assert self.source == 'biographies'
                from transformers import AutoTokenizer
                from data.utils import preprocess_pretraining_instance
                from official_knowledge.fact_values import validated_rows
                tokenizer = AutoTokenizer.from_pretrained(self.parent['hdd_original'], local_files_only=True)
                forget = validated_rows(forget, self.units, tokenizer, preprocess_pretraining_instance)
            elif protocol not in ('full_published_records_v1', 'published_qa_answers_only_v2'):
                raise ValueError('Unsupported formal forgetting-data protocol: ' + protocol)
            write(folder / 'forget.jsonl', forget)
            write(folder / 'retain.jsonl', retained)
            write(folder / 'eval-target.jsonl', self.evaluation_probes(unit))
            atomic_json(ready, {p.name: sha(p) for p in folder.glob('*.jsonl')})
        for filename, expected in json.loads(ready.read_text()).items():
            assert sha(folder / filename) == expected
        cfg = self.template(self.source + '_development_' + task['method'])
        self.output(cfg, name, out, True)
        cfg.model.model_args.pretrained_model_name_or_path = str(self.learned_parent)
        cfg.model.tokenizer_args.pretrained_model_name_or_path = self.parent['hdd_original']
        cfg.data.forget.published_records.args.hf_args.data_files = str(folder / 'forget.jsonl')
        cfg.data.retain.published_records.args.hf_args.data_files = str(folder / 'retain.jsonl')
        self.final_only(cfg)
        performance = self.root / 'PERFORMANCE_CHECK.json'
        if performance.exists():
            batch = json.loads(performance.read_text())['selected_microbatch']
            cfg.trainer.args.per_device_train_batch_size = batch
            cfg.trainer.args.gradient_accumulation_steps = 32 // batch
        if name in self.manifest['jobs']:
            cfg = OmegaConf.load(self.configs / (name + '.yaml'))
        else:
            policy_file = self.project / 'THROUGHPUT_POLICY.json'
            policy = json.loads(policy_file.read_text()) if policy_file.exists() else {}
            benchmark_info = policy.get('training_benchmarks', {}).get(self.root.name)
            if benchmark_info:
                benchmark = Path(benchmark_info['path'])
                if sha(benchmark) != benchmark_info['sha256']:
                    raise ValueError('Training throughput benchmark identity changed')
                profile = json.loads(benchmark.read_text())
                batch = profile['selected_microbatch']
                assert profile['passed'] and batch in (1, 2, 4, 8, 16) and profile['effective_batch'] == 32
                cfg.trainer.args.per_device_train_batch_size = batch
                cfg.trainer.args.gradient_accumulation_steps = 32 // batch
        gate_source = self.source + '-' + task['task_id']
        self.register(name, cfg, {'entry': 'src/train.py', 'kind': 'train', 'output': str(out),
            'requires': [self.source + '_learn'], 'primary_rows': len(forget), 'retain_rows': len(retained),
            'forget_data_protocol': protocol,
            'learning_gate': {'source': gate_source, 'phase': 'development'}})
        for assessment in ('knowledge', 'mmlu_validation'):
            ename = name + '_' + assessment
            existing = self.manifest['jobs'].get(ename, {})
            if existing.get('evaluation_execution_protocol') == 'joint-official-eval-v1':
                assert sha(self.configs / (ename + '.yaml')) == self.manifest['config_sha256'][ename + '.yaml']
                continue
            ecfg = self.template(self.source + '_development_' + task['method'] + '_' + assessment)
            eout = self.root / 'evals' / ename
            self.output(ecfg, ename, eout, False)
            ecfg.model.model_args.pretrained_model_name_or_path = str(self.learned_parent)
            ecfg.model.tokenizer_args.pretrained_model_name_or_path = self.parent['hdd_original']
            ecfg.model.lora.adapter_path = str(out)
            if performance.exists():
                choice = json.loads(performance.read_text())
                ecfg.eval.lm_eval.simple_evaluate_args.batch_size = choice.get('selected_eval_batch', 1)
                ecfg.eval.lm_eval.simple_evaluate_args.max_batch_size = 2
            job = {'entry': 'src/eval.py', 'kind': 'eval', 'output': str(eout),
                   'model': str(self.learned_parent), 'source': self.source, 'stage': name,
                   'assessment': assessment, 'requires': [name]}
            if assessment == 'knowledge':
                suffix = sha(folder / 'eval-target.jsonl')[:12]
                names = [f'formal_target_{suffix}', f'formal_retain_{suffix}']
                files = [folder / 'eval-target.jsonl', self.evaluation_file('retain')]
                for n, f in zip(names, files):
                    self.task(n, f)
                ecfg.eval.lm_eval.include_path = str(self.tasks)
                ecfg.eval.lm_eval.tasks = names
                ecfg.eval.lm_eval.simple_evaluate_args.num_fewshot = 0
                job['expected_task_counts'] = {n: len(records(f)) for n, f in zip(names, files)}
                job['expected_data_paths'] = {n: str(f) for n, f in zip(names, files)}
            self.register(ename, ecfg, job)
        self.combine_case_evaluations(name, task['method'])
        self.manifest.setdefault('shared_files_sha256', {}).update(
            {str(p): sha(p) for p in folder.glob('*.jsonl')})
        self.save()
        return name, gate_source
