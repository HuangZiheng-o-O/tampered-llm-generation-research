"""CPU contract checks for dispatch, provenance, activity I/O and batch options."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from official_knowledge import production_dispatch, activation_entry
from official_knowledge.production_worker import assessment, matching_parent_mmlu, FormalPilot
from official_knowledge.production_jobs import ProductionJobs, atomic_json
from official_knowledge.run_pilot import Pilot, sha
from official_pilot.launch import learning_configuration_matches


def check_registration_metadata(root):
    folder = root / 'registration'
    folder.mkdir()
    name = 'biographies_case_existing'
    output = folder / 'saved-model'
    output.mkdir()
    marker = output / 'prepared-job.json'
    marker.write_text('{"software_fixture_only": true}\n')
    cfg = OmegaConf.create({'trainer': {'args': {'num_train_epochs': 10}}})
    path = folder / (name + '.yaml')
    OmegaConf.save(cfg, path)
    original_config = path.read_bytes()
    original_marker = marker.read_bytes()
    builder = object.__new__(ProductionJobs)
    builder.configs = folder
    builder.manifest = {'jobs': {name: {'output': str(output), 'primary_rows': 120,
                                      'existing_provenance': 'preserved'}},
                        'config_sha256': {path.name: hashlib.sha256(original_config).hexdigest()}}
    metadata = {'output': str(output), 'primary_rows': 120, 'retain_rows': 20880}
    builder.register(name, cfg, metadata)
    assert builder.manifest['jobs'][name]['retain_rows'] == 20880
    assert builder.manifest['jobs'][name]['existing_provenance'] == 'preserved'
    registered = dict(builder.manifest['jobs'][name])
    builder.register(name, cfg, metadata)
    assert builder.manifest['jobs'][name] == registered
    assert path.read_bytes() == original_config and marker.read_bytes() == original_marker
    changed = OmegaConf.create({'trainer': {'args': {'num_train_epochs': 20}}})
    try:
        builder.register(name, changed, metadata)
    except ValueError:
        pass
    else:
        raise AssertionError('A prepared case adopted a different training recipe')
    assert path.read_bytes() == original_config and marker.read_bytes() == original_marker

    project = root / 'restore-project'
    pilot = project / 'selected-pilot'
    (pilot / 'configs').mkdir(parents=True)
    current_support = {'official_knowledge/production_jobs.py': 'selected-pilot-code-pin'}
    atomic_json(pilot / 'configs/jobs.json', {'support_sha256': current_support})
    data = project / 'published-data'
    data.mkdir()
    (data / 'units.jsonl').write_text('{"unit_id": "published-content"}\n')
    cell = project / 'formal-lora-400-v1/cells/example-biographies'
    prior = {'jobs': {'existing': {'output': 'preserved'},
                     'joint_mmlu': {'evaluation_alias': 'joint_knowledge',
                                    'evaluation_execution_protocol': 'joint-official-eval-v1'}},
             'source_pilot_suite': str(pilot),
             'support_sha256': {'official_knowledge/production_jobs.py': 'older-code-pin'}}
    atomic_json(cell / 'configs/jobs.json', prior)
    parent = {'base': 'example', 'source': 'biographies', 'data_dir': str(data)}
    with patch('official_knowledge.pilot_registry.active', return_value=(pilot, None)), \
         patch.object(ProductionJobs, 'refresh_learning_recipe'):
        restored = ProductionJobs(project, parent)
    saved = json.loads((cell / 'configs/jobs.json').read_text())
    assert restored.manifest['support_sha256'] == current_support
    expected_jobs = dict(prior['jobs'], joint_mmlu={
        'joint_evaluation_job': 'joint_knowledge', 'evaluation_execution_protocol': 'joint-official-eval-v1'})
    assert saved['support_sha256'] == current_support and saved['jobs'] == expected_jobs


def check_joint_evaluations(root):
    project = root / 'joint-evaluation-project'
    cell = project / 'cell'
    configs = cell / 'configs'
    configs.mkdir(parents=True)
    builder = object.__new__(ProductionJobs)
    builder.project, builder.root, builder.configs = project, cell, configs
    builder.manifest = {'jobs': {}, 'config_sha256': {}}
    name = 'biographies_case_fixture'
    for kind, shots, tasks in [('knowledge', 0, ['target', 'retain']),
                                ('mmlu_validation', 5, ['pilot_mmlu_validation'])]:
        n = name + '_' + kind
        out = cell / n
        cfg = OmegaConf.create({'task_name': n, 'model': {'fixture': 'same-saved-adapter'}, 'paths': {'output_dir': str(out)},
            'hydra': {'run': {'dir': str(out)}}, 'eval': {'lm_eval': {
                'handler': 'LMEvalEvaluator', 'output_dir': str(out), 'tasks': tasks,
                'include_path': '/frozen/' + kind,
                'simple_evaluate_args': {'num_fewshot': shots, 'batch_size': 1,
                                         'max_batch_size': 2}}}})
        builder.register(n, cfg, {'kind': 'eval', 'assessment': kind, 'output': str(out), 'requires': []})
    benchmark = project / 'benchmark.json'
    atomic_json(benchmark, {'passed': True, 'full_mmlu_questions': 1531, 'software_fixture_only': True, 'max_batch_size': 8})
    policy = {'evaluation_max_batch_size': 8, 'joint_evaluation_enabled': True, 'evaluation_benchmark': str(benchmark),
              'evaluation_benchmark_sha256': hashlib.sha256(benchmark.read_bytes()).hexdigest()}
    atomic_json(project / 'THROUGHPUT_POLICY.json', policy)
    # An already started evaluation keeps its exact original configuration.
    original = (configs / (name + '_knowledge.yaml')).read_bytes()
    original_mmlu = (configs / (name + '_mmlu_validation.yaml')).read_bytes()
    output = Path(builder.manifest['jobs'][name + '_knowledge']['output'])
    output.mkdir()
    marker = output / 'prepared-evaluation.json'
    marker.write_text('{}')
    builder.combine_case_evaluations(name)
    assert (configs / (name + '_knowledge.yaml')).read_bytes() == original
    marker.unlink()
    builder.combine_case_evaluations(name)
    first = (configs / (name + '_knowledge.yaml')).read_bytes()
    second = (configs / (name + '_mmlu_validation.yaml')).read_bytes()
    assert first == second
    cfg = OmegaConf.load(configs / (name + '_knowledge.yaml'))
    assert cfg.eval.lm_eval.simple_evaluate_args.num_fewshot == 0
    assert cfg.eval.lm_eval_mmlu.simple_evaluate_args.num_fewshot == 5
    assert list(cfg.eval.lm_eval.tasks) == ['target', 'retain']
    assert list(cfg.eval.lm_eval_mmlu.tasks) == ['pilot_mmlu_validation']
    assert cfg.eval.lm_eval.include_path == '/frozen/knowledge'
    assert cfg.eval.lm_eval_mmlu.include_path == '/frozen/mmlu_validation'
    assert all(e.simple_evaluate_args.max_batch_size == 8 for e in cfg.eval.values())
    assert builder.manifest['jobs'][name + '_mmlu_validation']['expected_mmlu_count'] == 1531
    assert builder.manifest['jobs'][name + '_mmlu_validation']['joint_evaluation_job'] == name + '_knowledge'
    assert 'evaluation_alias' not in builder.manifest['jobs'][name + '_mmlu_validation']
    marker.write_text(json.dumps({'job': name + '_knowledge'}))
    receipt = {'evidence_sha256': {str(marker.relative_to(cell)): 'fixture'},
               'config_sha256': dict(builder.manifest['config_sha256'])}
    assert learning_configuration_matches(cell, receipt, builder.manifest)
    marker.unlink()
    builder.combine_case_evaluations(name)
    assert (configs / (name + '_knowledge.yaml')).read_bytes() == first

    atomic_json(cell / 'accepted/first.json', {'task': {'method': 'NPO'}})
    policy['routine_mmlu_questions_per_subject'] = 3
    atomic_json(project / 'THROUGHPUT_POLICY.json', policy)
    next_name = 'biographies_case_next_fixture'
    for kind, raw in [('knowledge', original), ('mmlu_validation', original_mmlu)]:
        n = next_name + '_' + kind
        cfg = OmegaConf.create(raw.decode())
        out = cell / n
        builder.output(cfg, n, out, False)
        builder.register(n, cfg, {'kind': 'eval', 'assessment': kind, 'output': str(out)})
    builder.combine_case_evaluations(next_name, 'NPO')
    cfg = OmegaConf.load(configs / (next_name + '_knowledge.yaml'))
    assert cfg.eval.lm_eval_mmlu.simple_evaluate_args.limit == 3
    assert 'limit' not in cfg.eval.lm_eval.simple_evaluate_args
    assert builder.manifest['jobs'][next_name + '_mmlu_validation']['expected_mmlu_count'] == 171
    pilot = object.__new__(FormalPilot)
    pilot.manifest = builder.manifest
    with patch('official_knowledge.production_worker.assessment', side_effect=[({}, {}, {}), KeyError('partial MMLU')]):
        assert not pilot.completed(next_name + '_knowledge')
    with patch('official_knowledge.production_worker.assessment', return_value=({}, {}, {})):
        assert pilot.completed(next_name + '_knowledge')


def check_parent_question_subset(root):
    parent, after = {}, {}
    for index in range(57):
        subject = 'subject_' + str(index)
        rows = [{'doc_id': doc, 'doc': {'question': subject + str(doc)}, 'acc': int(doc % 2 == 0)}
                for doc in range(4)]
        parent[subject], after[subject] = rows, rows[:3]
    atomic_json(root / 'parent/LMEval_EVAL.json', {'pilot_mmlu_validation': parent})
    atomic_json(root / 'after/LMEval_EVAL.json', {'pilot_mmlu_validation': after})
    pilot = SimpleNamespace(manifest={'jobs': {
        'biographies_learn_mmlu_validation': {'output': str(root / 'parent')},
        'case_mmlu_validation': {'output': str(root / 'after'), 'expected_mmlu_count': 171}}})
    accuracy, count = matching_parent_mmlu(pilot, 'biographies', 'case')
    assert count == 171 and abs(accuracy - 2 / 3) < 1e-8
    after['subject_0'][0]['doc'] = {'question': 'different question'}
    atomic_json(root / 'after/LMEval_EVAL.json', {'pilot_mmlu_validation': after})
    try:
        matching_parent_mmlu(pilot, 'biographies', 'case')
    except AssertionError:
        pass
    else:
        raise AssertionError('Parent and adapter were compared on different questions')

def check_training_completion(root):
    folder = root / 'training-completion'
    output = folder / 'saved'
    output.mkdir(parents=True)
    name = 'benchmark_batch_4'
    cfg = OmegaConf.create({'paths': {'output_dir': str(output)}, 'model': {'lora': True},
        'trainer': {'args': {'per_device_train_batch_size': 4, 'gradient_accumulation_steps': 8,
                             'num_train_epochs': 10, 'max_steps': 2}}})
    config = folder / (name + '.yaml')
    OmegaConf.save(cfg, config)
    (output / 'adapter_config.json').write_text('{}')
    (output / 'adapter_model.safetensors').write_bytes(b'fixture' * 200)
    (output / 'prepared-job.json').write_text(json.dumps({'config_sha256': sha(config)}))
    atomic_json(output / 'trainer_state.json', {'global_step': 2})
    pilot = object.__new__(Pilot)
    pilot.data = {'sources': {}}
    pilot.manifest = {'jobs': {name: {'kind': 'train', 'primary_rows': 64}}, 'world_size': 1,
                      'config_sha256': {config.name: sha(config)}}
    pilot.configuration = lambda job: cfg
    assert pilot.completed(name)
    atomic_json(output / 'trainer_state.json', {'global_step': 1})
    assert not pilot.completed(name)
    cfg.trainer.args.max_steps = -1
    atomic_json(output / 'trainer_state.json', {'global_step': 20})
    assert pilot.completed(name)
    pilot.manifest['jobs'][name].pop('primary_rows')
    pilot.manifest['jobs'][name]['source'] = 'biographies'
    pilot.data['sources']['biographies'] = {'learning_rows': 64}
    assert pilot.completed(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / 'src'))
    from evals.lm_eval import LMEvalEvaluator
    import evals.lm_eval as official_eval
    for batch in (1, 2, 'auto'):
        evaluator = object.__new__(LMEvalEvaluator)
        evaluator.eval_cfg = OmegaConf.create({'simple_evaluate_args': {'batch_size': batch, 'max_batch_size': 2}})
        model = Mock()
        with patch.object(official_eval, 'HFLM') as hflm:
            evaluator.prepare_model(model)
            hflm.assert_called_once_with(model, batch_size=batch, max_batch_size=2)
            model.eval.assert_called_once()
    gate_fixture = Path(tempfile.mkdtemp(prefix='dispatch-software-contract-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    atomic_json(gate_fixture / 'formal-lora-400-v1/PRODUCTION_DATA_READY.json', {'software_fixture_only': True})
    atomic_json(gate_fixture / 'official-lora-pilot-v1-llama2-chat/RUN_STATUS-biographies.json',
                {'phase': 'running', 'qualified_methods': []})
    with patch.object(sys, 'argv', ['production_dispatch', '--root', str(gate_fixture), '--upstream', str(args.upstream)]), \
         patch.object(production_dispatch.subprocess, 'Popen') as popen:
        try:
            production_dispatch.main()
        except (AssertionError, FileNotFoundError):
            pass
        else:
            raise AssertionError('An unfinished pilot authorized formal generation')
        popen.assert_not_called()
    recomputed = {}
    for base in ('llama2-chat', 'llama3-instruct'):
        root = args.root / ('official-lora-pilot-v1-' + base)
        manifest = json.loads((root / 'configs/jobs.json').read_text())
        pilot = SimpleNamespace(root=root, manifest=manifest)
        score, _, _ = assessment(pilot, 'biographies_development_NPO_mmlu_validation')
        recomputed[base] = score

    root = Path(tempfile.mkdtemp(prefix='production-counter-contract-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    (root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
    check_registration_metadata(root)
    check_joint_evaluations(root)
    check_parent_question_subset(root)
    check_training_completion(root)
    duplicate = root / 'duplicate-count/cells/example/accepted'
    duplicate.mkdir(parents=True)
    for index in range(2):
        atomic_json(duplicate / (str(index) + '.json'), {'passed': True,
            'activations': {'paper-t2': {}, 't32-standard-chatfix': {}},
            'task': {'unit_id': 'same-content', 'primary_slot_id': 'slot-' + str(index)}})
    try:
        production_dispatch.reports(root / 'duplicate-count', {})
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate content inflated accepted model count')
    result = {'passed': True, 'proof_kind': 'software_fixture_only', 'model_loaded_by_fixture': False,
        'optimizer_steps': 0, 'fixture_directory': str(root), 'dispatch_fixture': str(gate_fixture),
        'recomputed_existing_real_mmlu': recomputed,
        'checks': ['configured_eval_batch_reaches_hflm', 'unfinished_pilots_block_dispatch',
                   'existing_real_mmlu_recomputed_question_by_question', 'duplicate_targets_do_not_inflate_400_count',
                   'existing_job_metadata_backfilled_without_changing_prepared_configuration',
                   'prepared_case_recipe_changes_remain_rejected',
                   'restored_cell_uses_selected_pilot_support_pins',
                   'legacy_joint_alias_migrates_without_affecting_parent_learning_guard',
                   'training_completion_honors_explicit_max_steps_and_lazy_primary_rows',
                   'joint_evaluation_preserves_all_tasks_and_fewshot_protocols',
                   'started_evaluations_keep_frozen_configuration',
                   'joint_evaluation_configuration_is_idempotent',
                   'routine_171_question_scope_is_explicit',
                   'parent_comparison_uses_identical_question_ids_and_content']}
    atomic_json(args.root / 'PRODUCTION_SOFTWARE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
