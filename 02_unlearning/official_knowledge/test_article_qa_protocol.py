"""CPU checks of QA case construction, isolation and fail-closed promotion."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from omegaconf import OmegaConf

from official_knowledge.article_qa_protocol import PROTOCOL
from official_knowledge.pilot_registry import active, promote_ready_qa
from official_knowledge.prepare_data import write
from official_knowledge.production_jobs import ProductionJobs, atomic_json, records
from official_knowledge.run_pilot import now, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    formal = a.root / 'formal-lora-400-v1'
    ready = json.loads((formal / 'ARTICLE_QA_PROTOCOL_READY.json').read_text())
    fixture = Path(tempfile.mkdtemp(prefix='article-qa-software-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    (fixture / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
    gate = fixture / 'gate'
    gate.mkdir()
    atomic_json(gate / 'formal-lora-400-v1/ARTICLE_QA_PROTOCOL_READY.json', {'passed': True, 'protocol': PROTOCOL})
    atomic_json(gate / 'ARTICLE_QA_SOFTWARE_CHECK.json', {'passed': True, 'proof_kind': 'software_fixture_only'})
    for base in ('llama2-chat', 'llama3-instruct'):
        atomic_json(gate / ('official-qa-unlearning-v2-' + base) / 'RUN_STATUS-squad.json', {'phase': 'running'})
    assert not promote_ready_qa(gate)
    assert not (gate / 'ACTIVE_PILOT_SUITES.json').exists()
    assert active(gate, 'llama2-chat', 'squad') == (gate / 'official-lora-pilot-v1-llama2-chat', None)
    summaries = {}
    for parent in json.loads((formal / 'parents.json').read_text()):
        if parent['source'] != 'squad':
            continue
        base, data = parent['base'], Path(parent['data_dir'])
        for file, expected in ready['bases'][base]['files_sha256'].items():
            assert sha(Path(file)) == expected
        units = records(data / 'units.jsonl')
        selected = units[:2]
        view = data / 'protocols' / PROTOCOL
        all_probes = records(view / 'eval-targets.jsonl') + records(view / 'eval-retain.jsonl')
        root = fixture / base
        root.mkdir()
        builder = object.__new__(ProductionJobs)
        builder.root, builder.project, builder.parent = root, a.root, parent
        builder.source, builder.pilot = 'squad', a.root / ('official-qa-unlearning-v2-' + base)
        builder.evaluation_protocol = PROTOCOL
        builder.configs, builder.tasks = root / 'configs', root / 'tasks'
        builder.configs.mkdir()
        builder.tasks.mkdir()
        builder.units = {u['unit_id']: u for u in selected}
        builder.manifest = copy.deepcopy(json.loads((builder.pilot / 'configs/jobs.json').read_text()))
        builder.manifest.update(jobs={}, config_sha256={}, task_sha256={}, evaluation_protocol=PROTOCOL)
        for role, unit in zip(('targets', 'retain'), selected):
            write(builder.evaluation_file(role), [q for q in all_probes if q['unit_id'] == unit['unit_id']])
        task = {'task_id': 'software-only-case', 'unit_id': selected[0]['unit_id'], 'method': 'NPO'}
        with patch.object(builder, 'data_protocol', return_value=PROTOCOL):
            name, _ = builder.case(task)
        folder = root / 'case-data-protocols' / PROTOCOL / task['task_id']
        forget, retained = records(folder / 'forget.jsonl'), records(folder / 'retain.jsonl')
        probes = records(folder / 'eval-target.jsonl')
        assert len(forget) == len(retained) == len(probes) == 40
        assert all(r['kind'] == 'published_qa' for r in forget + retained)
        assert {r['question_id'] for r in forget} == {q['probe_id'] for q in probes}
        assert not {q['probe_id'] for q in probes} & {q['probe_id'] for q in selected[0]['probes']}
        job = builder.manifest['jobs'][name]
        assert job['primary_rows'] == job['retain_rows'] == 40 and job['forget_data_protocol'] == PROTOCOL
        cfg = OmegaConf.load(builder.configs / (name + '.yaml'))
        assert cfg.trainer.args.save_strategy == 'no' and not cfg.checkpoint_first_step
        assert cfg.data.forget.published_records.args.hf_args.data_files == str(folder / 'forget.jsonl')
        knowledge = builder.manifest['jobs'][name + '_knowledge']
        assert list(knowledge['expected_task_counts'].values()) == [40, 40]
        builder.preserve_unstarted_configuration()
        assert list((root / 'receipts').glob('before-protocol-*/configs/jobs.json'))
        output = Path(job['output'])
        atomic_json(output / 'prepared-job.json', {'proof_kind': 'software_fixture_only'})
        try:
            builder.preserve_unstarted_configuration()
        except ValueError as error:
            assert 'started formal cell' in str(error)
        else:
            raise AssertionError('Started formal configuration was replaced')
        for file, expected in ready['bases'][base]['files_sha256'].items():
            assert sha(Path(file)) == expected
        project_fixture = fixture / (base + '-constructor')
        project_fixture.mkdir()
        original = a.root / ('official-lora-pilot-v1-' + base)
        with patch('official_knowledge.pilot_registry.active', return_value=(original, None)):
            first = ProductionJobs(project_fixture, parent)
        (first.root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
        assert list(first.manifest['jobs']['squad_base_knowledge']['expected_task_counts'].values()) == [3000, 100]
        with patch('official_knowledge.pilot_registry.active', return_value=(builder.pilot, {'data_protocol': PROTOCOL})):
            migrated = ProductionJobs(project_fixture, parent)
        assert migrated.manifest['source_pilot_suite'] == str(builder.pilot)
        assert migrated.manifest['evaluation_protocol'] == PROTOCOL
        assert list(migrated.manifest['jobs']['squad_base_knowledge']['expected_task_counts'].values()) == [5800, 200]
        assert list((migrated.root / 'receipts').glob('before-protocol-*/configs/jobs.json'))
        learning = OmegaConf.load(migrated.configs / 'squad_learn.yaml')
        assert learning.data.train.published_records.args.hf_args.data_files == str(migrated.root / 'data/learn.jsonl')
        assert learning.trainer.args.save_strategy == 'no' and learning.trainer.args.num_train_epochs == 20
        atomic_json(migrated.learned_adapter / 'prepared-job.json', {'proof_kind': 'software_fixture_only'})
        with patch('official_knowledge.pilot_registry.active', return_value=(original, None)):
            try:
                ProductionJobs(project_fixture, parent)
            except ValueError as error:
                assert 'started formal cell' in str(error)
            else:
                raise AssertionError('A started cell changed its pilot through the constructor')
        qa_only = a.root / ('official-qa-learning-v3-' + base)
        if qa_only.exists():
            project_v3 = fixture / (base + '-qa-only-constructor')
            project_v3.mkdir()
            entry = {'data_protocol': PROTOCOL, 'learning_data_protocol': 'published_qa_only_v1',
                'formal_learning_rows': ready['bases'][base]['learning_qa_rows'],
                'formal_learning_tokens': ready['bases'][base]['learning_qa_tokens']}
            with patch('official_knowledge.pilot_registry.active', return_value=(qa_only, entry)):
                focused = ProductionJobs(project_v3, parent)
            (focused.root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
            focused_cfg = OmegaConf.load(focused.configs / 'squad_learn.yaml')
            assert focused.parent['learning_rows'] == focused.manifest['jobs']['squad_learn']['primary_rows'] == 6000
            assert focused.parent['original_mixed_learning_rows'] == parent['learning_rows']
            assert focused_cfg.trainer.args.num_train_epochs == 10
            assert focused_cfg.data.train.published_records.args.hf_args.data_files == str(focused.evaluation_file('targets').parent / 'learn.jsonl')
            assert len(records(Path(focused_cfg.data.train.published_records.args.hf_args.data_files))) == 6000
        summaries[base] = {'forget_rows': 40, 'retain_rows_fixture': 40, 'target_eval_questions': 40,
                          'original_learning_rows_unchanged': parent['learning_rows']}
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'software_fixture_only',
        'model_loaded': False, 'optimizer_steps': 0, 'formal_production_authorized': False,
        'fixture': str(fixture), 'bases': summaries,
        'checks': ['unqualified_qa_pilot_cannot_activate', 'all_forget_and_retain_rows_are_published_qa',
                   'target_evaluation_matches_learning_questions', 'different_fact_questions_kept_separate',
                   'actual_case_row_counts_and_final_only_saving', 'unstarted_configs_archived',
                   'full_constructor_switches_pilot_and_all_parent_evaluations_together',
                   'qa_only_parent_uses_six_thousand_qa_and_ten_epochs_when_selected',
                   'started_cell_protocol_change_rejected', 'original_datasets_unchanged']}
    atomic_json(a.root / 'ARTICLE_QA_SOFTWARE_CHECK.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
