"""Select a qualified pilot and its matching formal data view with evidence."""
import fcntl
import json
import os
from pathlib import Path

from official_knowledge.pilot_recipes import selected, repaired, selected_control
from official_knowledge.run_pilot import now, sha
from official_knowledge.acceptance import unlearning_passed


METHODS = ('NPO', 'GradDiff', 'SimNPO', 'UNDIAL')
QA_PROTOCOL = 'published_qa_answers_only_v2'


def active(project, base, source):
    file = project / 'ACTIVE_PILOT_SUITES.json'
    entry = json.loads(file.read_text()).get('sources', {}).get(base + '/' + source) if file.exists() else None
    if entry:
        suite = Path(entry['suite']).resolve()
        assert suite.is_relative_to(project.resolve()) and suite.stat().st_uid == os.getuid()
        assert source == 'squad' and entry['data_protocol'] == QA_PROTOCOL
        ready = project / 'formal-lora-400-v1/ARTICLE_QA_PROTOCOL_READY.json'
        assert sha(ready) == entry['formal_data_receipt_sha256']
        receipt = json.loads(ready.read_text())
        assert receipt['passed'] and receipt['protocol'] == QA_PROTOCOL
        assert entry['formal_data_dir'] == receipt['bases'][base]['data_dir']
        if entry.get('learning_data_protocol') == 'published_qa_only_v1':
            assert entry['formal_learning_rows'] == receipt['bases'][base]['learning_qa_rows'] == 6000
        for path, expected in receipt['bases'][base]['files_sha256'].items():
            assert sha(Path(path)) == expected
        return suite, entry
    return project / ('official-lora-pilot-v1-' + base), None


def validate_suite(suite, source):
    """Validate measured outcomes, file evidence and the evaluated models."""
    state = json.loads((suite / ('RUN_STATUS-' + source + '.json')).read_text())
    assert state['phase'] == 'pilot_source_passed' and set(state['qualified_methods']) == set(METHODS)
    recipes = selected(suite, source)
    assert set(recipes) == set(METHODS)
    manifest = json.loads((suite / 'configs/jobs.json').read_text())
    policy = manifest['quality_policy']
    assert sha(suite / 'data-readiness.json') == manifest['data_readiness_sha256']
    evidence = {}
    for method, recipe in recipes.items():
        if recipe.get('repaired_after_original_confirmation'):
            repair = repaired(suite, source, method)
            assert repair and recipe['fresh_confirmation_suite'] == repair['fresh_confirmation_suite']
            fresh = Path(repair['fresh_confirmation_suite'])
            for file in [fresh / 'FRESH_PREPARED.json', fresh / 'FRESH_CONFIRMATION_RESULT.json',
                         *sorted((fresh / 'gates').glob(source + '-*.json'))]:
                evidence[str(file)] = sha(file)
    for phase in ('development', 'confirmation'):
        for method in (*METHODS, 'control_1e-4'):
            stage = selected_control(suite, source, phase) if method == 'control_1e-4' else recipes[method][phase + '_stage']
            if method != 'control_1e-4':
                assert recipes[method]['development_passed'] and recipes[method]['confirmation_passed']
            path = suite / 'gates' / (source + '-' + stage + '-quality.json')
            gate = json.loads(path.read_text())
            assert gate['proof_kind'] == 'official_model_evaluation' and gate['passed']
            assert gate['data_readiness_sha256'] == manifest['data_readiness_sha256']
            if 'quality_policy' in gate:
                assert gate['quality_policy'] == policy
                assert gate['quality_policy_id'] == manifest.get('quality_policy_id', 'legacy-strict-30pct')
            if method == 'control_1e-4':
                assert gate['retain_previously_correct_fraction_remaining'] >= policy['retain_previously_correct_min']
                assert gate['mmlu_before'] - gate['mmlu_after'] <= policy['mmlu_drop_max']
                assert gate['target_previously_correct_fraction_remaining'] >= policy['retain_previously_correct_min']
            else:
                assert unlearning_passed(policy, gate['target_previously_correct_fraction_remaining'],
                    gate['retain_previously_correct_fraction_remaining'], gate['mmlu_before'], gate['mmlu_after'])
            from official_pilot.launch import learning_configuration_matches
            assert learning_configuration_matches(suite, gate, manifest)
            for file, expected in gate['evidence_sha256'].items():
                assert sha(suite / file) == expected
            for file, expected in gate['model_files'].items():
                st = Path(file).stat()
                assert [st.st_size, st.st_mtime_ns] == expected
            evidence[str(path)] = sha(path)
    return evidence


def promote_ready_qa(project):
    """Activate an article protocol only after four real method confirmations."""
    ready = project / 'formal-lora-400-v1/ARTICLE_QA_PROTOCOL_READY.json'
    software = project / 'ARTICLE_QA_SOFTWARE_CHECK.json'
    if not ready.exists() or not software.exists():
        return []
    assert json.loads(software.read_text())['passed']
    receipt = json.loads(ready.read_text())
    assert receipt['passed'] and receipt['protocol'] == QA_PROTOCOL
    file = project / 'ACTIVE_PILOT_SUITES.json'
    with (project / 'active-pilot-suites.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(file.read_text()) if file.exists() else {'sources': {}}
        changed = []
        for base in ('llama2-chat', 'llama3-instruct'):
            key = base + '/squad'
            if key in registry['sources']:
                active(project, base, 'squad')
                continue
            candidates = project / 'QA_CANDIDATE_SUITES.json'
            choices = json.loads(candidates.read_text()) if candidates.exists() else {}
            suite = Path(choices.get(base, str(project / ('official-qa-unlearning-v2-' + base)))).resolve()
            assert suite.is_relative_to(project.resolve()) and suite.stat().st_uid == os.getuid()
            status = suite / 'RUN_STATUS-squad.json'
            if not status.exists() or json.loads(status.read_text()).get('phase') != 'pilot_source_passed':
                continue
            evidence = validate_suite(suite, 'squad')
            manifest = json.loads((suite / 'configs/jobs.json').read_text())
            for recipe in selected(suite, 'squad').values():
                name = 'squad_' + recipe['development_stage']
                assert manifest['jobs'][name]['forget_data_protocol'] == QA_PROTOCOL
            for path, expected in receipt['bases'][base]['files_sha256'].items():
                assert sha(Path(path)) == expected
            cell = project / 'formal-lora-400-v1/cells' / (base + '-squad')
            assert not list(cell.glob('runs/**/prepared-job.json'))
            assert not list(cell.glob('evals/**/prepared-evaluation.json'))
            registry['sources'][key] = {'suite': str(suite), 'data_protocol': QA_PROTOCOL,
                'learning_data_protocol': manifest.get('learning_data_protocol', 'mixed_body_qa_v1'),
                'formal_data_dir': receipt['bases'][base]['data_dir'],
                'formal_data_receipt_sha256': sha(ready), 'pilot_evidence_sha256': evidence,
                'activated_at_utc': now()}
            if manifest.get('learning_data_protocol') == 'published_qa_only_v1':
                registry['sources'][key].update(formal_learning_rows=receipt['bases'][base]['learning_qa_rows'],
                    formal_learning_tokens=receipt['bases'][base]['learning_qa_tokens'])
            changed.append(key)
        if changed:
            registry['updated_at_utc'] = now()
            from official_knowledge.production_jobs import atomic_json
            atomic_json(file, registry)
        return changed
