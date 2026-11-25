"""Published learned-QA views for formal article tasks; original files stay intact."""
import argparse
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer

from official_knowledge.prepare_data import digest, write
from official_knowledge.production_jobs import atomic_json, records
from official_knowledge.run_pilot import now, sha


PROTOCOL = 'published_qa_answers_only_v2'


def question_index(project):
    result = {}
    for split in ('train', 'dev'):
        data = json.loads((project / 'datasets/knowledge-20261101' / ('squad_' + split + '.json')).read_text())
        for article in data['data']:
            for paragraph in article['paragraphs']:
                for question in paragraph['qas']:
                    assert question['id'] not in result
                    result[question['id']] = (question, paragraph['context'], split)
    return result


def probes(learning, questions):
    output = []
    for row in learning:
        if row['kind'] != 'published_qa':
            continue
        question, context, split = questions[row['question_id']]
        answers = list(dict.fromkeys(a['text'] for a in question['answers']
            if context[a['answer_start']:a['answer_start'] + len(a['text'])] == a['text']))
        assert answers and row['evidence_sha256'] == digest(context)
        assert row['text'].lstrip().startswith(answers[0])
        output.append({'unit_id': row['unit_id'], 'fact_id': row['fact_id'], 'probe_id': row['question_id'],
            'prompt': row['prompt'], 'question': question['question'], 'answers': answers,
            'source': 'squad', 'source_split': split, 'evidence_sha256': row['evidence_sha256'],
            'evaluation_condition': 'published_question_used_during_learning'})
    return output


def prepare(project):
    formal = project / 'formal-lora-400-v1'
    ready = formal / 'ARTICLE_QA_PROTOCOL_READY.json'
    if ready.exists():
        receipt = json.loads(ready.read_text())
        for entry in receipt['bases'].values():
            for file, expected in entry['files_sha256'].items():
                assert sha(Path(file)) == expected
        if all('learning_qa_file' in entry for entry in receipt['bases'].values()):
            return receipt
        assert not (project / 'ACTIVE_PILOT_SUITES.json').exists()
        archive = formal / 'receipts' / ('before-qa-learning-view-' + sha(ready)[:12])
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ready, archive / ready.name)
    from data.utils import preprocess_pretraining_instance
    questions, outcomes = question_index(project), {}
    parents = [p for p in json.loads((formal / 'parents.json').read_text()) if p['source'] == 'squad']
    assert len(parents) == 2
    for parent in parents:
        folder = Path(parent['data_dir'])
        units = records(folder / 'units.jsonl')
        all_ids = {p['unit_id'] for p in records(folder / 'eval-targets.jsonl')}
        retained = {p['unit_id'] for p in records(folder / 'eval-retain.jsonl')}
        targets = all_ids - retained
        assert all_ids == {u['unit_id'] for u in units}
        assert len(units) == 150 and len(targets) == 145 and len(retained) == 5 and not targets & retained
        tokenizer = AutoTokenizer.from_pretrained(parent['hdd_original'], local_files_only=True)
        all_probes = []
        for unit in units:
            qa = [r for r in unit['learning'] if r['kind'] == 'published_qa']
            assert len(qa) == 40 and len({r['fact_id'] for r in qa}) == 40
            for row in qa:
                value = preprocess_pretraining_instance(tokenizer, row['prompt'], row['text'], 1024)
                labels = value['labels']
                supervised = labels[labels != -100]
                prefix = tokenizer.encode(row['prompt'], add_special_tokens=True)
                assert len(value['input_ids']) <= 1024 and supervised.numel() > 0
                assert value['input_ids'][:len(prefix)].tolist() == prefix
                assert (labels[:len(prefix)] == -100).all()
                decoded = tokenizer.decode(supervised.tolist(), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).strip()
                assert decoded == row['text'].replace(tokenizer.eos_token, '').strip()
            all_probes.extend(probes(qa, questions))
        assert len(all_probes) == 6000 and len({p['probe_id'] for p in all_probes}) == 6000
        view = folder / 'protocols' / PROTOCOL
        learning_qa = [r for unit in units for r in unit['learning'] if r['kind'] == 'published_qa']
        write(view / 'learn.jsonl', learning_qa)
        write(view / 'eval-targets.jsonl', [p for p in all_probes if p['unit_id'] in targets])
        write(view / 'eval-retain.jsonl', [p for p in all_probes if p['unit_id'] in retained])
        outcomes[parent['base']] = {'data_dir': str(view), 'articles': 150, 'qa_per_article': 40,
            'learning_qa_file': str(view / 'learn.jsonl'), 'learning_qa_rows': len(learning_qa),
            'learning_qa_tokens': sum(r['tokens'] for r in learning_qa),
            'target_articles_including_reserves': 145, 'target_questions': 5800,
            'retention_articles': 5, 'retention_questions': 200, 'retention_training_qa_per_case': 5960,
            'original_learning_rows': parent['learning_rows'], 'new_training_updates': 0,
            'files_sha256': {str(p): sha(p) for p in [folder / 'units.jsonl', folder / 'learn.jsonl',
                folder / 'eval-targets.jsonl', folder / 'eval-retain.jsonl', *view.glob('*.jsonl')]}}
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'published_data_and_official_tokenization_validation',
        'protocol': PROTOCOL, 'formal_production_authorized': False, 'optimizer_steps': 0,
        'original_files_changed': False, 'evaluation_condition': 'published_question_used_during_learning',
        'bases': outcomes}
    atomic_json(ready, receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.root)), flush=True)
