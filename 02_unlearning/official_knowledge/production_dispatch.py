"""Validate pilot evidence and dispatch four gated formal-production workers."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from official_knowledge.night_supervisor import owned_process
from official_knowledge.production_jobs import atomic_json
from official_knowledge.run_pilot import now, sha


METHODS = ('NPO', 'GradDiff', 'SimNPO', 'UNDIAL')


def pilot_gate(project):
    from official_knowledge.pilot_registry import active, validate_suite
    evidence = {}
    for base in ('llama2-chat', 'llama3-instruct'):
        for source in ('biographies', 'squad'):
            suite, _ = active(project, base, source)
            evidence.update(validate_suite(suite, source))
    return evidence


def reports(root, status):
    accepted = []
    for p in (root / 'cells').glob('*/accepted/*.json'):
        result = json.loads(p.read_text())
        assert result['passed'] and set(result['activations']) == {'paper-t2', 't32-standard-chatfix'}
        accepted.append(result)
    ids = [x['task']['unit_id'] for x in accepted]
    slots = [x['task']['primary_slot_id'] for x in accepted]
    assert len(ids) == len(set(ids)) and len(slots) == len(set(slots))
    counts = {'accepted': len(accepted), 'target': 400,
              'by_base': dict(collections.Counter(x['task']['base'] for x in accepted)),
              'by_method': dict(collections.Counter(x['task']['method'] for x in accepted)),
              'by_split': dict(collections.Counter(x['task']['split'] for x in accepted))}
    status.update(counts)
    status['phase'] = ('formal_cases_running' if accepted else
                       'formal_parent_learning_and_qualification' if status.get('started') else
                       'checking_saved_adapters')
    if len(accepted) == 400:
        assert set(counts['by_base'].values()) == {200} and set(counts['by_method'].values()) == {100}
        assert counts['by_split'] == {'train': 200, 'validation': 100, 'test': 100}
        for result in accepted:
            assert sha(Path(result['adapter']) / 'adapter_model.safetensors') == result['adapter_sha256']
        status.update(phase='complete', completed_at_utc=now())
    lines = ['# 400 个纯 LoRA 遗忘模型：真实生产进度', '', '更新时间：' + now(), '',
             f"流程完整的正式产物：**{len(accepted)} / 400**。正常对照单独计数。", '',
             '每个合格产物包含最终适配器、准确学习起点、数据与配置身份、逐题评估及两套分开标记的活动文件。', '',
             '| 方法 | 合格模型数 | 原先目标知识平均遗忘比例 | 其他原有知识平均保留比例 | 一般答题正确率平均变化 |',
             '|---|---:|---:|---:|---:|']
    en = ['# Formal LoRA production: measured progress', '', 'Updated: ' + now(), '',
          f'Complete formal artifacts: **{len(accepted)} / 400**. Normal controls are counted separately.', '',
          '| Method | Accepted models | Mean previously known target facts forgotten | Mean other known facts retained | Mean general-ability accuracy change |',
          '|---|---:|---:|---:|---:|']
    for method in METHODS:
        subset = [x for x in accepted if x['task']['method'] == method]
        if not subset:
            row = f'| {method} | 0 | — | — | — |'
        else:
            n = len(subset)
            forget = sum(1 - x['target_previously_correct_fraction_remaining'] for x in subset) / n
            retain = sum(x['retain_previously_correct_fraction_remaining'] for x in subset) / n
            delta = sum(x['general_ability_after'] - x['general_ability_before'] for x in subset) / n
            row = f'| {method} | {n} | {forget:.1%} | {retain:.1%} | {delta * 100:+.2f} pp |'
        lines.append(row)
        en.append(row)
    lines += ['', '| 方法 | 检查1,531题的模型数 | 检查171题的模型数 |', '|---|---:|---:|']
    en += ['', '| Method | Models evaluated on 1,531 questions | Models evaluated on 171 questions |', '|---|---:|---:|']
    for method in METHODS:
        scopes = collections.Counter(x.get('general_ability_questions', 1531) for x in accepted if x['task']['method'] == method)
        row = f'| {method} | {scopes[1531]} | {scopes[171]} |'
        lines.append(row)
        en.append(row)
    lines += ['', '原始基座和四份学习起点单独保存。每个模型的训练材料数量、待遗忘对象、检查题数量、一般能力题数量、文件哈希和活动协议均记录在对应 accepted 收据中。',
              '流程优先模式按保存、重载、数据身份与激活完整性统计交付。遗忘、知识保留和一般能力分数逐项记录。每个基座与材料组合完成自己的产物检查后接续正式训练。',
              '提速方案为每种基座、材料和方法的首份模型保留完整1,531题，后续模型检查57个科目各3题，共171题。每份模型与其学习起点按完全相同的问题比较，目标题和保留题全部检查。旧产物保留原检查题数。',
              'paper-t2 保留论文式取位；t32-standard-chatfix 使用已有内容取位协议，两者分别保存。活动记录数均为每模型 64 条、三层各 4,096 个数值。']
    en += ['', 'Shared original models and four learned parents are stored separately. Every accepted receipt identifies the training data and counts, target, question counts, general-ability evaluation, file hashes, and activation protocols.',
           'Pipeline-first delivery counts complete saved, reloaded, provenance-checked adapters and activations. Unlearning, retention and general-ability scores remain recorded individually. Each source/base cell proceeds after its own artifact checks.',
           'The throughput plan keeps all 1,531 general-ability questions for the first model of each base, material and method. Later models receive three questions per subject across 57 subjects, totaling 171. Each model and its parent are compared on identical questions. Target and retention questions remain complete. Historical artifacts retain their original question counts.',
           'Paper-style paper-t2 and the existing content-position t32-standard-chatfix are saved separately. Both formats have 64 records across three 4,096-value layers.']
    (root / 'PROGRESS_ZH.md').write_text('\n'.join(lines) + '\n')
    (root / 'PROGRESS_EN.md').write_text('\n'.join(en) + '\n')
    atomic_json(root / 'ACCEPTED_COUNTS.json', counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    project, upstream = args.root.resolve(), args.upstream.resolve()
    formal = project / 'formal-lora-400-v1'
    if not (formal / 'PRODUCTION_DATA_READY.json').exists():
        raise RuntimeError('Formal content preparation is incomplete')
    with (formal / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = pilot_gate(project)
        data = json.loads((formal / 'PRODUCTION_DATA_READY.json').read_text())
        assert data['primary_target'] == 400 and data['content_target_cross_split_overlap'] == 0
        assert sha(formal / 'inventory.jsonl') == data['inventory_sha256']
        assert sha(formal / 'reserve-targets.jsonl') == data['reserves_sha256']
        path = formal / 'PRODUCTION_STATUS.json'
        state = json.loads(path.read_text()) if path.exists() else {'started': False, 'workers': {}, 'phase': 'checking_saved_adapters'}
        state.update(at_utc=now(), primary_target=400, small_pilot_evidence_sha256=evidence)
        support = Path(__file__).resolve().parents[1]
        sources = sorted((support / 'official_knowledge').glob('*.py')) + sorted((support / 'official_pilot').glob('*.py'))
        version = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
        env = dict(os.environ, PYTHONPATH=str(support) + ':' + str(upstream / 'src'),
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
        for parent in json.loads((formal / 'parents.json').read_text()):
            key = parent['key']
            worker = state['workers'].setdefault(key, {})
            cell = formal / 'cells' / (parent['base'] + '-' + parent['source'])
            status_path = cell / 'PRODUCTION_WORKER.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            worker['last_status'] = status
            state['started'] = state['started'] or status.get('formal_batch_started', False)
            if status.get('phase') == 'formal_cell_complete':
                continue
            if owned_process(worker.get('pid', -1), 'official_knowledge.production_worker', project):
                continue
            if worker.get('started_version') == version:
                worker['phase'] = 'needs_repair'
                continue
            log = project / 'logs' / ('production-' + parent['base'] + '-' + parent['source'] + '.log')
            with log.open('a') as stream:
                process = subprocess.Popen([sys.executable, '-m', 'official_knowledge.production_worker', '--root', str(project),
                    '--upstream', str(upstream), '--cell', key], cwd=support, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            worker.update(pid=process.pid, phase='running', started_version=version, log=str(log), started_at_utc=now())
        reports(formal, state)
        atomic_json(path, state)
        print(json.dumps({'phase': state['phase'], 'started': state['started'], 'accepted': state['accepted'], 'target': 400}))


if __name__ == '__main__':
    main()
