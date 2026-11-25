"""Durable four-worker handoff and gated production dispatch for this account."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def owned_process(pid, module, suite):
    p = Path('/proc') / str(pid)
    try:
        if p.stat().st_uid != os.getuid():
            return False
        args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return module in args and str(suite) in args
    except (FileNotFoundError, ProcessLookupError):
        return False


def handoff_old(suite, pid):
    if not owned_process(pid, 'official_knowledge.run_pilot', suite):
        return True
    status = read(suite / 'RUN_STATUS.json', {})
    # Finish the already-running training before transferring its coordinator.
    if status.get('current_job') == 'biographies_development_GradDiff' and status.get('phase') in ('running', 'exporting_checkpoint'):
        return False
    unit = status.get('resource_scope', '')
    if unit and not unit.startswith(f'ulp-small-{pid}-'):
        raise RuntimeError('Old experiment scope does not match its verified coordinator')
    os.kill(pid, signal.SIGTERM)
    if unit:
        group = subprocess.run(['systemctl', '--user', 'show', unit, '--property=ControlGroup', '--value'],
                               capture_output=True, text=True).stdout.strip()
        if group:
            if not group.startswith(f'/user.slice/user-{os.getuid()}.slice/'):
                raise RuntimeError('Refusing to stop a scope outside this account')
            subprocess.run(['systemctl', '--user', 'stop', unit], check=True, capture_output=True)
    archive = suite / 'receipts' / 'night-handoff-20261106'
    archive.mkdir(parents=True, exist_ok=True)
    (archive / 'OLD_COORDINATOR.json').write_text(json.dumps({'at': now(), 'pid': pid, 'last_status': status,
        'completed_training_reused': True, 'other_users_stopped': False}, indent=2) + '\n')
    return True


def auxiliary_workers(root, support, upstream, config, state, env):
    """Adopt detached corrective pilots; restart failed code once per revision."""
    versions = sorted((support / 'official_knowledge').glob('*.py')) + sorted((support / 'official_pilot').glob('*.py'))
    code_digest = hashlib.sha256(b''.join(p.read_bytes() for p in versions)).digest()
    for spec in config.get('auxiliary_workers', []):
        module = spec['module']
        if module not in ('official_knowledge.run_source', 'official_knowledge.fresh_npo_confirmation'):
            raise ValueError('Unexpected auxiliary experiment module')
        marker, status = Path(spec['process_marker']).resolve(), Path(spec['status_file']).resolve()
        if not marker.is_relative_to(root) or not status.is_relative_to(root):
            raise ValueError('Auxiliary experiment lies outside the owned project')
        worker = state.setdefault('auxiliary_workers', {}).setdefault(spec['key'], {})
        report = read(status, {})
        worker['last_status'] = report
        pid = worker.get('pid', spec.get('adopt_pid', -1))
        version_file = Path(spec['version_file']) if spec.get('version_file') else None
        version = hashlib.sha256(code_digest + (version_file.read_bytes() if version_file else b'')).hexdigest()
        if owned_process(pid, module, marker):
            worker.update(pid=pid, phase='running')
            worker.setdefault('started_version', version)
            continue
        if report.get('phase') in spec.get('terminal_phases', []):
            worker['phase'] = report['phase']
            continue
        if worker.get('started_version') == version:
            worker['phase'] = 'needs_repair'
            continue
        log = root / 'logs' / spec['log_name']
        with log.open('a') as stream:
            proc = subprocess.Popen([sys.executable, '-m', module, *spec['arguments']], cwd=support, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        worker.update(pid=proc.pid, phase='running', started_version=version, started_at_utc=now(), log=str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    root, upstream = args.root.resolve(), args.upstream.resolve()
    if root != Path('/proj/rcs-hdd/model-tampering/zh2701/unlearning-400') or root.stat().st_uid != os.getuid():
        raise ValueError('Unexpected supervisor root or owner')
    support = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(support) + ':' + str(upstream / 'src'),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
    lock = (root / 'night-supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = read(root / 'NIGHT_STATUS.json', {})
    state = {'started_at_utc': now(), 'supervisor_pid': os.getpid(), 'primary_target': 400,
             'formal_batch_started': False, 'workers': prior.get('workers', {}),
             'auxiliary_workers': prior.get('auxiliary_workers', {}), 'phase': 'validating_pilots'}
    while True:
        config = read(root / 'NIGHT_CONFIG.json')
        if config is None:
            raise RuntimeError('Night execution configuration is absent')
        from official_knowledge.pilot_registry import active, promote_ready_qa
        try:
            promoted = promote_ready_qa(root)
            state.pop('qa_promotion_error', None)
            if promoted:
                state['qa_protocol_promoted'] = promoted
        except Exception as error:
            state['qa_promotion_error'] = repr(error)
        passed = []
        for base in ('llama2-chat', 'llama3-instruct'):
            original = root / ('official-lora-pilot-v1-' + base)
            old_pid = config['old_coordinator_pids'][base]
            ready = handoff_old(original, old_pid)
            for source in ('biographies', 'squad'):
                suite, _ = active(root, base, source)
                key = base + '/' + source
                worker = state['workers'].setdefault(key, {})
                report = read(suite / ('RUN_STATUS-' + source + '.json'), {})
                worker.update(last_status=report)
                recipes = read(suite / ('SELECTED_RECIPES-' + source + '.json'), {})
                if report.get('phase') == 'pilot_source_passed' and len(recipes) == 4 and all(
                        x.get('confirmation_passed') for x in recipes.values()):
                    passed.append(key)
                    continue
                if source == 'biographies' and not ready:
                    worker['phase'] = 'waiting_for_current_training_handoff'
                    continue
                if owned_process(worker.get('pid', -1), 'official_knowledge.run_source', suite):
                    worker['phase'] = 'running'
                    continue
                sources = sorted((support / 'official_knowledge').glob('*.py')) + sorted((support / 'official_pilot').glob('*.py'))
                repair = suite / ('REPAIR_RECIPES-' + source + '.json')
                version = hashlib.sha256((suite / 'configs/jobs.json').read_bytes() +
                                         (repair.read_bytes() if repair.exists() else b'') +
                                         b''.join(p.read_bytes() for p in sources)).hexdigest()
                if worker.get('started_version') == version:
                    worker['phase'] = 'needs_repair' if report.get('phase') == 'needs_repair' else 'needs_research'
                    continue
                log = suite / 'logs' / ('source-' + source + '-worker.log')
                with log.open('a') as stream:
                    proc = subprocess.Popen([sys.executable, '-m', 'official_knowledge.run_source', '--suite-root', str(suite),
                        '--source', source, '--upstream', str(upstream)], cwd=support, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                worker.update(pid=proc.pid, phase='running', started_version=version, started_at_utc=now(), log=str(log))
        auxiliary_workers(root, support, upstream, config, state, env)
        state.update(updated_at_utc=now(), source_pilots_passed=passed)
        if len(passed) == 4:
            dispatch = support / 'official_knowledge/production_dispatch.py'
            if dispatch.exists():
                gate = subprocess.run([sys.executable, str(dispatch), '--root', str(root), '--upstream', str(upstream)],
                                      cwd=support, env=env, capture_output=True, text=True)
                state.update(production_dispatch_returncode=gate.returncode,
                             production_dispatch_output=(gate.stdout + gate.stderr)[-4000:])
                state['phase'] = 'production_dispatched' if gate.returncode == 0 else 'production_preparation_required'
                state['formal_batch_started'] = bool(read(root / 'formal-lora-400-v1/PRODUCTION_STATUS.json', {}).get('started'))
            else:
                state['phase'] = 'production_preparation_required'
        path = root / 'NIGHT_STATUS.json'
        temporary = path.with_suffix('.pending.json')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
        time.sleep(30)


if __name__ == '__main__':
    main()
