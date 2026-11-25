"""Kill an owned toy evaluation and resume its official SQLite cache on HDD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from lm_eval.api.model import LM, CachingLM

from official_knowledge.production_jobs import atomic_json
from official_knowledge.response_cache import cache_prefix, install, install_if_ready, install_batched_writes
from official_knowledge.run_pilot import now


class ToyLM(LM):
    def __init__(self, pause=None, pause_after=1, commit_on_pause=True):
        super().__init__()
        self.calls, self.pause = 0, pause
        self.pause_after, self.commit_on_pause = pause_after, commit_on_pause

    def loglikelihood(self, requests, **kwargs):
        output = []
        for req in requests:
            self.calls += 1
            answer = (-float(sum(len(s) for s in req.args)), True)
            self.cache_hook.add_partial('loglikelihood', req.args, answer)
            output.append(answer)
            if self.pause and self.calls == self.pause_after:
                # Observe a durable partial result before interrupting this
                # owned toy process. Official HFLM also emits partial hooks.
                if self.commit_on_pause:
                    self.cache_hook.dbdict.commit()
                self.pause.write_text('toy pause point reached\n')
                time.sleep(60)
        return output

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError


def inputs():
    return [SimpleNamespace(args=('fixture question one', ' A')),
            SimpleNamespace(args=('fixture question two', ' B'))]


def identity(folder, name='first'):
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / 'not-a-real-model.safetensors'
    model.write_bytes(b'software fixture only; no trained model')
    st = model.stat()
    file = folder / 'prepared-evaluation.json'
    atomic_json(file, {'job': 'software_' + name, 'model': str(folder), 'proof_kind': 'software_fixture_only',
        'config_sha256': name, 'data_readiness_sha256': 'software-only',
        'model_files': {str(model): [st.st_size, st.st_mtime_ns]}})
    return file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--child', type=Path)
    p.add_argument('--batched', action='store_true')
    a = p.parse_args()
    if a.child:
        if a.batched:
            install_batched_writes()
        lm = ToyLM(a.child / 'partial-ready', pause_after=35 if a.batched else 1,
                   commit_on_pause=not a.batched)
        cached = CachingLM(lm, cache_prefix(a.child / 'prepared-evaluation.json') + '_rank0.db')
        requests = [SimpleNamespace(args=('batch fixture ' + str(i), ' A')) for i in range(40)] if a.batched else inputs()
        cached.loglikelihood(requests)
        return
    fixture = Path(tempfile.mkdtemp(prefix='response-cache-software-', dir=a.root / 'receipts'))
    (fixture / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No language model loaded or trained.\n')
    file = identity(fixture)
    with (fixture / 'child.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-m', 'official_knowledge.test_response_cache',
            '--root', str(a.root), '--child', str(fixture)], stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 45
            while not (fixture / 'partial-ready').exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Toy partial-response fixture failed to become ready')
                time.sleep(.05)
            assert (Path('/proc') / str(child.pid)).stat().st_uid == os.getuid()
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    prefix = cache_prefix(file)
    lm = ToyLM()
    cached = CachingLM(lm, prefix + '_rank0.db')
    result = cached.loglikelihood(inputs())
    assert lm.calls == 1 and len(result) == 2
    cached.dbdict.close()
    again = ToyLM()
    reused = CachingLM(again, prefix + '_rank0.db')
    assert reused.loglikelihood(inputs()) == result and again.calls == 0
    reused.dbdict.close()
    other = identity(fixture / 'another-model', 'second')
    assert cache_prefix(other) != prefix
    model = ToyLM()
    isolated = CachingLM(model, cache_prefix(other) + '_rank0.db')
    assert isolated.loglikelihood(inputs()) == result and model.calls == 2
    isolated.dbdict.close()
    batch_folder = fixture / 'batched'
    batch_identity = identity(batch_folder, 'batched')
    batch_db = cache_prefix(batch_identity) + '_rank0.db'
    with (batch_folder / 'child.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-m', 'official_knowledge.test_response_cache',
            '--root', str(a.root), '--child', str(batch_folder), '--batched'],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 45
            while not (batch_folder / 'partial-ready').exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Batched cache did not reach its first durable group')
                time.sleep(.05)
            with sqlite3.connect(batch_db, timeout=5) as db:
                assert db.execute('SELECT count(*) FROM unnamed').fetchone()[0] == 32
            assert (Path('/proc') / str(child.pid)).stat().st_uid == os.getuid()
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    install_batched_writes()
    requests = [SimpleNamespace(args=('batch fixture ' + str(i), ' A')) for i in range(40)]
    recovered_model = ToyLM()
    recovered = CachingLM(recovered_model, batch_db)
    responses = recovered.loglikelihood(requests)
    assert recovered_model.calls == 8
    assert responses == ToyLM().loglikelihood(requests)
    recovered.dbdict.close()
    with sqlite3.connect(batch_db) as db:
        assert db.execute('SELECT count(*) FROM unnamed').fetchone()[0] == 40
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    complete_model = ToyLM()
    complete = CachingLM(complete_model, batch_db)
    assert complete.loglikelihood(requests) == responses and complete_model.calls == 0
    complete.dbdict.close()
    install(file)
    from evals.lm_eval import LMEvalEvaluator
    from omegaconf import OmegaConf
    evaluator = LMEvalEvaluator(OmegaConf.create({'tasks': [], 'simple_evaluate_args': {'batch_size': 1}}))
    assert evaluator.simple_evaluate_args['use_cache'] == prefix
    assert evaluator.simple_evaluate_args['batch_size'] == 1
    performance = identity(fixture / 'performance', 'performance_eval_b1')
    with patch('official_knowledge.response_cache.install') as attach:
        install_if_ready(performance)
        attach.assert_not_called()
    weights = fixture / 'not-a-real-model.safetensors'
    weights.write_bytes(b'changed fixture weights')
    try:
        cache_prefix(file)
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed model files reused stale answers')
    receipt = {'at_utc': now(), 'passed': True, 'proof_kind': 'software_fixture_only',
        'language_model_loaded': False, 'optimizer_steps': 0, 'formal_production_authorized': False,
        'fixture': str(fixture), 'backend': 'lm-eval 0.4.11 official CachingLM/CacheHook',
        'response_commit_group': 32, 'maximum_recomputed_uncommitted_responses': 31,
        'checks': ['committed_partial_survives_owned_process_sigkill_on_hdd',
                   'only_missing_response_computed_after_restart', 'complete_repeat_computes_zero_responses',
                   'different_model_identity_has_isolated_cache', 'changed_weights_rejected',
                   'performance_benchmarks_never_reuse_responses',
                   'batched_partial_commit_survives_sigkill_with_at_most_31_recomputed',
                   'final_partial_batch_durable_and_same_responses_as_uncached',
                   'official_evaluator_receives_use_cache_without_other_argument_changes']}
    support = Path(__file__).resolve().parents[1]
    receipt['support_code_sha256'] = {name: hashlib.sha256((support / name).read_bytes()).hexdigest()
        for name in ['official_knowledge/response_cache.py', 'official_knowledge/runtime_entry.py', 'official_pilot/launch.py']}
    atomic_json(a.root / 'RESPONSE_CACHE_SOFTWARE_CHECK.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
