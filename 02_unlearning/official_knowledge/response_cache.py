"""Use lm-eval's official response cache for interruption-resilient evaluation."""
import hashlib
import json
import os
from pathlib import Path


def install_batched_writes():
    """Commit official response records in groups of 32 on the shared HDD.

    CachingLM still commits its final partial batch before returning responses.
    An interrupted inference may recompute at most 31 uncommitted records.
    Database format, request keys, model responses and scoring stay unchanged.
    """
    from lm_eval.api.model import CachingLM
    from sqlitedict import SqliteDict

    if getattr(CachingLM, '_ulp_batched_writes', False):
        return

    class CheckpointDict(SqliteDict):
        def __init__(self, filename):
            self.pending_writes = 0
            super().__init__(filename, autocommit=False)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self.pending_writes += 1
            if self.pending_writes >= 32:
                self.commit()

        def commit(self, blocking=True):
            super().commit(blocking=blocking)
            if blocking:
                self.pending_writes = 0

    original = CachingLM.__init__

    def initialize(self, lm, cache_db):
        original(self, lm, cache_db)
        self.dbdict.close()
        self.dbdict = CheckpointDict(cache_db)
        lm.set_cache_hook(self.get_cache_hook())

    CachingLM.__init__ = initialize
    CachingLM._ulp_batched_writes = True


def cache_prefix(identity_file):
    identity_file = Path(identity_file).resolve()
    project = Path('/proj/rcs-hdd/model-tampering/zh2701/unlearning-400')
    if not identity_file.is_relative_to(project) or identity_file.stat().st_uid != os.getuid():
        raise ValueError('Response-cache identity must belong to this experiment')
    raw = identity_file.read_bytes()
    identity = json.loads(raw)
    assert identity['model_files'] and identity['config_sha256'] and identity['data_readiness_sha256']
    for path, expected in identity['model_files'].items():
        p = Path(path).resolve()
        assert p.stat().st_uid == os.getuid()
        st = p.stat()
        assert [st.st_size, st.st_mtime_ns] == expected
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = identity_file.parent / 'response-cache' / key
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {'proof_kind': 'evaluation_response_cache_identity', 'identity_sha256': key,
        'prepared_evaluation': str(identity_file), 'config_sha256': identity['config_sha256'],
        'model_files': identity['model_files'], 'cache_prefix': str(folder / 'responses'),
        'official_backend': 'lm_eval.simple_evaluate(use_cache=...)', 'rank_count': 1}
    file = identity_file.parent / 'RESPONSE_CACHE_IDENTITY.json'
    if file.exists():
        assert json.loads(file.read_text()) == receipt
    else:
        file.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt['cache_prefix']


def install(identity_file):
    """Supply the official optional argument; model requests/scores stay intact."""
    prefix = cache_prefix(identity_file)
    install_batched_writes()
    from evals.lm_eval import LMEvalEvaluator
    original = LMEvalEvaluator.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        configured = self.simple_evaluate_args.get('use_cache')
        if configured is not None and configured != prefix:
            raise ValueError('An evaluation already specifies a different response cache')
        self.simple_evaluate_args['use_cache'] = prefix

    LMEvalEvaluator.__init__ = initialize
    print('OFFICIAL_RESPONSE_CACHE ' + prefix + ' commit_every=32', flush=True)


def install_if_ready(identity_file):
    identity = json.loads(Path(identity_file).read_text())
    if '_performance_' in identity.get('job', ''):
        return  # Throughput measurements must compute every response afresh.
    receipt = Path('/proj/rcs-hdd/model-tampering/zh2701/unlearning-400/RESPONSE_CACHE_SOFTWARE_CHECK.json')
    if not receipt.exists():
        return
    checked = json.loads(receipt.read_text())
    if not checked.get('passed'):
        return
    support = Path(__file__).resolve().parents[1]
    for name, expected in checked['support_code_sha256'].items():
        if hashlib.sha256((support / name).read_bytes()).hexdigest() != expected:
            print('RESPONSE_CACHE_DISABLED_UNVALIDATED_CODE ' + name, flush=True)
            return
    install(identity_file)
