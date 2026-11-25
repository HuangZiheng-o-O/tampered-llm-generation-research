"""Resolve an explicit, read-only reuse of an identical owned MMLU evaluation."""
import hashlib
import json
import os
from pathlib import Path


def resolve(root, manifest, name):
    job = manifest['jobs'][name]
    alias = job.get('evaluation_alias')
    if not alias:
        return None
    if job['kind'] != 'eval' or not job['assessment'].startswith('mmlu_'):
        raise ValueError('Only identical general-ability evaluations can be reused')
    original = Path(alias['suite']).resolve()
    project = Path('/proj/rcs-hdd/model-tampering/zh2701/unlearning-400')
    if original == root.resolve() or not original.is_relative_to(project) or original.stat().st_uid != os.getuid():
        raise ValueError('Evaluation reuse must reference another owned suite in this project')
    source = json.loads((original / 'configs/jobs.json').read_text())
    prior = source['jobs'][name]
    if prior.get('evaluation_alias') or prior['assessment'] != job['assessment'] or prior['model'] != job['model']:
        raise ValueError('The reused model, assessment or source changed')
    expected = alias['source_config_sha256']
    digests = [source['config_sha256'][name + '.yaml'], manifest['config_sha256'][name + '.yaml'],
        hashlib.sha256((original / 'configs' / (name + '.yaml')).read_bytes()).hexdigest(),
        hashlib.sha256((root / 'configs' / (name + '.yaml')).read_bytes()).hexdigest()]
    if any(value != expected for value in digests) or Path(job['output']) != Path(prior['output']):
        raise ValueError('The reused evaluation configuration or output changed')
    return original, source
