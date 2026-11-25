"""Run the existing private BackdoorID extractor in a bounded worker process."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def compatible_parent(root, parent, tokenizer_source=None):
    """Bridge HF 5's RoPE serialization to the existing HF 4 extraction loader."""
    config_path = parent / 'config.json'
    config = json.loads(config_path.read_text())
    parameters = config.get('rope_parameters')
    if not parameters and tokenizer_source is None:
        return parent, {'configuration_bridge_required': False}
    if parameters and parameters.get('rope_type', 'default') != 'default':
        raise ValueError('This configuration bridge has only been validated for the selected default-RoPE bases')
    theta = parameters['rope_theta'] if parameters else config.get('rope_theta', 10000.)
    tokenizer_source = Path(tokenizer_source).resolve() if tokenizer_source else parent
    if not tokenizer_source.is_relative_to(root):
        raise ValueError('The original tokenizer escapes this project')
    token_names = {p.name for p in tokenizer_source.iterdir() if p.is_file() and
                   (p.name.startswith('tokenizer') or p.name in ('special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja'))}
    tokenizer_identity = {name: hashlib.sha256((tokenizer_source / name).read_bytes()).hexdigest() for name in sorted(token_names)}
    key = hashlib.sha256(json.dumps([str(parent), hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                    str(tokenizer_source), tokenizer_identity], sort_keys=True).encode()).hexdigest()[:24]
    folder = root / 'configuration-views'
    folder.mkdir(exist_ok=True)
    if folder.is_symlink() or folder.stat().st_uid != os.getuid():
        raise ValueError('Unexpected configuration-view directory ownership')
    with (folder / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        view = folder / key
        view.mkdir(exist_ok=True)
        if view.is_symlink() or view.stat().st_uid != os.getuid():
            raise ValueError('Unexpected configuration-view ownership')
        files = {p.name: p for p in parent.iterdir() if p.is_file()}
        files.update({name: tokenizer_source / name for name in token_names})
        for name, path in files.items():
            if name == 'config.json':
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError('A parent-model file escapes this project')
            target = view / path.name
            if not target.exists():
                target.symlink_to(path)
            elif target.resolve() != path.resolve():
                raise ValueError('Configuration view points to different weights or tokenizer files')
        config['rope_theta'] = theta
        pending = view / f'config.{os.getpid()}.pending.json'
        pending.write_text(json.dumps(config, indent=2) + '\n')
        pending.replace(view / 'config.json')
    return view, {'configuration_bridge_required': True, 'original_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  'extraction_config_sha256': hashlib.sha256((view / 'config.json').read_bytes()).hexdigest(),
                  'rope_theta': theta, 'original_parent': str(parent), 'weight_files_copied': False,
                  'tokenizer_source': str(tokenizer_source), 'tokenizer_sha256': tokenizer_identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    from official_knowledge.runtime_entry import apply_limits
    apply_limits()
    spec = json.loads(args.spec.read_text())
    root = Path(spec['asset_root']).resolve()
    adapter, parent = Path(spec['adapter']).resolve(), Path(spec['parent']).resolve()
    if not adapter.is_relative_to(root) or not parent.is_relative_to(root):
        raise ValueError('The extraction model escapes this project')
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if Path(config['base_model_name_or_path']).resolve() != parent:
        raise ValueError('The adapter does not belong to the selected learned parent')
    code = Path(spec['extractor_code'])
    actual = hashlib.sha256((code / 'tamperid/extract.py').read_bytes()).hexdigest()
    if actual != spec['extractor_sha256']:
        raise ValueError('Frozen private extraction code changed')
    sys.path.insert(0, str(code))
    import torch
    from tamperid.catalog import ModelRecord
    from tamperid.extract import extract_one
    from tamperid.features import feature_paths, load_features, save_features
    model_parent, configuration_bridge = compatible_parent(root, parent, spec.get('tokenizer_source'))
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'ACTIVATION_RESULT.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        features, receipt = load_features(Path(result['feature_directory']), spec['record_id'])
        assert receipt['source']['adapter_sha256'] == hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
        print(json.dumps(result), flush=True)
        return
    data_path, receipt_path = feature_paths(output, spec['record_id'])
    feature_dir = output
    if data_path.exists() or receipt_path.exists():
        feature_dir = output / ('retry-' + str(time.time_ns()))
    record = ModelRecord.from_dict({'record_id': spec['record_id'], 'model_id': 'local/' + spec['record_id'],
        'base_model': spec['base_model_id'], 'sample_path': str(adapter.relative_to(root)),
        'peft_adapter': True, 'source': 'OpenUnlearning-local', 'group': spec['record_id'],
        'split': spec.get('split', 'pilot_only'), 'labels': {'backdoor': None, 'unlearning': spec.get('unlearning_label', 1)}})
    features, protocol, diagnostics = extract_one(record, root, model_parent, 'cuda:0', 42,
                                                  probe_protocol=spec['protocol'])
    unique = {name: int(torch.unique(tensor.reshape(64, -1), dim=0).shape[0]) for name, tensor in features.items()}
    assert len(features) == 3 and all(tuple(t.shape) == (64, 1, 4096) and torch.isfinite(t).all() for t in features.values())
    if spec['protocol'] == 't32-standard-chatfix' and min(unique.values()) < 16:
        raise ValueError('Content probes produced fewer than 16 distinct activity records')
    source = {'base_model': spec['base_model_id'], 'learning_parent': str(parent), 'adapter_path': str(adapter),
              'adapter_sha256': hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
              'extractor_sha256': actual, 'unique_records_per_layer': unique,
              'labels': record.labels, 'configuration_compatibility': configuration_bridge, **diagnostics}
    receipt = save_features(feature_dir, spec['record_id'], features, source, protocol)
    result = {'passed': True, 'protocol': spec['protocol'], 'samples': 64,
              'feature_directory': str(feature_dir), 'activation_sha256': receipt['sha256'],
              'adapter_sha256': source['adapter_sha256'], 'unique_records_per_layer': unique,
              'input_template_repetition_observed': min(unique.values()) == 1}
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
