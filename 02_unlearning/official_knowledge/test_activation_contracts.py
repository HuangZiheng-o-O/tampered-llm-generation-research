"""CPU fixtures for the existing extraction environment; no model is loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from official_knowledge import activation_entry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    detector = Path('/local/rcs/zh2701/LLM-safety/code/LLM_ULP-BackdoorID-AcT')
    sys.path.insert(0, str(detector))
    import torch
    from transformers import AutoConfig
    import tamperid.extract as extractor
    from tamperid.features import load_features
    root = Path(tempfile.mkdtemp(prefix='activation-software-contract-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    (root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No model loaded or trained.\n')
    parent, adapter = root / 'parent', root / 'adapter'
    parent.mkdir()
    adapter.mkdir()
    write(parent / 'config.json', {'model_type': 'llama', 'rope_parameters': {'rope_type': 'default', 'rope_theta': 500000.0}})
    write(adapter / 'adapter_config.json', {'base_model_name_or_path': str(parent)})
    (adapter / 'adapter_model.safetensors').write_bytes(b'not-a-model-software-fixture')
    view, bridge = activation_entry.compatible_parent(root, parent)
    assert AutoConfig.from_pretrained(view, local_files_only=True).rope_theta == 500000.0
    features = {f'model.layers.{i}': torch.randn(64, 1, 4096) for i in (0, 15, 31)}
    spec = {'asset_root': str(root), 'adapter': str(adapter), 'parent': str(parent),
            'extractor_code': str(detector), 'extractor_sha256': hashlib.sha256((detector / 'tamperid/extract.py').read_bytes()).hexdigest(),
            'record_id': 'software-control', 'base_model_id': 'software/llama', 'protocol': 't32-standard-chatfix',
            'output': str(root / 'features'), 'unlearning_label': 0}
    spec_path = root / 'spec.json'
    write(spec_path, spec)
    with patch.object(sys, 'argv', ['activation_entry', '--spec', str(spec_path)]), \
         patch('official_knowledge.runtime_entry.apply_limits'), \
         patch.object(extractor, 'extract_one', return_value=(features, {'name': 'software_fixture'}, {})):
        activation_entry.main()
        activation_entry.main()
        _, receipt = load_features(root / 'features', spec['record_id'])
        assert receipt['source']['labels']['unlearning'] == 0
        assert receipt['source']['configuration_compatibility']['rope_theta'] == 500000.0
        write(adapter / 'adapter_config.json', {'base_model_name_or_path': str(root / 'wrong-parent')})
        try:
            activation_entry.main()
        except ValueError as error:
            assert 'parent' in str(error)
        else:
            raise AssertionError('Wrong parent accepted')
        write(adapter / 'adapter_config.json', {'base_model_name_or_path': str(parent)})
    spec['output'] = str(root / 'repeated-features')
    write(spec_path, spec)
    repeated = {key: torch.ones_like(value) for key, value in features.items()}
    with patch.object(sys, 'argv', ['activation_entry', '--spec', str(spec_path)]), \
         patch('official_knowledge.runtime_entry.apply_limits'), \
         patch.object(extractor, 'extract_one', return_value=(repeated, {'name': 'software_fixture'}, {})):
        try:
            activation_entry.main()
        except ValueError as error:
            assert 'distinct' in str(error)
        else:
            raise AssertionError('Repeated content activities accepted')
    result = {'passed': True, 'proof_kind': 'software_fixture_only', 'language_model_loaded': False,
              'optimizer_steps': 0, 'fixture_directory': str(root), 'checks': ['save_and_reload_features',
              'normal_control_label_preserved', 'correct_parent_identity_required', 'repeated_content_rejected',
              'llama3_rope_parameter_bridge_preserves_500000']}
    write(args.root / 'ACTIVATION_SOFTWARE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
