"""Validate one longer-learning retry without loading models or running updates."""
import argparse
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from omegaconf import OmegaConf

from official_knowledge.learning_retry import prepare_longer_article_learning
from official_knowledge.production_jobs import ProductionJobs
from official_knowledge.run_pilot import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    original = args.root / 'official-lora-pilot-v1-llama2-chat'
    root = Path(tempfile.mkdtemp(prefix='article-retry-software-', dir='/local/rcs/zh2701/LLM-safety/tmp'))
    (root / 'PROOF_KIND.txt').write_text('SOFTWARE FIXTURE ONLY. No models loaded or trained.\n')
    (root / 'configs').mkdir()
    with (original / 'configs/recipe-preparation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        manifest = json.loads((original / 'configs/jobs.json').read_text().replace(str(original), str(root)))
        for name, job in manifest['jobs'].items():
            p = root / 'configs' / (name + '.yaml')
            p.write_text((original / 'configs' / p.name).read_text().replace(str(original), str(root)))
            manifest['config_sha256'][p.name] = sha(p)
    # The fixture always starts at the original five-epoch budget.
    learn_file = root / 'configs/squad_learn.yaml'
    learn = OmegaConf.load(learn_file)
    learn.trainer.args.num_train_epochs = 5
    old = root / 'runs/squad_learn_fixture_old'
    old.mkdir(parents=True)
    (old / 'historical.txt').write_text('keep this historical artifact')
    old_declared = manifest['jobs']['squad_learn']['output']
    for p in (root / 'configs').glob('*.yaml'):
        p.write_text(p.read_text().replace(old_declared, str(old)))
    learn = OmegaConf.load(learn_file)
    learn.trainer.args.num_train_epochs = 5
    OmegaConf.save(learn, learn_file)
    for n, job in manifest['jobs'].items():
        for key in ('output', 'model'):
            if key in job:
                job[key] = job[key].replace(old_declared, str(old))
        manifest['config_sha256'][n + '.yaml'] = sha(root / 'configs' / (n + '.yaml'))
    (root / 'configs/jobs.json').write_text(json.dumps(manifest))
    bio_hash = sha(root / 'configs/biographies_learn.yaml')
    base_hash = sha(root / 'configs/squad_base_knowledge.yaml')
    pilot = SimpleNamespace(root=root, source='squad')
    assert prepare_longer_article_learning(pilot)
    updated = json.loads((root / 'configs/jobs.json').read_text())
    cfg = OmegaConf.load(learn_file)
    assert cfg.trainer.args.num_train_epochs == 20 and cfg.trainer.args.save_strategy == 'no'
    assert Path(cfg.trainer.args.output_dir) == Path(updated['jobs']['squad_learn']['output'])
    assert (old / 'historical.txt').read_text() == 'keep this historical artifact'
    assert sha(root / 'configs/biographies_learn.yaml') == bio_hash
    assert sha(root / 'configs/squad_base_knowledge.yaml') == base_hash
    for name, job in updated['jobs'].items():
        if name.startswith('squad_') and job.get('stage') in ('learn', 'learn_adapter'):
            c = OmegaConf.load(root / 'configs' / (name + '.yaml'))
            assert str(c.paths.output_dir) == job['output'] and job['output'].endswith('_20epochs')
    assert not prepare_longer_article_learning(pilot)
    builder = object.__new__(ProductionJobs)
    builder.root, builder.pilot, builder.source = root / 'formal-cell', root, 'squad'
    builder.configs, builder.tasks = builder.root / 'configs', builder.root / 'tasks'
    builder.configs.mkdir(parents=True)
    builder.tasks.mkdir()
    formal_file = builder.configs / 'squad_learn.yaml'
    formal_cfg = OmegaConf.load(learn_file)
    formal_cfg.trainer.args.num_train_epochs = 5
    OmegaConf.save(formal_cfg, formal_file)
    output = builder.root / 'runs/learn'
    builder.manifest = {'jobs': {'squad_learn': {'output': str(output)}},
                        'config_sha256': {formal_file.name: sha(formal_file)}}
    builder.refresh_learning_recipe()
    assert OmegaConf.load(formal_file).trainer.args.num_train_epochs == 20
    output.mkdir(parents=True)
    (output / 'prepared-job.json').write_text('{}')
    formal_cfg = OmegaConf.load(formal_file)
    formal_cfg.trainer.args.num_train_epochs = 5
    OmegaConf.save(formal_cfg, formal_file)
    try:
        builder.refresh_learning_recipe()
    except ValueError:
        pass
    else:
        raise AssertionError('A started formal parent adopted a changed learning recipe')
    result = {'passed': True, 'proof_kind': 'software_fixture_only', 'optimizer_steps': 0,
        'fixture_directory': str(root), 'checks': ['five_to_twenty_epochs_with_final_only_save',
        'old_outputs_preserved', 'biography_and_original_model_evaluations_unchanged',
        'new_evaluations_use_fresh_directories', 'one_automatic_retry_only',
        'unstarted_formal_parent_adopts_qualified_learning_budget', 'started_parent_recipe_locked']}
    (args.root / 'LEARNING_RETRY_SOFTWARE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
