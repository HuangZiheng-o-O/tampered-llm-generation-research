"""One explicit longer-learning retry, with all original artifacts preserved."""
import fcntl
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

from official_knowledge.run_pilot import now, sha


def prepare_longer_article_learning(pilot):
    if pilot.source != 'squad':
        return False
    root = pilot.root
    lock = (root / 'configs/recipe-preparation.lock').open('a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / 'configs/jobs.json'
        manifest = json.loads(manifest_path.read_text())
        name = 'squad_learn'
        cfg = OmegaConf.load(root / 'configs' / (name + '.yaml'))
        if int(cfg.trainer.args.num_train_epochs) != 5:
            return False
        # No unlearning model may already depend on the parent being changed.
        for n, job in manifest['jobs'].items():
            if n.startswith('squad_') and job['kind'] == 'train' and n != name:
                if (Path(job['output']) / 'prepared-job.json').exists():
                    raise ValueError('An article unlearning task already uses the old parent')
        archive = root / 'receipts/article-learning-5ep-20261106'
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archive / 'jobs.json')
        gate = root / 'gates/squad-learning.json'
        if gate.exists():
            shutil.copy2(gate, archive / gate.name)
        old_parent = Path(manifest['jobs'][name]['output'])
        new_parent = root / 'runs/squad_learn_20epochs'
        if new_parent.exists():
            raise ValueError('The longer-learning output already exists without its prepared configuration')
        for n, job in manifest['jobs'].items():
            if not n.startswith('squad_') or n.startswith('squad_base_'):
                continue
            file = root / 'configs' / (n + '.yaml')
            shutil.copy2(file, archive / file.name)
            text = file.read_text().replace(str(old_parent), str(new_parent))
            c = OmegaConf.create(text)
            if n == name:
                job['output'] = str(new_parent)
                c.task_name = name
                c.trainer.args.num_train_epochs = 20
                c.trainer.args.save_strategy = 'no'
                c.trainer.args.save_only_model = True
                c.checkpoint_first_step = False
            elif job.get('stage') in ('learn', 'learn_adapter'):
                # Original evaluations stay in place as historical evidence.
                output = root / 'evals' / (n + '_20epochs')
                job['output'] = str(output)
                c.paths.output_dir = c.eval.lm_eval.output_dir = c.hydra.run.dir = str(output)
            if 'model' in job:
                job['model'] = job['model'].replace(str(old_parent), str(new_parent))
            OmegaConf.save(c, file)
            manifest['config_sha256'][file.name] = sha(file)
        pending = manifest_path.with_suffix('.learning-retry.pending.json')
        pending.write_text(json.dumps(manifest, indent=2) + '\n')
        pending.replace(manifest_path)
        receipt = {'at_utc': now(), 'source': 'squad', 'old_parent': str(old_parent),
            'new_parent': str(new_parent), 'old_epochs': 5, 'new_epochs': 20,
            'data_changed': False, 'acceptance_thresholds_changed': False,
            'old_artifacts_preserved': True, 'reason': 'article knowledge remained below learning gate',
            'next_failure_requires_research': True}
        (root / 'ARTICLE_LEARNING_RETRY.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return True
