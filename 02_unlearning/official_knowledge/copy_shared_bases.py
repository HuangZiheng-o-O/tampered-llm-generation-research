"""Copy the two existing model snapshots to HDD and verify every copied file."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from official_knowledge.prepare_production_data import BASES
from official_knowledge.run_pilot import now


def copy_snapshot(source, destination):
    if (destination / 'HDD_COPY_VERIFIED.json').exists():
        return json.loads((destination / 'HDD_COPY_VERIFIED.json').read_text())
    if destination.exists():
        raise ValueError('An unverified destination exists: ' + str(destination))
    pending = destination.with_name('.' + destination.name + '-copying')
    pending.mkdir(exist_ok=True)
    if pending.stat().st_uid != os.getuid() or pending.is_symlink():
        raise ValueError('The copy staging directory is not owned by this account')
    record = {'source': str(source), 'files': {}, 'at_utc': now()}
    files = [p for p in source.iterdir() if p.is_file() and not p.name.startswith('.') and
             p.suffix in ('.safetensors', '.json', '.model', '.txt', '.jinja')]
    for src in sorted(files):
        target = pending / src.name
        if target.is_symlink():
            raise ValueError('Unexpected symlink in model copy staging')
        digest = hashlib.sha256()
        with src.open('rb') as reader, target.open('wb') as writer:
            while chunk := reader.read(8 * 1024**2):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        with target.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest.hexdigest()
        assert target.stat().st_size == src.stat().st_size
        record['files'][src.name] = {'bytes': target.stat().st_size, 'sha256': digest.hexdigest()}
        print(json.dumps({'copied': src.name, 'bytes': target.stat().st_size}), flush=True)
    assert 'config.json' in record['files'] and any(n.endswith('.safetensors') for n in record['files'])
    (pending / 'HDD_COPY_VERIFIED.json').write_text(json.dumps(record, indent=2) + '\n')
    pending.rename(destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root.resolve() / 'shared-bases'
    out.mkdir(exist_ok=True)
    with (out / 'copy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for base, (_, directory) in BASES.items():
            source = Path('/local/rcs/zh2701/LLM-safety/models') / directory
            results[base] = copy_snapshot(source, out / directory)
        (out / 'READY.json').write_text(json.dumps({'at_utc': now(), 'bases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
