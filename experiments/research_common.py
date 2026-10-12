"""Isolated, checksummed research transactions; never use publication data/main."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import uuid

import numba
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='2d89df0569c05846b7b93f3925cf495bb80162a2'


def canonical_json(value: object) -> str:
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def research_root(path: str | Path) -> Path:
    root=Path(path).resolve()
    if root.name != 'nucleation' or any(p.name=='main' and p.parent.name=='data' for p in (root,*root.parents)):
        raise ValueError('research root must end in nucleation and cannot be inside data/main')
    root.mkdir(parents=True,exist_ok=True)
    return root


def write_json(path: Path, value: object) -> None:
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w') as stream:
        stream.write(canonical_json(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def write_npz(path: Path, arrays: dict, metadata: dict) -> None:
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays,metadata_json=np.array(canonical_json(metadata)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def provenance(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(['git',*args],cwd=PROJECT_ROOT,text=True).strip()
    sources=sorted([*PROJECT_ROOT.glob('code/*.py'),*PROJECT_ROOT.glob('experiments/*.py'),
                    *PROJECT_ROOT.glob('configs/nucleation/*.json'),*PROJECT_ROOT.glob('tests/*.py')])
    fingerprints={str(p.relative_to(PROJECT_ROOT)):file_hash(p) for p in sources}
    digest=config_hash(fingerprints)
    source_dir=root/'_sources'
    source_dir.mkdir(exist_ok=True)
    archive=source_dir/f'{digest}.tar.gz'
    if not archive.exists():
        temporary=archive.with_suffix('.tmp-'+uuid.uuid4().hex)
        with tarfile.open(temporary,'w:gz') as output:
            for source in sources:
                output.add(source,arcname=str(source.relative_to(PROJECT_ROOT)))
        os.replace(temporary,archive)
    return dict(base_commit=BASE_COMMIT,run_commit=git('rev-parse','HEAD'),
                working_tree_dirty=bool(git('status','--porcelain')),
                source_hash=digest,source_files=fingerprints,source_archive=str(archive),
                python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,
                rng_backend='Numba np.random; one seeded uninterrupted job',
                restart_policy='replay from initial state and seed; no RNG checkpoint')


def verify_complete(directory: Path) -> dict:
    marker=directory/'COMPLETE'
    manifest=json.loads(marker.read_text())
    required={'config.json','summary.json','trajectory.npz','events.npz','provenance.json',
              'states/initial.npz','states/final.npz','clusters.npz','lineage.npz'}
    if not required.issubset(manifest):
        raise ValueError(f'incomplete manifest: {directory}')
    for name,digest in manifest.items():
        path=directory/name
        if not path.is_file() or file_hash(path)!=digest:
            raise ValueError(f'completed task checksum mismatch: {path}')
    return json.loads((directory/'summary.json').read_text())


def start_run(root: Path, config: dict, seed: int) -> tuple[Path,Path]:
    experiment=config['experiment_id']
    if experiment not in ('N00','N01','N02','N03','N06'):
        raise ValueError('research runner only supports N00/N01/N02/N03/N06')
    parent=research_root(root)/experiment/config_hash(config)
    parent.mkdir(parents=True,exist_ok=True)
    final=parent/f'seed-{seed:010d}'
    if final.exists():
        raise FileExistsError(f'task already exists: {final}')
    staging=parent/f'.{final.name}.partial-{uuid.uuid4().hex}'
    staging.mkdir()
    (staging/'states').mkdir()
    write_json(staging/'config.json',config)
    return staging,final


def finish_run(staging: Path, final: Path) -> None:
    manifest={str(p.relative_to(staging)):file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    write_json(staging/'COMPLETE',manifest)
    verify_complete(staging)
    if final.exists():
        raise FileExistsError(f'refusing to overwrite completed task: {final}')
    staging.rename(final)
