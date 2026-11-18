#!/usr/bin/env python3
"""Create a private full study backup, including trained checkpoints, after verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_dir',type=Path,help='Directory directly containing protocol/, predictions/ and checkpoints/.')
    parser.add_argument('--output',type=Path,required=True,help='ZIP outside the study directory; do not commit this private backup.')
    args=parser.parse_args()
    root=args.study_dir.resolve()
    output=args.output.resolve()
    if output.exists():
        parser.error('Backup already exists. Choose a new filename.')
    if output==root or root in output.parents:
        parser.error('The output ZIP must be outside the study directory.')
    frozen=json.loads((root/'protocol'/'frozen_protocol.json').read_text())
    for name, expected in frozen['files'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or changed protected file: '+name)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    z.write(path,path.relative_to(root))
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError('Backup ZIP CRC verification failed.')
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Full backup saved: {output}')
    print('SHA-256:',hashlib.sha256(output.read_bytes()).hexdigest())
    print('Created UTC:',datetime.now(timezone.utc).isoformat())
    print('Store this full backup privately; the public analysis ZIP excludes trained weights.')


if __name__=='__main__':
    main()
