#!/usr/bin/env python3
"""Independently check the recorded study using only the Python standard library."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / 'artifacts' / 'Knee_OA_study_outputs.zip'
WEIGHTS = {'checkpoints/cnn_full.pt', 'checkpoints/cnn_20.pt',
           'checkpoints/lora_20/adapter_model.safetensors'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(rows):
    """Compute a five-class confusion matrix independently of notebook utilities."""
    cm = [[0 for _ in range(5)] for _ in range(5)]
    for row in rows:
        cm[int(row['y'])][int(row['pred'])] += 1
    n = len(rows)
    actual = [sum(r) for r in cm]
    predicted = [sum(cm[i][j] for i in range(5)) for j in range(5)]
    observed_disagreement = sum((i-j)**2 * cm[i][j] for i in range(5) for j in range(5))
    expected_disagreement = sum((i-j)**2 * actual[i] * predicted[j] / n
                                for i in range(5) for j in range(5))
    return {
        'accuracy': sum(cm[k][k] for k in range(5)) / n,
        'QWK': 1-observed_disagreement/expected_disagreement if expected_disagreement else math.nan,
        'macro_F1': statistics.mean(2*cm[k][k]/(actual[k]+predicted[k])
                                    if actual[k]+predicted[k] else 0 for k in range(5)),
        'balanced_accuracy': statistics.mean(cm[k][k]/actual[k] for k in range(5) if actual[k]),
        'MAE': sum(abs(i-j)*cm[i][j] for i in range(5) for j in range(5))/n,
        'large_error': sum(cm[i][j] for i in range(5) for j in range(5) if abs(i-j)>=2)/n,
        'KL1_recall': cm[1][1]/actual[1] if actual[1] else math.nan,
    }


def normalized(scores, temperature=1):
    maximum = max(scores)
    values = [math.exp((x-maximum)/temperature) for x in scores]
    return [x/sum(values) for x in values]


def verify(archive=DEFAULT_ARCHIVE, require_weights=False):
    archive = Path(archive).resolve()
    provenance = json.loads((REPO/'docs'/'SOURCE_PROVENANCE.json').read_text())
    if archive == DEFAULT_ARCHIVE.resolve():
        require(hashlib.sha256(archive.read_bytes()).hexdigest() == provenance['result_archive_sha256'],
                'The immutable result archive SHA-256 changed.')
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'ZIP CRC error.')
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP paths.')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                    'Unsafe archive path: '+name)
        checksums = json.loads(z.read('OUTPUT_CHECKSUMS.json'))
        for name, expected in checksums.items():
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, 'Output hash mismatch: '+name)
        frozen = json.loads(z.read('protocol/frozen_protocol.json'))
        missing = set(frozen['files']) - set(names)
        require(missing <= WEIGHTS, 'Unexpected missing frozen files: '+str(missing))
        require(not require_weights or not missing, 'Full backup is missing trained weights.')
        for name, expected in frozen['files'].items():
            if name not in missing:
                require(hashlib.sha256(z.read(name)).hexdigest() == expected, 'Frozen hash mismatch: '+name)
        contract = json.loads(z.read('protocol/run_contract.json'))
        require(contract['hash'] == frozen['run_contract'] == provenance['recorded_run_contract'],
                'Recorded run contracts disagree.')
        csv_rows = lambda name: list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8'))))
        json_rows = lambda name: [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]
        manifest = csv_rows('audit/manifest.csv')
        test = {r['case_id']: (r['group_id'], int(r['y'])) for r in manifest if r['split']=='test'}
        require(len(test)==1656, 'Unexpected test cohort size.')
        require([sum(y==k for _, y in test.values()) for k in range(5)] == [639,296,447,223,51],
                'Unexpected reference grade counts.')
        recorded_tables = {}
        for table in [5,7,8,9]:
            for row in csv_rows(f'tables/Table_{table:02d}.csv'):
                recorded_tables.setdefault(row['condition'], []).append(row)
        temperatures = json.loads(z.read('protocol/temperatures.json'))
        summary = {}
        for name in frozen['available']:
            rows = json_rows(f'predictions/{name}__test__original.jsonl')
            require(len(rows)==len(test) and len({r['case_id'] for r in rows})==len(rows),
                    'Incomplete or duplicated test predictions: '+name)
            require({r['case_id'] for r in rows}==set(test), 'Cohort mismatch: '+name)
            scored = {r['case_id']: r for r in csv_rows(f'predictions/{name}__test_scored.csv')}
            require(set(scored)==set(test), 'Scored CSV cohort mismatch: '+name)
            for row in rows:
                require(row['run_contract']==contract['hash'], 'Prediction contract mismatch.')
                require(row['status']=='ok', 'Completed snapshot contains a failed test row.')
                require((row['group_id'],int(row['y']))==test[row['case_id']], 'Reference mismatch.')
                scores=[row[f'score_{k}'] for k in range(5)]
                require(all(isinstance(s,(int,float)) and math.isfinite(s) for s in scores), 'Invalid scores.')
                require(row['pred']==max(range(5),key=lambda k:scores[k]), 'Argmax mismatch.')
                for k, p in enumerate(normalized(scores)):
                    require(math.isclose(p,row[f'raw_p_{k}'],abs_tol=1e-10), 'Raw probability mismatch.')
                cal = normalized(scores, temperatures[name]['T'])
                for k, p in enumerate(cal):
                    key=f'calibrated_p_{k}'
                    require(key in scored[row['case_id']], 'Missing calibrated probability column: '+key)
                    require(math.isclose(p,float(scored[row['case_id']][key]),abs_tol=1e-10),
                            'Calibrated probability mismatch.')
            got = metrics(rows)
            for table_row in recorded_tables[name]:
                for key, value in got.items():
                    if key in table_row:
                        require(math.isclose(value,float(table_row[key]),abs_tol=1e-10), f'{name}: {key} differs from table.')
            summary[name]=got
            timed=json_rows(f'logs/timing_{name}.jsonl')
            require(len(timed)==300 and len({r['case_id'] for r in timed})==100, 'Unexpected timing cohort.')
            require(len({(r['case_id'],r['repeat']) for r in timed})==300, 'Duplicate timed attempt.')
            require(all(r['status']=='ok' and r['run_contract']==contract['hash'] for r in timed), 'Timing status/contract mismatch.')
            timing_table=next(r for r in csv_rows('tables/Table_17.csv') if r['condition']==name)
            require(math.isclose(statistics.mean(r['total_s'] for r in timed),float(timing_table['mean_s']),abs_tol=1e-10),
                    'Timing mean does not match Table 17.')
        for i in range(1,15):
            require(f'figures/Figure_{i:02d}.png' in names and f'figures/Figure_{i:02d}.pdf' in names, 'Missing figure.')
        for i in range(5,19):
            require(f'tables/Table_{i:02d}.csv' in names, 'Missing table.')
        # Every browsable copy must match the original snapshot byte-for-byte.
        for path in (REPO/'results').rglob('*'):
            if path.is_file():
                rel=path.relative_to(REPO/'results').as_posix()
                require(path.read_bytes()==z.read(rel), 'Browsable output differs from snapshot: '+rel)
    notebook=json.loads((REPO/'notebooks'/'Knee_OA_Study.ipynb').read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']),filename=f'notebook-cell-{i}')
    for path in (REPO/'src').rglob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    return {'archive':archive.name,'output_hashes_verified':len(checksums),
            'frozen_files_verified':len(frozen['files'])-len(missing),
            'omitted_weight_files':sorted(missing),'test_cases_per_condition':len(test),
            'conditions':summary,'notebook_cells':len(notebook['cells'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    parser.add_argument('--require-weights',action='store_true',help='Require all frozen weight files in a private full backup.')
    parser.add_argument('--json',type=Path,help='Optional verification report path.')
    args=parser.parse_args()
    try:
        report=verify(args.archive,args.require_weights)
    except (ValueError,KeyError,zipfile.BadZipFile) as error:
        print('Verification FAILED:',error,file=sys.stderr)
        return 1
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Verified {report['output_hashes_verified']} output hashes, {report['frozen_files_verified']} protected files, "
          f"8 × {report['test_cases_per_condition']} test predictions, 2,400 timed attempts and notebook syntax.")
    for name, values in report['conditions'].items():
        print(f"{name:10s} accuracy={values['accuracy']:.6f} QWK={values['QWK']:.6f} macro_F1={values['macro_F1']:.6f}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
