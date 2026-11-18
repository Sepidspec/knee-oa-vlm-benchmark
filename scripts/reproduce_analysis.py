#!/usr/bin/env python3
"""Regenerate Tables 5–18 and Figures 1–14 from recorded predictions on a CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

from verify_results import DEFAULT_ARCHIVE, REPO, verify
sys.path.insert(0,str(REPO/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    parser.add_argument('--output',type=Path,default=REPO/'reproduced')
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--rq',choices=['all','1','2','3','4','5','6','7'],default='all')
    args=parser.parse_args()
    if args.bootstrap<1:
        parser.error('--bootstrap must be positive.')
    verify(args.archive)
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty. Use a new directory to preserve prior results.')
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        # Path validation and checksums have already passed in verify().
        z.extractall(output)
    from knee_oa import analysis
    import numpy as np
    import pandas as pd
    analysis.initialize(output,args.bootstrap)
    started=time.perf_counter()
    selected=range(1,8) if args.rq=='all' else [int(args.rq)]
    for rq in selected:
        print(f'Regenerating RQ{rq} with {args.bootstrap} bootstrap resamples ...',flush=True)
        getattr(analysis,f'rq{rq}')()
    comparisons=[]
    if args.bootstrap==2000:
        table_ranges={1:[5,6],2:[7,8],3:[9,10],4:[11,12],5:[13,14],6:[15,16],7:[17,18]}
        for rq in selected:
            for number in table_ranges[rq]:
                filename=f'Table_{number:02d}.csv'
                reference=pd.read_csv(REPO/'results'/'tables'/filename)
                reproduced=pd.read_csv(output/'tables'/filename)
                if list(reference.columns)!=list(reproduced.columns) or reference.shape!=reproduced.shape:
                    raise AssertionError('Table structure mismatch: '+filename)
                for column in reference:
                    if pd.api.types.is_numeric_dtype(reference[column]):
                        np.testing.assert_allclose(reference[column].to_numpy(float),reproduced[column].to_numpy(float),
                                                   rtol=1e-8,atol=1e-10,equal_nan=True,err_msg=filename+': '+column)
                    else:
                        pd.testing.assert_series_equal(reference[column],reproduced[column],check_dtype=False)
                comparisons.append(filename)
        # Secondary contrasts carry the multiplicity corrections used in the thesis.
        contrast_files={1:['primary_estimand.json'],2:['RQ2_paired_contrasts.csv'],3:['RQ3_paired_contrasts.csv'],
                        4:['RQ4_calibration_contrasts.csv'],5:['RQ5_confidence_vs_random.csv'],6:['RQ6_perturbation_effects.csv'],7:[]}
        for rq in selected:
            for filename in contrast_files[rq]:
                if filename.endswith('.csv'):
                    a=pd.read_csv(REPO/'results'/'tables'/filename)
                    b=pd.read_csv(output/'tables'/filename)
                    pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-8,atol=1e-10)
                else:
                    a=json.loads((REPO/'results'/'tables'/filename).read_text())
                    b=json.loads((output/'tables'/filename).read_text())
                    for key in a:
                        if isinstance(a[key],(float,int)):
                            np.testing.assert_allclose(a[key],b[key],rtol=1e-8,atol=1e-10)
                        else:
                            assert a[key]==b[key],key
                comparisons.append(filename)
    report={'bootstrap':args.bootstrap,'research_questions':list(selected),'seconds':time.perf_counter()-started,
            'matched_recorded_outputs':comparisons,'numerical_comparison':'tolerance rtol=1e-8, atol=1e-10; figure bytes not compared',
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib']}}
    (output/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Completed. Outputs: {output}',flush=True)
    print(f'{len(comparisons)} recorded tables/contrasts matched.' if comparisons else
          'Nonstandard bootstrap count: these outputs are a smoke check, not the recorded intervals.',flush=True)


if __name__=='__main__':
    main()
