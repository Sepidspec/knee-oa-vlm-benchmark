"""CPU analysis functions extracted from the publication notebook.

The mathematical function bodies are shared with the full Kaggle workflow.
Use scripts/reproduce_analysis.py to initialize an extracted result snapshot.
"""
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=42
COLORS=['#22577a','#d76a36','#50805f','#8e649e','#ae862d','#5697a1','#bc6388']
LABELS={'cnn_full':'CNN full','cnn_20':'CNN 20%','vlm_p0':'Frozen P0','vlm_p1':'Rubric P1',
        'vlm_s1':'Few shot S1','vlm_s2':'Few shot S2','vlm_s3':'Few shot S3','lora_20':'LoRA 20%'}
PERTURBATIONS=['blur_0.5','blur_1.0','noise_0.01','noise_0.03','jpeg_80','jpeg_60']
CAPTIONS={}

def display(*items):
    """Notebook display is suppressed in the command-line analysis."""

def save_table(number, frame, title):
    frame.to_csv(ROOT/'tables'/f'Table_{number:02d}.csv', index=False)
    return frame

def save_figure(number, figure, caption):
    figure.tight_layout()
    figure.savefig(ROOT/'figures'/f'Figure_{number:02d}.pdf')
    figure.savefig(ROOT/'figures'/f'Figure_{number:02d}.png', dpi=160)
    CAPTIONS[str(number)]=caption
    dump_json(ROOT/'figures'/'captions.json',CAPTIONS)
    plt.close(figure)

def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def digest(x):
    return hashlib.sha256(json.dumps(clean_json(x),sort_keys=True).encode()).hexdigest()

def dump_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean_json(obj),indent=2,sort_keys=True),encoding='utf-8')
    tmp.replace(path)

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def metric_values(y,pred):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    if len(y)==0:
        return dict(QWK=np.nan,macro_F1=np.nan,balanced_accuracy=np.nan,
                    accuracy=np.nan,MAE=np.nan,large_error=np.nan)
    cm=np.bincount(5*y+pred,minlength=25).reshape(5,5).astype(float)
    actual=cm.sum(1); guessed=cm.sum(0); diag=np.diag(cm)
    recall=np.divide(diag,actual,out=np.zeros(5),where=actual>0)
    f1=np.divide(2*diag,actual+guessed,out=np.zeros(5),where=actual+guessed>0)
    w=(np.arange(5)[:,None]-np.arange(5)[None,:])**2/16
    expected=np.outer(actual,guessed)/len(y)
    den=(w*expected).sum()
    return dict(QWK=1-(w*cm).sum()/den if den>0 else np.nan,
                macro_F1=f1.mean(),balanced_accuracy=recall[actual>0].mean(),
                accuracy=float((y==pred).mean()),MAE=float(abs(y-pred).mean()),
                large_error=float((abs(y-pred)>=2).mean()))

def valid_rows(df):
    return df.loc[df.status.eq('ok')].copy()

def logits_array(df): return df[[f'score_{j}' for j in range(5)]].to_numpy(float)

def probabilities(df,T=1.): return softmax(log_softmax(logits_array(df),axis=1)/T,axis=1)

def calibration_metrics(y,p):
    y=np.asarray(y,int)
    if len(y)==0: return dict(NLL=np.nan,Brier=np.nan,ECE=np.nan)
    nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()
    brier=((p-np.eye(5)[y])**2).sum(1).mean()
    bins=reliability(y,p,np.linspace(0,1,16))
    ece=(bins.n/len(y)*(bins.confidence-bins.correctness).abs()).sum()
    return dict(NLL=nll,Brier=brier,ECE=ece)

def reliability(y,p,edges):
    conf=p.max(1); correct=p.argmax(1)==np.asarray(y)
    idx=np.clip(np.searchsorted(edges,conf,side='right')-1,0,len(edges)-2)
    rows=[]
    for b in range(len(edges)-1):
        m=idx==b
        rows.append(dict(lower=edges[b],upper=edges[b+1],n=int(m.sum()),
                         confidence=float(conf[m].mean()) if m.any() else np.nan,
                         correctness=float(correct[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)

def fit_temperature(df):
    v=valid_rows(df)
    if not len(v): raise ValueError('No complete calibration predictions; cannot fit temperature.')
    y=v.y.to_numpy(int); z=log_softmax(logits_array(v),axis=1)
    fit=minimize_scalar(lambda t:-log_softmax(z/np.exp(t),axis=1)[np.arange(len(y)),y].mean(),
                        bounds=(-4,4),method='bounded')
    assert fit.success
    T=float(np.exp(fit.x)); p=softmax(z/T,axis=1)
    assert np.array_equal(p.argmax(1),z.argmax(1))
    return dict(T=T,n=len(v),eligible=len(df),at_bound=abs(fit.x)>3.95)

def cluster_indices(groups,B,seed=SEED):
    groups=np.asarray(groups); unique=np.unique(groups)
    if not len(unique): return
    buckets=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(seed)
    for _ in range(B):
        yield np.concatenate([buckets[k] for k in rng.integers(0,len(buckets),len(buckets))])

def interval(values):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    return (np.quantile(values,.025),np.quantile(values,.975),len(values)) if len(values) else (np.nan,np.nan,0)

def summary_row(name,df,B=None):
    v=valid_rows(df); out=dict(condition=name,eligible=len(df),valid=len(v),
          completion=len(v)/len(df) if len(df) else np.nan,
          all_case_accuracy=float((v.y.to_numpy()==v.pred.to_numpy()).sum()/len(df)) if len(df) else np.nan)
    out.update(metric_values(v.y,v.pred))
    if len(v):
        samples=defaultdict(list)
        for ix in cluster_indices(v.group_id,CFG['bootstrap'] if B is None else B):
            for k,val in metric_values(v.y.to_numpy()[ix],v.pred.to_numpy()[ix]).items(): samples[k].append(val)
        for k,vals in samples.items():
            lo,hi,n=interval(vals); out.update({k+'_lo':lo,k+'_hi':hi,k+'_bootstrap_valid':n})
    return out

def paired_effect(a,b,label,metric='QWK',B=None):
    # Positive = A minus B. Inner join AFTER failure filtering: common-complete estimand.
    d=valid_rows(a).merge(valid_rows(b),on='case_id',suffixes=('_a','_b'),validate='one_to_one')
    if not len(d): return dict(contrast=label,metric=metric,n=0,effect=np.nan,lo=np.nan,hi=np.nan,p=np.nan)
    assert np.array_equal(d.y_a,d.y_b) and np.array_equal(d.group_id_a,d.group_id_b)
    y=d.y_a.to_numpy(); pa=d.pred_a.to_numpy(); pb=d.pred_b.to_numpy()
    est=metric_values(y,pa)[metric]-metric_values(y,pb)[metric]
    samples=[metric_values(y[ix],pa[ix])[metric]-metric_values(y[ix],pb[ix])[metric]
             for ix in cluster_indices(d.group_id_a,CFG['bootstrap'] if B is None else B)]
    lo,hi,n=interval(samples); vals=np.asarray(samples); vals=vals[np.isfinite(vals)]
    # Approximate two-sided centred bootstrap test. Declared secondary families use Holm.
    p=(1+int((abs(vals-est)>=abs(est)).sum()))/(len(vals)+1) if len(vals) and np.isfinite(est) else np.nan
    return dict(contrast=label,metric=metric,n=len(d),excluded_a=len(a)-len(d),excluded_b=len(b)-len(d),
                effect=est,lo=lo,hi=hi,p=p,bootstrap_valid=n)

def holm(df):
    df=df.copy(); df['p_holm']=np.nan
    idx=df.index[df.p.notna()].tolist(); idx=sorted(idx,key=lambda i:df.loc[i,'p'])
    running=0
    for k,i in enumerate(idx):
        running=max(running,min(1,(len(idx)-k)*df.loc[i,'p'])); df.loc[i,'p_holm']=running
    return df

def prediction_path(name,split,variant='original'):
    return ROOT/'predictions'/f'{name}__{split}__{variant}.jsonl'

def read_records(path):
    if not path.exists(): return pd.DataFrame()
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        # A torn last line after a killed runtime is not silently discarded.
        rows.append(json.loads(line))
    df=pd.DataFrame(rows)
    if len(df):
        assert df.case_id.is_unique, f'Duplicate records in {path}'
        assert df.run_contract.eq(RUN_CONTRACT).all(), 'Cache protocol mismatch'
    return df

def common_valid(names,results=None):
    results=RESULTS if results is None else results
    names=[n for n in names if n in results]
    if not names: return {}
    keys=set.intersection(*(set(valid_rows(results[n]).case_id) for n in names))
    return {n:valid_rows(results[n]).set_index('case_id').loc[sorted(keys)].reset_index() for n in names}

def recall_interval(df,grade):
    v=valid_rows(df); y=v.y.to_numpy(int); p=v.pred.to_numpy(int)
    if not len(v): return np.nan,np.nan,np.nan,0
    den=(y==grade).sum(); est=float((p[y==grade]==grade).mean()) if den else np.nan
    vals=[]
    for ix in cluster_indices(v.group_id,CFG['bootstrap']):
        m=y[ix]==grade
        vals.append(float((p[ix][m]==grade).mean()) if m.any() else np.nan)
    lo,hi,n=interval(vals); return est,lo,hi,n

def plot_metric_points(ax,df,metric,x=None):
    x=np.arange(len(df)) if x is None else np.asarray(x)
    for i,(_,r) in enumerate(df.iterrows()):
        y=r.get(metric,np.nan); lo=r.get(metric+'_lo',np.nan); hi=r.get(metric+'_hi',np.nan)
        if np.isfinite(y):
            ax.plot(x[i],y,'o',color=COLORS[i%len(COLORS)])
            if np.isfinite(lo) and np.isfinite(hi): ax.vlines(x[i],lo,hi,color=COLORS[i%len(COLORS)])
    ax.set_xticks(x,df.condition.map(lambda n:LABELS.get(n,n)),rotation=20,ha='right')
    ax.set_ylabel(metric.replace('_',' '))

def rq1():
    names=[n for n in ['cnn_full','vlm_p0'] if n in RESULTS]
    allrows=pd.DataFrame([summary_row(n,RESULTS[n]) for n in names])
    shared=common_valid(names)
    primary=paired_effect(RESULTS['vlm_p0'],RESULTS['cnn_full'],'Frozen P0 minus CNN full') if len(names)==2 else dict(status='primary contrast unavailable')
    dump_json(ROOT/'tables'/'primary_estimand.json',primary); display(pd.DataFrame([primary]))
    save_table(5,allrows,'Overall grading and completion on all eligible test cases; grade metrics use valid cases')
    fig,axes=plt.subplots(1,max(1,len(names)),figsize=(6*max(1,len(names)),4.6),squeeze=False)
    for ax,n in zip(axes[0],names):
        d=shared[n]; cm=np.bincount(5*d.y.to_numpy(int)+d.pred.to_numpy(int),minlength=25).reshape(5,5)
        ax.imshow(cm,cmap='Blues')
        for i in range(5):
            for j in range(5): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xlabel='Predicted KL grade',ylabel='Reference KL grade',title=f'{LABELS[n]} · common n={len(d)}')
    save_figure(1,fig,'Paired confusion matrices on the common-complete test cohort. Failures are reported separately in Table 5; rows are reference grades.')
    rows=[]
    for n,d in shared.items():
        for g in range(5):
            y=d.y.to_numpy(int); p=d.pred.to_numpy(int); actual=y==g; guessed=p==g
            rec,lo,hi,bn=recall_interval(d,g)
            precision=float((y[guessed]==g).mean()) if guessed.any() else 0.
            f1=2*precision*rec/(precision+rec) if precision+rec>0 else 0.
            rows.append(dict(condition=n,grade=g,support=int(actual.sum()),recall=rec,recall_lo=lo,recall_hi=hi,
                precision=precision,F1=f1,large_errors=int((abs(y[actual]-p[actual])>=2).sum()),bootstrap_valid=bn))
    t6=save_table(6,pd.DataFrame(rows),'Grade recognition and large ordinal errors on the paired common-complete cohort')
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for i,(n,d) in enumerate(shared.items()):
        error=abs(d.y.to_numpy()-d.pred.to_numpy()).astype(int)
        axes[0].bar(np.arange(5)+(i-.5)*.34,np.bincount(error,minlength=5)/len(d),.34,label=LABELS[n],color=COLORS[i])
        sub=t6[t6.condition.eq(n)]
        axes[1].plot(sub.grade,sub.recall,'o-',label=LABELS[n],color=COLORS[i])
        axes[1].vlines(sub.grade,sub.recall_lo,sub.recall_hi,color=COLORS[i],alpha=.7)
    axes[0].set(xlabel='Absolute KL error',ylabel='Proportion of paired valid cases',xticks=range(5)); axes[0].legend()
    axes[1].set(xlabel='Reference KL grade',ylabel='Recall',ylim=(0,1),xticks=range(5)); axes[1].legend()
    save_figure(2,fig,f'Ordinal error distances and grade-specific recall. Recall bars are 95% {GROUP_LABEL}-bootstrap intervals; rare-grade resamples without support are excluded and counted in Table 6.')

def rq2():
    names=[n for n in ['vlm_p0','vlm_p1','vlm_s1'] if n in RESULTS]
    t7=pd.DataFrame([summary_row(n,RESULTS[n]) for n in names]); t7['examples']=[5 if n=='vlm_s1' else 0 for n in names]
    save_table(7,t7,'Fixed prompt and main demonstration comparisons; all-case completion is explicit')
    paired=[]
    for a,b,label in [('vlm_p1','vlm_p0','Rubric minus minimal'),('vlm_s1','vlm_p1','Five examples minus rubric')]:
        if a in RESULTS and b in RESULTS: paired.append(paired_effect(RESULTS[a],RESULTS[b],label))
    if paired:
        p=holm(pd.DataFrame(paired)); p.to_csv(ROOT/'tables'/'RQ2_paired_contrasts.csv',index=False); display(p)
    shared=common_valid(names)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    plot_metric_points(axes[0],t7,'QWK')
    axes[0].plot(np.arange(len(t7)),t7.macro_F1,'v--',color='#646464',label='Macro-F1')
    axes[0].vlines(np.arange(len(t7)),t7.macro_F1_lo,t7.macro_F1_hi,color='#646464')
    axes[0].plot([],[],'o',color=COLORS[0],label='QWK')
    axes[0].set_ylabel('QWK / macro-F1'); axes[0].legend()
    for i,(n,d) in enumerate(shared.items()):
        rec=[recall_interval(d,g) for g in range(5)]
        axes[1].plot(range(5),[x[0] for x in rec],'o-',color=COLORS[i],label=LABELS[n])
        axes[1].vlines(np.arange(5)+i*.02,[x[1] for x in rec],[x[2] for x in rec],color=COLORS[i],alpha=.5)
    axes[1].set(xlabel='Reference KL grade',ylabel='Recall on shared complete cases',ylim=(0,1),xticks=range(5)); axes[1].legend()
    save_figure(3,fig,f'Prompt and demonstration comparisons. QWK uses each configuration’s complete cases; recall uses their common-complete cohort. Bars are 95% {GROUP_LABEL}-bootstrap intervals.')
    rows=[]
    for n in ['vlm_s1','vlm_s2','vlm_s3']:
        s=SUPPORTS[CONDITIONS[n]['support']]
        if n in RESULTS:
            r=summary_row(n,RESULTS[n])
        else:
            r=dict(condition=n,status='omitted at development feasibility gate',QWK=np.nan,QWK_lo=np.nan,QWK_hi=np.nan,
                   macro_F1=np.nan,macro_F1_lo=np.nan,macro_F1_hi=np.nan)
        r.update(support_set=CONDITIONS[n]['support'],support_images=len(s),support_groups=s.group_id.nunique()); rows.append(r)
    t8=save_table(8,pd.DataFrame(rows),'Prespecified support-set sensitivity; S1 remains the main set')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,metric in zip(axes,['QWK','macro_F1']):
        plot_metric_points(ax,t8,metric)
        if 'vlm_p1' in RESULTS:
            ref=metric_values(valid_rows(RESULTS['vlm_p1']).y,valid_rows(RESULTS['vlm_p1']).pred)[metric]
            ax.axhline(ref,color='gray',ls='--',label='Rubric, no examples'); ax.legend()
    save_figure(4,fig,'Three fixed training-only support sets evaluated on the same test targets. Intervals quantify case sampling uncertainty; differences between the three sets are descriptive and do not justify selecting the best test set.')

def read_training_resources():
    p=ROOT/'logs'/'training_resources.jsonl'
    return pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()]) if p.exists() else pd.DataFrame()

def rq3():
    rows=[]
    budgets={'cnn_full':1.,'cnn_20':SUBSET.group_id.nunique()/TRAIN.group_id.nunique(),'vlm_p0':0.,
             'lora_20':SUBSET.group_id.nunique()/TRAIN.group_id.nunique()}
    for n in budgets:
        if n in RESULTS:
            r=summary_row(n,RESULTS[n]); r['downstream_group_fraction']=budgets[n]; rows.append(r)
    if 'lora_20' not in RESULTS:
        rows.append(dict(condition='lora_20',status=ADAPTATION['status'],reason=ADAPTATION.get('reason','See adaptation gate log'),
                         downstream_group_fraction=budgets['lora_20']))
    t9=save_table(9,pd.DataFrame(rows),'Training-budget comparison; no optional full-data adapter is assumed')
    pairs=[]
    for b in ['vlm_p0','cnn_20']:
        if 'lora_20' in RESULTS and b in RESULTS:
            pairs.append(paired_effect(RESULTS['lora_20'],RESULTS[b],f'LoRA 20% minus {LABELS[b]}'))
    if pairs:
        effects=holm(pd.DataFrame(pairs)); effects.to_csv(ROOT/'tables'/'RQ3_paired_contrasts.csv',index=False); display(effects)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,metric in zip(axes,['QWK','macro_F1']):
        for i,r in enumerate(rows):
            if metric not in r: continue
            x=r['downstream_group_fraction']; y=r[metric]
            ax.plot(x,y,'o',color=COLORS[i],label=LABELS[r['condition']])
            ax.vlines(x,r.get(metric+'_lo',np.nan),r.get(metric+'_hi',np.nan),color=COLORS[i])
        ax.set(xlabel='Fraction of downstream training groups',ylabel=metric,xticks=[0,.2,1]); ax.legend(fontsize=8)
    save_figure(5,fig,'Measured downstream label budgets and performance, with grouped 95% intervals. Zero denotes no KL gradient fitting, not absence of foundation pretraining. No scaling law or full-data adapter result is inferred.')
    resources=read_training_resources()
    if not resources.empty:
        resources['grouping_unit']=GROUP_LABEL
        resources.to_csv(ROOT/'tables'/'training_resources_all_attempts.csv',index=False)
        t10=resources.copy()
    else: t10=pd.DataFrame([dict(status='No completed training-resource record')])
    if 'lora_20' not in RESULTS:
        t10=pd.concat([t10,pd.DataFrame([dict(condition='lora_20',status=ADAPTATION['status'],error=ADAPTATION.get('reason','gradient gate failed'))])],ignore_index=True)
    save_table(10,t10,'Measured selection and refit resources, including failed attempts; validation-label use is additional')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    if 'lora_20' in RESULTS and 'vlm_p0' in RESULTS:
        d=common_valid(['vlm_p0','lora_20']); a=d['vlm_p0']; b=d['lora_20']
        before=a.y.to_numpy()==a.pred.to_numpy(); after=b.y.to_numpy()==b.pred.to_numpy()
        counts=[int((before&after).sum()),int((~before&after).sum()),int((before&~after).sum()),int((~before&~after).sum())]
        axes[0].bar(['Both correct','Improved','Deteriorated','Both wrong'],counts,color=COLORS[:4])
        axes[0].tick_params(axis='x',rotation=20); axes[0].set_ylabel('Common-complete cases')
        pd.DataFrame(dict(category=['both correct','improved','deteriorated','both wrong'],n=counts)).to_csv(
            ROOT/'tables'/'RQ3_correctness_transitions.csv',index=False)
        for i,(n,s) in enumerate(common_valid(['vlm_p0','lora_20','cnn_20']).items()):
            rec=[recall_interval(s,g) for g in range(5)]
            axes[1].plot(range(5),[x[0] for x in rec],'o-',color=COLORS[i],label=LABELS[n])
            axes[1].vlines(range(5),[x[1] for x in rec],[x[2] for x in rec],color=COLORS[i],alpha=.5)
        axes[1].set(xlabel='Reference KL grade',ylabel='Recall',ylim=(0,1)); axes[1].legend()
    else:
        for ax in axes:
            ax.axis('off'); ax.text(.5,.5,'Adapter effect not estimated\n'+ADAPTATION['status'],ha='center',va='center',transform=ax.transAxes)
    save_figure(6,fig,'Paired correctness transitions and grade recall after adaptation, when feasible. Failed or omitted adaptation is stated explicitly and is not replaced by simulated performance.')

def calibration_interval(y,p,groups):
    samples=defaultdict(list)
    for ix in cluster_indices(groups,CFG['bootstrap']):
        for key,val in calibration_metrics(y[ix],p[ix]).items(): samples[key].append(val)
    result={}
    for key,vals in samples.items():
        lo,hi,n=interval(vals); result.update({key+'_lo':lo,key+'_hi':hi,key+'_bootstrap_valid':n})
    return result

def paired_scalar_interval(values,groups):
    if not len(values): return dict(effect=np.nan,lo=np.nan,hi=np.nan,p=np.nan,n=0,bootstrap_valid=0)
    values=np.asarray(values,float); est=float(values.mean())
    samples=np.array([values[ix].mean() for ix in cluster_indices(groups,CFG['bootstrap'])])
    lo,hi,bn=interval(samples)
    p=(1+int((abs(samples-est)>=abs(est)).sum()))/(len(samples)+1)
    return dict(effect=est,lo=lo,hi=hi,p=p,n=len(values),bootstrap_valid=bn)

def rq4():
    rows=[]; effects=[]; bin_records=[]
    for name,d in RESULTS.items():
        v=valid_rows(d); y=v.y.to_numpy(int); raw=probabilities(v); T=TEMPERATURES[name]['T']; scaled=probabilities(v,T)
        assert np.array_equal(raw.argmax(1),scaled.argmax(1))
        for state,p,t in [('Raw',raw,1.),('Scaled',scaled,T)]:
            r=dict(condition=name,state=state,T=t,eligible=len(d),valid=len(v),accuracy=float((p.argmax(1)==y).mean()),
                   temperature_at_search_bound=TEMPERATURES[name]['at_bound'])
            r.update(calibration_metrics(y,p)); r.update(calibration_interval(y,p,v.group_id.to_numpy())); rows.append(r)
            bins=reliability(y,p,np.linspace(0,1,16)); bins['condition']=name; bins['state']=state; bin_records.append(bins)
        raw_nll=-np.log(np.clip(raw[np.arange(len(y)),y],1e-15,1))
        cal_nll=-np.log(np.clip(scaled[np.arange(len(y)),y],1e-15,1))
        effects.append(dict(condition=name,contrast='Scaled minus raw NLL',**paired_scalar_interval(cal_nll-raw_nll,v.group_id.to_numpy())))
    save_table(11,pd.DataFrame(rows),'Raw and temperature-scaled five-candidate calibration on complete test predictions')
    effects=holm(pd.DataFrame(effects)); effects.to_csv(ROOT/'tables'/'RQ4_calibration_contrasts.csv',index=False)
    pd.concat(bin_records).to_csv(ROOT/'tables'/'reliability_15_bins_all_conditions.csv',index=False)
    chosen=[n for n in ['vlm_p0','lora_20','cnn_full'] if n in RESULTS][:2]
    fig,axes=plt.subplots(1,len(chosen),figsize=(6*len(chosen),4.7),squeeze=False)
    for ax,name in zip(axes[0],chosen):
        v=valid_rows(RESULTS[name]); y=v.y.to_numpy(int)
        ax.plot([0,1],[0,1],':',color='gray')
        for j,(state,T) in enumerate([('Raw',1.),('Scaled',TEMPERATURES[name]['T'])]):
            b=reliability(y,probabilities(v,T),np.linspace(0,1,16)); b=b[b.n>0]
            ax.plot(b.confidence,b.correctness,'-',color=COLORS[j],alpha=.7)
            ax.scatter(b.confidence,b.correctness,s=18+3*np.sqrt(b.n),color=COLORS[j],label=state)
        ax.set(xlabel='Mean maximum class probability',ylabel='Observed correctness',title=LABELS[name],xlim=(0,1),ylim=(0,1)); ax.legend()
    save_figure(7,fig,'Top-label reliability in 15 fixed equal-width bins; empty bins are omitted, marker area reflects support. Model-level metric intervals appear in Table 11. Scaling does not change the selected grade.')
    name='vlm_p0'; v=valid_rows(RESULTS[name]); y=v.y.to_numpy(int)
    scaled=probabilities(v,TEMPERATURES[name]['T'])
    bins=reliability(y,scaled,np.array([0,.4,.5,.6,.7,1.]))
    save_table(12,bins,'Scaled frozen P0 in five readable display bands; ECE still uses 15 bins')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,(state,T) in zip(axes,[('Raw',1.),('Scaled',TEMPERATURES[name]['T'])]):
        p=probabilities(v,T); ok=p.argmax(1)==y
        for j,(mask,label) in enumerate([(ok,'Correct'),(~ok,'Incorrect')]):
            ax.hist(p.max(1)[mask],bins=np.linspace(0,1,21),alpha=.55,label=f'{label} n={mask.sum()}',color=COLORS[j])
        ax.set(xlabel='Maximum candidate probability',ylabel='Cases',title=state); ax.legend()
    save_figure(8,fig,'Frozen P0 confidence distributions before and after temperature scaling. Correctness is unchanged; reduced confidence alone is not improved grading.')

def selection_metrics(y,pred):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    return dict(n=len(y),errors=int((y!=pred).sum()),large_errors=int((abs(y-pred)>=2).sum()),
                error_risk=float((y!=pred).mean()) if len(y) else np.nan,
                large_risk=float((abs(y-pred)>=2).mean()) if len(y) else np.nan,
                retained_QWK=metric_values(y,pred)['QWK'])

def rank_data(name,d,random_order=False):
    v=valid_rows(d).copy(); v['confidence']=probabilities(v,TEMPERATURES[name]['T']).max(1)
    v['random_key']=[int(hashlib.sha256(f'{SEED}:random:{x}'.encode()).hexdigest()[:12],16) for x in v.case_id]
    return v.sort_values(['random_key','case_id'] if random_order else ['confidence','case_id'],
                         ascending=[True,True] if random_order else [False,True])

def retained_at(name,d,coverage,random_order=False):
    v=rank_data(name,d,random_order); k=math.ceil(coverage*len(d)); attainable=k<=len(v)
    if not attainable:
        return dict(condition=name,policy='random' if random_order else 'confidence',target=coverage,
                    attainable=False,eligible=len(d),valid=len(v),n=0,coverage=np.nan,
                    max_attainable_coverage=len(v)/len(d))
    kept=v.iloc[:k]; r=selection_metrics(kept.y,kept.pred)
    r.update(condition=name,policy='random' if random_order else 'confidence',target=coverage,attainable=True,
             eligible=len(d),valid=len(v),coverage=k/len(d),review_n=len(d)-k)
    return r

def selective_bootstrap(name,d,c):
    # Resample eligible groups, then re-rank each resample; failures remain in coverage denominator.
    samples=defaultdict(list)
    for ix in cluster_indices(d.group_id,CFG['bootstrap']):
        sample=d.iloc[ix].copy()
        a=retained_at(name,sample,c); b=retained_at(name,sample,c,True)
        if not a['attainable'] or not b['attainable']: continue
        samples['error_risk'].append(a['error_risk']); samples['large_risk'].append(a['large_risk'])
        samples['effect'].append(a['error_risk']-b['error_risk'])
    result={}
    for key in ['error_risk','large_risk']:
        lo,hi,bn=interval(samples[key]); result.update({key+'_lo':lo,key+'_hi':hi,key+'_bootstrap_valid':bn})
    est=retained_at(name,d,c)['error_risk']-retained_at(name,d,c,True)['error_risk']
    lo,hi,bn=interval(samples['effect']); values=np.asarray(samples['effect'])
    p=(1+int((abs(values-est)>=abs(est)).sum()))/(len(values)+1)
    effect=dict(condition=name,coverage=c,effect=est,lo=lo,hi=hi,p=p,bootstrap_valid=bn)
    return result,effect

def rq5():
    rows=[]; effects=[]
    for name,d in RESULTS.items():
        for c in [.6,.8,.9]:
            r=retained_at(name,d,c)
            if r['attainable']:
                ci,effect=selective_bootstrap(name,d,c); r.update(ci); effects.append(effect)
            rows.append(r); rows.append(retained_at(name,d,c,True))
    t13=save_table(13,pd.DataFrame(rows),'Matched 60%, 80% and 90% retention, with seeded random references and attainable coverage')
    if effects: holm(pd.DataFrame(effects)).to_csv(ROOT/'tables'/'RQ5_confidence_vs_random.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,name in enumerate([n for n in ['cnn_full','vlm_p0','lora_20'] if n in RESULTS]):
        curve=pd.DataFrame([retained_at(name,RESULTS[name],float(c)) for c in np.linspace(.1,1,46)])
        curve.to_csv(ROOT/'tables'/f'risk_coverage_{name}.csv',index=False)
        for ax,key in zip(axes,['error_risk','large_risk']):
            if key not in curve: continue
            ax.plot(curve.coverage,curve.get(key),label=LABELS[name],color=COLORS[i])
            points=t13[(t13.condition==name)&(t13.policy=='confidence')&t13.attainable]
            if len(points): ax.vlines(points.coverage,points[key+'_lo'],points[key+'_hi'],color=COLORS[i])
    if 'vlm_p0' in RESULTS:
        random_curve=pd.DataFrame([retained_at('vlm_p0',RESULTS['vlm_p0'],float(c),True) for c in np.linspace(.1,1,46)])
        for ax,key in zip(axes,['error_risk','large_risk']):
            if key in random_curve: ax.plot(random_curve.coverage,random_curve[key],':',color='gray',label='P0 seeded random')
    for ax,label in zip(axes,['Any-grade-error risk','Large-ordinal-error risk']):
        ax.set(xlabel='Retained fraction of all eligible cases',ylabel=label,xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=8)
    save_figure(9,fig,f'Risk versus all-case coverage. Rankings use calibrated confidence; failures are never released. Bars at 60%, 80% and 90% show 95% {GROUP_LABEL}-bootstrap intervals with ranking recomputed within each resample. Random ordering is a descriptive reference.')
    threshold_rows=[]; accounts=[]
    for name,d in RESULTS.items():
        v=rank_data(name,d); threshold=THRESHOLDS[name]['threshold']; kept=v[v.confidence>=threshold]
        r=selection_metrics(kept.y,kept.pred)
        correct=(v.y.to_numpy()==v.pred.to_numpy()); keep=v.confidence.to_numpy()>=threshold
        accounts.append(dict(condition=name,retained_correct=int((correct&keep).sum()),
            retained_wrong=int((~correct&keep).sum()),referred_correct=int((correct&~keep).sum()),
            referred_wrong=int((~correct&~keep).sum()),failed=len(d)-len(v)))
        r.update(condition=name,threshold=threshold,eligible=len(d),coverage=len(kept)/len(d),
                 development_coverage=THRESHOLDS[name]['dev_coverage'],review_n=len(d)-len(kept))
        vals=defaultdict(list)
        for ix in cluster_indices(d.group_id,CFG['bootstrap']):
            s=d.iloc[ix]; vv=rank_data(name,s); kk=vv[vv.confidence>=threshold]
            vals['coverage'].append(len(kk)/len(s))
            vals['error_risk'].append(selection_metrics(kk.y,kk.pred)['error_risk'])
        for key in vals:
            lo,hi,bn=interval(vals[key]); r.update({key+'_lo':lo,key+'_hi':hi,key+'_bootstrap_valid':bn})
        threshold_rows.append(r)
    save_table(14,pd.DataFrame(threshold_rows),'Unchanged development thresholds at realised test coverage; all ties are retained')
    account=pd.DataFrame(accounts); account.to_csv(ROOT/'tables'/'RQ5_release_review_accounting.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,4.8)); left=np.zeros(len(account))
    for key,color,label in zip(['retained_correct','retained_wrong','referred_correct','referred_wrong','failed'],
        [COLORS[2],COLORS[1],'#b9d5bd','#edc4a9','#9c9c9c'],
        ['Retained correct','Retained wrong','Referred correct','Referred wrong','Inference failed']):
        ax.barh(account.condition.map(LABELS),account[key],left=left,color=color,label=label); left+=account[key]
    ax.set_xlabel('Eligible test cases'); ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left')
    save_figure(10,fig,'Correctness and review accounting under fixed development thresholds. Test coverage may differ from 80%; inference failures always enter review. Thresholds were selected without test labels.')

def rq6():
    robust={}; effects=[]; rows=[]
    for name in ['cnn_full','vlm_p0','lora_20']:
        if name not in RESULTS: continue
        robust[name]={variant:read_records(prediction_path(name,'robustness',variant))
                      for variant in ['original']+PERTURBATIONS}
        for variant,d in robust[name].items():
            r=summary_row(name,d); r['target_condition']=variant; rows.append(r)
            if variant!='original':
                for metric in ['QWK','large_error']:
                    effect=paired_effect(d,robust[name]['original'],f'{name}: {variant} minus original',metric)
                    effect.update(condition=name,target_condition=variant); effects.append(effect)
    t15=save_table(15,pd.DataFrame(rows),'Fixed 100-group perturbation cohort; denominators and grouped intervals are explicit')
    effects=holm(pd.DataFrame(effects)); effects.to_csv(ROOT/'tables'/'RQ6_perturbation_effects.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for ax,metric in zip(axes,['QWK','large_error']):
        for j,name in enumerate(robust):
            sub=effects[(effects.condition==name)&(effects.metric==metric)].set_index('target_condition').reindex(PERTURBATIONS)
            y=np.arange(len(sub))+(j-1)*.16
            ax.plot(sub.effect,y,'o',label=LABELS[name],color=COLORS[j]); ax.hlines(y,sub.lo,sub.hi,color=COLORS[j])
        ax.axvline(0,color='gray',ls=':'); ax.set_yticks(range(len(PERTURBATIONS)),PERTURBATIONS)
        ax.set_xlabel('Change in '+('QWK' if metric=='QWK' else 'large-error risk')); ax.legend(fontsize=8)
    save_figure(11,fig,f'Paired changes from each model’s own unmodified image results. Bars are 95% {GROUP_LABEL}-bootstrap intervals on common-complete pairs. Holm-adjusted tests are supplied separately. Transformations are engineering controls, not validated acquisition shifts.')
    base=robust['vlm_p0']['original']
    wordings=[base.set_index('case_id').reindex(ROBUST.case_id)]
    for prompt in ['P2','P3','P4','P5']:
        d=pd.read_csv(ROOT/'predictions'/f'wording_{prompt}.csv')
        wordings.append(d.set_index('case_id').reindex(ROBUST.case_id))
    complete=np.column_stack([d.status.eq('ok').to_numpy() for d in wordings]).all(1)
    predictions=np.column_stack([pd.to_numeric(d.pred,errors='coerce').to_numpy() for d in wordings])
    stable=complete&(predictions==predictions[:,[0]]).all(1)
    reference=ROBUST.y.to_numpy(int); categories=np.full(len(ROBUST),'Incomplete',dtype=object)
    categories[complete&~stable]='Changed grade'
    categories[stable&(predictions[:,0]==reference)]='Stable correct'
    categories[stable&(predictions[:,0]!=reference)]='Stable wrong'
    category_names=['Stable correct','Stable wrong','Changed grade','Incomplete']
    t16=pd.DataFrame([dict(category=c,n=int((categories==c).sum()),
         percent=100*float((categories==c).mean()),eligible=len(ROBUST)) for c in category_names])
    save_table(16,t16,'Correctness-aware stability across canonical P0 and four fixed paraphrases')
    detail=pd.DataFrame(dict(case_id=ROBUST.case_id,group_id=ROBUST.group_id,y=reference,category=categories))
    detail.to_csv(ROOT/'tables'/'RQ6_case_stability.csv',index=False)
    controls={kind:read_records(prediction_path('vlm_p0','robustness',kind)) for kind in ['original','blank','shuffled']}
    control_summary=pd.DataFrame([summary_row(kind,d) for kind,d in controls.items()])
    control_summary.to_csv(ROOT/'tables'/'RQ6_visual_input_controls.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7)); bottom=np.zeros(5)
    grade_counts=[]
    for j,c in enumerate(category_names):
        counts=[int(((categories==c)&(reference==g)).sum()) for g in range(5)]
        axes[0].bar(range(5),counts,bottom=bottom,label=c,color=COLORS[j]); bottom+=counts
        for g,n in enumerate(counts): grade_counts.append(dict(grade=g,category=c,n=n))
    pd.DataFrame(grade_counts).to_csv(ROOT/'tables'/'RQ6_stability_by_grade.csv',index=False)
    axes[0].set(xlabel='Reference KL grade',ylabel='Index knees (one per group)',xticks=range(5)); axes[0].legend(fontsize=8)
    axes[1].bar(control_summary.condition,control_summary.all_case_accuracy,color=COLORS[:3])
    axes[1].set(ylabel='Accuracy over all eligible index cases',ylim=(0,1),xlabel='Visual-input control')
    save_figure(12,fig,'Correctness-aware prompt stability by reference grade and all-case accuracy under target, blank and shuffled-image controls. Stable errors are counted separately. Controls use original references and are not pooled with the main cohort; they do not establish causal explanations.')

def rq7():
    rows=[]; components=[]
    quality={n:summary_row(n,d) for n,d in RESULTS.items()}
    for name in list(FROZEN['available'])+(['vlm_p0_8bit'] if CFG['optional_8bit'] else []):
        path=ROOT/'logs'/f'timing_{name}.jsonl'
        if not path.exists(): continue
        d=pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
        v=valid_rows(d)
        if name in quality: q=quality[name]; pool='full held-out test, valid predictions'
        else:
            q=summary_row(name,read_records(prediction_path(name,'quantization_sensitivity')))
            pool='100-group robustness subset, eight-bit sensitivity only'
        r=dict(condition=name,QWK=q.get('QWK'),QWK_lo=q.get('QWK_lo'),QWK_hi=q.get('QWK_hi'),
            quality_pool=pool,quality_n=q.get('valid'),attempts=len(d),complete=len(v),
            completion=len(v)/len(d),oom=int(d.status.eq('oom').sum()),
            cases=d.case_id.nunique(),repeats=CFG['timing_repeats'],load_seconds=float(d.load_seconds.iloc[0]),
            precision='float32' if name.startswith('cnn') else ('int8-float16' if name.endswith('8bit') else 'NF4-float16'))
        if len(v):
            r.update(mean_s=float(v.total_s.mean()),p95_s=float(v.total_s.quantile(.95)),peak_GiB=float(v.peak_GiB.max()))
            means=[]; p95=[]
            for ix in cluster_indices(v.group_id,CFG['bootstrap']):
                means.append(v.total_s.to_numpy()[ix].mean()); p95.append(np.quantile(v.total_s.to_numpy()[ix],.95))
            lo,hi,bn=interval(means); r.update(mean_s_lo=lo,mean_s_hi=hi,bootstrap_valid=bn)
            lo,hi,bn=interval(p95); r.update(p95_s_lo=lo,p95_s_hi=hi)
            c=dict(condition=name,valid_measurements=len(v),
                **{k:float(v[k].mean()) for k in ['preprocess_s','image_s','language_s','scoring_s','total_s']})
            assert abs(sum(c[k] for k in ['preprocess_s','image_s','language_s','scoring_s'])-c['total_s'])<.01
            c.update(context_tokens_mean=float(v.context_tokens.mean()),context_tokens_max=int(v.context_tokens.max()),
                     visual_tokens_mean=float(v.visual_tokens.mean()),support_cache=False)
            components.append(c)
        rows.append(r)
    t17=save_table(17,pd.DataFrame(rows),'Warm batch-one inference, loading cost, completion and quality with cohort labels')
    t18=save_table(18,pd.DataFrame(components),'Synchronized component means on the identical successful timed attempts')
    fig,ax=plt.subplots(figsize=(9,5))
    for i,r in enumerate(rows):
        if not np.isfinite(r.get('mean_s',np.nan)) or r.get('QWK') is None: continue
        x=r['mean_s']; y=r['QWK']
        ax.scatter(x,y,s=35+20*r['peak_GiB'],alpha=.8,color=COLORS[i%len(COLORS)],
                   label=LABELS.get(r['condition'],r['condition']))
        ax.hlines(y,r['mean_s_lo'],r['mean_s_hi'],color=COLORS[i%len(COLORS)])
        if r.get('QWK_lo') is not None: ax.vlines(x,r['QWK_lo'],r['QWK_hi'],color=COLORS[i%len(COLORS)])
    ax.set(xscale='log',xlabel='Mean warm latency (seconds, log scale)',ylabel='QWK')
    ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left',markerscale=.6)
    save_figure(13,fig,'Measured quality and runtime. Horizontal and vertical bars are grouped 95% intervals; marker area increases with peak allocated GPU memory. Quality and timing cohorts are separately named in Table 17. No optimal frontier is asserted.')
    fig,ax=plt.subplots(figsize=(12,5)); left=np.zeros(len(t18))
    for key,color,label in zip(['preprocess_s','image_s','language_s','scoring_s'],COLORS,
        ['Processing and transfers','Visual encoder','Language and model overhead','Head / answer scoring']):
        ax.barh(t18.condition.map(lambda n:LABELS.get(n,n)),t18[key],left=left,label=label,color=color); left+=t18[key]
    ax.set_xlabel('Mean seconds per successful attempt'); ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left')
    save_figure(14,fig,'Synchronized component means sum to end-to-end warm means. Processing/transfers are shown as an additional component so their cost is not silently assigned to image encoding. Loading, target disk reads and training are separate; support images are reprocessed without prefix caching. Timing hooks add synchronization overhead.')

def initialize(root, bootstrap=None):
    global ROOT, CFG, SEED, RUN_CONTRACT, FROZEN, TEMPERATURES, THRESHOLDS
    global MANIFEST, TRAIN, SUBSET, SUPPORTS, ROBUST, RESULTS, CONDITIONS
    global ADAPTATION, GROUP_LABEL
    ROOT=Path(root)
    contract=json.loads((ROOT/'protocol'/'run_contract.json').read_text())
    CFG=contract['config'].copy()
    if bootstrap is not None:
        CFG['bootstrap']=bootstrap
    SEED=CFG['seed']
    RUN_CONTRACT=contract['hash']
    FROZEN=json.loads((ROOT/'protocol'/'frozen_protocol.json').read_text())
    TEMPERATURES=json.loads((ROOT/'protocol'/'temperatures.json').read_text())
    THRESHOLDS=json.loads((ROOT/'protocol'/'review_thresholds.json').read_text())
    CONDITIONS=json.loads((ROOT/'protocol'/'conditions.json').read_text())
    GROUP_LABEL='participant' if CFG['grouping_verified'] else 'provisional prefix'
    MANIFEST=pd.read_csv(ROOT/'audit'/'manifest.csv')
    TRAIN=MANIFEST[MANIFEST.split.eq('train')].copy()
    SUBSET=pd.read_csv(ROOT/'protocol'/'adapter_subset_manifest.csv')
    ROBUST=pd.read_csv(ROOT/'protocol'/'robustness_manifest.csv')
    SUPPORTS={f'S{i}':pd.read_csv(ROOT/'protocol'/f'support_S{i}.csv') for i in range(1,4)}
    ADAPTATION=json.loads((ROOT/'checkpoints'/'lora_20_result.json').read_text())
    RESULTS={name:read_records(prediction_path(name,'test')) for name in FROZEN['available']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':110,'savefig.bbox':'tight','pdf.fonttype':42})
