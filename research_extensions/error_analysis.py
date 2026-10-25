"""Post-hoc diagnostics of frozen predictions. Never trains or changes thresholds."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import ROOT, KINDS
from research.features import NAMES
from research.report import markdown_table

OUT = ROOT / 'results' / 'error_analysis_v1'
KEYS = ['cutoff', 'src', 'dst']
FEATURES = ['hours_since_pair_event', 'pair_count_24h', 'pair_count_168h',
            'pair_count_lifetime', 'pair_age_days', 'active_days_30d',
            'reverse_count_24h', 'src_out_168h', 'dst_in_168h',
            'common_neighbors_30d', 'gap_mean_hours', 'burstiness',
            *[f'type_{k}_last' for k in KINDS]]
GROUP_NAMES = {
    'recency': '마지막 연결 후 시간', 'lifetime': '과거 동일 방향 총 연결 횟수',
    'frequency24': '과거 24시간 동일 방향 연결 횟수', 'reciprocity': '과거 24시간 반대 방향 연결',
    'last_type': '마지막 행동 유형', 'src_activity': '보낸 사용자 주간 활동',
    'dst_activity': '받은 사용자 주간 활동', 'neighbors': '30일 공통 이웃 수',
    'score_ratio': '점수 / 해당 구간의 판정 기준',
}


def rate(a, b):
    return float(a / b) if b else None


def confusion(frame):
    y = frame.label.to_numpy(bool); p = frame.prediction.to_numpy(bool)
    tp = int((y & p).sum()); fn = int((y & ~p).sum())
    fp = int((~y & p).sum()); tn = int((~y & ~p).sum())
    return dict(n=len(frame), positive=tp+fn, predicted_positive=tp+fp,
                tp=tp, fn=fn, fp=fp, tn=tn, precision=rate(tp,tp+fp),
                recall=rate(tp,tp+fn), miss_rate=rate(fn,tp+fn),
                false_discovery_rate=rate(fp,tp+fp), false_positive_rate=rate(fp,fp+tn),
                f1=rate(2*tp,2*tp+fp+fn))


def categories(f):
    return {
        'recency': pd.cut(f.hours_since_pair_event,[-np.inf,1,4,12,24],labels=['0-1h','1-4h','4-12h','12-24h']),
        'lifetime': pd.cut(f.pair_count_lifetime,[0,1,3,np.inf],labels=['1','2-3','4+']),
        'frequency24': pd.cut(f.pair_count_24h,[0,1,3,np.inf],labels=['1','2-3','4+']),
        'reciprocity': pd.Series(np.where(f.reverse_count_24h>0,'yes','no'),index=f.index),
        'last_type': pd.Series(np.array(KINDS)[f[[f'type_{k}_last' for k in KINDS]].to_numpy().argmax(axis=1)],index=f.index),
        'src_activity': pd.cut(f.src_out_168h,[0,5,20,np.inf],labels=['1-5','6-20','21+']),
        'dst_activity': pd.cut(f.dst_in_168h,[0,5,20,np.inf],labels=['1-5','6-20','21+']),
        'neighbors': pd.cut(f.common_neighbors_30d,[-1,0,3,np.inf],labels=['0','1-3','4+']),
        'score_ratio': pd.cut(f.score/f.threshold,[-np.inf,.5,.8,1,1.25,2,np.inf],right=False,
                              labels=['<0.5','0.5-0.8','0.8-1','1-1.25','1.25-2','2+']),
    }


def next_delays(events, candidates, horizon=48):
    """Outcome inspection ONLY: first same-direction future event; handle censoring."""
    index = {key:g.time.to_numpy() for key,g in events.groupby(['src','dst'],sort=False)}
    delays = np.full(len(candidates),np.inf)
    for key, indexes in candidates.groupby(['src','dst'],sort=False).indices.items():
        times = index[key]; q = candidates.iloc[indexes].cutoff.to_numpy()
        pos = np.searchsorted(times,q,side='left'); available = pos<len(times)
        delays[indexes[available]] = (times[pos[available]]-q[available])/3600
    complete = candidates.cutoff.to_numpy()+horizon*3600 <= events.time.max()
    return delays, complete


def future_activity(events, candidates, direction):
    """Diagnostic outcome only: did the endpoint act anywhere in the next 24h?"""
    index={int(key):g.time.to_numpy() for key,g in events.groupby(direction,sort=False)}
    active=np.zeros(len(candidates),dtype=bool)
    for key,ix in candidates.groupby(direction,sort=False).indices.items():
        times=index[int(key)];q=candidates.iloc[ix].cutoff.to_numpy()
        active[ix]=np.searchsorted(times,q+86400,side='left')>np.searchsorted(times,q,side='left')
    return active


def raw_events(dataset):
    # Read only local public files. No downloads or changes to training manifests.
    parts=[]; hashes={}
    for kind in KINDS:
        p=ROOT/'data'/'typed'/dataset/f'sx-{dataset}-{kind}.txt'
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        g=pd.read_csv(p,sep=r'\s+',names=['src','dst','time']);parts.append(g)
    g=pd.concat(parts,ignore_index=True);g=g[g.src!=g.dst].copy()
    users=np.unique(g[['src','dst']].to_numpy())
    for k in ('src','dst'):g[k]=np.searchsorted(users,g[k])
    return g.sort_values('time',kind='stable').reset_index(drop=True),hashes


def load_features(dataset):
    manifest=json.loads((ROOT/'results'/'lookback_density_v1'/dataset/'feature_validation.json').read_text())
    path=ROOT/'data'/'cache'/manifest['cache']
    with np.load(path,allow_pickle=False) as z:
        daily=z['cutoff']%86400==0
        f=pd.DataFrame({k:z[k][daily] for k in KEYS+['label']})
        x=z['x']
        for name in FEATURES:f[name]=x[daily,NAMES.index(name)]
    assert not f.duplicated(KEYS).any()
    return f,hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(dataset):
    folder=OUT/dataset;folder.mkdir(parents=True,exist_ok=True)
    features,cache_hash=load_features(dataset);events,raw_hashes=raw_events(dataset)
    totals=[];groups=[];folds=[];calibration=[];examples=[];medians=[];timings=[];paired=[];mechanisms=[];prediction_hashes={}
    diagnostic={};seed_frames={}
    for condition in ('lookback_4h_train_24h','lookback_24h_train_6h'):
        for seed in (2026,2027,2028):
            parts=[]
            for fold in range(5):
                run=ROOT/'results'/'lookback_density_v1'/dataset/condition/f'fold_{fold}'/f'seed_{seed}'
                path=run/'predictions.csv.gz'
                prediction_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                p=pd.read_csv(path,float_precision='round_trip');p=p[p.model=='Full LightGBM'].copy()
                m=pd.read_csv(run/'metrics.csv',float_precision='round_trip')
                threshold=float(m.loc[(m.model=='Full LightGBM')&(m.cohort=='all'),'threshold'].iloc[0])
                p['threshold']=threshold
                np.testing.assert_array_equal(p.prediction,p.score>=threshold)
                parts.append(p)
            p=pd.concat(parts,ignore_index=True)
            f=p.merge(features,on=KEYS+['label'],validate='one_to_one')
            assert len(f)==len(p)
            f['outcome']=np.where(f.label==1,np.where(f.prediction,'TP','FN'),np.where(f.prediction,'FP','TN'))
            cohorts={'all':f}
            if condition=='lookback_24h_train_6h':cohorts.update(common_4h=f[f.common_4h],added_4_24h=f[~f.common_4h])
            seed_frames[(condition,seed)]=f[KEYS+['label','prediction','common_4h']].copy()
            for cohort,g in cohorts.items():
                meta=dict(dataset=dataset,condition=condition,seed=seed,cohort=cohort)
                total=confusion(g);totals.append({**meta,**total})
                for fold,h in g.groupby('fold'):
                    folds.append({**meta,'fold':fold,'threshold':h.threshold.iloc[0],**confusion(h)})
                for dimension,values in categories(g).items():
                    for group,h in g.groupby(values,observed=True):
                        c=confusion(h)
                        groups.append({**meta,'dimension':dimension,'group':str(group),**c,
                                       'fn_share':rate(c['fn'],total['fn']),'fp_share':rate(c['fp'],total['fp'])})
                for score_bin,h in g.groupby(pd.cut(g.score,[0,.05,.1,.2,.3,.5,1],include_lowest=True),observed=True):
                    calibration.append({**meta,'score_bin':str(score_bin),'mean_score':h.score.mean(),
                                        'observed_positive_rate':h.label.mean(),**confusion(h)})
                if seed==2026:
                    for outcome,h in g.groupby('outcome'):
                        medians.append({**meta,'outcome':outcome,'n':len(h),**h[FEATURES+['score']].median().to_dict()})
            if seed==2026:
                delay,complete=next_delays(events,f)
                np.testing.assert_array_equal((delay<24).astype(int),f.label)
                f['future_delay_hours']=delay;f['complete_48h']=complete
                f['future_src_out']=future_activity(events,f,'src')
                f['future_dst_in']=future_activity(events,f,'dst')
                assert f.loc[f.label==1,['future_src_out','future_dst_in']].all().all()
                for (outcome,source,target),h in f.groupby(['outcome','future_src_out','future_dst_in']):
                    mechanisms.append(dict(condition=condition,outcome=outcome,source_sends_anywhere=bool(source),
                                           target_receives_from_anyone=bool(target),n=len(h)))
                for outcome,h in f.groupby('outcome'):
                    eligible=h[h.complete_48h]
                    timings.append(dict(condition=condition,outcome=outcome,n=len(h),complete_48h_n=len(eligible),
                                        reconnected_24_48h=int(((eligible.future_delay_hours>=24)&(eligible.future_delay_hours<48)).sum()),
                                        delay_q25=float(h.future_delay_hours.quantile(.25)) if outcome in ('TP','FN') else None,
                                        delay_median=float(h.future_delay_hours.median()) if outcome in ('TP','FN') else None,
                                        delay_q75=float(h.future_delay_hours.quantile(.75)) if outcome in ('TP','FN') else None))
                # Deterministic illustrative extremes, explicitly NOT representative samples.
                f['relative_score']=f.score/f.threshold
                for outcome,ascending in (('FN',True),('FP',False)):
                    h=f[f.outcome==outcome].sort_values(['relative_score',*KEYS],ascending=[ascending,True,True,True]).head(5)
                    for row in h.to_dict('records'):
                        examples.append({k:row[k] for k in KEYS+['fold','outcome','label','score','threshold','future_delay_hours',*FEATURES]}|{'condition':condition})
                diagnostic[condition]=f
            print(f'{dataset} {condition} seed={seed} diagnosed',flush=True)
    for seed in (2026,2027,2028):
        a=seed_frames[('lookback_4h_train_24h',seed)]
        b=seed_frames[('lookback_24h_train_6h',seed)]
        b=b[b.common_4h]
        h=a.merge(b,on=KEYS+['label'],suffixes=('_old','_new'),validate='one_to_one');assert len(h)==len(a)==len(b)
        for label,g in h.groupby('label'):
            for (old,new),j in g.groupby(['prediction_old','prediction_new']):
                paired.append(dict(seed=seed,label=int(label),old=bool(old),new=bool(new),n=len(j)))
    tables={'overall':totals,'subgroups':groups,'per_fold':folds,'calibration':calibration,'examples':examples,
            'feature_medians':medians,'future_timing':timings,'prediction_changes':paired,'future_activity':mechanisms}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(folder/f'{name}.csv',index=False)
    write_report(dataset,tables)
    draw(dataset,pd.DataFrame(groups))
    manifest={'dataset':dataset,'model':'Full LightGBM','primary_display_seed':2026,'sensitivity_seeds':[2026,2027,2028],
              'aggregation':'pooled five disjoint time folds; no seed counts pooled',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cache_sha256':cache_hash,
              'raw_sha256':raw_hashes,'prediction_sha256':prediction_hashes,
              'scope':'post-hoc descriptive error analysis; no retraining, threshold changes, or causal attribution',
              'all_primary_labels_reconstructed':sum(len(f) for f in diagnostic.values())}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return tables


def write_report(dataset,tables):
    folder=OUT/dataset
    overview=pd.DataFrame(tables['overall']);overview=overview[overview.seed==2026]
    sub=pd.DataFrame(tables['subgroups']);sub=sub[(sub.seed==2026)&(sub.condition=='lookback_24h_train_6h')&(sub.cohort=='all')]
    lines=[f'# {dataset}: 오류 분석','',
           'Full LightGBM의 저장된 예측과 당시 보정 구간에서 정한 판정 기준을 그대로 사용했습니다. 재학습·임계값 변경은 하지 않았습니다.',
           '아래 건수는 시드 2026의 5개 테스트 구간을 합친 쌍-날짜 수입니다. 시드 3개를 합쳐 표본 수를 부풀리지 않았습니다. 기존 구간 평균 성능과 집계 방식이 다릅니다.',
           '', '## 전체 혼동행렬', '',markdown_table(overview[['condition','cohort','n','positive','tp','fn','fp','tn','precision','recall','f1']]),
           '', 'TP=맞춘 재연결, FN=놓친 재연결, FP=재연결로 예측했지만 안 일어남, TN=안 일어날 것으로 맞춤.',
           '누락률=FN/(TP+FN), 예측 중 오답률=FP/(TP+FP), 음성 중 오탐률=FP/(FP+TN). 오답률 80%가 모든 음성의 80%를 잘못 예측했다는 뜻은 아닙니다.',
           '', '## 확장 모델 오류의 세부 패턴', '']
    for dim,title in GROUP_NAMES.items():
        g=sub[sub.dimension==dim]
        lines += [f'### {title}', '',markdown_table(g[['group','n','positive','tp','fn','fp','tn','miss_rate','false_discovery_rate','false_positive_rate','fn_share','fp_share']]),'']
    lines += ['## 판정 기준과 시드 민감도','',markdown_table(pd.DataFrame(tables['per_fold'])[['condition','cohort','seed','fold','threshold','precision','recall','f1']]),
              '', '## 원래 후보에서 예측이 어떻게 바뀌었는가?', '',markdown_table(pd.DataFrame(tables['prediction_changes'])),
              '', 'label=1에서 old=False,new=True는 복구한 누락, old=True,new=False는 새 누락입니다. label=0에서 old=False,new=True는 새 오탐입니다.',
              '', '## 실제 다음 연결 시점 (사후 정답 분석 전용)', '',markdown_table(pd.DataFrame(tables['future_timing'])),
              '', '48시간 확인 가능한 후보만 24~48시간 재연결 분모에 포함했습니다. 24시간 정답을 48시간으로 바꾸거나 오류를 정답으로 처리하지 않습니다.',
              '', '## 오탐 당시 사용자는 다른 곳에서 활동했는가? (사후 분석)', '',markdown_table(pd.DataFrame(tables['future_activity'])),
              '', 'source_sends_anywhere=False는 A가 다음 24시간 누구에게도 보내는 행동을 하지 않았다는 뜻입니다. 읽기·로그인 등 모든 활동이 없었다는 뜻은 아닙니다. target_receives_from_anyone은 B가 누구에게서든 해당 방향 행동을 받은 여부입니다. 이 미래 정보는 모델 입력에 사용하지 않았습니다.',
              '', '## 해석 제한', '',
              '- 행동 유형 a2q=질문에 답변, c2a=답변에 댓글, c2q=질문에 댓글. 연결은 동일 방향이며 세 유형 중 어느 하나가 발생하면 양성입니다.',
              '- 한 번만 연결된 관계는 관측 데이터 안에서 처음인 관계입니다. 현실의 첫 만남을 뜻하지 않습니다.',
              '- 구간·시드별 자료는 per_fold.csv, subgroups.csv에 모두 저장했습니다. 희소 집단은 분모를 함께 보세요.',
              '- 그룹별 차이는 상관관계이며 원인이나 모델의 내부 의사결정 설명으로 단정할 수 없습니다. 다중 비교를 한 탐색적 진단입니다.',
              '- 실제 게시물·스레드 ID나 내용이 없어 같은 대화가 끝났는지, 주제가 같은지는 이 데이터로 확정할 수 없습니다.',
              '- examples.csv는 상대 점수가 극단적인 오류를 보여주는 예시일 뿐 대표 표본이 아닙니다. src/dst는 내부 재매핑 ID이며 계정 주소가 아닙니다.',
              '- 미래 연결 시간은 오류 설명에만 쓰였으며 입력 특징이나 모델 조정에 쓰지 않았습니다.',
              '- 테스트 오류를 보고 개선안을 고르면 기존 테스트는 더 이상 새 확인용 평가가 아닙니다. 미사용 기간 또는 별도 데이터가 필요합니다.']
    (folder/'report_ko.md').write_text('\n'.join(lines),encoding='utf-8')


def draw(dataset,sub):
    g=sub[(sub.seed==2026)&(sub.condition=='lookback_24h_train_6h')&(sub.cohort=='all')]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,dimension,title in zip(axes.flat,['recency','lifetime','last_type','reciprocity'],
                                ['Hours since last interaction','Lifetime directed pair count','Last interaction type','Reverse interaction in past 24h']):
        h=g[g.dimension==dimension];pos=np.arange(len(h))
        ax.bar(pos-.18,h.miss_rate,.36,label='Miss rate FN / actual positives',color='#217A8B')
        ax.bar(pos+.18,h.false_discovery_rate,.36,label='False discovery FP / predicted positives',color='#BA654A')
        ax.set_xticks(pos,h.group);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=7,loc='lower right')
    fig.suptitle(f'{dataset}: 24h lookback / 6h training; frozen seed 2026, pooled test folds')
    fig.tight_layout();fig.savefig(OUT/dataset/'error_patterns.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for dataset in ('mathoverflow','askubuntu'):analyze(dataset)
    print('Error analysis complete. No model or threshold changed.',flush=True)
