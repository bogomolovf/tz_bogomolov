import harness as H, numpy as np, pandas as pd, pickle
o=pickle.load(open('oof_s1.pkl','rb'))

def entropy_vals(s):
    p=s.value_counts(normalize=True).values; return float(-(p*np.log(p)).sum()) if len(p) else np.nan

def timing_features(ev):
    ev=ev[['cookie_id','event_ts','event_name']].copy()
    g=ev.groupby('cookie_id', sort=False)
    ev['dt']=g.event_ts.diff().dt.total_seconds()
    ev['dwell']=-g.event_ts.diff(-1).dt.total_seconds()           # время на событии до следующего
    f=pd.DataFrame(index=pd.Index(H.meta.cookie_id,name='cookie_id'))
    sh=ev[ev.dt<60].copy()                                          # интервалы внутри активности
    sh['prev']=sh.groupby('cookie_id').dt.shift()
    pair=sh.dropna(subset=['prev'])
    gp=pair.groupby('cookie_id')
    # 1) точный повтор интервала подряд (скрипт с фиксированной/дискретной задержкой)
    f['share_same_dt']=gp.apply(lambda d:(d.dt==d.prev).mean())
    # 2) "гладкость" ритма: насколько соседние интервалы похожи
    lr=np.abs(np.log((pair.dt+0.5)/(pair.prev+0.5)))
    f['dt_logratio_median']=lr.groupby(pair.cookie_id).median()
    f['dt_logratio_mean']=lr.groupby(pair.cookie_id).mean()
    # 3) лаг-1 автокорреляция интервалов
    f['dt_autocorr1']=gp.apply(lambda d:d.dt.corr(d.prev) if len(d)>=5 else np.nan)
    # 4) разнообразие интервалов: энтропия округлённых значений и доля моды
    gs=sh.groupby('cookie_id').dt
    f['dt_entropy']=gs.agg(lambda s:entropy_vals(s.round()))
    f['dt_mode_share']=gs.agg(lambda s:s.round().value_counts(normalize=True).iloc[0])
    f['dt_iqr']=gs.quantile(0.75)-gs.quantile(0.25)
    f['dt_burstiness']=(gs.std()-gs.mean())/(gs.std()+gs.mean())
    # 5) время на объявлении vs на выдаче (человек "читает" объявление)
    dw=ev[ev.dwell<120].groupby(['cookie_id','event_name']).dwell.median().unstack()
    f['dwell_item_median']=dw.get('item_view')
    f['dwell_search_median']=dw.get('search_results_view')
    f['dwell_item_to_search']=(dw.get('item_view')+1)/(dw.get('search_results_view')+1)
    f['dwell_std_across_types']=dw.std(axis=1)
    return f.reset_index()

F=timing_features(H.ev); new=H.add(F); pickle.dump(F,open('feat_s3.pkl','wb'))
base=H.FEATURES
_,o3=H.evaluate(base+new,'3a. + тонкий тайминг (все)'); H.paired_bootstrap(o3,o['o0'])
groups={'повторы/гладкость':['share_same_dt','dt_logratio_median','dt_logratio_mean','dt_autocorr1'],
        'разнообразие интервалов':['dt_entropy','dt_mode_share','dt_iqr','dt_burstiness'],
        'время на объявлении':['dwell_item_median','dwell_search_median','dwell_item_to_search','dwell_std_across_types']}
for k,v in groups.items():
    _,oo=H.evaluate(base+v,f'3. + {k}'); H.paired_bootstrap(oo,o['o0'])
pickle.dump(o3,open('oof_s3.pkl','wb'))
