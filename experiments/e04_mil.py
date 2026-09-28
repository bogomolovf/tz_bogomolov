"""Multi-instance: модель на уровне событий -> агрегаты скоров по куке (стекинг-признаки, строго внутри фолда)."""
from lab import *
from sklearn.model_selection import GroupKFold
A=load_oof('A3')

def event_table(ev=ev):
    e=ev.copy(); g=e.groupby('cookie_id',sort=False)
    dt=g.event_ts.diff().dt.total_seconds(); dn=-g.event_ts.diff(-1).dt.total_seconds()
    T=pd.DataFrame({'cookie_id':e.cookie_id,
        'etype':e.event_name.astype('category').cat.codes,'prev_etype':g.event_name.shift().astype('category').cat.codes,
        'dt_prev':np.log1p(dt),'dt_next':np.log1p(dn),
        'dt_roll_std':np.log1p(dt).groupby(e.cookie_id).transform(lambda s:s.rolling(5,min_periods=2).std()),
        'dt_roll_med':dt.groupby(e.cookie_id).transform(lambda s:s.rolling(5,min_periods=2).median()),
        'same_dt':(dt==g.event_ts.diff().dt.total_seconds().groupby(e.cookie_id).shift()).astype(float),
        'pos':g.cumcount(),'n':g.event_ts.transform('size'),
        'px':e.pointer_x,'py':e.pointer_y,'pstep':np.hypot(g.pointer_x.diff(),g.pointer_y.diff()),
        'plat':e.platform.astype('category').cat.codes,'uac':e.ua_class.astype('category').cat.codes,
        'page':e.search_page,'page_diff':g.search_page.diff(),
        'loc_change':(g.item_location.shift()!=e.item_location).astype(float),
        'hour':e.event_ts.dt.hour})
    return T
T=event_table(); ECOLS=[c for c in T.columns if c!='cookie_id']
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row)            # NaN для тестовых кук
Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
P=dict(LGB_A,n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=50)

def agg(rows,p):
    lo=np.log(p/(1-p))
    d=pd.DataFrame({'row':rows,'lo':lo,'p':p}).groupby('row')
    return pd.DataFrame({'mil_mean_logit':d.lo.mean(),'mil_max':d.p.max(),'mil_q90':d.p.quantile(.9),'mil_q10':d.p.quantile(.1),
                         'mil_share_hi':d.p.apply(lambda s:(s>0.5).mean())})

def fit_ev(Te):
    w=1.0/Te.n.values                                              # каждая кука весит одинаково
    return lgb.LGBMClassifier(**P,random_state=0).fit(Te[ECOLS],Te.y,sample_weight=w)

def fold_fn(tr,va):
    Tf=Ttr[tr[Ttr.row.values]]; Ta=Ttr[va[Ttr.row.values]]
    parts=[]
    for a,b in GroupKFold(5).split(Tf,groups=Tf.row):
        m=fit_ev(Tf.iloc[a]); parts.append(agg(Tf.row.values[b],m.predict_proba(Tf.iloc[b][ECOLS])[:,1]))
    ftr=pd.concat(parts)
    fva=agg(Ta.row.values,fit_ev(Tf).predict_proba(Ta[ECOLS])[:,1])
    idx_tr,idx_va=np.where(tr)[0],np.where(va)[0]
    Xf=Xtr.loc[idx_tr,FEATURES].join(ftr); Xa=Xtr.loc[idx_va,FEATURES].join(fva)
    return Xf,Xa
t=time.time()
o=run(lgb_fp(LGB_A),None,fold_fn=fold_fn)
report('+ MIL: модель на событиях -> агрегаты (5 пр.)',o,ref=A,group='features',save_as='mil',note=f'{time.time()-t:.0f}s')
