"""Динамика оценок событий: кэш event-level OOF-скоров по фолдам + агрегаты второго порядка."""
from lab import *
import mil
from sklearn.model_selection import GroupKFold
# референс v2 = равновзвешенный ансамбль 5 моделей (совпадает с ноутбуком)
M=[load_oof(k) for k in ['m_lgbA_v1','m_lgbB_v1','m_lgbA_v12','m_xgb_v1','m_cat_v1']]
V2={s:np.mean([m[s] for m in M],0) for s in ('FWD','LDO')}; pickle.dump(V2,open('oof/V2.pkl','wb'))
report('v2 финал (референс)',V2,group='ref')
T=pd.read_pickle('event_table.pkl')
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row); Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
Ttr['ts']=ev.set_index(ev.index).event_ts.values[:0].tolist() if False else np.nan
# время события для динамики: берём из отсортированных событий тем же порядком, что в event_table
e=ev.sort_values(['cookie_id','event_ts','eid'],kind='mergesort').reset_index(drop=True)
T['ts']=(e.event_ts.astype('int64')//10**9).values; Ttr['ts']=T.loc[Ttr.index,'ts'].values
# 1) кэш event-level скоров (v1): для train-части OOF по куке, для валидации - модель на train-части
EVP={}
import os
if os.path.exists('evp_v1.pkl'): EVP=pickle.load(open('evp_v1.pkl','rb'))
for sname,sch in [('FWD',FWD),('LDO',LDO)]:
    for k,(tr,va) in enumerate(masks(sch)):
        if (sname,k) in EVP: continue
        Tf=Ttr[tr[Ttr.row.values]]; Ta=Ttr[va[Ttr.row.values]]
        pf=np.zeros(len(Tf))
        for a,b in GroupKFold(5).split(Tf,groups=Tf.row):
            pf[b]=mil.fit_event_model(Tf.iloc[a],mil.ECOLS_V1).predict_proba(Tf.iloc[b][mil.ECOLS_V1])[:,1]
        pa=mil.fit_event_model(Tf,mil.ECOLS_V1).predict_proba(Ta[mil.ECOLS_V1])[:,1]
        EVP[(sname,k)]=(Tf.index.values,pf,Ta.index.values,pa)
        print(sname,k,flush=True)
pickle.dump(EVP,open('evp_v1.pkl','wb'))

def dyn_agg(idx,p):
    d=Ttr.loc[idx,['row','ts']].copy(); d['lo']=np.log(p/(1-p)); d['hi']=(p>0.5).astype(int)
    g0=d.groupby('row',sort=False)
    d['pos']=g0.cumcount()/g0.row.transform('size').clip(lower=2).sub(1)
    g=d.groupby('row',sort=False)
    out=pd.DataFrame(index=pd.Index(d.row.unique(),name='row'))
    out['dyn_trend']=g.apply(lambda x:np.corrcoef(x.pos,x.lo)[0,1] if len(x)>=4 and x.lo.std()>0 else np.nan)
    out['dyn_roll5_max']=g.lo.apply(lambda s:s.rolling(5,min_periods=1).mean().max())
    out['dyn_roll5_min']=g.lo.apply(lambda s:s.rolling(5,min_periods=1).mean().min())
    out['dyn_longest_hi_run']=g.hi.apply(lambda s:(s.groupby((s!=s.shift()).cumsum()).transform('size')*s).max())
    out['dyn_n_hi08']=g.apply(lambda x:(x.lo>np.log(4)).sum())
    out['dyn_last_minus_first']=g.apply(lambda x:x.lo[x.pos>=2/3].mean()-x.lo[x.pos<=1/3].mean())
    # "пик": максимум среднего логита за любые 10 минут
    d['t10']=d.ts//600
    out['dyn_peak10m']=d.groupby(['row','t10']).lo.mean().groupby('row').max()
    return out

REF=load_oof('m_lgbA_v1'); BC=FEATURES+EXTRA
MILC=pickle.load(open('mil_cache.pkl','rb'))
DYN={}
for key,(itr,pf,iva,pa) in EVP.items():
    DYN[key]=(dyn_agg(itr,pf),dyn_agg(iva,pa)); print('dyn',key,flush=True)
pickle.dump(DYN,open('dyn_v1.pkl','wb'))
def ff(tr,va,sname,k):
    Xf=X2.loc[np.where(tr)[0],BC]; Xa=X2.loc[np.where(va)[0],BC]
    a,b=MILC[('v1',sname,k)]; Xf=Xf.join(a); Xa=Xa.join(b)
    c,d=DYN[(sname,k)]; return Xf.join(c),Xa.join(d)
o=run(lgb_fp(LGB_A),None,fold_fn=ff); report('LGB A | ext+MILv1 + динамика оценок событий (7)',o,ref=REF,group='mil',save_as='m_lgbA_v1_dyn')
