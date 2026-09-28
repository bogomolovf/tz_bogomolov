"""Кэш MIL-признаков для каждого фолда (FWD, LDO) и для финала (train OOF / test)."""
from lab import *
import mil
T=pd.read_pickle('event_table.pkl')
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row)
Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
ECOLS_V2=[c for c in T.columns if c not in ('cookie_id','row')]
cache={}
for ver,ecols in [('v1',mil.ECOLS_V1),('v2',ECOLS_V2)]:
    for sname,sch in [('FWD',FWD),('LDO',LDO)]:
        for k,(tr,va) in enumerate(masks(sch)):
            t=time.time()
            ftr,fva=mil.mil_features(Ttr[tr[Ttr.row.values]],Ttr[va[Ttr.row.values]],ecols)
            cache[(ver,sname,k)]=(ftr.add_suffix('_'+ver),fva.add_suffix('_'+ver))
            print(ver,sname,k,round(time.time()-t),'s',flush=True)
    # финал: train OOF + test
    Tte=T[T.row.isna()].copy(); Tte['row']=Tte.cookie_id.map({c:i for i,c in enumerate(Xte.cookie_id)}).astype(int)
    ftr,fte=mil.mil_features(Ttr,Tte,ecols)
    cache[(ver,'FINAL')]=(ftr.add_suffix('_'+ver),fte.add_suffix('_'+ver))
pickle.dump(cache,open('mil_cache.pkl','wb')); print('done')
