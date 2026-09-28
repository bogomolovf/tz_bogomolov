from lab import *
from feats_new import *
A=load_oof('A3')
X=Xtr.copy()
for name,fn in [('курсор: траектория',pointer_feats),('сессии',session_feats),('n-граммы событий',ngram_feats)]:
    t=time.time(); Fn=fn(); cols=[c for c in Fn.columns if c!='cookie_id']
    Fn.to_pickle(f'feat_{fn.__name__}.pkl')
    X=X.drop(columns=[c for c in cols if c in X]).merge(Fn,on='cookie_id',how='left')
    o=run(lgb_fp(LGB_A),FEATURES+cols,X=X)
    report(f'+ {name} ({len(cols)} пр.)',o,ref=A,group='features',save_as=f'feat_{fn.__name__}',note=f'{time.time()-t:.0f}s')
uf,cats=ua_feats(); uf.to_pickle('feat_ua.pkl'); cats.to_pickle('cats_ua.pkl')
cols=[c for c in uf.columns if c!='cookie_id']; X=X.merge(uf,on='cookie_id',how='left')
o=run(lgb_fp(LGB_A),FEATURES+cols,X=X); report(f'+ UA семейство/версия ({len(cols)} пр.)',o,ref=A,group='features',save_as='feat_ua')
