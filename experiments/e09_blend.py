from lab import *
import itertools
ref=load_oof('AB_v1')
M={k:load_oof(k) for k in ['m_lgbA_v1','m_lgbB_v1','m_lgbA_v12','m_xgb_v1','m_cat_v1']}
def bl(keys): return {s:np.mean([M[k][s] for k in keys],0) for s in ('FWD','LDO')}
combos=[('lgbA_v1','m_lgbA_v1'),]
for r in [2,3,4,5]:
    for c in itertools.combinations(M,r):
        o=bl(c); sc=scores(o)
        print(f"{'+'.join(k[2:] for k in c):45s} FWD={sc['fwd_mean']:.4f} {sc['fwd_folds']} LDO={sc['ldo_pooled']:.4f} PR-AUC={sc['ldo_prauc']:.4f} AUC={sc['ldo_auc']:.4f}")
for c in [('m_lgbA_v1','m_lgbB_v1'),('m_lgbA_v1','m_lgbA_v12','m_xgb_v1'),tuple(M)]:
    report('БЛЕНД '+'+'.join(k[2:] for k in c)+' vs финал v1',bl(c),ref=ref,group='blend')
