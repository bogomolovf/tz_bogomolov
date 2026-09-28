"""(1) Вложенный стекинг: LR поверх логитов 5 моделей, обученная на OOF других фолдов. (2) Веса свежих дней."""
from lab import *
from sklearn.linear_model import LogisticRegression
V2=load_oof('V2'); REF=load_oof('m_lgbA_v1'); BC=FEATURES+EXTRA
keys=['m_lgbA_v1','m_lgbB_v1','m_lgbA_v12','m_xgb_v1','m_cat_v1']; M=[load_oof(k) for k in keys]
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
out={}
for sname,sch in [('FWD',FWD),('LDO',LDO)]:
    o=np.full(len(y),np.nan); vas=[va for _,va in masks(sch)]
    for k,va in enumerate(vas):
        other=np.zeros(len(y),bool)
        for j,vj in enumerate(vas):
            if (sname=='LDO' and j!=k) or (sname=='FWD' and j<k): other|=vj
        Z=np.c_[[lg(m[sname]) for m in M]].T
        if other.sum()==0: o[va]=np.mean([m[sname][va] for m in M],0); continue    # первый FWD-фолд: мета-модели учиться не на чем
        lr=LogisticRegression(C=1.0,max_iter=2000).fit(Z[other],y[other]); o[va]=lr.predict_proba(Z[va])[:,1]
        if sname=='LDO' and k==0: print('веса мета-модели:',dict(zip(keys,lr.coef_[0].round(2))))
    out[sname]=o
report('вложенный стекинг LR поверх 5 моделей',out,ref=V2,group='blend',save_as='stack_lr')
# (2) веса свежих дней: w = exp(-(последний день train-части - день)/tau)
ff=mil_fold_fn(BC,vers=('v1',))
dayn=((DAY-DAY.min()).dt.days).values
for tau in [5,10]:
    def rec_fp(Xf,yf,Xa,tau=tau):
        d=dayn[Xf.index.values]; w=np.exp(-(d.max()-d)/tau)
        return lgb_fp(LGB_A)(Xf,yf,Xa,w)
    o=run(rec_fp,None,fold_fn=ff); report(f'LGB A | веса свежих дней, tau={tau} дн.',o,ref=REF,group='weights')
