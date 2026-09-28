"""Шум разметки (confident learning, Northcutt et al. 2021 / cleanlab) и отбор признаков (null importance).
Обе процедуры выполняются ВНУТРИ фолда только на train-части: fp получает (Xf, yf) и сам решает, что выкинуть."""
from lab import *
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues
A=load_oof('A3')

def inner_oof(Xf,yf,params=LGB_A,seed=0):
    p=np.zeros(len(yf))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xf,yf):
        p[b]=lgb.LGBMClassifier(**params,random_state=seed).fit(Xf.iloc[a],yf[a]).predict_proba(Xf.iloc[b])[:,1]
    return p

def cl_fp(mode='drop',which='both',w_low=0.3):
    base=lgb_fp(LGB_A)
    def fp(Xf,yf,Xa):
        p=inner_oof(Xf,yf)
        iss=find_label_issues(yf,np.c_[1-p,p],filter_by='prune_by_noise_rate')
        if which=='pos': iss&=(yf==1)
        if which=='neg': iss&=(yf==0)
        fp.stats.append((int(iss.sum()),int((iss&(yf==1)).sum())))
        if mode=='drop': return base(Xf[~iss],yf[~iss],Xa)
        w=np.where(iss,w_low,1.0); return base(Xf,yf,Xa,w)
    fp.stats=[]; return fp

for mode,which in [('drop','both'),('drop','neg'),('drop','pos'),('weight','both')]:
    f=cl_fp(mode,which); o=run(f,FEATURES)
    report(f'cleanlab: {mode} {which}',o,ref=A,group='label_noise',note=f'issues(total,pos) per fold: {f.stats}')

def nullimp_fp(n_null=8,q=75):
    base=lgb_fp(LGB_A)
    def fp(Xf,yf,Xa):
        P=dict(LGB_A,n_estimators=300,learning_rate=0.05)
        real=lgb.LGBMClassifier(**P,random_state=0).fit(Xf,yf).booster_.feature_importance('gain')
        rng=np.random.default_rng(0)
        null=np.array([lgb.LGBMClassifier(**P,random_state=i).fit(Xf,rng.permutation(yf)).booster_.feature_importance('gain') for i in range(n_null)])
        keep=np.array(Xf.columns)[real>np.percentile(null,q,axis=0)]
        fp.kept.append(len(keep))
        return base(Xf[keep],yf,Xa[keep])
    fp.kept=[]; return fp
for q in [75,95]:
    f=nullimp_fp(q=q); o=run(f,FEATURES)
    report(f'null importance отбор (порог p{q})',o,ref=A,group='feature_selection',note=f'kept per fold: {f.kept}')
