import harness as H, numpy as np, pickle, lightgbm as lgb
from metric import precision_at_recall
o4=pickle.load(open('oof_s4.pkl','rb'))
cols=H.FEATURES; seeds=(101,102,103)
def fitpred(Xf,yf,Xa,params,w=None):
    return np.mean([lgb.LGBMClassifier(**params,random_state=s).fit(Xf,yf,sample_weight=w).predict_proba(Xa)[:,1] for s in seeds],0)
def run(top_pos,bot_neg,weight,name):
    per,op=[],[]
    for a,b in H.FOLDS:
        tr,va=(H.DAY<a).values,((H.DAY>=a)&(H.DAY<b)).values
        Xf,yf,Xv=H.Xtr.loc[tr,cols],H.y[tr],H.Xtr.loc[va,cols]
        p0=(fitpred(Xf,yf,Xv,H.LGB_A)+fitpred(Xf,yf,Xv,H.LGB_B))/2
        r=p0.argsort().argsort()/len(p0)            # ранг в [0,1)
        pos=r>=1-top_pos; neg=r<bot_neg              # уверенные боты / уверенные люди; разметку валидации НЕ используем
        Xp=np.vstack([Xf.values,Xv.values[pos|neg]]); yp=np.r_[yf,pos[pos|neg].astype(int)]
        w=np.r_[np.ones(len(yf)),np.full((pos|neg).sum(),weight)]
        import pandas as pd
        Xp=pd.DataFrame(Xp,columns=cols)
        p=(fitpred(Xp,yp,Xv,H.LGB_A,w)+fitpred(Xp,yp,Xv,H.LGB_B,w))/2
        per.append(precision_at_recall(H.y[va],p)); op.append(p)
    print(f'{name:50s} folds={np.round(per,4)} mean={np.mean(per):.4f}',flush=True)
    return op
AB=[(a+b)/2 for a,b in zip(o4['oA'],o4['oB'])]
yf=[H.y[((H.DAY>=a)&(H.DAY<b)).values] for a,b in H.FOLDS]
print('A+B без псевдоразметки: mean',round(np.mean([precision_at_recall(v,p) for v,p in zip(yf,AB)]),4))
for tp,bn,w in [(0.03,0.5,1.0),(0.05,0.5,0.5),(0.03,0.7,0.5)]:
    op=run(tp,bn,w,f'псевдо: топ-{int(tp*100)}% боты, низ-{int(bn*100)}% люди, вес {w}')
    H.paired_bootstrap(op,AB)
