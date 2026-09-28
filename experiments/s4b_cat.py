import harness as H, numpy as np, pickle
from catboost import CatBoostClassifier
from metric import precision_at_recall
o4=pickle.load(open('oof_s4.pkl','rb'))
class CB(CatBoostClassifier): pass
cols=H.FEATURES
params=dict(iterations=1500,learning_rate=0.03,depth=6,l2_leaf_reg=5,verbose=0,thread_count=2)
_,oC=H.evaluate(cols,'CatBoost',params=None,seeds=()) if False else (None,None)
# evaluate() работает с LGBMClassifier; для CatBoost - короткий цикл
per,op=[],[]
for a,b in H.FOLDS:
    tr,va=(H.DAY<a).values,((H.DAY>=a)&(H.DAY<b)).values
    p=np.mean([CatBoostClassifier(**params,random_seed=s).fit(H.Xtr.loc[tr,cols],H.y[tr]).predict_proba(H.Xtr.loc[va,cols])[:,1] for s in (101,102,103)],0)
    per.append(precision_at_recall(H.y[va],p)); op.append(p)
print('CatBoost folds',np.round(per,4),'mean',round(np.mean(per),4))
AB=[(a+b)/2 for a,b in zip(o4['oA'],o4['oB'])]
ABC=[(a+b+c)/3 for a,b,c in zip(o4['oA'],o4['oB'],op)]
yf=[H.y[((H.DAY>=a)&(H.DAY<b)).values] for a,b in H.FOLDS]
for n,o in [('A+B (текущий финал)',AB),('A+B+CatBoost',ABC)]:
    print(n,[round(precision_at_recall(yv,p),4) for yv,p in zip(yf,o)],'mean',round(np.mean([precision_at_recall(yv,p) for yv,p in zip(yf,o)]),4))
H.paired_bootstrap(ABC,AB)
pickle.dump(op,open('oof_cat.pkl','wb'))
