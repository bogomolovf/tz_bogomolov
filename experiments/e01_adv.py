"""Adversarial validation: можно ли отличить train от test по признакам? Если да - какие признаки "плывут"."""
from lab import *
from sklearn.model_selection import StratifiedKFold
Xall=pd.concat([Xtr[FEATURES],Xte[FEATURES]],ignore_index=True); t=np.r_[np.zeros(len(Xtr)),np.ones(len(Xte))]
p=np.zeros(len(t))
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(Xall,t):
    p[b]=lgb.LGBMClassifier(**LGB_A,random_state=0).fit(Xall.iloc[a],t[a]).predict_proba(Xall.iloc[b])[:,1]
print('Adversarial AUC train vs test:',round(roc_auc_score(t,p),4))
m=lgb.LGBMClassifier(**LGB_A,random_state=0).fit(Xall,t)
imp=pd.Series(m.booster_.feature_importance('gain'),FEATURES).sort_values(ascending=False)
print((imp/imp.sum()).head(10).round(3).to_string())
# по-признаково: AUC одного признака (насколько он сам по себе отличает train/test)
single={c:abs(roc_auc_score(t,Xall[c].fillna(-999))-0.5)+0.5 for c in FEATURES}
print(pd.Series(single).sort_values(ascending=False).head(10).round(3).to_string())
# доля ботов в тесте - оценка по модели (среднее предсказание A)
A5=load_oof('A5')
# а также внутри train: отличается ли последняя неделя от первой
w=(DAY>='2026-04-13').astype(int).values; p2=np.zeros(len(w))
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(Xtr,w):
    p2[b]=lgb.LGBMClassifier(**LGB_A,random_state=0).fit(Xtr.loc[a,FEATURES],w[a]).predict_proba(Xtr.loc[b,FEATURES])[:,1]
print('Adversarial AUC train неделя1 vs неделя2:',round(roc_auc_score(w,p2),4))
