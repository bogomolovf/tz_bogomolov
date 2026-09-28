"""Модели поверх расширенного набора (базовые + курсор + сессии + MIL v1)."""
from lab import *
import xgboost as xgb
from catboost import CatBoostClassifier
A=load_oof('A3'); S3=(42,43,44)
BC=FEATURES+EXTRA
ff=mil_fold_fn(BC,vers=('v1',)); ff12=mil_fold_fn(BC,vers=('v1','v2'))
def xgb_fp(seeds=S3):
    P=dict(n_estimators=800,learning_rate=0.02,max_depth=4,min_child_weight=3,subsample=0.8,colsample_bytree=0.6,reg_lambda=2.0,
           tree_method='hist',n_jobs=2,eval_metric='aucpr')
    return lambda Xf,yf,Xa:np.mean([xgb.XGBClassifier(**P,random_state=s).fit(Xf,yf).predict_proba(Xa)[:,1] for s in seeds],0)
def cat_fp(seeds=S3):
    P=dict(iterations=1000,learning_rate=0.04,depth=5,l2_leaf_reg=5,verbose=0,thread_count=2)
    return lambda Xf,yf,Xa:np.mean([CatBoostClassifier(**P,random_seed=s).fit(Xf,yf).predict_proba(Xa)[:,1] for s in seeds],0)
jobs=[('LGB A | ext+MILv1',lgb_fp(LGB_A),ff,'m_lgbA_v1'),
      ('LGB B | ext+MILv1',lgb_fp(LGB_B),ff,'m_lgbB_v1'),
      ('LGB A | ext+MILv1+v2',lgb_fp(LGB_A),ff12,'m_lgbA_v12'),
      ('XGBoost | ext+MILv1',xgb_fp(),ff,'m_xgb_v1'),
      ('CatBoost | ext+MILv1',cat_fp(),ff,'m_cat_v1')]
for name,fp,fn,key in jobs:
    t=time.time(); o=run(fp,None,fold_fn=fn); report(name,o,ref=A,group='models',save_as=key,note=f'{time.time()-t:.0f}s')
