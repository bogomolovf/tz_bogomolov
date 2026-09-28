import harness as H, numpy as np, pandas as pd, lightgbm as lgb, optuna, json
from sklearn.metrics import average_precision_score
from metric import precision_at_recall
optuna.logging.set_verbosity(optuna.logging.WARNING)
cols=H.FEATURES
def objective(trial):
    p=dict(n_estimators=trial.suggest_int('n_estimators',200,1500,step=100),
           learning_rate=trial.suggest_float('learning_rate',0.005,0.08,log=True),
           num_leaves=trial.suggest_int('num_leaves',4,48),
           min_child_samples=trial.suggest_int('min_child_samples',5,100,log=True),
           subsample=trial.suggest_float('subsample',0.5,1.0), subsample_freq=1,
           colsample_bytree=trial.suggest_float('colsample_bytree',0.2,0.9),
           reg_lambda=trial.suggest_float('reg_lambda',1e-3,30,log=True),
           reg_alpha=trial.suggest_float('reg_alpha',1e-3,10,log=True),
           min_split_gain=trial.suggest_float('min_split_gain',0,0.5),
           verbose=-1,n_jobs=2,deterministic=True,force_row_wise=True,random_state=42)
    ap,pr=[],[]
    for a,b in H.FOLDS:
        tr,va=(H.DAY<a).values,((H.DAY>=a)&(H.DAY<b)).values
        pp=lgb.LGBMClassifier(**p).fit(H.Xtr.loc[tr,cols],H.y[tr]).predict_proba(H.Xtr.loc[va,cols])[:,1]
        ap.append(average_precision_score(H.y[va],pp)); pr.append(precision_at_recall(H.y[va],pp))
    trial.set_user_attr('p_at_r',float(np.mean(pr)))
    return float(np.mean(ap))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
st.enqueue_trial(dict(n_estimators=500,learning_rate=0.02,num_leaves=15,min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_lambda=1.0,reg_alpha=0.001,min_split_gain=0))
st.optimize(objective,n_trials=60,show_progress_bar=False)
df=st.trials_dataframe().sort_values('value',ascending=False)
print(df[['number','value','user_attrs_p_at_r']].head(10).to_string())
print('trial0 (текущий A):',st.trials[0].value,st.trials[0].user_attrs)
json.dump([t.params for t in sorted(st.trials,key=lambda t:-t.value)[:5]],open('optuna_top5.json','w'),indent=1)
