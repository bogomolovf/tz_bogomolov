import harness as H, json, pickle
o=pickle.load(open('oof_s1.pkl','rb'))
top=json.load(open('optuna_top5.json'))
seeds=(101,102,103)
_,oA=H.evaluate(H.FEATURES,'A (текущий), сиды 101-103',seeds=seeds)
_,oB=H.evaluate(H.FEATURES,'B (текущий), сиды 101-103',params=H.LGB_B,seeds=seeds)
res={}
for i,p in enumerate(top):
    params={**p,'subsample_freq':1,'verbose':-1,'n_jobs':2,'deterministic':True,'force_row_wise':True}
    _,oo=H.evaluate(H.FEATURES,f'optuna #{i+1} (lr={p["learning_rate"]:.3f}, leaves={p["num_leaves"]})',params=params,seeds=seeds)
    H.paired_bootstrap(oo,oA); res[i]=(params,oo)
pickle.dump(dict(oA=oA,oB=oB,res=res),open('oof_s4.pkl','wb'))
