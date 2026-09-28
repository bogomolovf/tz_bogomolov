"""Лоссы под ранжирование (P@R - ранговая метрика) + target encoding UA поверх нового набора."""
from lab import *
from sklearn.model_selection import StratifiedKFold
A=load_oof('A3'); REF=load_oof('m_lgbA_v1'); BC=FEATURES+EXTRA
ff=mil_fold_fn(BC,vers=('v1',))
# 1) ranking objectives: вся выборка = одна "группа" (упорядочиваем всех кук)
def rank_fp(obj,seeds=(42,43,44)):
    P={**LGB_A,'objective':obj}
    def fp(Xf,yf,Xa):
        return np.mean([lgb.LGBMRanker(**P,random_state=s).fit(Xf,yf,group=[len(yf)]).predict(Xa) for s in seeds],0)
    return fp
# 2) focal loss (Lin et al. 2017) через кастомную цель
def focal_fp(gamma=2.0,alpha=0.5,seeds=(42,43,44)):
    def obj(yt,raw):
        p=1/(1+np.exp(-raw)); a=np.where(yt==1,alpha,1-alpha); pt=np.where(yt==1,p,1-p); s=np.where(yt==1,1,-1)
        # градиент/гессиан focal loss по raw (численно устойчивая аппроксимация)
        g=a*s*(1-pt)**gamma*(gamma*pt*np.log(np.clip(pt,1e-12,1))-(1-pt))
        h=a*(1-pt)**gamma*pt*(1-pt)*np.maximum(1+gamma*(-np.log(np.clip(pt,1e-12,1)))-gamma,0.1)
        return g,h
    P={**LGB_A,'objective':obj}
    return lambda Xf,yf,Xa:np.mean([lgb.LGBMClassifier(**P,random_state=s).fit(Xf,yf).predict_proba(Xa,raw_score=True) for s in seeds],0)
# 3) class weights
def cw_fp(spw): return lgb_fp({**LGB_A,'scale_pos_weight':spw})
for name,fp in [('LGB lambdarank',rank_fp('lambdarank')),('LGB rank_xendcg',rank_fp('rank_xendcg')),
                ('LGB focal gamma=2',focal_fp()),('LGB scale_pos_weight=4',cw_fp(4))]:
    try:
        o=run(fp,None,fold_fn=ff); report(name+' | ext+MILv1',o,ref=REF,group='objective')
    except Exception as e: print(name,'ERROR',e)
# 4) UA target encoding внутри фолда (вложенный 5-fold для train-части)
cats=pd.read_pickle('cats_ua.pkl').set_index('cookie_id'); uf=pd.read_pickle('feat_ua.pkl')
Xu=X2.merge(uf,on='cookie_id',how='left'); UAC=[c for c in uf.columns if c!='cookie_id']
def te(kf,yf,ka,alpha=20):
    pr=yf.mean(); st=pd.DataFrame({'k':kf,'y':yf}).groupby('k').y.agg(['sum','count'])
    return pd.Series(ka).map((st['sum']+alpha*pr)/(st['count']+alpha)).fillna(pr).values
base_ff=mil_fold_fn(BC+UAC,X=Xu,vers=('v1',))
def ua_ff(tr,va,sname,k):
    Xf,Xa=base_ff(tr,va,sname,k); itr,iva=np.where(tr)[0],np.where(va)[0]
    for c in ['ua_string','ua_model','ua_os','ua_family']:
        kc=Xtr.cookie_id.map(cats[c]).values; Xa['te_'+c]=te(kc[itr],y[itr],kc[iva])
        col=np.zeros(len(itr))
        for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(itr,y[itr]):
            col[b]=te(kc[itr[a]],y[itr[a]],kc[itr[b]])
        Xf['te_'+c]=col
    return Xf,Xa
o=run(lgb_fp(LGB_A),None,fold_fn=ua_ff); report('LGB A | ext+MILv1 + UA (семейство, версия, TE)',o,ref=REF,group='features',save_as='m_lgbA_v1_ua')
