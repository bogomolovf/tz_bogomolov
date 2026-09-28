"""Мобильные боты: (a) раздельные модели куки desktop/mobile, (b) event-модели раздельно по платформе события."""
from lab import *
import mil
from sklearn.model_selection import GroupKFold
REF=load_oof('m_lgbA_v1'); BC=FEATURES+EXTRA
MILC=pickle.load(open('mil_cache.pkl','rb'))
base_ff=mil_fold_fn(BC,vers=('v1',))
is_desk=(X2.plat_desktop>=0.5).values
print('desktop-кук',is_desk.mean().round(3),'| ботов среди mobile',y[~is_desk].mean().round(3),'desktop',y[is_desk].mean().round(3))
# (a) раздельные модели: каждая обучается на своей подвыборке; для маленькой mobile-выборки - общая модель как fallback-признак
def split_fp(Xf,yf,Xa):
    d_f=Xf.plat_desktop.values>=0.5; d_a=Xa.plat_desktop.values>=0.5
    p=np.zeros(len(Xa)); fp=lgb_fp(LGB_A)
    p[d_a]=fp(Xf[d_f],yf[d_f],Xa[d_a]); p[~d_a]=fp(Xf[~d_f],yf[~d_f],Xa[~d_a]); return p
# (b) event-модели раздельно для desktop- и mobile-событий, агрегаты объединяются
T=pd.read_pickle('event_table.pkl'); cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row); Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
desk_code=pd.Categorical(ev.sort_values(['cookie_id','event_ts','eid'],kind='mergesort').platform).categories.get_loc('desktop')
Ttr['is_desk']=(Ttr.plat==desk_code)
def mil_split(Tf,Ta):
    pf=np.zeros(len(Tf)); pa=np.zeros(len(Ta))
    for flag in (True,False):
        mf=(Tf.is_desk==flag).values; ma=(Ta.is_desk==flag).values
        F_,A_=Tf[mf],Ta[ma]; tmp=np.zeros(mf.sum())
        for a,b in GroupKFold(5).split(F_,groups=F_.row):
            tmp[b]=mil.fit_event_model(F_.iloc[a],mil.ECOLS_V1).predict_proba(F_.iloc[b][mil.ECOLS_V1])[:,1]
        pf[mf]=tmp; pa[ma]=mil.fit_event_model(F_,mil.ECOLS_V1).predict_proba(A_[mil.ECOLS_V1])[:,1]
    return mil.aggregate(Tf.row.values,pf).add_suffix('_plat'),mil.aggregate(Ta.row.values,pa).add_suffix('_plat')
C={}
def ffb(tr,va,sname,k,replace=True):
    if (sname,k) not in C: C[(sname,k)]=mil_split(Ttr[tr[Ttr.row.values]],Ttr[va[Ttr.row.values]])
    a,b=C[(sname,k)]
    Xf=X2.loc[np.where(tr)[0],BC].join(a); Xa=X2.loc[np.where(va)[0],BC].join(b)
    if not replace:
        c,d=MILC[('v1',sname,k)]; Xf=Xf.join(c); Xa=Xa.join(d)
    return Xf,Xa
o=run(lgb_fp(LGB_A),None,fold_fn=lambda tr,va,s,k:ffb(tr,va,s,k,True)); report('(b) MIL: event-модели раздельно по платформе',o,ref=REF,group='platform',save_as='m_milplat')
o=run(lgb_fp(LGB_A),None,fold_fn=lambda tr,va,s,k:ffb(tr,va,s,k,False)); report('(b2) MIL общий + MIL по платформам',o,ref=REF,group='platform',save_as='m_milplat2')
pickle.dump(C,open('mil_plat_cache.pkl','wb'))
