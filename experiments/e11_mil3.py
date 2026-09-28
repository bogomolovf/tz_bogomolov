from lab import *
import mil
REF=load_oof('m_lgbA_v1'); BC=FEATURES+EXTRA
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
t=time.time(); S=mil.segment_table(ev); print('segments',S.shape,round(time.time()-t),'s',flush=True); S.to_pickle('segment_table.pkl')
S['row']=S.cookie_id.map(cid2row); Str=S[S.row.notna()].copy(); Str['row']=Str.row.astype(int); Str['y']=y[Str.row.values]
SCOLS=[c for c in S.columns if c not in ('cookie_id','row','n')]
T=pd.read_pickle('event_table.pkl'); T['row']=T.cookie_id.map(cid2row); Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
cache=pickle.load(open('mil_cache.pkl','rb'))
BIG=dict(mil.EV_PARAMS,num_leaves=63,n_estimators=600,learning_rate=0.03,min_child_samples=30)
def mil_bag(Tf,Ta,ecols,params,seeds):
    """MIL-признаки с бэггингом event-модели по сидам (усредняем скоры событий)."""
    from sklearn.model_selection import GroupKFold
    pf=np.zeros(len(Tf))
    for a,b in GroupKFold(5).split(Tf,groups=Tf.row):
        pf[b]=np.mean([mil.fit_event_model(Tf.iloc[a],ecols,params,s).predict_proba(Tf.iloc[b][ecols])[:,1] for s in seeds],0)
    pa=np.mean([mil.fit_event_model(Tf,ecols,params,s).predict_proba(Ta[ecols])[:,1] for s in seeds],0)
    return mil.aggregate(Tf.row.values,pf),mil.aggregate(Ta.row.values,pa)
new_cache={}
def make_ff(tag,table,cols,params,seeds,with_v1=False):
    def fn(tr,va,sname,k):
        key=(tag,sname,k)
        if key not in new_cache:
            ftr,fva=mil_bag(table[tr[table.row.values]],table[va[table.row.values]],cols,params,seeds)
            new_cache[key]=(ftr.add_suffix('_'+tag),fva.add_suffix('_'+tag))
        ftr,fva=new_cache[key]
        Xf=X2.loc[np.where(tr)[0],BC].join(ftr); Xa=X2.loc[np.where(va)[0],BC].join(fva)
        if with_v1:
            a,b=cache[('v1',sname,k)]; Xf=Xf.join(a); Xa=Xa.join(b)
        return Xf,Xa
    return fn
jobs=[('MIL сегменты (8 событий)', make_ff('seg',Str,SCOLS,mil.EV_PARAMS,(0,)), 'm_lgbA_seg'),
      ('MIL сегменты + MIL v1', make_ff('seg',Str,SCOLS,mil.EV_PARAMS,(0,),with_v1=True), 'm_lgbA_seg_v1'),
      ('MIL v1, event-модель x3 сида', make_ff('v1b',Ttr,mil.ECOLS_V1,mil.EV_PARAMS,(0,1,2)), 'm_lgbA_v1bag'),
      ('MIL v1, большая event-модель', make_ff('v1big',Ttr,mil.ECOLS_V1,BIG,(0,)), 'm_lgbA_v1big')]
for name,fn,key in jobs:
    t=time.time(); o=run(lgb_fp(LGB_A),None,fold_fn=fn); report('LGB A | ext + '+name,o,ref=REF,group='mil',save_as=key,note=f'{time.time()-t:.0f}s')
    pickle.dump(new_cache,open('mil_cache2.pkl','wb'))
