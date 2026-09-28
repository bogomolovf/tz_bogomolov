from lab import *
import mil
A=load_oof('A3')
t=time.time(); T=mil.event_table(ev,meta); print('event table',round(time.time()-t),'s',flush=True)
T.to_pickle('event_table.pkl')
ECOLS_V2=[c for c in T.columns if c!='cookie_id']
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row); Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
fc=pd.read_pickle('feat_pointer_feats.pkl'); fs=pd.read_pickle('feat_session_feats.pkl')
X2=Xtr.merge(fc,on='cookie_id',how='left').merge(fs,on='cookie_id',how='left')
EXTRA=[c for c in fc.columns if c!='cookie_id']+[c for c in fs.columns if c!='cookie_id']
def make_fold_fn(ecols,base_cols,X):
    def fold_fn(tr,va):
        ftr,fva=mil.mil_features(Ttr[tr[Ttr.row.values]],Ttr[va[Ttr.row.values]],ecols)
        return X.loc[np.where(tr)[0],base_cols].join(ftr), X.loc[np.where(va)[0],base_cols].join(fva)
    return fold_fn
for name,ecols,bc,X in [('+ MIL v2 (богаче контекст события)',ECOLS_V2,FEATURES,Xtr),
                        ('+ курсор + сессии + MIL v2',ECOLS_V2,FEATURES+EXTRA,X2)]:
    t=time.time(); o=run(lgb_fp(LGB_A),None,fold_fn=make_fold_fn(ecols,bc,X))
    report(name,o,ref=A,group='features',save_as=name.replace(' ','_').replace('+','p'),note=f'{time.time()-t:.0f}s')
