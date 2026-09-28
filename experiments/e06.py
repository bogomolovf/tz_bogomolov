from lab import *
import mil
A=load_oof('A3')
T=pd.read_pickle('event_table.pkl')
cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row); Ttr=T[T.row.notna()].copy(); Ttr['row']=Ttr.row.astype(int); Ttr['y']=y[Ttr.row.values]
fc=pd.read_pickle('feat_pointer_feats.pkl'); fs=pd.read_pickle('feat_session_feats.pkl')
X2=Xtr.merge(fc,on='cookie_id',how='left').merge(fs,on='cookie_id',how='left')
EXTRA=[c for c in fc.columns if c!='cookie_id']+[c for c in fs.columns if c!='cookie_id']
o=run(lgb_fp(LGB_A),FEATURES+EXTRA,X=X2); report('+ курсор + сессии (без MIL)',o,ref=A,group='features',save_as='cur_sess')
def fold_fn(tr,va):
    ftr,fva=mil.mil_features(Ttr[tr[Ttr.row.values]],Ttr[va[Ttr.row.values]],mil.ECOLS_V1)
    return X2.loc[np.where(tr)[0],FEATURES+EXTRA].join(ftr), X2.loc[np.where(va)[0],FEATURES+EXTRA].join(fva)
o=run(lgb_fp(LGB_A),None,fold_fn=fold_fn); report('+ курсор + сессии + MIL v1',o,ref=A,group='features',save_as='cur_sess_mil1')
