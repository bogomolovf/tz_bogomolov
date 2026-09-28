"""GRU по последовательности событий куки (внутри окна). ~50k параметров."""
import harness as H, numpy as np, pandas as pd, torch, torch.nn as nn, pickle, time
from metric import precision_at_recall
torch.set_num_threads(2)
MAXLEN=128
ev=H.ev.copy()
g=ev.groupby('cookie_id',sort=False)
ev['dt']=g.event_ts.diff().dt.total_seconds().fillna(0)
ev['etype']=ev.event_name.astype('category').cat.codes+1                  # 0 = padding
num=pd.DataFrame({
  'log_dt':np.log1p(ev.dt.clip(upper=36000))/10,
  'dt_3_10':ev.dt.between(3,10).astype(float),
  'desktop':(ev.platform=='desktop').astype(float),'android':(ev.platform=='android').astype(float),
  'has_ptr':ev.pointer_x.notna().astype(float),
  'px':(ev.pointer_x.fillna(-1)/1920),'py':(ev.pointer_y.fillna(-1)/1080),
  'page':np.log1p(ev.search_page.fillna(0))/4,
  'loc_change':(g.item_location.shift()!=ev.item_location).astype(float)*ev.item_location.notna(),
  'new_item':(~ev.duplicated(['cookie_id','item_id'])&ev.item_id.notna()).astype(float),
  'hour_sin':np.sin(2*np.pi*ev.event_ts.dt.hour/24),'hour_cos':np.cos(2*np.pi*ev.event_ts.dt.hour/24),
  'ua_bad':ev.ua_class.isin(['headless','http_lib']).astype(float),
})
NUMF=num.shape[1]
ids=H.Xtr.cookie_id.values
pos={c:i for i,c in enumerate(ids)}
E=np.zeros((len(ids),MAXLEN),np.int64); X=np.zeros((len(ids),MAXLEN,NUMF),np.float32); L=np.zeros(len(ids),np.int64)
ev['row']=ev.cookie_id.map(pos); ev['k']=g.cumcount()
m=ev.row.notna()&(ev.k<MAXLEN)                                             # первые 128 событий окна
r=ev.row[m].astype(int).values; k=ev.k[m].values
E[r,k]=ev.etype[m].values; X[r,k]=num[m].values.astype(np.float32)
L[:]=np.maximum(1,np.bincount(r,minlength=len(ids)))

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(12,8); s.gru=nn.GRU(8+NUMF,48,batch_first=True,bidirectional=True)
        s.head=nn.Sequential(nn.Linear(96*2,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(s,e,x,l):
        h=torch.cat([s.emb(e),x],-1)
        pk=nn.utils.rnn.pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False)
        o,_=s.gru(pk); o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True,total_length=MAXLEN)
        mask=(torch.arange(MAXLEN)[None,:]<l[:,None]).float()[...,None]
        mean=(o*mask).sum(1)/mask.sum(1); mx=(o-1e9*(1-mask)).max(1).values
        return s.head(torch.cat([mean,mx],-1)).squeeze(-1)

def train_predict(tr,va,seed,epochs=12):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-4)
    yt=torch.tensor(H.y,dtype=torch.float32); Et,Xt,Lt=torch.tensor(E),torch.tensor(X),torch.tensor(L)
    pw=torch.tensor((1-H.y[tr].mean())/H.y[tr].mean())**0.5
    idx=np.where(tr)[0]
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for i in range(0,len(idx),128):
            b=idx[i:i+128]; opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(net(Et[b],Xt[b],Lt[b]),yt[b],pos_weight=pw)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    net.eval(); v=np.where(va)[0]
    with torch.no_grad(): return torch.sigmoid(net(Et[v],Xt[v],Lt[v])).numpy()

if __name__=='__main__':
    print('params',sum(p.numel() for p in Net().parameters()))
    o4=pickle.load(open('oof_s4.pkl','rb'))
    yf=[H.y[((H.DAY>=a)&(H.DAY<b)).values] for a,b in H.FOLDS]
    t=time.time(); op=[]
    for a,b in H.FOLDS:
        tr,va=(H.DAY<a).values,((H.DAY>=a)&(H.DAY<b)).values
        op.append(np.mean([train_predict(tr,va,s) for s in (0,1,2)],0))
    print('time',round(time.time()-t))
    pickle.dump(op,open('oof_gru.pkl','wb'))
    sc=lambda o:[round(precision_at_recall(v,p),4) for v,p in zip(yf,o)]
    print('GRU alone',sc(op),np.mean(sc(op)))
    from sklearn.metrics import roc_auc_score
    print('GRU AUC',roc_auc_score(np.concatenate(yf),np.concatenate(op)))
    AB=[(a+b)/2 for a,b in zip(o4['oA'],o4['oB'])]
    for w in [0.15,0.3]:
        # смешиваем ранги внутри фолда (шкалы вероятностей у NN и GBM разные)
        bl=[(1-w)*pd.Series(ab).rank(pct=True).values+w*pd.Series(g_).rank(pct=True).values+1e-9*ab for ab,g_ in zip(AB,op)]
        print(f'A+B + GRU (вес {w})',sc(bl),round(np.mean(sc(bl)),4)); H.paired_bootstrap(bl,AB)
