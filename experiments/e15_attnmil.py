"""Attention-MIL (Ilse et al., 2018): эмбеддинг событий -> gated attention pooling -> классификатор куки."""
from lab import *
import torch, torch.nn as nn
torch.set_num_threads(2)
V2=load_oof('V2')
T=pd.read_pickle('event_table.pkl'); cid2row={c:i for i,c in enumerate(Xtr.cookie_id)}
T['row']=T.cookie_id.map(cid2row); T=T[T.row.notna()].copy(); T['row']=T.row.astype(int)
CAT=['etype','prev_etype','next_etype','plat','uac','cat','seller']
NUM=[c for c in T.columns if c not in CAT+['cookie_id','row']]
MAXLEN=128
T['k']=T.groupby('row').cumcount(); T=T[T.k<MAXLEN]
N=len(Xtr); L=np.maximum(1,np.bincount(T.row,minlength=N)).astype(np.int64)
Ci=np.zeros((N,MAXLEN,len(CAT)),np.int64); Xn=np.zeros((N,MAXLEN,2*len(NUM)),np.float32)
r,k=T.row.values,T.k.values
Ci[r,k]=(T[CAT].values+2).clip(0,63)
raw=T[NUM].values.astype(np.float64)
def build_num(fit_rows):
    # стандартизация по train-части фолда (медиана/IQR), пропуски -> 0 + флаг пропуска
    m=np.isin(r,fit_rows); med=np.nanmedian(raw[m],0); iqr=np.nanpercentile(raw[m],75,0)-np.nanpercentile(raw[m],25,0)+1e-6
    z=np.clip((raw-med)/iqr,-5,5); nanf=np.isnan(z); z[nanf]=0
    out=np.zeros_like(Xn); out[r,k]=np.c_[z,nanf].astype(np.float32); return out
class AttnMIL(nn.Module):
    def __init__(s,nnum,ncat,d=48):
        super().__init__(); s.emb=nn.ModuleList([nn.Embedding(64,6) for _ in range(ncat)])
        s.enc=nn.Sequential(nn.Linear(nnum+6*ncat,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU())
        s.V=nn.Linear(d,32); s.U=nn.Linear(d,32); s.w=nn.Linear(32,1)
        s.head=nn.Sequential(nn.Linear(2*d,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(s,c,x,l):
        h=s.enc(torch.cat([x]+[e(c[...,i]) for i,e in enumerate(s.emb)],-1))
        mask=torch.arange(MAXLEN)[None,:]<l[:,None]
        a=s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))).squeeze(-1).masked_fill(~mask,-1e9)
        a=torch.softmax(a,1)                                           # веса внимания по событиям
        att=(a[...,None]*h).sum(1); mx=h.masked_fill(~mask[...,None],-1e9).max(1).values
        return s.head(torch.cat([att,mx],-1)).squeeze(-1)
def fit_predict(tr,va,seed,epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(build_num(np.where(tr)[0])); C=torch.tensor(Ci); Lt=torch.tensor(L); yt=torch.tensor(y,dtype=torch.float32)
    net=AttnMIL(X.shape[-1],len(CAT)); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
    pw=torch.tensor(((1-y[tr].mean())/y[tr].mean())**0.5); idx=np.where(tr)[0]
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=epochs*int(np.ceil(len(idx)/128)))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for i in range(0,len(idx),128):
            b=idx[i:i+128]; opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(net(C[b],X[b],Lt[b]),yt[b],pos_weight=pw)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
    net.eval(); v=np.where(va)[0]
    with torch.no_grad(): return torch.sigmoid(net(C[v],X[v],Lt[v])).numpy()
t=time.time(); o={}
for sname,sch in [('FWD',FWD),('LDO',LDO)]:
    oo=np.full(N,np.nan)
    for tr,va in masks(sch): oo[va]=np.mean([fit_predict(tr,va,s) for s in (0,1,2)],0)
    o[sname]=oo; print(sname,round(time.time()-t),'s',flush=True)
report('attention-MIL NN (одна)',o,ref=V2,group='nn',save_as='attnmil')
rk=lambda a:pd.Series(a).rank(pct=True).values
for w in [0.1,0.2,0.3]:
    b={s:(1-w)*rk(V2[s][~np.isnan(V2[s])]) if False else None for s in ()}
    bl={}
    for s in ('FWD','LDO'):
        m=~np.isnan(V2[s]); z=np.full(N,np.nan)
        # смешиваем ранги внутри каждого фолда (у NN и бустинга разные шкалы вероятностей)
        for tr,va in masks(FWD if s=='FWD' else LDO):
            z[va]=(1-w)*rk(V2[s][va])+w*rk(o[s][va])+1e-9*V2[s][va]
        bl[s]=z
    report(f'v2 + attention-MIL, вес {w}',bl,ref=V2,group='nn')
