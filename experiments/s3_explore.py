import harness as H, numpy as np, pandas as pd
pd.set_option('display.width',250)
ev=H.ev.merge(H.train[['cookie_id','target']],on='cookie_id').copy()
g=ev.groupby('cookie_id')
ev['dt_next']=-g.event_ts.diff(-1).dt.total_seconds()      # сколько "провёл" на событии до следующего
s=ev[ev.dt_next<120]
print(s.groupby(['event_name','target']).dt_next.median().unstack().round(1))
# per-cookie dwell ratio item vs search
d=s.groupby(['cookie_id','event_name']).dt_next.median().unstack()
c=d.join(H.train.set_index('cookie_id').target)
c['ratio']=c.item_view/c.search_results_view
print(c.groupby('target').ratio.describe())
# lag-1 autocorrelation & repeated dt
ev['dt']=g.event_ts.diff().dt.total_seconds()
sh=ev[ev.dt<60].copy(); sh['prev']=sh.groupby('cookie_id').dt.shift()
sh['same']=(sh.dt==sh.prev)
print(sh.groupby('target').same.mean())
sh['lr']=np.abs(np.log((sh.dt+0.5)/(sh.prev+0.5)))
print(sh.groupby('target').lr.describe())
print(pd.crosstab(ev.event_ts.dt.second//10,ev.target,normalize='columns').round(3))
