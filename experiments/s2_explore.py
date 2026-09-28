import harness as H, numpy as np, pandas as pd
pd.set_option('display.width',250)
ev=H.ev.merge(H.train[['cookie_id','target']],on='cookie_id')
def spread(col):
    c=ev.dropna(subset=[col]).groupby([col,'cookie_id']).target.first().groupby(col).agg(['mean','size'])
    c=c[c['size']>=30]; p=ev.groupby('cookie_id').target.first().mean()
    noise=np.sqrt(p*(1-p)/c['size']).mean()
    print(f'{col:15s} n_cat={len(c):5d} std(bot rate)={c["mean"].std():.3f} vs binomial noise~{noise:.3f}  min/max={c["mean"].min():.3f}/{c["mean"].max():.3f}')
for col in ['search_query','item_location','item_category','seller_type','item_id']: spread(col)
# query x location combos
ev['ql']=ev.search_query+'|'+ev.item_location; spread('ql')
# synchronisation: same query+page viewed by another cookie within 2 min
s=H.ev[H.ev.event_name=='search_results_view'][['cookie_id','event_ts','search_query','search_page']].copy()
s['m']=s.event_ts.dt.floor('2min')
s['n_same']=s.groupby(['search_query','search_page','m']).cookie_id.transform('nunique')
c=s.groupby('cookie_id').n_same.mean().rename('sync').to_frame().join(H.train.set_index('cookie_id').target,how='inner')
print(c.groupby('target').sync.describe())
it=H.ev.dropna(subset=['item_id'])[['cookie_id','event_ts','item_id']].copy(); it['m']=it.event_ts.dt.floor('10min')
it['n']=it.groupby(['item_id','m']).cookie_id.transform('nunique')
c=it.groupby('cookie_id').n.apply(lambda s:(s>1).mean()).rename('item_sync').to_frame().join(H.train.set_index('cookie_id').target,how='inner')
print(c.groupby('target').item_sync.describe())
