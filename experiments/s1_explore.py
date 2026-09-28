import harness as H, numpy as np, pandas as pd
pd.set_option('display.width',250)
ev=H.ev.merge(H.train[['cookie_id','target']],on='cookie_id')
ua=ev.user_agent
ev['ua_os']=np.select([ua.str.contains('Windows'),ua.str.contains('Macintosh'),ua.str.contains('iPhone|iOS'),ua.str.contains('Android'),ua.str.contains('X11|Linux')],
                      ['windows','mac','ios','android','linux'],'none')
print(pd.crosstab([ev.platform,ev.ua_os],ev.target))
c=ev.groupby('cookie_id').agg(t=('target','first'),n_ua=('user_agent','nunique'),n_os=('ua_os','nunique'),n_pl=('platform','nunique'))
for k in ['n_ua','n_os','n_pl']: print(pd.crosstab(c[k],c.t))
# UA switches over time
ev['sw']=ev.groupby('cookie_id').user_agent.shift().ne(ev.user_agent)&ev.groupby('cookie_id').user_agent.shift().notna()
c['sw']=ev.groupby('cookie_id').sw.sum(); print(pd.crosstab(c.sw.clip(upper=5),c.t))
