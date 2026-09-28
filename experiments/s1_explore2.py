import harness as H, numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
ev=H.ev.merge(H.train[['cookie_id','target']],on='cookie_id')
c=ev.groupby('cookie_id').agg(t=('target','first'),ua=('user_agent','first'))
ua=c.ua
c['fam']=np.select([ua.str.contains('HeadlessChrome'),ua.str.contains('YaBrowser'),ua.str.contains('Firefox'),ua.str.contains('Avito/'),ua.str.contains('Chrome'),ua.str.contains('Safari')],['headless','yandex','firefox','app','chrome','safari'],'lib')
c['ver']=ua.str.extract(r'(?:Chrome|Firefox|Avito|Version)/(\d+)')[0]
c['model']=ua.str.extract(r'; ([A-Z0-9\-]{5,})[;)]')[0]
c['osv']=ua.str.extract(r'(Android \d+|iOS \d+[\._]\d|OS \d+_\d)')[0]
def rate(col):
    r=c.groupby(col).t.agg(['mean','size']); print(r[r['size']>=40].sort_values('mean').round(3).to_string()); print()
rate('fam'); rate('model'); rate('osv'); rate(['fam','ver'])
# UA string-level: how many cookies share exact UA (train+test) vs bot rate
print(c.groupby('ua').t.agg(['mean','size']).sort_values('size').describe())
