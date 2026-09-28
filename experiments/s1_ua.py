import harness as H, numpy as np, pandas as pd
ev=H.ev
first=ev.groupby('cookie_id').user_agent.agg(lambda s:s.mode().iloc[0])   # основной UA куки
ua=first
fam=pd.Series(np.select([ua.str.contains('HeadlessChrome'),~ua.str.startswith(('Mozilla','Avito/')),ua.str.contains('YaBrowser'),ua.str.contains('Firefox'),ua.str.contains('Avito/'),ua.str.contains('Chrome'),ua.str.contains('Safari')],
                        ['headless','lib','yandex','firefox','app','chrome','safari'],'other'),index=ua.index)
ver=ua.str.extract(r'(?:Chrome|Firefox|Avito|Version)/(\d+)')[0].astype(float)
model=ua.str.extract(r'; ([A-Z0-9\-]{5,})[;)]')[0].fillna('none')
osv=ua.str.extract(r'(Android \d+|iOS \d+[\._]\d|OS \d+_\d|Windows NT [\d.]+|Mac OS X [\d_]+|X11)')[0].fillna('none').str.replace('_','.')
F=pd.DataFrame({'cookie_id':ua.index,
   'ua_major_version':ver.values,
   **{f'ua_fam_{f}':(fam==f).astype(int).values for f in ['headless','lib','yandex','firefox','app','chrome','safari']}})
new=H.add(F)
base=H.FEATURES
H.evaluate(base,'0. текущее')
H.evaluate(base+new,'1a. + семейство браузера и версия (без TE)')
H.evaluate_te(base,{'te_ua_family':fam},'1b. + TE(семейство)')
H.evaluate_te(base,{'te_ua_family':fam,'te_device_model':model,'te_os_version':osv},'1c. + TE(семейство, модель, версия ОС)')
H.evaluate_te(base,{'te_ua_string':ua},'1d. + TE(полная строка UA)')
H.evaluate_te(base+['ua_major_version'],{'te_ua_family':fam,'te_device_model':model,'te_os_version':osv,'te_ua_string':ua},'1e. всё вместе')
