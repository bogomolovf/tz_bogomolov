"""Новые группы признаков (все - по событиям окна, без таргета)."""
from lab import *

def entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum()) if len(p) else np.nan

def pointer_feats(ev=ev):
    """Траектория курсора: скорость, прямолинейность, распределение по экрану, "сетка"."""
    p = ev.dropna(subset=['pointer_x'])[['cookie_id', 'event_ts', 'pointer_x', 'pointer_y']].copy()
    g = p.groupby('cookie_id', sort=False)
    dx, dy = g.pointer_x.diff(), g.pointer_y.diff()
    dt = g.event_ts.diff().dt.total_seconds()
    p['step'] = np.hypot(dx, dy); p['speed'] = p.step / (dt + 1)
    p['ang'] = np.arctan2(dy, dx); p['turn'] = np.abs(np.angle(np.exp(1j * (p.ang - p.groupby('cookie_id').ang.shift()))))
    f = pd.DataFrame(index=pd.Index(meta.cookie_id, name='cookie_id'))
    G = p.groupby('cookie_id')
    for q in [0.1, 0.5, 0.9]:
        f[f'ptr_x_q{int(q*100)}'] = G.pointer_x.quantile(q); f[f'ptr_y_q{int(q*100)}'] = G.pointer_y.quantile(q)
    f['ptr_x_range'] = G.pointer_x.max() - G.pointer_x.min(); f['ptr_y_range'] = G.pointer_y.max() - G.pointer_y.min()
    f['ptr_xy_corr'] = G.apply(lambda d: d.pointer_x.corr(d.pointer_y) if len(d) >= 4 else np.nan)
    f['ptr_speed_median'] = G.speed.median(); f['ptr_speed_cv'] = G.speed.std() / (G.speed.mean() + 1e-9)
    f['ptr_step_cv'] = G.step.std() / (G.step.mean() + 1e-9)
    f['ptr_turn_mean'] = G.turn.mean()
    f['ptr_same_as_prev'] = G.step.apply(lambda s: (s == 0).mean())
    f['ptr_round10_share'] = G.apply(lambda d: ((d.pointer_x % 10 == 0) & (d.pointer_y % 10 == 0)).mean())
    f['ptr_edge_share'] = G.apply(lambda d: ((d.pointer_x < 5) | (d.pointer_y < 5) | (d.pointer_x > 1915) | (d.pointer_y > 1075)).mean())
    f['ptr_cells_entropy'] = G.apply(lambda d: entropy((d.pointer_x // 240).astype(int).astype(str) + '_' + (d.pointer_y // 135).astype(int).astype(str)))
    f['ptr_n'] = G.size()
    return f.reset_index()

def ngram_feats(ev=ev, min_df=0.01):
    """Нормированные частоты биграмм типов событий + биграмм "событие|интервал" (скрипты = повторяющиеся шаблоны)."""
    e = ev[['cookie_id', 'event_name', 'event_ts']].copy()
    g = e.groupby('cookie_id', sort=False)
    e['prev'] = g.event_name.shift()
    dt = g.event_ts.diff().dt.total_seconds()
    e['dtb'] = pd.cut(dt, [-1, 2, 10, 60, 600, 1e9], labels=['0-2', '3-10', '11-60', '1-10m', '10m+']).astype(str)
    e = e.dropna(subset=['prev'])
    ab = e.prev.str[:6] + '>' + e.event_name.str[:6]
    ct = pd.crosstab(e.cookie_id, ab)
    ct = ct.loc[:, (ct > 0).mean() > min_df]
    ct = ct.div(ct.sum(1), axis=0).add_prefix('bg_')
    ct2 = pd.crosstab(e.cookie_id, e.event_name.str[:6] + '|' + e.dtb)
    ct2 = ct2.loc[:, (ct2 > 0).mean() > min_df]
    ct2 = ct2.div(ct2.sum(1), axis=0).add_prefix('edt_')
    f = pd.DataFrame(index=pd.Index(meta.cookie_id, name='cookie_id')).join(ct).join(ct2)
    return f.reset_index()

def session_feats(ev=ev, gap=1800):
    """Сессии (разрыв > 30 мин): длина, темп, однородность между сессиями."""
    e = ev[['cookie_id', 'event_ts', 'event_name']].copy()
    g = e.groupby('cookie_id', sort=False)
    dt = g.event_ts.diff().dt.total_seconds()
    e['sid'] = (dt.isna() | (dt > gap)).groupby(e.cookie_id).cumsum()
    s = e.groupby(['cookie_id', 'sid']).agg(n=('event_ts', 'size'), t0=('event_ts', 'min'), t1=('event_ts', 'max'),
                                             ntypes=('event_name', 'nunique'))
    s['dur'] = (s.t1 - s.t0).dt.total_seconds(); s['rate'] = s.n / (s.dur / 60 + 1)
    G = s.groupby('cookie_id')
    f = pd.DataFrame(index=pd.Index(meta.cookie_id, name='cookie_id'))
    f['sess_n_max'] = G.n.max(); f['sess_n_mean'] = G.n.mean(); f['sess_n_cv'] = G.n.std() / (G.n.mean() + 1e-9)
    f['sess_dur_max'] = G.dur.max(); f['sess_dur_mean'] = G.dur.mean()
    f['sess_rate_max'] = G.rate.max(); f['sess_rate_cv'] = G.rate.std() / (G.rate.mean() + 1e-9)
    f['sess_types_mean'] = G.ntypes.mean()
    f['sess_single_share'] = G.n.apply(lambda x: (x == 1).mean())
    gaps = s.reset_index().groupby('cookie_id').t0.diff().dt.total_seconds()
    f['sess_gap_cv'] = gaps.groupby(s.reset_index().cookie_id).agg(lambda x: x.std() / (x.mean() + 1e-9))
    return f.reset_index()

def ua_feats(ev=ev):
    """Детали User-Agent: семейство, мажорная версия, модель устройства, версия ОС (+ сырые категории для TE)."""
    ua = ev.groupby('cookie_id').user_agent.agg(lambda s: s.mode().iloc[0])
    fam = pd.Series(np.select([ua.str.contains('HeadlessChrome'), ~ua.str.startswith(('Mozilla', 'Avito/')), ua.str.contains('YaBrowser'),
                               ua.str.contains('Firefox'), ua.str.contains('Avito/'), ua.str.contains('Chrome'), ua.str.contains('Safari')],
                              ['headless', 'lib', 'yandex', 'firefox', 'app', 'chrome', 'safari'], 'other'), index=ua.index)
    f = pd.DataFrame({'cookie_id': ua.index, 'ua_major_version': ua.str.extract(r'(?:Chrome|Firefox|Avito|Version)/(\d+)')[0].astype(float).values})
    for x in ['headless', 'lib', 'yandex', 'firefox', 'app', 'chrome', 'safari']:
        f[f'ua_fam_{x}'] = (fam == x).astype(int).values
    cats = pd.DataFrame({'cookie_id': ua.index, 'ua_string': ua.values, 'ua_family': fam.values,
                         'ua_model': ua.str.extract(r'; ([A-Z0-9\-]{5,})[;)]')[0].fillna('none').values,
                         'ua_os': ua.str.extract(r'(Android \d+|iOS \d+[\._]\d|OS \d+_\d|Windows NT [\d.]+|Mac OS X [\d_]+|X11)')[0].fillna('none').values})
    return f, cats
