"""Multi-instance learning: event-level LightGBM -> агрегаты скоров по куке. Используется в экспериментах и в финале."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

def event_table(ev, meta):
    """Признаки одного события с локальным контекстом (соседние интервалы, темп за последнюю минуту, курсор)."""
    e = ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort').reset_index(drop=True)
    g = e.groupby('cookie_id', sort=False)
    dt = g.event_ts.diff().dt.total_seconds()
    dt_prev2 = dt.groupby(e.cookie_id).shift()
    dn = -g.event_ts.diff(-1).dt.total_seconds()
    ts = e.event_ts.astype('int64') // 10**9
    # сколько событий куки было за последние 60 / 300 секунд (темп "здесь и сейчас")
    def n_last(sec):
        out = np.zeros(len(e))
        for _, idx in g.indices.items():
            t = ts.values[idx]; out[idx] = np.arange(len(t)) - np.searchsorted(t, t - sec, side='left')
        return out
    m = meta.set_index('cookie_id')
    T = pd.DataFrame({
        'cookie_id': e.cookie_id,
        'etype': e.event_name.astype('category').cat.codes,
        'prev_etype': g.event_name.shift().astype('category').cat.codes,
        'next_etype': g.event_name.shift(-1).astype('category').cat.codes,
        'dt_prev': np.log1p(dt), 'dt_next': np.log1p(dn), 'dt_prev2': np.log1p(dt_prev2),
        'dt_logratio': np.abs(np.log1p(dt) - np.log1p(dt_prev2)),
        'same_dt': (dt == dt_prev2).astype(float),
        'dt_roll_std': np.log1p(dt).groupby(e.cookie_id).transform(lambda s: s.rolling(5, min_periods=2).std()),
        'dt_roll_med': dt.groupby(e.cookie_id).transform(lambda s: s.rolling(5, min_periods=2).median()),
        'n_last60': n_last(60), 'n_last300': n_last(300),
        'pos': g.cumcount(), 'n': g.event_ts.transform('size'),
        'px': e.pointer_x, 'py': e.pointer_y,
        'pstep': np.hypot(g.pointer_x.diff(), g.pointer_y.diff()),
        'pspeed': np.hypot(g.pointer_x.diff(), g.pointer_y.diff()) / (dt + 1),
        'plat': e.platform.astype('category').cat.codes, 'uac': e.ua_class.astype('category').cat.codes,
        'page': e.search_page, 'page_diff': g.search_page.diff(),
        'loc_change': (g.item_location.shift() != e.item_location).astype(float),
        'same_item': (g.item_id.shift() == e.item_id).astype(float),
        'cat': e.item_category.astype('category').cat.codes,
        'seller': e.seller_type.astype('category').cat.codes,
        'hour': e.event_ts.dt.hour,
        'cookie_age_h': ((e.event_ts - e.cookie_id.map(m.cookie_created_at)).dt.total_seconds() / 3600).values,
    })
    return T

ECOLS_V1 = ['etype', 'prev_etype', 'dt_prev', 'dt_next', 'dt_roll_std', 'dt_roll_med', 'same_dt', 'pos', 'n', 'px', 'py', 'pstep',
            'plat', 'uac', 'page', 'page_diff', 'loc_change', 'hour']
EV_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True)

def aggregate(rows, p):
    lo = np.log(p / (1 - p))
    d = pd.DataFrame({'row': rows, 'lo': lo, 'p': p}).groupby('row')
    return pd.DataFrame({'mil_mean_logit': d.lo.mean(), 'mil_std_logit': d.lo.std(), 'mil_max': d.p.max(),
                         'mil_q90': d.p.quantile(.9), 'mil_q50': d.p.median(), 'mil_q10': d.p.quantile(.1),
                         'mil_share_hi': d.p.apply(lambda s: (s > 0.5).mean())})

def fit_event_model(Te, ecols, params=EV_PARAMS, seed=0):
    # вес события 1/n_events: каждая кука вносит одинаковый вклад, длинные сессии не доминируют
    return lgb.LGBMClassifier(**params, random_state=seed).fit(Te[ecols], Te.y, sample_weight=1.0 / Te.n.values)

def mil_features(T_fit, T_apply, ecols, inner_folds=5):
    """Признаки для train-кук - out-of-fold (GroupKFold по куке), для apply-кук - моделью на всём T_fit."""
    parts = []
    for a, b in GroupKFold(inner_folds).split(T_fit, groups=T_fit.row):
        m = fit_event_model(T_fit.iloc[a], ecols)
        parts.append(aggregate(T_fit.row.values[b], m.predict_proba(T_fit.iloc[b][ecols])[:, 1]))
    f_fit = pd.concat(parts)
    m = fit_event_model(T_fit, ecols)
    f_apply = aggregate(T_apply.row.values, m.predict_proba(T_apply[ecols])[:, 1])
    return f_fit, f_apply


def segment_table(ev, win=8, stride=4):
    """Инстансы = окна из `win` подряд идущих событий куки (шаг `stride`); у коротких кук - одно окно из всех событий."""
    e = ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort').reset_index(drop=True)
    g = e.groupby('cookie_id', sort=False)
    e['dt'] = g.event_ts.diff().dt.total_seconds()
    e['same_dt'] = (e.dt == g.dt.shift()).astype(float)
    e['pstep'] = np.hypot(g.pointer_x.diff(), g.pointer_y.diff())
    e['locch'] = (g.item_location.shift() != e.item_location).astype(float)
    et = pd.get_dummies(e.event_name).astype(float)
    rows = []
    for cid, idx in g.indices.items():
        n = len(idx)
        starts = [0] if n <= win else list(range(0, n - win + 1, stride))
        for s in starts:
            w = idx[s:s + win]
            d = e.dt.values[w][1:]; d = d[~np.isnan(d)]
            px = e.pointer_x.values[w]
            r = {'cookie_id': cid, 'n_cookie': n, 'len': len(w)}
            r.update({f'seg_{k}': v for k, v in zip(et.columns, et.values[w].mean(0))})
            if len(d):
                r.update(seg_dt_med=np.median(d), seg_dt_std=d.std(), seg_dt_min=d.min(), seg_dt_max=d.max(),
                         seg_dt_3_10=np.mean((d >= 3) & (d <= 10)), seg_dur=d.sum())
            r.update(seg_same_dt=e.same_dt.values[w].mean(), seg_ptr_present=np.mean(~np.isnan(px)),
                     seg_ptr_std=np.nanstd(px) if np.any(~np.isnan(px)) else np.nan,
                     seg_pstep=np.nanmean(e.pstep.values[w]) if np.any(~np.isnan(e.pstep.values[w])) else np.nan,
                     seg_page_max=np.nanmax(e.search_page.values[w]) if np.any(~np.isnan(e.search_page.values[w])) else np.nan,
                     seg_locch=e.locch.values[w].mean(), seg_items=len(set(e.item_id.values[w][~pd.isna(e.item_id.values[w])])),
                     seg_desktop=np.mean(e.platform.values[w] == 'desktop'), seg_pos=s / n)
            rows.append(r)
    T = pd.DataFrame(rows)
    T['n'] = T.groupby('cookie_id').cookie_id.transform('size')   # число инстансов куки (для веса 1/n)
    return T
