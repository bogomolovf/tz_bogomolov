import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from metric import precision_at_recall   # официальная реализация метрики из архива

SEED = 42
np.random.seed(SEED)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)

DATES = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
train = pd.read_csv('data/train.csv', parse_dates=DATES)
test = pd.read_csv('data/test.csv', parse_dates=DATES)
events = pd.read_csv('data/events.csv.gz', parse_dates=['event_ts'])
meta = pd.concat([train, test], ignore_index=True)

(lambda *a,**k:None)('train', train.shape, 'test', test.shape, 'events', events.shape)
(lambda *a,**k:None)('доля ботов в train:', round(train.target.mean(), 4))
(lambda *a,**k:None)('окна train:', train.window_start_ts.min().date(), '-', train.window_start_ts.max().date())
(lambda *a,**k:None)('окна test: ', test.window_start_ts.min().date(), '-', test.window_start_ts.max().date())
(lambda *a,**k:None)('дубли cookie_id: train', train.cookie_id.duplicated().sum(), 'test', test.cookie_id.duplicated().sum(),
      '| пересечение train/test:', len(set(train.cookie_id) & set(test.cookie_id)))

PLATFORM_MAP = {'desktop': 'desktop', 'web': 'desktop', 'android': 'android', 'ios': 'ios', 'iphone': 'ios'}


def clean_events(events, meta):
    # точные дубли - артефакт логирования (одинаковая доля у ботов и людей)
    ev = events.drop_duplicates()
    ev = ev.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']], on='cookie_id')
    # строго [start, end): признаки должны быть доступны на момент окончания окна
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)].copy()
    ev['platform'] = ev.platform.str.lower().map(PLATFORM_MAP)
    # UA -> класс клиента (версии браузера/приложения - шум)
    ua = ev.user_agent
    ev['ua_class'] = np.select(
        [ua.str.contains('HeadlessChrome'),
         ~ua.str.startswith(('Mozilla', 'Avito/')),      # python-requests, curl, Scrapy, Go-http-client...
         ua.str.startswith('Avito/')],
        ['headless', 'http_lib', 'app'], 'browser')
    # файл перемешан: сортируем внутри куки по времени (eid - стабильный tie-break)
    return ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort').reset_index(drop=True)


ev = clean_events(events, meta)
assert ev.event_name.ne('captcha_shown').all()
(lambda *a,**k:None)('событий после очистки:', len(ev), '| кук с событиями:', ev.cookie_id.nunique(), 'из', len(meta))
(lambda *a,**k:None)(ev.platform.value_counts().to_string())
(lambda *a,**k:None)(pd.crosstab(ev.merge(train[['cookie_id', 'target']]).ua_class,
                  ev.merge(train[['cookie_id', 'target']]).target, normalize='index').round(3))

EVENTS = ['search_results_view', 'item_view', 'photo_swipe', 'seller_page_view', 'favorite_add',
          'contact_phone_show', 'contact_chat_open', 'contact_message_sent', 'login']
CONTACT = ['contact_phone_show', 'contact_chat_open', 'contact_message_sent']
SCRAPE_EVENTS = {'search_results_view', 'item_view', 'seller_page_view'}


def entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def by_cookie(values, cookie):
    # короткий помощник: группировка произвольного Series по cookie_id
    return values.groupby(cookie)


def build_features(ev, meta):
    ev = ev.copy()
    g = ev.groupby('cookie_id', sort=False)
    ck = ev.cookie_id
    ev['dt'] = g.event_ts.diff().dt.total_seconds()        # интервал до предыдущего события куки
    f = pd.DataFrame(index=pd.Index(meta.cookie_id, name='cookie_id'))

    # --- 1. Тайминг
    dtg = ev.dropna(subset=['dt']).groupby('cookie_id').dt
    f['dt_min'], f['dt_median'], f['dt_mean'], f['dt_std'] = dtg.min(), dtg.median(), dtg.mean(), dtg.std()
    f['dt_cv'] = f.dt_std / (f.dt_mean + 1e-9)
    f['dt_q10'], f['dt_q90'] = dtg.quantile(0.1), dtg.quantile(0.9)
    has_dt = ev.dt.notna()
    for name, cond in [('dt_3_10', ev.dt.between(3, 10)), ('dt_le1', ev.dt <= 1), ('dt_gt300', ev.dt > 300)]:
        f[f'share_{name}'] = cond[has_dt].groupby(ck[has_dt]).mean()
    short = ev.dt[ev.dt < 60]                                  # темп внутри активности, без долгих пауз
    sg = short.groupby(ck[short.index])
    f['dt_short_median'], f['dt_short_std'] = sg.median(), sg.std()
    f['dt_short_cv'] = f.dt_short_std / (sg.mean() + 1e-9)
    f['dt_short_mad'] = sg.agg(lambda s: np.median(np.abs(s - np.median(s))))
    span = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds()
    f['span_h'] = span / 3600
    f['n_events'] = g.size()
    f['events_per_min_active'] = f.n_events / (span / 60 + 1)
    f['max_events_5min'] = ev.groupby(['cookie_id', ev.event_ts.dt.floor('5min')]).size().groupby('cookie_id').max()
    f['n_sessions'] = by_cookie(ev.dt.isna() | (ev.dt > 1800), ck).sum()   # сессия = разрыв > 30 мин
    hour = ev.event_ts.dt.hour
    f['share_night'] = by_cookie(hour < 6, ck).mean()
    f['hour_nunique'] = by_cookie(hour, ck).nunique()
    f['hour_entropy'] = by_cookie(hour, ck).agg(entropy)

    # --- 2. Состав событий и переходы
    cnt = pd.crosstab(ck, ev.event_name).reindex(columns=EVENTS, fill_value=0)
    for e in EVENTS:
        f[f'share_{e}'] = cnt[e] / f.n_events
    f['n_contact'] = cnt[CONTACT].sum(1)
    f['share_contact'] = f.n_contact / f.n_events
    f['event_type_entropy'] = g.event_name.agg(entropy)
    f['photo_per_item'] = cnt['photo_swipe'] / (cnt['item_view'] + 1)
    f['search_per_item'] = cnt['search_results_view'] / (cnt['item_view'] + 1)
    prev = g.event_name.shift()
    hp = prev.notna()
    loop = prev.isin(SCRAPE_EVENTS) & ev.event_name.isin(SCRAPE_EVENTS)   # выдача/объявление/продавец без вовлечения
    f['share_scrape_transitions'] = loop[hp].groupby(ck[hp]).mean()
    f['share_same_event_repeat'] = (prev == ev.event_name)[hp].groupby(ck[hp]).mean()
    is_scr = ev.event_name.isin(SCRAPE_EVENTS)
    run_id = ((~is_scr) | (ck != ck.shift())).cumsum()
    runs = ev[is_scr].groupby(run_id[is_scr]).agg(c=('cookie_id', 'first'), n=('eid', 'size'))
    f['max_scrape_run'] = runs.groupby('c').n.max()

    # --- 3. Разнообразие контента
    f['item_nunique'] = g.item_id.nunique()
    f['item_repeat'] = g.item_id.count() / (f.item_nunique + 1e-9)
    f['cat_nunique'], f['loc_nunique'] = g.item_category.nunique(), g.item_location.nunique()
    f['loc_per_item'] = f.loc_nunique / (f.item_nunique + 1)
    f['cat_entropy'], f['loc_entropy'] = g.item_category.agg(entropy), g.item_location.agg(entropy)
    f['query_nunique'] = g.search_query.nunique()
    f['share_seller_pro'] = by_cookie(ev.seller_type == 'pro', ck).sum() / (by_cookie(ev.seller_type.notna(), ck).sum() + 1)
    it = ev.dropna(subset=['item_id'])
    pop = it.groupby([it.window_start_ts, it.item_id]).cookie_id.transform('nunique')   # сколько кук смотрели объявление в тот же день
    f['item_pop_mean'] = pop.groupby(it.cookie_id).mean()
    f['item_shared_share'] = (pop > 1).groupby(it.cookie_id).mean()

    # --- 4. Пагинация выдачи
    s = ev[ev.event_name == 'search_results_view']
    sp = s.groupby('cookie_id').search_page
    f['page_max'], f['page_mean'] = sp.max(), sp.mean()
    f['share_page_gt5'] = by_cookie(s.search_page > 5, s.cookie_id).mean()
    f['page_jump_mean'] = sp.diff().abs().groupby(s.cookie_id).mean()
    f['pages_per_query'] = s.groupby('cookie_id').size() / (s.groupby('cookie_id').search_query.nunique() + 1)
    f['search_loc_nunique'] = s.groupby('cookie_id').item_location.nunique()

    # --- 5. Курсор (есть только на desktop)
    d = ev[ev.platform == 'desktop']
    f['pointer_missing_desktop'] = by_cookie(d.pointer_x.isna(), d.cookie_id).mean()
    p = ev.dropna(subset=['pointer_x'])
    pg = p.groupby('cookie_id')
    f['ptr_x_std'], f['ptr_y_std'], f['ptr_x_mean'] = pg.pointer_x.std(), pg.pointer_y.std(), pg.pointer_x.mean()
    f['ptr_x_max'], f['ptr_y_max'] = pg.pointer_x.max(), pg.pointer_y.max()
    f['ptr_zero_share'] = by_cookie((p.pointer_x == 0) & (p.pointer_y == 0), p.cookie_id).mean()
    xy = p.pointer_x.astype(str) + '_' + p.pointer_y.astype(str)
    f['ptr_unique_share'] = by_cookie(xy, p.cookie_id).nunique() / pg.size()
    f['ptr_step_mean'] = np.hypot(pg.pointer_x.diff(), pg.pointer_y.diff()).groupby(p.cookie_id).mean()

    # --- 6. Клиент
    f['desktop_share'] = by_cookie(ev.platform == 'desktop', ck).mean()
    for pl in ['desktop', 'android', 'ios']:
        f[f'plat_{pl}'] = by_cookie(ev.platform == pl, ck).mean()
    for uc in ['headless', 'http_lib', 'app', 'browser']:
        f[f'ua_{uc}'] = by_cookie(ev.ua_class == uc, ck).mean()
    f['ua_nunique'] = g.user_agent.nunique()

    # --- 7. Кука
    m = meta.set_index('cookie_id')
    f['cookie_age_h'] = (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 3600
    f['cookie_new'] = (f.cookie_age_h < 0).astype(int)
    f['first_event_after_create_s'] = (g.event_ts.min() - m.cookie_created_at).dt.total_seconds()
    return f.reset_index()


F = build_features(ev, meta)
FEATURES = [c for c in F.columns if c != 'cookie_id']
Xtr = train[['cookie_id', 'window_start_ts', 'target']].merge(F, on='cookie_id', how='left')
Xte = test[['cookie_id']].merge(F, on='cookie_id', how='left')
assert len(Xtr) == len(train) and len(Xte) == len(test)
y = Xtr.target.values
(lambda *a,**k:None)('признаков:', len(FEATURES))
Xtr[FEATURES].describe().T.head(15)

DAY = Xtr.window_start_ts
FOLDS = [('2026-04-10', '2026-04-13'), ('2026-04-13', '2026-04-16'), ('2026-04-16', '2026-04-20')]
for a, b in FOLDS:
    va = (DAY >= a) & (DAY < b)
    (lambda *a,**k:None)(f'train < {a}: {int((DAY < a).sum()):5d} кук | valid [{a}, {b}): {int(va.sum()):4d} кук, ботов {int(y[va].sum())}')


def cross_val(make_models, cols, fillna=False):
    # make_models() -> список моделей; их предсказания усредняются (бэггинг по сидам)
    oof_y, oof_p, per_fold = [], [], []
    X = Xtr[cols].fillna(-1) if fillna else Xtr[cols]
    for a, b in FOLDS:
        tr_m, va_m = (DAY < a).values, ((DAY >= a) & (DAY < b)).values
        p = np.mean([m.fit(X[tr_m], y[tr_m]).predict_proba(X[va_m])[:, 1] for m in make_models()], axis=0)
        per_fold.append(precision_at_recall(y[va_m], p))
        oof_y.append(y[va_m]); oof_p.append(p)
    return per_fold, oof_y, oof_p


def summarize(name, per_fold, oof_y, oof_p):
    yy, pp = np.concatenate(oof_y), np.concatenate(oof_p)
    return dict(model=name, **{f'fold{i+1}': round(v, 4) for i, v in enumerate(per_fold)},
                mean_P_at_R07=round(np.mean(per_fold), 4), pooled_P_at_R07=round(precision_at_recall(yy, pp), 4),
                ROC_AUC=round(roc_auc_score(yy, pp), 4), PR_AUC=round(average_precision_score(yy, pp), 4))

LGB_A = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True)
LGB_B = {**LGB_A, 'num_leaves': 7, 'n_estimators': 1500, 'learning_rate': 0.01, 'colsample_bytree': 0.5}
SEEDS = [SEED + i for i in range(5)]

