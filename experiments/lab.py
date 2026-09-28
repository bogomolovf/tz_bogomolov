"""Экспериментальный стенд.

Две схемы валидации:
  * FWD  - "вперёд по времени", как в ноутбуке: 3 фолда, обучение только на прошлом (главная, честная схема);
  * LDO  - leave-2-days-out: 7 фолдов по 2 дня, обучение на остальных 12 днях (в т.ч. будущих).
           Не имитирует прод, но оценивает ВСЕ 899 ботов трейна => меньше шум при СРАВНЕНИИ вариантов.
Правило принятия изменения: delta > 0 в обеих схемах и P(delta>0) по парному бутстрапу (LDO) >= 0.9.
Все результаты дописываются в experiments/experiments.csv (журнал идёт в документацию).
"""
import os, sys, json, pickle, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from metric import precision_at_recall

_c = pickle.load(open(os.path.join(HERE, 'cache.pkl'), 'rb'))
train, test, meta, ev, F, FEATURES, Xtr, Xte, y, DAY = (_c[k] for k in
    ['train', 'test', 'meta', 'ev', 'F', 'FEATURES', 'Xtr', 'Xte', 'y', 'DAY'])
events = _c['events']
Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)

LGB_A = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True)
LGB_B = {**LGB_A, 'num_leaves': 7, 'n_estimators': 1500, 'learning_rate': 0.01, 'colsample_bytree': 0.5}

FWD = [('2026-04-10', '2026-04-13'), ('2026-04-13', '2026-04-16'), ('2026-04-16', '2026-04-20')]
_days = sorted(DAY.dt.normalize().unique())
LDO = [(_days[i], _days[i + 1] + pd.Timedelta(days=1)) for i in range(0, 14, 2)]


def masks(scheme):
    for a, b in scheme:
        va = ((DAY >= a) & (DAY < b)).values
        tr = (DAY < a).values if scheme is FWD else ~va
        yield tr, va


def lgb_fp(params=LGB_A, seeds=(42, 43, 44)):
    """Фабрика fit_predict для LightGBM с бэггингом по сидам."""
    def fp(Xf, yf, Xa, w=None):
        return np.mean([lgb.LGBMClassifier(**params, random_state=s).fit(Xf, yf, sample_weight=w).predict_proba(Xa)[:, 1]
                        for s in seeds], 0)
    return fp


def run(fp, cols, X=None, fold_fn=None, schemes=('FWD', 'LDO'), weights=None):
    """OOF-предсказания. fold_fn(tr_mask, va_mask) -> (X_fit, X_apply) позволяет строить признаки внутри фолда
    (target encoding, стекинг) без утечки. Возвращает {'FWD': oof, 'LDO': oof} (NaN вне валидации)."""
    X = Xtr if X is None else X
    out = {}
    for sname in schemes:
        oof = np.full(len(X), np.nan)
        for k, (tr, va) in enumerate(masks(FWD if sname == 'FWD' else LDO)):
            if fold_fn is not None:
                Xf, Xa = fold_fn(tr, va, sname, k) if fold_fn.__code__.co_argcount == 4 else fold_fn(tr, va)
            else:
                Xf, Xa = X.loc[tr, cols], X.loc[va, cols]
            w = None if weights is None else weights[tr]
            oof[va] = fp(Xf, y[tr], Xa, w) if w is not None else fp(Xf, y[tr], Xa)
        out[sname] = oof
    return out


def scores(oof):
    r = {}
    m = ~np.isnan(oof['FWD'])
    r['fwd_folds'] = [round(precision_at_recall(y[va], oof['FWD'][va]), 4) for _, va in masks(FWD)]
    r['fwd_mean'] = float(np.mean(r['fwd_folds']))
    r['fwd_pooled'] = precision_at_recall(y[m], oof['FWD'][m])
    if 'LDO' in oof:
        r['ldo_pooled'] = precision_at_recall(y, oof['LDO'])
        r['ldo_fold_mean'] = float(np.mean([precision_at_recall(y[va], oof['LDO'][va]) for _, va in masks(LDO)]))
        r['ldo_auc'] = roc_auc_score(y, oof['LDO']); r['ldo_prauc'] = average_precision_score(y, oof['LDO'])
    return r


def boot(new, ref, n=1000, seed=0):
    """Парный бутстрап delta по кукам (стратифицированно внутри фолдов FWD; по всем кукам для LDO)."""
    rng = np.random.default_rng(seed)
    res = {}
    fwd_va = [va for _, va in masks(FWD)]
    d = []
    for _ in range(n):
        dd = []
        for va in fwd_va:
            idx = np.where(va)[0]; i = rng.choice(idx, len(idx))
            dd.append(precision_at_recall(y[i], new['FWD'][i]) - precision_at_recall(y[i], ref['FWD'][i]))
        d.append(np.mean(dd))
    res['fwd_p'] = float(np.mean(np.array(d) > 0))
    if 'LDO' in new and 'LDO' in ref:
        d = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            d.append(precision_at_recall(y[i], new['LDO'][i]) - precision_at_recall(y[i], ref['LDO'][i]))
        d = np.array(d)
        res['ldo_p'] = float(np.mean(d > 0)); res['ldo_ci'] = (float(np.percentile(d, 5)), float(np.percentile(d, 95)))
    return res


LOG = os.path.join(HERE, 'experiments.csv')


def report(name, oof, ref=None, group='', note='', save_as=None):
    r = scores(oof)
    line = f"{name:52s} FWD mean={r['fwd_mean']:.4f} {r['fwd_folds']}"
    if 'ldo_pooled' in r:
        line += f" | LDO pooled={r['ldo_pooled']:.4f} AUC={r['ldo_auc']:.4f} PR-AUC={r['ldo_prauc']:.4f}"
    row = dict(time=time.strftime('%H:%M'), group=group, name=name, note=note, **{k: v for k, v in r.items() if k != 'fwd_folds'},
               fwd_folds=json.dumps(r['fwd_folds']))
    if ref is not None:
        rr = scores(ref); b = boot(oof, ref)
        row.update(d_fwd=r['fwd_mean'] - rr['fwd_mean'], p_fwd=b['fwd_p'])
        line += f"\n{'':52s} delta_fwd={row['d_fwd']:+.4f} P(delta>0)={b['fwd_p']:.2f}"
        if 'ldo_p' in b:
            row.update(d_ldo=r['ldo_pooled'] - rr['ldo_pooled'], p_ldo=b['ldo_p'], ldo_ci=json.dumps([round(v, 4) for v in b['ldo_ci']]))
            line += f" | delta_ldo={row['d_ldo']:+.4f} 90%CI[{b['ldo_ci'][0]:+.4f},{b['ldo_ci'][1]:+.4f}] P(delta>0)={b['ldo_p']:.2f}"
    print(line, flush=True)
    pd.DataFrame([row]).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
    if save_as:
        pickle.dump(oof, open(os.path.join(HERE, 'oof', save_as + '.pkl'), 'wb'))
    return r


def load_oof(name):
    return pickle.load(open(os.path.join(HERE, 'oof', name + '.pkl'), 'rb'))


os.makedirs(os.path.join(HERE, 'oof'), exist_ok=True)


# ---- расширенный набор признаков: базовые + курсор + сессии + MIL (из кэша)
FC = pd.read_pickle(os.path.join(HERE, 'feat_pointer_feats.pkl')) if os.path.exists(os.path.join(HERE, 'feat_pointer_feats.pkl')) else None
FS = pd.read_pickle(os.path.join(HERE, 'feat_session_feats.pkl')) if os.path.exists(os.path.join(HERE, 'feat_session_feats.pkl')) else None
if FC is not None:
    X2 = Xtr.merge(FC, on='cookie_id', how='left').merge(FS, on='cookie_id', how='left')
    X2te = Xte.merge(FC, on='cookie_id', how='left').merge(FS, on='cookie_id', how='left')
    EXTRA = [c for c in FC.columns if c != 'cookie_id'] + [c for c in FS.columns if c != 'cookie_id']
_MIL = None

def mil_fold_fn(base_cols, X=None, vers=('v1',)):
    """fold_fn, который добавляет кэшированные MIL-признаки нужных версий."""
    global _MIL
    if _MIL is None:
        _MIL = pickle.load(open(os.path.join(HERE, 'mil_cache.pkl'), 'rb'))
    X = X2 if X is None else X
    def fn(tr, va, sname, k):
        Xf = X.loc[np.where(tr)[0], base_cols]; Xa = X.loc[np.where(va)[0], base_cols]
        for v in vers:
            ftr, fva = _MIL[(v, sname, k)]
            Xf = Xf.join(ftr); Xa = Xa.join(fva)
        return Xf, Xa
    return fn
