"""Общий стенд для экспериментов: признаки из solution.ipynb (кэш) + одинаковая оценка."""
import os, sys, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = 'experiments/cache.pkl'
if os.path.exists(CACHE):
    globals().update(pickle.load(open(CACHE, 'rb')))
    import numpy as np, pandas as pd, lightgbm as lgb
    from metric import precision_at_recall
    from sklearn.metrics import roc_auc_score, average_precision_score
    exec(open('experiments/base.py').read().split("DATES =")[0])          # импорты
    exec("def cross_val" + open('experiments/base.py').read().split("def cross_val")[1].split("def summarize")[0])
    exec("def summarize" + open('experiments/base.py').read().split("def summarize")[1].split("LGB_A =")[0])
    exec("LGB_A =" + open('experiments/base.py').read().split("LGB_A =")[1])
else:
    exec(open('experiments/base.py').read())
    pickle.dump(dict(train=train, test=test, events=events, meta=meta, ev=ev, F=F, FEATURES=FEATURES,
                     Xtr=Xtr, Xte=Xte, y=y, DAY=DAY, FOLDS=FOLDS, PLATFORM_MAP=PLATFORM_MAP), open(CACHE, 'wb'))

import numpy as np, pandas as pd, lightgbm as lgb

def add(Fnew):
    """Добавить новые признаки (DataFrame с cookie_id) в Xtr/Xte, вернуть их имена."""
    global Xtr, Xte
    cols = [c for c in Fnew.columns if c != 'cookie_id']
    Xtr = Xtr.drop(columns=[c for c in cols if c in Xtr]).merge(Fnew, on='cookie_id', how='left')
    Xte = Xte.drop(columns=[c for c in cols if c in Xte]).merge(Fnew, on='cookie_id', how='left')
    globals()['Xtr'], globals()['Xte'] = Xtr, Xte
    return cols

def evaluate(cols, name, params=None, seeds=(42, 43, 44), X=None, yv=None):
    """LightGBM (по умолчанию конфиг A, 3 сида) на тех же 3 временных фолдах."""
    X = Xtr if X is None else X
    yv = y if yv is None else yv
    params = params or LGB_A
    per, oy, op = [], [], []
    for a, b in FOLDS:
        tr, va = (DAY < a).values, ((DAY >= a) & (DAY < b)).values
        p = np.mean([lgb.LGBMClassifier(**params, random_state=s).fit(X.loc[tr, cols], yv[tr]).predict_proba(X.loc[va, cols])[:, 1] for s in seeds], 0)
        per.append(precision_at_recall(yv[va], p)); oy.append(yv[va]); op.append(p)
    r = summarize(name, per, oy, op)
    print(f"{name:45s} folds={[r['fold1'], r['fold2'], r['fold3']]} mean={r['mean_P_at_R07']:.4f} pooled={r['pooled_P_at_R07']:.4f} AUC={r['ROC_AUC']:.4f} PR-AUC={r['PR_AUC']:.4f}", flush=True)
    return r, op


def evaluate_te(cols, te_specs, name, params=None, seeds=(42, 43, 44), alpha=20):
    """Как evaluate, но с out-of-fold target encoding.
    te_specs: {имя_признака: Series(cookie_id -> категория)}. Кодирование считается ТОЛЬКО по train-части фолда
    (сглаженное среднее таргета к глобальному), внутри train-части - через 5-fold, чтобы не переобучаться на себе."""
    from sklearn.model_selection import StratifiedKFold
    params = params or LGB_A
    per, oy, op = [], [], []
    for a, b in FOLDS:
        tr, va = (DAY < a).values, ((DAY >= a) & (DAY < b)).values
        X = Xtr[['cookie_id'] + cols].copy()
        for fname, cat in te_specs.items():
            X[fname] = np.nan
            k = X.cookie_id.map(cat)
            X.loc[va, fname] = te_fit_apply(k[tr], y[tr], k[va], alpha)
            idx = np.where(tr)[0]
            for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx]):
                X.loc[idx[i_app], fname] = te_fit_apply(k.iloc[idx[i_fit]], y[idx[i_fit]], k.iloc[idx[i_app]], alpha)
        fc = cols + list(te_specs)
        p = np.mean([lgb.LGBMClassifier(**params, random_state=s).fit(X.loc[tr, fc], y[tr]).predict_proba(X.loc[va, fc])[:, 1] for s in seeds], 0)
        per.append(precision_at_recall(y[va], p)); oy.append(y[va]); op.append(p)
    r = summarize(name, per, oy, op)
    print(f"{name:45s} folds={[r['fold1'], r['fold2'], r['fold3']]} mean={r['mean_P_at_R07']:.4f} pooled={r['pooled_P_at_R07']:.4f} AUC={r['ROC_AUC']:.4f} PR-AUC={r['PR_AUC']:.4f}", flush=True)
    return r, op


def te_fit_apply(k_fit, y_fit, k_apply, alpha=20):
    prior = np.mean(y_fit)
    st = pd.DataFrame({'k': np.asarray(k_fit), 'y': np.asarray(y_fit)}).groupby('k').y.agg(['sum', 'count'])
    enc = (st['sum'] + alpha * prior) / (st['count'] + alpha)
    return pd.Series(np.asarray(k_apply)).map(enc).fillna(prior).values


def paired_bootstrap(op_new, op_old, n=1000, seed=0):
    """Парный бутстрап: ресэмплим куки внутри каждого фолда, считаем delta(mean P@R по фолдам).
    Возвращает (delta, 90% ДИ, доля ресэмплов с delta>0)."""
    rng = np.random.default_rng(seed)
    yf = [y[((DAY >= a) & (DAY < b)).values] for a, b in FOLDS]
    diffs = []
    for _ in range(n):
        d = []
        for yv, pn, po in zip(yf, op_new, op_old):
            i = rng.integers(0, len(yv), len(yv))
            d.append(precision_at_recall(yv[i], pn[i]) - precision_at_recall(yv[i], po[i]))
        diffs.append(np.mean(d))
    diffs = np.array(diffs)
    base = np.mean([precision_at_recall(yv, pn) - precision_at_recall(yv, po) for yv, pn, po in zip(yf, op_new, op_old)])
    lo, hi = np.percentile(diffs, [5, 95])
    print(f"   delta={base:+.4f}  90% ДИ [{lo:+.4f}, {hi:+.4f}]  P(delta>0)={np.mean(diffs > 0):.2f}", flush=True)
    return base, lo, hi, np.mean(diffs > 0)


def evaluate_stack(cols, fold_feats, name, params=None, seeds=(42, 43, 44), X=None):
    """Как evaluate, но fold_feats(X_fit, y_fit, X_apply) -> DataFrame новых признаков считается внутри фолда:
    для train-части - 5-fold out-of-fold, для валидации - по всей train-части."""
    from sklearn.model_selection import StratifiedKFold
    params = params or LGB_A
    X0 = Xtr if X is None else X
    per, oy, op = [], [], []
    for a, b in FOLDS:
        tr, va = (DAY < a).values, ((DAY >= a) & (DAY < b)).values
        idx = np.where(tr)[0]
        parts = []
        for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx]):
            parts.append(fold_feats(X0.iloc[idx[i_fit]], y[idx[i_fit]], X0.iloc[idx[i_app]]).set_index(X0.index[idx[i_app]]))
        f_tr = pd.concat(parts).loc[X0.index[idx]]
        f_va = fold_feats(X0[tr], y[tr], X0[va]).set_index(X0.index[va])
        fc = cols + list(f_tr.columns)
        Xa = pd.concat([X0.loc[tr, cols].join(f_tr), X0.loc[va, cols].join(f_va)])
        p = np.mean([lgb.LGBMClassifier(**params, random_state=s).fit(Xa.loc[X0.index[tr], fc], y[tr]).predict_proba(Xa.loc[X0.index[va], fc])[:, 1] for s in seeds], 0)
        per.append(precision_at_recall(y[va], p)); oy.append(y[va]); op.append(p)
    r = summarize(name, per, oy, op)
    print(f"{name:45s} folds={[r['fold1'], r['fold2'], r['fold3']]} mean={r['mean_P_at_R07']:.4f} pooled={r['pooled_P_at_R07']:.4f} AUC={r['ROC_AUC']:.4f} PR-AUC={r['PR_AUC']:.4f}", flush=True)
    return r, op
