import harness as H, numpy as np, pandas as pd, pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
o=pickle.load(open('oof_s1.pkl','rb'))
base=H.FEATURES
# признаки для kNN: сильные поведенческие группы (тайминг, курсор, контент, пагинация)
KNN_COLS=['dt_median','dt_q10','dt_q90','dt_short_median','dt_short_cv','share_dt_3_10','max_events_5min','n_events',
          'loc_nunique','cat_nunique','cat_entropy','item_repeat','page_mean','page_max','share_contact','share_photo_swipe',
          'share_favorite_add','share_scrape_transitions','desktop_share','pointer_missing_desktop','ptr_x_std','ptr_x_max','ptr_step_mean']
def knn_feats(k_list=(10,50)):
    def f(Xf, yf, Xa):
        qt=QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=0)
        A=qt.fit_transform(Xf[KNN_COLS].fillna(-1)); B=qt.transform(Xa[KNN_COLS].fillna(-1))
        nn=NearestNeighbors(n_neighbors=max(k_list)).fit(A); d,i=nn.kneighbors(B)
        yb=np.asarray(yf)[i]
        out={f'knn{k}_bot_rate':yb[:,:k].mean(1) for k in k_list}
        # расстояние до ближайшего известного бота
        nb=NearestNeighbors(n_neighbors=1).fit(A[np.asarray(yf)==1]); out['dist_nearest_bot']=nb.kneighbors(B)[0][:,0]
        out['dist_ratio_bot_vs_any']=out['dist_nearest_bot']/(d[:,0]+1e-6)
        return pd.DataFrame(out)
    return f
_,o2=H.evaluate_stack(base,knn_feats(),'2a. + kNN-похожесть на ботов'); H.paired_bootstrap(o2,o['o0'])
pickle.dump(o2,open('oof_s2.pkl','wb'))
