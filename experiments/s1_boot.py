exec(open('s1_ua.py').read().split("base=H.FEATURES")[0])
import pickle
base=H.FEATURES
_,o0=H.evaluate(base,'0. текущее')
_,o1a=H.evaluate(base+new,'1a. семейство+версия'); H.paired_bootstrap(o1a,o0)
_,o1d=H.evaluate_te(base,{'te_ua_string':ua},'1d. TE(строка UA)'); H.paired_bootstrap(o1d,o0)
_,o1e=H.evaluate_te(base+['ua_major_version'],{'te_ua_family':fam,'te_device_model':model,'te_os_version':osv,'te_ua_string':ua},'1e. всё вместе'); H.paired_bootstrap(o1e,o0)
pickle.dump(dict(o0=o0,o1a=o1a,o1d=o1d,o1e=o1e),open('oof_s1.pkl','wb'))
