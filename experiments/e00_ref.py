from lab import *
S5=(42,43,44,45,46)
A=run(lgb_fp(LGB_A),FEATURES); report('A x3 (скрининг-референс)',A,group='ref',save_as='A3')
A2=run(lgb_fp(LGB_A,seeds=(7,8,9)),FEATURES); report('A x3 другие сиды (оценка шума)',A2,ref=A,group='ref')
A5=run(lgb_fp(LGB_A,S5),FEATURES); B5=run(lgb_fp(LGB_B,S5),FEATURES)
pickle.dump(A5,open('oof/A5.pkl','wb')); pickle.dump(B5,open('oof/B5.pkl','wb'))
AB={k:(A5[k]+B5[k])/2 for k in A5}
report('ФИНАЛ v1: A+B x5 (текущий сабмит)',AB,group='ref',save_as='AB_v1')
