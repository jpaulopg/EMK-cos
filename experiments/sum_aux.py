import json, numpy as np
names=[("iris","Iris"),("wine","Wine"),("bc","Breast Cancer"),("digits8","Digits $8\\times8$"),("vehicle","Vehicle"),("oliv_pool8","Olivetti, $8\\times8$ pooled"),("oliv_pca40","Olivetti, PCA-40"),("news50","20 Newsgroups, SVD-50")]
rows=[];S=[]
for k,lab in names:
    L=[json.loads(l)['res'] for l in open(f"res_aux_{k}.jsonl")]
    me=[np.mean([l['A']['method_err'][m] for l in L]) for m in ('EMK-cos','RatioMeans','Plug-in')]
    B=np.mean([l['B']['rmse']['EMK-cos'] for l in L]); C=np.mean([l['C']['rmse']['EMK-cos'] for l in L]); A=np.mean([l['A']['rmse']['EMK-cos'] for l in L])
    Ar=np.mean([l['A']['rmse']['RatioMeans'] for l in L]); Ap=np.mean([l['A']['rmse']['Plug-in'] for l in L])
    S.append((k,me,B,C,A,Ar,Ap)); f=lambda x:f"{x:.4f}"
    rows.append(f"{lab} & {f(me[0])} & {f(me[1])} & {f(me[2])} & {f(B)} & {f(C)} & {f(A)} \\\\")
print("\n".join(rows))
for s in S: print(s[0], 'share method^2/A^2 %.2e'%(s[1][0]**2/s[4]**2), 'B/A %.2f'%(s[2]/s[4]), 'A ratio %.4f plug %.4f'%(s[5],s[6]))
