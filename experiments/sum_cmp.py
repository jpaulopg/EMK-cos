import json,numpy as np
from scipy import stats
D=["iris","wine","bc","digits8","vehicle","oliv_pool8","oliv_pca40","news50"]
nm=["Iris","Wine","Breast Cancer","Digits","Vehicle","Olivetti pooled","Olivetti PCA-40","Newsgroups"]
M=["EMK-cos","EMK-cos g1","RatioMeans","Taylor2","Plug-in","MeanImp","ObsSubset"]
T=[];A=[];neg=collections=None
print(f"{'':16s}"+"".join(f"{m:>11s}" for m in M))
import collections
for d,n in zip(D,nm):
    R=[json.loads(l) for l in open(f"res_cmp_{d}.jsonl")]; assert len(R)==9
    r=[np.mean([x["res"][m]["rmse"] for x in R]) for m in M]; T.append(r)
    A.append([np.mean([x["res"][m]["knn"] for x in R]) for m in M])
    print(f"{n:16s}"+"".join(f"{v:11.4f}" for v in r), " PSD violations:", {m:sum(x['res'][m]['mineig']<-1e-8 for x in R) for m in ("Taylor2","ObsSubset")})
T=np.array(T); A=np.array(A); print("mean".ljust(16)+"".join(f"{v:11.4f}" for v in T.mean(0)))
print("\nEMK-cos better than each method on N of 8 sets, mean rel diff:")
for j,m in enumerate(M[1:],1):
    rel=(T[:,0]-T[:,j])/T[:,j]; print(f"  vs {m:10s}: {(T[:,0]<T[:,j]).sum()}/8  mean {100*rel.mean():+.1f}%  range [{100*rel.min():+.1f},{100*rel.max():+.1f}]")
print("\nkNN acc (5-NN LOO):"); print("".join(f"{m:>11s}" for m in M)); 
for n,a in zip(nm,A): print(f"{n:16s}"+"".join(f"{v:11.3f}" for v in a))
print("mean".ljust(16)+"".join(f"{v:11.3f}" for v in A.mean(0)))
print("picks:", {d:dict(collections.Counter(json.loads(l)['res']['_sel'].split('_')[0]+'_'+json.loads(l)['res']['_sel'].split('_')[1] for l in open(f'res_cmp_{d}.jsonl'))) for d in D})
