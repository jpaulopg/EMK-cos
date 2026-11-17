import json,numpy as np
from scipy import stats
S=["full","diag","lowrank3","lowrank8"]
tot=[]
for d in ["iris","wine","bc","digits8","vehicle","oliv_pool8","oliv_pca40","news50"]:
    A=[json.loads(l) for l in open(f"res_select_{d}.jsonl")]; B=[json.loads(l) for l in open(f"res_select2_{d}.jsonl")]
    f=[];b=[];v=[];o=[];pb=[];pv=[];kf=[];kb=[];kv=[]
    for a,c in zip(A,B):
        x=a["res"]; val=c["val"]
        sb=min(S,key=lambda s:x[s]["bic"]); sv=max(S,key=lambda s:val[s])
        f.append(x["full"]["rmse"]); b.append(x[sb]["rmse"]); v.append(x[sv]["rmse"]); o.append(min(x[s]["rmse"] for s in S))
        pv.append(sv); kf.append(x["full"]["knn1"]); kv.append(x[sv]["knn1"]); kb.append(x[sb]["knn1"])
    m=lambda z:float(np.mean(z)); tot.append((m(f),m(b),m(v),m(o)))
    print(f"{d:11s} full={m(f):.4f} BIC={m(b):.4f} val={m(v):.4f} oracle={m(o):.4f} | knn1 full={m(kf):.3f} BIC={m(kb):.3f} val={m(kv):.3f} | val picks { {s:pv.count(s) for s in S} }")
t=np.array(tot)
print("mean",t.mean(0)); print("val<full",(t[:,2]<t[:,0]).sum(),"BIC<full",(t[:,1]<t[:,0]).sum())
print("rel change val vs full",np.mean((t[:,2]-t[:,0])/t[:,0]),"wilcoxon",stats.wilcoxon(t[:,2],t[:,0]).pvalue)
print("rel to oracle: full",np.mean(t[:,0]/t[:,3]),"val",np.mean(t[:,2]/t[:,3]),"BIC",np.mean(t[:,1]/t[:,3]))
