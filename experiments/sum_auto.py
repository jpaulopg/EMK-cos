import json,numpy as np,os,collections
D=["iris","wine","bc","digits8","vehicle","oliv_pool8","oliv_pca40","news50"]
tot=[];names={"iris":"Iris","wine":"Wine","bc":"Breast Cancer","digits8":"Digits","vehicle":"Vehicle","oliv_pool8":"Olivetti pooled","oliv_pca40":"Olivetti PCA-40","news50":"Newsgroups"}
print(f"{'data':16s} {'fullC4':>7s} {'GMMauto':>8s} {'kNNauto':>8s} {'AUTO':>7s} {'oracle':>7s}  picks")
for d in D:
    if not os.path.exists(f"res_auto_{d}.jsonl"): continue
    R=[json.loads(l) for l in open(f"res_auto_{d}.jsonl")]
    if len(R)<9: print(d,"incompleto",len(R)); continue
    a=collections.defaultdict(list); picks=[]; acc=collections.defaultdict(list)
    for r in R:
        x=r["res"]; g=[k for k in x if k.startswith("gmm")]; n=[k for k in x if k.startswith("knn")]
        sel=lambda ks:min(ks,key=lambda k:x[k]["mse"])
        a["full"].append(x["gmm_full_C4"]["rmse"]); a["gmm"].append(x[sel(g)]["rmse"]); a["knn"].append(x[sel(n)]["rmse"])
        b=sel(g+n); a["auto"].append(x[b]["rmse"]); a["orc"].append(min(v["rmse"] for v in x.values())); picks.append(b.split("_")[0]+("" if b.startswith("knn") else "_"+b.split("_")[1]))
        acc["full"].append(x["gmm_full_C4"]["knn1"]); acc["auto"].append(x[b]["knn1"])
    m={k:np.mean(v) for k,v in a.items()}; tot.append((m["full"],m["gmm"],m["knn"],m["auto"],m["orc"],np.mean(acc["full"]),np.mean(acc["auto"])))
    print(f"{names[d]:16s} {m['full']:7.4f} {m['gmm']:8.4f} {m['knn']:8.4f} {m['auto']:7.4f} {m['orc']:7.4f}  {dict(collections.Counter(picks))}  knn-acc full={np.mean(acc['full']):.3f} auto={np.mean(acc['auto']):.3f}")
t=np.array(tot); print("mean",np.round(t.mean(0),4)); print("AUTO<full on",(t[:,3]<t[:,0]).sum(),"of",len(t),"; AUTO<=best(GMMauto,kNNauto) on",(t[:,3]<=np.minimum(t[:,1],t[:,2])+1e-9).sum())
