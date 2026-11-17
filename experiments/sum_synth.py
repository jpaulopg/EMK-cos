import json, glob, collections, numpy as np
rows=[json.loads(l) for f in glob.glob("res_synth_*.jsonl") for l in open(f)]
M=["EMK-cos","RatioMeans","Taylor2","CMI"]
def table(exp,keys,pref=""):
    g=collections.defaultdict(list)
    for r in rows:
        if r["exp"]==exp: g[tuple(r.get(k) for k in keys)].append(r["res"])
    print("==",exp,pref or "realized",keys,"|"," | ".join(M))
    for k in sorted(g,key=str):
        if pref and pref+"EMK-cos" not in g[k][0]: continue
        print(k," | ".join(f"{np.mean([x[pref+m] for x in g[k]]):.4f}" for m in M), f"(n={len(g[k])})")
for e,k in [("S1",["model","ntrain"]),("S2",["a","rate"]),("S3",["rho","model"]),("S6",["rho","rate"])]:
    table(e,k); table(e,k,"ref_")
table("S4",["D"]);table("S5",["kind"])
for e in ["S1","S2","S3","S6"]:
    o=[r for r in rows if r["exp"]==e and r.get("model","oracle")=="oracle"]
    d=np.array([r["res"]["RatioMeans"]-r["res"]["EMK-cos"] for r in o])
    print(e,"oracle: realized RMSE Ratio-EMK mean",d.mean().round(5),"min",d.min().round(5),"EMK<=Ratio in",(d>=-1e-4).sum(),"/",len(d))
