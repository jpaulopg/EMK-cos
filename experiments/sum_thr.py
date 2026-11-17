import json, numpy as np
R=[json.loads(open(f"res_thr_{s}.jsonl").read())["res"] for s in range(3)]
ms=["EMK-cos","RatioMeans","Taylor2","Plug-in"]
def avg(tau,m,k): return np.array([r[f"tau{tau}"][m][k] for r in R])
for tau in (0.25,1.0,4.0,16.0):
    print(f"\n== tau {tau}  (th@FAR5% exact = {np.mean([r[f'tau{tau}']['_th_far'] for r in R]):.3f}, exact AUC {np.mean([r[f'tau{tau}']['_auc_ref'] for r in R]):.4f})")
    print("method      AUC    | FAR@exact_th  TPR  disc | disagree% threshold 0.3 0.5 0.7 0.9 | accept(0.5): est/exact")
    for m in ms:
        a=lambda k: avg(tau,m,k).mean()
        print(f"{m:11s} {a('auc'):.4f} | {a('far_at_exact_th')*100:5.2f}% {a('tpr_at_exact_th')*100:5.1f}% {a('dis_at_exact_th')*100:5.2f}% | "+" ".join(f"{a(f'dis{t}')*100:5.2f}" for t in (0.3,0.5,0.7,0.9))+f" | {a('acc0.5')*100:5.1f}/{a('acc0_0.5')*100:5.1f}")
