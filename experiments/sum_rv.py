import json, numpy as np
ms=["EMK-cos","RatioMeans","Taylor2","Plug-in"]
for kind,keys in (("gp",["nobs3","nobs6","nobs12"]),("blr",["nk15","nk30","nk60"])):
    R=[json.loads(open(f"res_rv_{kind}_{s}.jsonl").read())["res"] for s in range(3)]
    for k in keys:
        g=lambda m,f: np.mean([r[k][m][f] for r in R])
        extra=f"  cond~{np.mean([r[k]['_cond'] for r in R]):.0f}" if kind=="blr" else ""
        print(f"\n== {kind} {k}  SNR {np.mean([r[k]['_snr'] for r in R]):.2f}{extra}  exact AUC {g('MC ref','auc'):.4f}  | MC vs true: rmse {g('MC ref','rmse_true'):.4f}")
        print("method      rmse_exact  max   rmse_true | mineig | AUC   FAR@exact_th  dis(th.5 vs exact)  dis(vs true)")
        for m in ms:
            print(f"{m:11s} {g(m,'rmse_ref'):.4f}   {g(m,'maxerr'):.3f} {g(m,'rmse_true'):.4f}  | {g(m,'mineig'):7.3f} | {g(m,'auc'):.4f} {g(m,'far_at_exact_th')*100:6.2f}%       {g(m,'dis_th')*100:5.2f}%            {g(m,'dis_true')*100:5.2f}%")
