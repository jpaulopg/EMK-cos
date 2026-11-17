import json,numpy as np
R=[json.loads(l) for l in open("res_vae.jsonl")]
print("active dims",[r["res"]["active"] for r in R])
M=["EMK-cos","RatioMeans","Taylor2","Plug-in"]
for tau in ("0.25","1.0","4.0","16.0"):
    k=f"tau{tau}"; snr=np.mean([r["res"][k]["_snr"] for r in R]); nn=np.mean([r["res"][k]["_normnu"] for r in R])
    print(f"\ntau={tau}  SNR={snr:.2f}  mean||nu||={nn:.3f}")
    print(f"{'':12s}{'RMSE vs exact (MC)':>20s}{'max err':>10s}{'RMSE vs realized':>18s}{'5-NN acc':>10s}")
    for m in M:
        g=lambda f:np.mean([r["res"][k][m][f] for r in R])
        print(f"{m:12s}{g('rmse_ref'):20.4f}{g('maxerr'):10.3f}{g('rmse_real'):18.4f}{g('knn'):10.3f}")
    print(f"{'EMK-cos g1':12s}{'':48s}{np.mean([r['res'][k]['EMK-cos g1']['knn'] for r in R]):.3f}  | MC-ref acc {np.mean([r['res'][k]['MC ref']['knn'] for r in R]):.3f}")
