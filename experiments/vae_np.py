import numpy as np
def train_vae(X, zdim=8, h=64, epochs=400, bs=128, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed); n, D = X.shape
    P = {"W1": rng.normal(0, 1/np.sqrt(D), (D, h)), "b1": np.zeros(h), "Wm": rng.normal(0, .1/np.sqrt(h), (h, zdim)), "bm": np.zeros(zdim),
         "Wv": rng.normal(0, .1/np.sqrt(h), (h, zdim)), "bv": np.zeros(zdim), "W2": rng.normal(0, 1/np.sqrt(zdim), (zdim, h)), "b2": np.zeros(h),
         "W3": rng.normal(0, 1/np.sqrt(h), (h, D)), "b3": np.zeros(D)}
    M = {k: np.zeros_like(v) for k, v in P.items()}; V = {k: np.zeros_like(v) for k, v in P.items()}; t = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            xb = X[idx[s:s+bs]]; B = len(xb)
            a1 = np.tanh(xb @ P["W1"] + P["b1"]); mu = a1 @ P["Wm"] + P["bm"]; lv = np.clip(a1 @ P["Wv"] + P["bv"], -8, 4)
            eps = rng.normal(size=mu.shape); z = mu + np.exp(.5*lv) * eps
            a2 = np.tanh(z @ P["W2"] + P["b2"]); xr = a2 @ P["W3"] + P["b3"]
            # Gaussian likelihood with fixed variance 0.05^2*... use sigma^2=0.1 -> loss 0.5/sigma2*||x-xr||^2
            s2 = 0.1; dxr = (xr - xb) / s2 / B
            g = {}
            g["W3"] = a2.T @ dxr; g["b3"] = dxr.sum(0); da2 = dxr @ P["W3"].T * (1 - a2**2)
            g["W2"] = z.T @ da2; g["b2"] = da2.sum(0); dz = da2 @ P["W2"].T
            dmu = dz + mu / B; dlv = dz * eps * .5 * np.exp(.5*lv) + .5 * (np.exp(lv) - 1) / B
            g["Wm"] = a1.T @ dmu; g["bm"] = dmu.sum(0); g["Wv"] = a1.T @ dlv; g["bv"] = dlv.sum(0)
            da1 = (dmu @ P["Wm"].T + dlv @ P["Wv"].T) * (1 - a1**2)
            g["W1"] = xb.T @ da1; g["b1"] = da1.sum(0)
            t += 1
            for k in P:
                M[k] = .9*M[k] + .1*g[k]; V[k] = .999*V[k] + .001*g[k]**2
                P[k] -= lr * (M[k]/(1-.9**t)) / (np.sqrt(V[k]/(1-.999**t)) + 1e-8)
    return P
def encode(P, X):
    a1 = np.tanh(X @ P["W1"] + P["b1"]); return a1 @ P["Wm"] + P["bm"], np.clip(a1 @ P["Wv"] + P["bv"], -8, 4)
