"""Rebuild vehicle.npz from the original Statlog (Vehicle Silhouettes) files of the UCI repository.

Download the data set (files xaa.dat ... xai.dat) from
https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes
and run:  python prepare_vehicle.py /path/to/folder/with/xa*.dat
"""
import glob, os, sys
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for f in sorted(glob.glob(os.path.join(folder, "xa*.dat"))):
    rows += [line.split() for line in open(f) if line.strip()]
X = np.array([[float(v) for v in r[:18]] for r in rows])
classes = sorted({r[18] for r in rows})            # bus, opel, saab, van
y = np.array([classes.index(r[18]) for r in rows])
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicle.npz"), X=X, y=y)
print(X.shape, classes)
