# Data

Iris, Wine, Breast Cancer, Digits and Diabetes are loaded from scikit-learn. The remaining sets are stored here.

| File | Source | Content |
| --- | --- | --- |
| `vehicle.npz` | Statlog (Vehicle Silhouettes), UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes | 846 × 18 attributes, 4 classes (bus, opel, saab, van). Rebuild with `prepare_vehicle.py`. |
| `olivetti.npz` | Olivetti faces (AT&T Laboratories Cambridge), as distributed by `sklearn.datasets.fetch_olivetti_faces` | 400 images of 64 × 64 pixels, 40 subjects. The loader derives the 8 × 8 pooled (D = 64) and 40-component PCA versions. |
| `news50.npz` | 20 Newsgroups, as distributed by `sklearn.datasets.fetch_20newsgroups` | 2000 documents, TF-IDF reduced to 50 dimensions by truncated SVD, 20 labels. The experiments subsample 600 documents. |

The projections of Olivetti (PCA) and 20 Newsgroups (SVD) are computed on complete data: these two sets define feature
spaces in which entries are then deleted.
