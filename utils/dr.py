import numpy as np
import umap
import umap.plot
import umap.utils as utils
import umap.aligned_umap

def find_min(l):
    min = l[0][0]
    for i, val in enumerate(l):
        if val[0] < min:
            min = val[0]
    for i, val in enumerate(l):
        l[i][0] = l[i][0] + abs(min)
    return l

def basic_umap_dr(data, n_neighbors=15, min_dist=0.1, spread=1, n_components=1, metric='cosine'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=1000
    )
    u = fit.fit_transform(data)
    u = find_min(u)
    return u
