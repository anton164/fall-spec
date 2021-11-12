import numpy as np
import umap
import umap.plot
import umap.utils as utils
import umap.aligned_umap

def basic_umap_dr(data, n_neighbors=15, min_dist=0.1, spread=1, n_components=1, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        metric=metric,
        random_state=1000
    )
    u = fit.fit_transform(data)
    return u
