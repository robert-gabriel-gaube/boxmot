import numpy as np
from sklearn.cluster import DBSCAN

# def cluster_detections(detections, embeddings, img_shape,
#                        eps=0.5, min_samples=3,
#                        spatial_weight=0.5):
#     """
#     detections: Nx6 array of [x1,y1,x2,y2,conf,cls]
#     embeddings: NxD array of ReID features
#     img_shape: (h, w)
#     returns: clusters: dict cid→list of det indices, noise: list of indices
#     """
#     H, W = img_shape[:2]
#     # normalize center coords + w,h to [0,1]
#     xywh = np.copy(detections[:, :4])
#     xywh[:, 2:] -= xywh[:, :2]
#     xywh[:, :2] += xywh[:, 2:] / 2
#     xywh[:, [0,2]] /= W
#     xywh[:, [1,3]] /= H

#     X = np.concatenate([
#         embeddings, 
#         spatial_weight * xywh
#     ], axis=1)

#     clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
#     labels = clustering.labels_
#     clusters, noise = {}, []
#     for i, l in enumerate(labels):
#         if l == -1:
#             noise.append(i)
#         else:
#             clusters.setdefault(l, []).append(i)
#     return clusters, noise

def cluster_detections(detections, img_shape,
                       eps=0.05, min_samples=3):
    """
    Cluster detections purely by their normalized box geometry.
    detections: Nx4 array of [x1,y1,x2,y2]
    
    img_shape: (H, W)
    eps: DBSCAN eps in normalized units (0-1)
    min_samples: min points per cluster
    returns: clusters dict cid→list of indices, noise list
    """
    H, W = img_shape
    # convert to center + size
    xyxy = detections.astype(float)
    widths  = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    cx = xyxy[:, 0] + widths  / 2
    cy = xyxy[:, 1] + heights / 2

    # normalize to [0,1]
    cx /= W;      widths  /= W
    cy /= H;      heights /= H

    # feature vector = [cx, cy, w, h]
    X = np.stack([cx, cy, widths, heights], axis=1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(X)

    clusters, noise = {}, []
    for i, l in enumerate(labels):
        if l == -1:
            noise.append(i)
        else:
            clusters.setdefault(l, []).append(i)
    return clusters, noise