import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import cv2

PALETTE = [
    (255,  0,   0  ),  # red
    (  0, 255,   0 ),  # green
    (  0,   0, 255 ),  # blue
    (255,255,  0  ),  # yellow
    (255,  0, 255 ),  # magenta
    (  0,255,255 ),   # cyan
    (128,128,128 )   # gray
    # …add as many as you like…
]

def show_clusters(frame_count, img: np.ndarray, dets: np.ndarray):
    """
    img: HxWx3 BGR
    dets: Nx8 array of [x1,y1,x2,y2, conf, cls, cluster_id]
    """
    img = img.copy()
    if dets.size == 0:
        return img

    # build a color map for each cluster ID (except noise)
    cluster_ids = np.unique(dets[:, 6].astype(int))
    rng = np.random.RandomState(25)
    colors = {}
    for cid in cluster_ids:
        if cid == -1:
            colors[cid] = (200,200,200)  # noise always gray
        else:
            colors[cid] = PALETTE[cid % len(PALETTE)]

    for det in dets:
        x1, y1, x2, y2, _, _, cid = det
        x1, y1, x2, y2, cid = map(int, (x1, y1, x2, y2, cid))

        col = colors[cid]

        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            img,
            f"C{cid}",
            (x1, max(y1-5,0)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            col,
            1,
            cv2.LINE_AA
        )
    success = cv2.imwrite(f'/home/ubuntu/boxmot/cluster_viz/frame_{frame_count}.png', img)

    cv2.namedWindow("grupari", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("grupari", 1200, 680)
    cv2.imshow("grupari", img)
    if cv2.waitKey(1) == 27:
        return


def build_x(detections, img_shape):
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

    return X

def cluster_detections(detections, img_shape,
                       eps=0.05, min_samples=3):
    """
    Cluster detections purely by their normalized box geometry.
    detections: Nx4 array of [x1,y1,x2,y2]

    img_shape: (H, W)
    eps: DBSCAN eps in normalized units (0-1)
    min_samples: min points per cluster
    returns: cluster labels for each detection (-1 is noise)
    """
    X = build_x(detections, img_shape)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    return clustering.fit_predict(X)

def count_clusters(dets: np.ndarray) -> dict[int, int]:
    """
    Count how many detections per cluster (excluding noise = -1).

    Args:
        dets: NxK array, where dets[:, -1] is the cluster_id for each detection.

    Returns:
        A dict mapping cluster_id -> count.
    """
    # pull out the cluster IDs as ints
    labels = dets[:, -2].astype(int)
    # ignore noise
    labels = labels[labels != -1]
    # get unique labels and their counts
    unique_ids, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_ids.tolist(), counts.tolist()))

def cluster_detections_optics(detections, img_shape,
                              min_samples=3, eps=0.1):
    X = build_x(detections, img_shape)

    clustering = OPTICS(min_samples=min_samples,
                        max_eps=eps,   # maximum epsilon to consider
                        cluster_method='dbscan')
    
    return clustering.fit_predict(X)

