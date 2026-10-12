"""Read-only periodic components; lifted coordinates distinguish seams from winding."""
import numpy as np
from numba import njit


@njit(cache=True)
def _components(mask):
    size = mask.shape[0]
    labels = np.zeros((size,size),np.int32)
    lifted = np.zeros((size,size,2),np.int64)
    queue = np.empty((size*size,2),np.int32)
    areas = np.zeros(size*size,np.int64)
    perimeters = np.zeros(size*size,np.int64)
    winding = np.zeros((size*size,2),np.bool_)
    moves = ((-1,0),(1,0),(0,-1),(0,1))
    count = 0
    for row in range(size):
        for col in range(size):
            if not mask[row,col] or labels[row,col]:
                continue
            count += 1
            head,tail = 0,1
            queue[0] = row,col
            labels[row,col] = count
            while head < tail:
                r,c = queue[head]
                head += 1
                areas[count-1] += 1
                for dr,dc in moves:
                    nr,nc = (r+dr)%size,(c+dc)%size
                    if not mask[nr,nc]:
                        perimeters[count-1] += 1
                    elif labels[nr,nc] == 0:
                        labels[nr,nc] = count
                        lifted[nr,nc,0] = lifted[r,c,0]+dr
                        lifted[nr,nc,1] = lifted[r,c,1]+dc
                        queue[tail] = nr,nc
                        tail += 1
                    else:
                        if lifted[nr,nc,0] != lifted[r,c,0]+dr:
                            winding[count-1,0] = True
                        if lifted[nr,nc,1] != lifted[r,c,1]+dc:
                            winding[count-1,1] = True
    return labels,areas[:count],perimeters[:count],winding[:count]


def analyze_clusters(mask: np.ndarray) -> dict[str,np.ndarray]:
    field = np.asarray(mask,dtype=np.bool_)
    if field.ndim != 2 or field.shape[0] != field.shape[1] or field.shape[0] < 3:
        raise ValueError('clusters require a square periodic grid with L >= 3')
    labels,areas,perimeters,winding = _components(field)
    compactness = np.divide(4*np.pi*areas,perimeters.astype(float)**2,
                            out=np.full(len(areas),np.nan),where=perimeters>0)
    return dict(labels=labels,areas=areas,perimeters=perimeters,
                winding=winding,compactness=compactness)


def cluster_overlaps(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return (previous label, current label, shared nodes) at sampled times."""
    if previous.shape != current.shape:
        raise ValueError('label fields must have the same shape')
    both = (previous>0)&(current>0)
    pairs = np.column_stack((previous[both],current[both]))
    unique,counts = np.unique(pairs,axis=0,return_counts=True)
    return np.column_stack((unique,counts)).astype(np.int64)
