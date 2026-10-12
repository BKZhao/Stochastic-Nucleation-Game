"""Exact-cost damage layouts on a periodic square, using a separate Python RNG."""
import numpy as np
from research_clusters import analyze_clusters


def _distance_order(size: int, center: tuple[int,int]) -> np.ndarray:
    row,col=np.indices((size,size))
    dr=(row-center[0]+size//2)%size-size//2
    dc=(col-center[1]+size//2)%size-size//2
    # Lexicographic local ties preserve periodic translation equivariance.
    return np.lexsort((dc.ravel(),dr.ravel(),(dr*dr+dc*dc).ravel()))


def seed_mask(size: int, count: int, shape: str, center: tuple[int,int],
              rng: np.random.Generator) -> np.ndarray:
    if size < 3 or not 0 <= count <= size*size:
        raise ValueError('invalid lattice size or damage count')
    mask=np.zeros((size,size),bool)
    if shape == 'compact':
        mask.ravel()[_distance_order(size,(0,0))[:count]]=True
    elif shape == 'random':
        mask.ravel()[rng.choice(size*size,count,replace=False)]=True
    elif shape == 'stripe':
        if count > (size-1)**2:
            raise ValueError('nonwinding stripe does not fit')
        length=min(size-1,max(1,int(np.ceil(np.sqrt(4*count)))))
        for index in range(count):
            row,col=divmod(index,length)
            mask[row,col]=True
    elif shape == 'four_patches':
        if count < 4:
            raise ValueError('four separated patches require at least four nodes')
        centers=[(0,0),(0,size//2),(size//2,0),(size//2,size//2)]
        for index,point in enumerate(centers):
            budget=count//4+int(index<count%4)
            nodes=_distance_order(size,point)[:budget]
            if mask.ravel()[nodes].any():
                raise ValueError('four patches overlap at this damage budget')
            mask.ravel()[nodes]=True
        if len(analyze_clusters(mask)['areas']) != 4:
            raise ValueError('four patches are not separated at this damage budget')
    else:
        raise ValueError(f'unknown seed shape: {shape}')
    return np.roll(mask,(int(center[0])%size,int(center[1])%size),(0,1))
