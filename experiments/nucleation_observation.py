"""Read-only sampled cluster records; overlap links are not causal nucleus labels."""
import numpy as np
from research_clusters import analyze_clusters, cluster_overlaps

CLUSTER_COLUMNS=['time','kind','label','area','perimeter','winding_row','winding_col','resource_mean','outer_ring_resource']
LINK_COLUMNS=['previous_time','time','kind','previous_label','label','shared_nodes']


def observe_clusters(state, time: int, cutoff: float, previous: dict) -> tuple[dict,list,list,dict]:
    strategy=state.strategy.reshape(state.size,state.size)
    resource=state.resource.reshape(state.size,state.size)
    records,links,updated=[],[],{}
    metrics={}
    for kind,mask in enumerate(((strategy==0)&(resource<cutoff),strategy==0)):
        clusters=analyze_clusters(mask)
        labels=clusters['labels']
        areas=clusters['areas']
        metrics[f'n_max_{kind}']=int(areas.max()) if areas.size else 0
        metrics[f'n_clusters_{kind}']=int(areas.size)
        if kind == 0:
            metrics['depleted_area']=int(areas.sum())
        if kind in previous:
            old_time,old_labels=previous[kind]
            links.extend([old_time,time,kind,*row] for row in cluster_overlaps(old_labels,labels).tolist())
        # Resource rings are measured only for extant components, without changing arrays.
        for index,area in enumerate(areas):
            component=labels==index+1
            ring=(np.roll(component,1,0)|np.roll(component,-1,0)|
                  np.roll(component,1,1)|np.roll(component,-1,1))&~component
            ring_resource=float(resource[ring].mean()) if ring.any() else np.nan
            records.append([time,kind,index+1,int(area),int(clusters['perimeters'][index]),
                            *clusters['winding'][index].astype(int).tolist(),
                            float(resource[component].mean()),ring_resource])
        updated[kind]=(time,labels)
    return metrics,records,links,updated
