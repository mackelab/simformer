
import jax
from probjax.utils.graph import faithfull_mask, min_faithfull_mask, moralize


def get_edge_mask_fn(name, task):

    if name.lower() == "faithfull":
        base_mask_fn = task.get_base_mask_fn()
        def faithfull_edge_mask(node_id, condition_mask, meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)
            return faithfull_mask(base_mask, condition_mask)

        return faithfull_edge_mask
    elif name.lower() == "min_faithfull":
        base_mask_fn = task.get_base_mask_fn()        
        def min_faithfull_edge_mask(node_id, condition_mask,meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)

            return min_faithfull_mask(base_mask, condition_mask)

        return min_faithfull_edge_mask
    elif name.lower() == "undirected":
        base_mask_fn = task.get_base_mask_fn()        
        def undirected_edge_mask(node_id, condition_mask, meta_data=None):
            base_mask = base_mask_fn(node_id, meta_data)
            return moralize(base_mask)
        
        return undirected_edge_mask
    elif name.lower() == "none":
        return lambda node_id, condition_mask, *args, **kwargs: None
    else:
        raise NotImplementedError()