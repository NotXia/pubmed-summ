from modules.Cluster import Cluster


"""
    Interface of a generic module.
    Modules take as input a list of clusters and return a list of clusters.
"""
class ModuleInterface:
    def __init__(self): 
        pass

    """
        Do something on the clusters.
        
        Parameters
        ----------
            clusters : list[Cluster]
        
        Returns
        -------
            out_clusters : list[Cluster]
    """
    def __call__(self, clusters: list[Cluster]) -> list[Cluster]:
        raise NotImplementedError("__call__ not implemented")