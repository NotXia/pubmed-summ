from modules.Cluster import Cluster


"""
    Interface of Fetch modules.
    This type of module takes arbitrary arguments
    and retrieves documents to summarize.
"""
class FetchInterface:
    def __init__(self): 
        pass

    """
        Retrieve the documents to summarize

        Returns
        -------
            documents : list[Cluster]
                A list containing all retrieved documents in a single or multiple clusters.
    """
    def __call__(self, **kwargs) -> list[Cluster]:
        raise NotImplementedError("__call__ not implemented")