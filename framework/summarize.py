from typing import Callable
from modules.Cluster import Cluster
from modules.Document import Document
from modules.fetch.FetchFromPubMed import FetchFromPubMed
from modules.clusterer.Clusterer import Clusterer
from modules.summarizer.ExtractiveSummarizer import ExtractiveSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from umap import UMAP
from sklearn.cluster import OPTICS
from bertopic.vectorizers import ClassTfidfTransformer


class ClustererSummarizer():
    def __init__(self, fs_cache: bool=False, fs_cache_dir:str="./.cache"):
        self.fetcher = FetchFromPubMed(fs_cache=fs_cache, cache_dir=fs_cache_dir)

        self.clusterer = Clusterer(
            vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, stop_words="english", ngram_range=(1, 3)),
            dim_reduction = UMAP(n_components=25, random_state=42),
            clustering = OPTICS(min_samples=5),
            topic_vectorizer = CountVectorizer(stop_words="english", max_df=0.5, min_df=0.05, ngram_range=(1, 3)),
            topic_extraction = ClassTfidfTransformer(reduce_frequent_words=True),
            representation = None
        )

        self.summarizer = ExtractiveSummarizer()


    """
        Clusters and summarizes PubMed articles related to a query.

        Parameters
        ----------
            query : str

            batch_size : int
                Refer to `FetchFromPubMed`.

            max_fetched : int|None 
                Refer to `FetchFromPubMed`.

            min_date: str|None
                Refer to `FetchFromPubMed`.

            max_date: str|None
                Refer to `FetchFromPubMed`.
        
            onClustersCreated: Callable[[list[Cluster]], None] | None
                A function that will be called at the end of the clusterization phase.
                The list of Cluster objects will be provided as argument.
            
            onDocumentSummary : Callable[[Document], None] | None
                Refer to `ExtractiveSummarizer`.

            onClusterSummary : Callable[[Cluster], None] | None
                Refer to `ExtractiveSummarizer`.

        Returns
        -------
            cluster : list[Cluster]
                The final clusters after the elaboration
    """
    def __call__(self, 
        query: str, 
        batch_size: int = 5000, 
        max_fetched: int|None = None,
        min_date: str|None = None,
        max_date: str|None = None,
        onClustersCreated: Callable[[list[Cluster]], None]|None = None,
        onDocumentSummary: Callable[[Document], None]|None = None,
        onClusterSummary: Callable[[Cluster], None]|None = None
    ) -> list[Cluster]:
        clusters = self.fetcher(
            query = query, 
            batch_size = batch_size, 
            max_fetched = max_fetched, 
            min_date = min_date,
            max_date = max_date
        )

        clusters = self.clusterer(clusters)
        if onClustersCreated is not None: onClustersCreated(clusters)

        clusters = self.summarizer(clusters, onDocumentSummary=onDocumentSummary, onClusterSummary=onClusterSummary)

        return clusters