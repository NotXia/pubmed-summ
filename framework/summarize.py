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

            document_summary_sents : int
                Number of sentences to summarize a document.

            cluster_summary_sents : int
                Number of sentences to summarize a cluster.

            overall_summary_sents : int|None
                Number of sentences in the overall summary.
                If None, the overall summary will not be created.

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

            overall_summary : str|list[str]
                Overall summary for all the clusters.
    """
    def __call__(self, 
        query: str, 
        batch_size: int = 5000, 
        document_summary_sents: int = 3,
        cluster_summary_sents: int = 5,
        overall_summary_sents: int|None = None,
        max_fetched: int|None = None,
        min_date: str|None = None,
        max_date: str|None = None,
        onClustersCreated: Callable[[list[Cluster]], None]|None = None,
        onDocumentSummary: Callable[[Document], None]|None = None,
        onClusterSummary: Callable[[Cluster], None]|None = None
    ) -> list[Cluster] | tuple[list[Cluster], str|list[str]]:
        clusters = self.fetcher(
            query = query, 
            batch_size = batch_size, 
            max_fetched = max_fetched, 
            min_date = min_date,
            max_date = max_date
        )

        clusters = self.clusterer(clusters)
        if onClustersCreated is not None: onClustersCreated(clusters)

        if overall_summary_sents is None:
            return self.summarizer(
                clusters, 
                onDocumentSummary = onDocumentSummary, 
                onClusterSummary = onClusterSummary,
                document_summary_len = document_summary_sents,
                cluster_summary_len = cluster_summary_sents
            )
        else:
            return self.summarizer(
                clusters, 
                onDocumentSummary = onDocumentSummary, 
                onClusterSummary = onClusterSummary,
                document_summary_len = document_summary_sents,
                cluster_summary_len = cluster_summary_sents,
                overall_summary_len = overall_summary_sents,
                overall_summary = True
            )
