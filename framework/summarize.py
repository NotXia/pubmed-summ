from typing import Callable
from modules.Cluster import Cluster
from modules.Document import Document
from modules.fetch.FetchFromPubMed import FetchFromPubMed
from modules.clusterer.Clusterer import Clusterer
from modules.summarizer.ExtractiveSummarizer import ExtractiveSummarizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.cluster import OPTICS
from bertopic.vectorizers import ClassTfidfTransformer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize 
from nltk.corpus import stopwords
from nltk import ngrams
import re

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")
TOPIC_EXTRACTION_STOPWORDS = stopwords.words("english")


class ClustererSummarizer():
    def __init__(self, fs_cache: bool=False, fs_cache_dir:str="./.cache"):
        self.fetcher = FetchFromPubMed(fs_cache=fs_cache, cache_dir=fs_cache_dir)

        def topic_extraction_tokenizer(doc):
            wnl = WordNetLemmatizer()
            doc = re.sub("[,\"'`()]", "", doc)
            doc = word_tokenize(doc)
            # Lowercasing
            doc = [w.lower() for w in doc]
            # Short words removal
            doc = [w for w in doc if len(w) > 1]
            # Stopwords removal
            doc = [w for w in doc if w not in TOPIC_EXTRACTION_STOPWORDS]
            # Lemmatization
            doc = [wnl.lemmatize(w) for w in doc]
            # n-gram chunking
            doc = list(ngrams(doc, 1)) + list(ngrams(doc, 2)) + list(ngrams(doc, 3))
            doc = [" ".join(ngram) for ngram in doc]
            return doc

        self.clusterer = Clusterer(
            vectorizer = SentenceTransformer("all-MiniLM-L12-v2"),
            dim_reduction = UMAP(n_components=25, random_state=42),
            clustering = OPTICS(min_samples=5),
            topic_vectorizer = CountVectorizer(tokenizer=topic_extraction_tokenizer, max_df=0.8, min_df=0.05),
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
