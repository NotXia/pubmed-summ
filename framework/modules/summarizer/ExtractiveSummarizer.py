from typing import Callable
from modules.Cluster import Cluster
from modules.Document import Document
from modules.ModuleInterface import ModuleInterface
import torch
from transformers import pipeline, AutoTokenizer, Pipeline
import spacy
import copy
import itertools


class ExtractiveSummarizer(ModuleInterface):
    """
        Parameters
        ----------
            model_name : str
                Model for the summarization pipeline

            pipeline : Pipeline|None
                Summarization pipeline. Has priority over model_name.
    """
    def __init__(self, 
        model_name: str = "NotXia/longformer-bio-ext-summ", 
        pipeline: Pipeline|None = None
    ):
        super().__init__()
        if pipeline is not None:
            self.summarizer = pipeline
        else:
            self.summarizer = pipeline("summarization",
                model = model_name,
                tokenizer = AutoTokenizer.from_pretrained(model_name),
                trust_remote_code = True,
                device = 0 if torch.cuda.is_available() else -1
            ) # type: ignore
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.select_pipes(enable=['tok2vec', "parser", "senter"])

    
    """
        Summarizes each cluster in a list.

        Parameters
        ----------
            clusters : list[Cluster]
                Clusters to summarize.
            
            onDocumentSummary : Callable[[Document], None] | None
                A function that will be called when the summary of an article is ready.
                The Document object of the article will be passed as argument.

            onClusterSummary : Callable[[Cluster], None] | None
                A function that will be called when the summary of a cluster is ready.
                The Cluster object will be passed as argument.

            document_summary_len : int
                Length of document summary.

            cluster_summary_len : int
                Length of cluster summary.

            overall_summary : bool
                If True, the overall summary will be generated.

            overall_summary_len : int
                Length of overall summary.
        
        Returns
        -------
            cluster : list[Cluster]
                Clusters with summaries.

            overall_summary : list[str]
                Sentences selected for the overall summary.
    """
    def __call__(self, 
        clusters: list[Cluster],
        onDocumentSummary: Callable[[Document], None]|None = None,
        onClusterSummary: Callable[[Cluster], None]|None = None,
        document_summary_len: int = 3,
        cluster_summary_len: int = 5,
        overall_summary: bool = False,
        overall_summary_len: int = 5
    ) -> list[Cluster] | tuple[list[Cluster], str|list[str]]:
        def _doc2sents(document):
            return [s.text for s in self.nlp(document).sents]
        
        out_cluster: list[Cluster] = []

        for cluster in clusters:
            new_cluster = copy.deepcopy(cluster)

            # Summarizes each individual abstract
            abstracts_sents = [_doc2sents(doc.abstract) for doc in new_cluster.docs]
            summarizer_out: list[tuple[list[str], list[int]]] = self.summarizer(
                [{ "sentences": abs } for abs in abstracts_sents], 
                strategy = "count", 
                strategy_args = document_summary_len
            ) # type: ignore

            # Saves the summary of each document
            for i in range(len(new_cluster.docs)):
                new_cluster.docs[i].summary = summarizer_out[i][0]
                if onDocumentSummary is not None: onDocumentSummary(new_cluster.docs[i])

            # Summarizes all (summarized) abstracts as a single document
            full_doc_sentences = []
            for selected_sents, _ in summarizer_out:
                full_doc_sentences = full_doc_sentences + selected_sents
            selected_sents: list[str] = []
            selected_sents, _ = self.summarizer({ "sentences": full_doc_sentences }, strategy="count", strategy_args=cluster_summary_len) # type: ignore

            new_cluster.summary = [s.strip() for s in selected_sents]
            out_cluster.append(new_cluster)
            if onClusterSummary is not None: onClusterSummary(new_cluster)

        if overall_summary:
            clusters_summ_sents = list(itertools.chain.from_iterable([c.summary for c in out_cluster])) # type: ignore
            final_summary_sents: list[str] = []
            final_summary_sents, _ = self.summarizer({ "sentences": clusters_summ_sents }, strategy="count", strategy_args=overall_summary_len) # type: ignore
            return out_cluster, final_summary_sents
        else:
            return out_cluster