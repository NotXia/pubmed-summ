from modules.Cluster import Cluster
from modules.Document import Document
from modules.ModuleInterface import ModuleInterface
import torch
from transformers import pipeline, AutoTokenizer
import spacy
import copy


class ExtractiveSummarizer(ModuleInterface):
    def __init__(self, model_name: str="NotXia/longformer-bio-ext-summ"):
        super().__init__()
        self.summarizer = pipeline("summarization",
            model = model_name,
            tokenizer = AutoTokenizer.from_pretrained(model_name),
            trust_remote_code = True,
            device = 0 if torch.cuda.is_available() else -1
        ) # type: ignore
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.select_pipes(enable=['tok2vec', "parser", "senter"])

    
    def __call__(self, clusters: list[Cluster]) -> list[Cluster]:
        def _doc2sents(document):
            return [s.text for s in self.nlp(document).sents]
        
        out_cluster = []

        for cluster in clusters:
            new_cluster = copy.deepcopy(cluster)

            # Summarize each individual abstract
            abstracts_sents = [_doc2sents(doc.abstract) for doc in new_cluster.docs]
            summarizer_out: list[tuple[list[str], list[int]]] = self.summarizer(
                [{ "sentences": abs } for abs in abstracts_sents], 
                strategy = "count", 
                strategy_args = 3
            ) # type: ignore

            # Saves the summary of each document
            for i in range(len(new_cluster.docs)):
                new_cluster.docs[i].summary = summarizer_out[i][0]

            # Summarize all (summarized) abstracts as a single document
            full_doc_sentences = []
            for selected_sents, _ in summarizer_out:
                full_doc_sentences = full_doc_sentences + selected_sents
            selected_sents: list[str] = []
            selected_sents, _ = self.summarizer({ "sentences": full_doc_sentences }, strategy="count", strategy_args=5) # type: ignore

            new_cluster.summary = [s.strip() for s in selected_sents]
            out_cluster.append(new_cluster)

            print(new_cluster.summary)

        return out_cluster