from modules.Cluster import Cluster
from modules.ModuleInterface import ModuleInterface
from bertopic import BERTopic


class Clusterer(ModuleInterface):
    def __init__(self, vectorizer, dim_reduction, clustering, topic_vectorizer, topic_extraction, representation):
        super().__init__()
        self.vectorizer = vectorizer
        self.dim_reduction = dim_reduction
        self.clustering = clustering
        self.topic_vectorizer = topic_vectorizer
        self.topic_extraction = topic_extraction
        self.representation = representation
    

    def __call__(self, clusters: list[Cluster]) -> list[Cluster]:
        out_clusters: list[Cluster] = []

        for cluster in clusters:
            documents = [doc.abstract for doc in cluster.docs]

            topic_model = BERTopic(
                embedding_model = self.vectorizer,
                umap_model = self.dim_reduction,
                hdbscan_model = self.clustering,
                vectorizer_model = self.topic_vectorizer,
                ctfidf_model = self.topic_extraction,
                representation_model = self.representation
            )
            labels, _ = topic_model.fit_transform(documents)

            for label in set(labels):
                new_cluster_docs = [doc for i, doc in enumerate(cluster.docs) if labels[i] == label]
                new_cluster_topics = [t for t, _ in topic_model.get_topic(label)] # type: ignore
                
                if label == -1:
                    out_clusters.append( Cluster(new_cluster_docs, topics=new_cluster_topics, is_outlier=True, metadata=cluster.metadata) )
                else:
                    out_clusters.append( Cluster(new_cluster_docs, topics=new_cluster_topics, metadata=cluster.metadata) )

        return out_clusters
