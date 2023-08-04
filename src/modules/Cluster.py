from modules.Document import Document


class Cluster:
    def __init__(self, 
            docs: list[Document] = [], 
            topics: list[str] = [], 
            metadata = {}
        ):
        self.docs = docs
        self.topics = topics
        self.metadata = metadata