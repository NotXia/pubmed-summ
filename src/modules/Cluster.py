from modules.Document import Document


class Cluster:
    def __init__(self, docs: list[Document]=[], metadata={}):
        self.docs: list[Document] = docs
        self.metadata = metadata