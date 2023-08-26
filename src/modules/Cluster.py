from modules.Document import Document


class Cluster:
    def __init__(self, 
            docs: list[Document] = [], 
            topics: list[str] = [],
            is_outlier: bool = False,
            summary: str|list[str]|None = None,
            metadata = {}
        ):
        self.docs = docs
        self.topics = topics
        self.is_outlier = is_outlier
        self.summary = summary
        self.metadata = metadata