from datetime import datetime


class Document:
    """
        Parameters
        ----------
            title : str
            abstract : str
            date : datetime|None
            authors : list[str]|None
                List of strings with format 'surname, name'
            doi : str|None
            pmid : str|None
    """
    def __init__(self, 
            title: str, 
            abstract: str, 
            date: datetime|None = None, 
            authors: list[str]|None = None, 
            doi: str|None = None, 
            pmid: str|None = None
        ):
        self.title = title
        self.abstract = abstract
        self.date = date
        self.authors = authors
        self.doi = doi
        self.pmid = pmid
