from datetime import datetime


class Document:
    """
        Parameters
        ----------
            title : str
            abstract : str
            date : datetime
            authors : list[str]
                List of strings with format 'surname, name'
            doi : str
            pmid : str
    """
    def __init__(self, title: str, abstract: str, date: datetime, authors: list[str], doi: str="", pmid: str=""):
        self.title: str = title
        self.abstract: str = abstract
        self.date: datetime = date
        self.authors: list[str] = authors
        self.doi: str = doi
        self.pmid: str = pmid
