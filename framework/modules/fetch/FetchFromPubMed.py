from modules.FetchInterface import FetchInterface
from modules.Cluster import Cluster
from modules.Document import Document
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pickle as pickle
import os


class FetchFromPubMed(FetchInterface):
    """
        Parameters
        ----------
            batch_size : int
                Number of documents to fetch with a single API call to PubMed.

            max_fetched : int|None
                Maximum number of fetched articles.
                All available if None.

            fs_cache : bool
                If true, retrieved articles will be cached locally.

            batch_size : str
                Path of cache directory.
    """
    def __init__(self, batch_size: int=5000, max_fetched: int|None=None, fs_cache: bool=False, cache_dir: str="./.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.max_fetched = max_fetched
        self.fs_cache = fs_cache
        self.cache_dir = cache_dir


    """
        Retrieve PubMed abstracts.

        Parameters
        ----------
            query : str
                Search query.

        Returns
        -------
            documents : list[Cluster]
                A list containing a single cluster with all retrieved documents.
    """
    def __call__(self, query: str) -> list[Cluster]:
        def _esearch(query, retstart=0, retmax=self.batch_size):
            res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": retmax,
                "retstart": retstart,
                "term": query
            })
            return res.json()
        
        def _efetch(pmids):
            res = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", data={
                "db": "pubmed",
                "retmode": "xml",
                "rettype": "abstract",
                "id": ",".join(pmids)
            })
            return res.content

        def _getDocumentFromXML(xml):
            def _getInnerText(xml):
                # Gets the content of a tag as text without other inner tags
                return ''.join(xml.itertext())
            
            # Abstract not available, discard article
            if xml.find("MedlineCitation/Article/Abstract") is None: 
                return None

            # Title
            title = _getInnerText(xml.find("MedlineCitation/Article/ArticleTitle"))
            
            # Date
            date: datetime|None = None
            try:
                date_xml = xml.find("MedlineCitation/Article/ArticleDate")
                date = datetime( int(date_xml.find("Year").text), int(date_xml.find("Month").text), int(date_xml.find("Day").text) )
            except: date = None

            if date is None:
                try:
                    date_xml = xml.find("MedlineCitation/DateCompleted")
                    date = datetime( int(date_xml.find("Year").text), int(date_xml.find("Month").text), int(date_xml.find("Day").text) )
                except: date = None

            if date is None:
                try:
                    date_xml = xml.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
                    year = date_xml.find('Year').text
                    month = date_xml.find('Month').text if date_xml.find('Month') is not None else "Jan"
                    day = date_xml.find('Day').text if date_xml.find('Day') is not None else "1"
                    date = datetime.strptime(f"{year}-{month}-{day}", '%Y-%b-%d')
                except: date = None
            
            # Identifiers
            id_xml = xml.find("PubmedData/ArticleIdList")
            doi = id_xml.find("ArticleId[@IdType='doi']").text if id_xml.find("ArticleId[@IdType='doi']") is not None else ""
            pmid = id_xml.find("ArticleId[@IdType='pubmed']").text if id_xml.find("ArticleId[@IdType='pubmed']") is not None else ""
            
            # Authors
            authors_xml = xml.find("MedlineCitation/Article/AuthorList")
            authors = []
            for author_xml in authors_xml:
                try:
                    authors.append(f"{author_xml.find('LastName').text}, {author_xml.find('ForeName').text}")
                except:
                    continue

            # Abstract
            abstract_sections = xml.find("MedlineCitation/Article/Abstract").findall("AbstractText")
            abstract = ""
            for section_xml in abstract_sections:
                section_text = _getInnerText(section_xml)
                if "Label" in section_xml.attrib: # Appends section title if set
                    section_text = f"{section_xml.attrib['Label']} {section_text}"
                abstract = f"{abstract}\n{section_text}"
            abstract = abstract.strip()

            return Document(title=title, abstract=abstract, date=date, authors=authors, doi=doi, pmid=pmid)
        
        def _getCached(pmid) -> Document|None:
            if not self.fs_cache: return None
            
            cache_file = os.path.join(self.cache_dir, f"{pmid}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as fin:
                    return pickle.load(fin)
            return None

        def _cacheDocument(document: Document):
            if not self.fs_cache: return

            if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)
            with open(os.path.join(self.cache_dir, f"{document.pmid}.pkl"), "wb") as fout:
                pickle.dump(document, fout, pickle.HIGHEST_PROTOCOL)


        documents: list[Document] = []
        batch_pmids = ["pad"]
        retstart = 0
        while len(batch_pmids) > 0:
            res: dict
            if self.max_fetched is not None:
                res = _esearch(query, retstart=retstart, retmax=min(self.batch_size, self.max_fetched-len(documents)))
            else:
                res = _esearch(query, retstart=retstart, retmax=self.batch_size)
            batch_pmids = res["esearchresult"]["idlist"]
            retstart += len(batch_pmids)

            # Retrieves cached documents first
            i = 0
            while i < len(batch_pmids):
                doc = _getCached(batch_pmids[i])
                if doc is not None:
                    del batch_pmids[i]
                    documents.append(doc)
                else:
                    i += 1

            # Retrieves remaining documents from PubMed
            query_xml = _efetch(batch_pmids)
            query_tree = ET.fromstring(query_xml)
            for article_tree in query_tree:
                doc = _getDocumentFromXML(article_tree)
                if doc is not None:
                    _cacheDocument(doc)
                    documents.append(doc)

            # Reached maximum number of required documents
            if (self.max_fetched is not None) and (len(documents) >= self.max_fetched):
                break
            
        return [Cluster(documents)]
        