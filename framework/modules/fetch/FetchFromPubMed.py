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
            fs_cache : bool
                If true, retrieved articles will be cached locally.
            
            cache_dir : str
                Directory for cached articles.
    """
    def __init__(self, fs_cache: bool=False, cache_dir: str="./.cache"):
        super().__init__()
        self.fs_cache = fs_cache
        self.cache_dir = cache_dir


    """
        Retrieve PubMed abstracts.

        Parameters
        ----------
            query : str
                Search query.

            batch_size : int
                Number of documents to fetch with a single API call to PubMed.

            max_fetched : int|None
                Maximum number of fetched articles.
                All available if None.

            min_date: str|None
                Minimum publication date of the articles (Format: YYYY/MM/DD, YYYY/MM, YYYY).

            max_date: str|None
                Maximun publication date of the articles (Format: YYYY/MM/DD, YYYY/MM, YYYY).

        Returns
        -------
            documents : list[Cluster]
                A list containing a single cluster with all retrieved documents.
    """
    def __call__(self, 
        query: str, 
        batch_size: int = 5000, 
        max_fetched: int|None = None,
        min_date: str|None = None,
        max_date: str|None = None
    ) -> list[Cluster]:
        def _esearch(query, retstart=0, retmax=batch_size, min_date=None, max_date=None):
            res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": retmax,
                "retstart": retstart,
                "term": query,
                "sort": "pub_date",
                "datetype": "pdat", # Publication date
                "mindate": min_date, 
                "maxdate": max_date
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
                date_xml = xml.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
                year = date_xml.find('Year').text
                month = date_xml.find('Month').text if date_xml.find('Month') is not None else "Jan"
                day = date_xml.find('Day').text if date_xml.find('Day') is not None else "1"
                date = datetime.strptime(f"{year}-{month}-{day}", '%Y-%b-%d')
            except: date = None

            if date is None:
                try:
                    date_xml = xml.find("MedlineCitation/Article/ArticleDate")
                    date = datetime( int(date_xml.find("Year").text), int(date_xml.find("Month").text), int(date_xml.find("Day").text) )
                except: date = None

            if date is None:
                try:
                    date_xml = xml.find("MedlineCitation/DateCompleted")
                    date = datetime( int(date_xml.find("Year").text), int(date_xml.find("Month").text), int(date_xml.find("Day").text) )
                except: date = None

            # Identifiers
            id_xml = xml.find("PubmedData/ArticleIdList")
            doi = id_xml.find("ArticleId[@IdType='doi']").text if id_xml.find("ArticleId[@IdType='doi']") is not None else ""
            pmid = id_xml.find("ArticleId[@IdType='pubmed']").text if id_xml.find("ArticleId[@IdType='pubmed']") is not None else ""
            
            # Authors
            authors_xml = xml.find("MedlineCitation/Article/AuthorList")
            authors = []
            if authors_xml is not None:
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

            # Keywords and MeSH headings
            keywords = []
            mesh_headings_xml = xml.find("MedlineCitation/MeshHeadingList")
            if mesh_headings_xml is not None: 
                for mesh_xml in mesh_headings_xml.findall("MeshHeading"):
                    keywords.append(mesh_xml.find("DescriptorName").text) # type: ignore
            keywords_xml = xml.find("MedlineCitation/KeywordList")
            if keywords_xml is not None: 
                for keyword_xml in keywords_xml.findall("Keyword"):
                    keywords.append(keyword_xml.text)

            return Document(title=title, abstract=abstract, date=date, authors=authors, doi=doi, pmid=pmid, keywords=keywords)
        
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
            if max_fetched is not None:
                res = _esearch(query, retstart=retstart, retmax=min(batch_size, max_fetched-len(documents)), min_date=min_date, max_date=max_date)
            else:
                res = _esearch(query, retstart=retstart, retmax=batch_size, min_date=min_date, max_date=max_date)
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
            if (max_fetched is not None) and (len(documents) >= max_fetched):
                break
            
        return [Cluster(documents)]
        