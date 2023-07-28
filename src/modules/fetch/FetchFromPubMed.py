from modules.FetchInterface import FetchInterface
from modules.Cluster import Cluster
from modules.Document import Document
import requests
import xml.etree.ElementTree as ET
from datetime import datetime


class FetchFromPubMed(FetchInterface):
    """
        Parameters
        ----------
            batch_size : int
                Number of documents to fetch with a single API call to PubMed
    """
    def __init__(self, batch_size=5000):
        super().__init__()
        self.batch_size = batch_size


    """
        Retrieve PubMed abstracts

        Parameters
        ----------
            query : str
                Search query

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
        
        def _efetch(ids):
            res = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", data={
                "db": "pubmed",
                "retmode": "xml",
                "rettype": "abstract",
                "id": ",".join(ids)
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
            date = None
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
        

        documents = []
        pmids = ["pad"]
        retstart = 0
        while len(pmids) > 0:
            res = _esearch(query, retstart=retstart)
            pmids = res["esearchresult"]["idlist"]
            retstart += len(pmids)

            query_xml = _efetch(pmids)
            query_tree = ET.fromstring(query_xml)
            for article_tree in query_tree:
                doc = _getDocumentFromXML(article_tree)
                if doc is not None:
                    documents.append(doc)
                    
        return [Cluster(documents)]