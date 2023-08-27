export interface Article {
    title: string;
    abstract: string; 
    date?: Date; 
    authors?: string[];
    doi: string; 
    pmid: string;
    summary: string[];
}