export interface Article {
    title: string;
    abstract: string; 
    date?: Date|null; 
    authors?: string[];
    doi: string|null; 
    pmid: string;
    summary: string[]|null;
}