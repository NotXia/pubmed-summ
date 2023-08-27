import { Article } from "./Article"; 

export interface Cluster {
    docs: Article[]; 
    topics: string[];
    is_outlier: boolean;
    summary: string[];
    metadata?: any;
}