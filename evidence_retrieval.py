# evidence_retrieval.py - Evidence Retrieval via SerpAPI

import os
import time
import requests
import pandas as pd
from typing import List, Dict, Tuple
import ollama
import config
from kg_construction import construct_kg_from_snippets


class EvidenceRetriever:
    """
    Retrieves evidence from SerpAPI and constructs External KG (Ge).
    """
    
    def __init__(self, api_key: str, kg_model: str = "gpt-oss:20b"):
        """
        Initialize evidence retriever.
        
        Parameters:
        -----------
        api_key : str
            SerpAPI key
        kg_model : str
            Ollama model for KG construction
        """
        self.api_key = api_key
        self.kg_model = kg_model
        
    
    def generate_search_query(self, claim: str) -> str:
        """
        Use LLM to generate effective search query from claim.
        
        Parameters:
        -----------
        claim : str
            The claim to generate query for
            
        Returns:
        --------
        str
            Optimized search query
        """
        try:
            prompt = config.CLAIM_QUERY_PROMPT.format(claim=claim)
            
            resp = ollama.chat(
                model=self.kg_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3}
            )
            
            query = resp["message"]["content"].strip()
            
            # Clean up the query
            query = query.replace('"', '').replace("'", "").strip()
            
            # If too long, truncate
            if len(query.split()) > 10:
                query = ' '.join(query.split()[:10])
            
            return query
            
        except Exception as e:
            print(f"Error generating search query: {e}")
            # Fallback to first 60 characters of claim
            return claim[:60]
    
    
    def search_serpapi(self, 
                      query: str,
                      num_results: int = 5) -> pd.DataFrame:
        """
        Query SerpAPI and return results as DataFrame.
        
        Parameters:
        -----------
        query : str
            Search query
        num_results : int
            Number of results to retrieve
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: id, title, snippet, link, source
        """
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": max(1, num_results),
            "hl": "en",
            "gl": "us",
            "safe": "off",
        }
        
        url = "https://serpapi.com/search.json"
        
        try:
            response = requests.get(url, params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
            
            # Check for errors
            if isinstance(data, dict) and data.get("error"):
                print(f"SerpAPI Error: {data.get('error')}")
                return pd.DataFrame()
            
            # Extract organic results
            organic = data.get("organic_results", []) if isinstance(data, dict) else []
            
            results = []
            for idx, item in enumerate(organic[:num_results]):
                result = {
                    'id': idx,
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', ''),
                    'source': item.get('source', '')
                }
                results.append(result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error querying SerpAPI: {e}")
            return pd.DataFrame()
    
    
    def retrieve_evidence(self,
                         claim: str,
                         num_results: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Complete evidence retrieval pipeline:
        1. Generate search query
        2. Query SerpAPI
        3. Build External KG from snippets
        
        Parameters:
        -----------
        claim : str
            The claim to find evidence for
        num_results : int
            Number of search results to retrieve
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            (External KG DataFrame, search query used)
        """
        # Step 1: Generate search query
        search_query = self.generate_search_query(claim)
        print(f"Generated search query: '{search_query}'")
        
        # Step 2: Query SerpAPI
        search_results = self.search_serpapi(search_query, num_results)
        
        if search_results.empty:
            print("No search results returned")
            return pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object']), search_query
        
        print(f"Retrieved {len(search_results)} search results")
        
        # Step 3: Build External KG from snippets
        # Combine title and snippet for better context
        search_results['text'] = search_results['title'] + '. ' + search_results['snippet']
        
        external_kg = construct_kg_from_snippets(
            search_results,
            snippet_col='text',
            id_col='id',
            model_name=self.kg_model
        )
        
        print(f"Constructed External KG with {len(external_kg)} triples")
        
        return external_kg, search_query
