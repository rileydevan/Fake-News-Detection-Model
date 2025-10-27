# kg_construction.py - Knowledge Graph Certificate Construction

import re
import html
import ast
import pandas as pd
from typing import List, Tuple
import ollama
import config

def clean_tweet_for_search(text: str) -> str:
    """
    Clean a tweet/claim for processing:
    - Remove links entirely
    - Drop @ and # symbols, keep the words
    - Unescape HTML entities (&amp; -> &)
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    s = html.unescape(text)
    
    # Remove URLs
    URL_RE = re.compile(r"https?://\S+")
    s = URL_RE.sub(" ", s)
    
    # Remove @ and # symbols but keep the words
    MENT_HASH_RE = re.compile(r"[@#]")
    s = MENT_HASH_RE.sub("", s)
    
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


def extract_triples_from_response(response_text: str) -> List[List[str]]:
    """
    Extract triples from the LLM response.
    Handles both <python> tagged and raw list formats.
    """
    try:
        # First try to find content within <python> tags
        python_match = re.search(r'<python>(.*?)</python>', response_text, re.DOTALL)
        if python_match:
            content = python_match.group(1).strip()
        else:
            # If no python tags, use the whole response
            content = response_text.strip()

        # Try to parse as Python literal
        triples = ast.literal_eval(content)

        # Ensure it's a list of lists
        if isinstance(triples, list) and all(isinstance(t, list) for t in triples):
            # Ensure each triple has exactly 3 elements
            valid_triples = [t for t in triples if len(t) == 3]
            return valid_triples
        else:
            return []
    except:
        # If parsing fails, try alternative formats
        try:
            # Try to find lists in the text
            list_pattern = r'\[([^\]]+)\]'
            matches = re.findall(list_pattern, response_text)
            triples = []
            for match in matches:
                # Clean and split by comma
                items = [item.strip().strip('"').strip("'") for item in match.split('",')]
                if len(items) == 3:
                    triples.append(items)
            return triples
        except:
            return []


def construct_kg_certificate(text: str, 
                            source_id: str = None,
                            model_name: str = "gpt-oss:20b") -> pd.DataFrame:
    """
    Construct a KG certificate (Gt) from a claim/tweet.
    
    Parameters:
    -----------
    text : str
        The raw claim/tweet text to process
    source_id : str, optional
        An identifier for the source
    model_name : str
        The Ollama model to use for triple extraction
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: source_id, subject, predicate, object
    """
    # Clean the text
    cleaned_text = clean_tweet_for_search(text)
    
    if not cleaned_text:
        print(f"Warning: Text cleaned to empty string")
        return pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
    
    all_triples = []
    
    try:
        # Prepare the prompt
        prompt = config.SYSTEM_PROMPT_TRIPLE_EXTRACTION.format(input=cleaned_text)
        
        # Call LLM to extract triples
        resp = ollama.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1}
        )
        
        response_content = resp["message"]["content"]
        
        # Extract triples from response
        triples = extract_triples_from_response(response_content)
        
        # Build dataframe rows
        for triple in triples:
            if len(triple) == 3:
                all_triples.append({
                    'source_id': source_id if source_id else 'unknown',
                    'subject': str(triple[0]).strip(),
                    'predicate': str(triple[1]).strip(),
                    'object': str(triple[2]).strip()
                })
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
    
    # Create dataframe
    kg_df = pd.DataFrame(all_triples)
    if kg_df.empty:
        kg_df = pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
    
    return kg_df


def construct_kg_from_snippets(snippets_df: pd.DataFrame,
                               snippet_col: str = 'snippet',
                               id_col: str = 'id',
                               model_name: str = "gpt-oss:20b") -> pd.DataFrame:
    """
    Construct KG from multiple search result snippets.
    
    Parameters:
    -----------
    snippets_df : pd.DataFrame
        DataFrame containing snippets
    snippet_col : str
        Name of column containing snippets
    id_col : str
        Name of ID column
    model_name : str
        Ollama model for triple extraction
        
    Returns:
    --------
    pd.DataFrame
        Combined KG from all snippets
    """
    all_triples = []
    
    for idx, row in snippets_df.iterrows():
        snippet = row[snippet_col]
        source_id = row[id_col] if id_col in snippets_df.columns else idx
        
        if pd.isna(snippet) or snippet == '':
            continue
        
        # Extract triples from this snippet
        snippet_kg = construct_kg_certificate(snippet, str(source_id), model_name)
        
        if not snippet_kg.empty:
            all_triples.append(snippet_kg)
    
    # Combine all dataframes
    if all_triples:
        combined_kg = pd.concat(all_triples, ignore_index=True)
        # Remove duplicates
        combined_kg = combined_kg.drop_duplicates(subset=['subject', 'predicate', 'object'])
        return combined_kg
    else:
        return pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
