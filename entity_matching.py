# entity_matching.py - Hybrid Entity Matching (Aliases + FAISS)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import faiss
import config


class EntityMatcherHybrid:
    """
    Matches entities between KG Certificate (Gt) and External KG (Ge) using:
    1. Domain-specific alias matching (highest priority)
    2. FAISS-accelerated semantic similarity (fallback)
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the hybrid entity matcher.
        
        Parameters:
        -----------
        similarity_threshold : float
            Minimum cosine similarity to consider entities as matched
        """
        self.threshold = similarity_threshold
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Storage for entities and embeddings
        self.ge_entities = None
        self.ge_embeddings = None
        self.gt_entities = None
        self.gt_embeddings = None
        
        # FAISS index
        self.faiss_index = None
        self.embedding_dim = 768  # all-mpnet-base-v2 dimension
        
        # Matching results
        self.entity_matches = {}
        self.unmatched_entities = []
        
        # Load domain-specific aliases
        self.aliases = config.ENTITY_ALIASES.copy()
        
        # Build reverse lookup: alias -> canonical form
        self.entity_to_canonical = {}
        for canonical, alias_list in self.aliases.items():
            self.entity_to_canonical[canonical.lower()] = canonical
            for alias in alias_list:
                self.entity_to_canonical[alias.lower()] = canonical
    
    
    def normalize_entity(self, entity: str) -> str:
        """
        Normalize entity to its canonical form using alias dictionary.
        
        Parameters:
        -----------
        entity : str
            Entity to normalize
            
        Returns:
        --------
        str
            Canonical form if found in aliases, otherwise original entity
        """
        entity_lower = entity.lower().strip()
        return self.entity_to_canonical.get(entity_lower, entity)
    
    
    def load_external_kg(self, ge_df: pd.DataFrame):
        """
        Load and embed entities from the External KG (Ge), then build FAISS index.
        
        Parameters:
        -----------
        ge_df : pd.DataFrame
            External KG dataframe with columns: source_id, subject, predicate, object
        """
        # Extract unique entities from subjects and objects
        subjects = set(ge_df['subject'].unique())
        objects = set(ge_df['object'].unique())
        self.ge_entities = list(subjects.union(objects))
        
        print(f"Found {len(self.ge_entities)} unique entities in External KG")
        
        # Compute embeddings
        self.ge_embeddings = self.model.encode(
            self.ge_entities,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        # Build FAISS index
        self._build_faiss_index()
    
    
    def _build_faiss_index(self):
        """
        Build FAISS index for External KG embeddings.
        Uses IndexFlatIP (Inner Product) since embeddings are already normalized.
        """
        # IndexFlatIP: exact search using inner product (cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.faiss_index.add(self.ge_embeddings.astype('float32'))
    
    
    def load_kg_certificate(self, gt_df: pd.DataFrame):
        """
        Load and embed entities from the KG Certificate (Gt).
        
        Parameters:
        -----------
        gt_df : pd.DataFrame
            KG Certificate dataframe with columns: source_id, subject, predicate, object
        """
        # Extract unique entities from subjects and objects
        subjects = set(gt_df['subject'].unique())
        objects = set(gt_df['object'].unique())
        self.gt_entities = list(subjects.union(objects))
        
        print(f"Found {len(self.gt_entities)} unique entities in KG Certificate")
        
        # Compute embeddings
        self.gt_embeddings = self.model.encode(
            self.gt_entities,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
    
    
    def match_entities(self, k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Match entities from Gt to Ge using hybrid approach:
        1. First try alias-based exact matching
        2. Fallback to FAISS similarity search
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to retrieve for FAISS search
            
        Returns:
        --------
        Dict[str, List[Tuple[str, float]]]
            Mapping of Gt entities to list of (Ge entity, similarity score) tuples
        """
        if self.faiss_index is None or self.gt_embeddings is None:
            raise ValueError("Must load both External KG and KG Certificate first!")
        
        self.entity_matches = {}
        self.unmatched_entities = []
        
        alias_match_count = 0
        embedding_match_count = 0
        
        # Pre-compute FAISS search for all entities (efficiency)
        similarities, indices = self.faiss_index.search(
            self.gt_embeddings.astype('float32'),
            k
        )
        
        # Process each Gt entity
        for i, gt_entity in enumerate(self.gt_entities):
            # STEP 1: Try alias matching
            normalized_gt = self.normalize_entity(gt_entity)
            
            alias_matches = []
            for j, ge_entity in enumerate(self.ge_entities):
                normalized_ge = self.normalize_entity(ge_entity)
                if normalized_gt.lower() == normalized_ge.lower():
                    alias_matches.append((ge_entity, 1.0))  # Perfect match score
            
            if alias_matches:
                self.entity_matches[gt_entity] = alias_matches
                alias_match_count += 1
                continue
            
            # STEP 2: Fallback to embedding similarity via FAISS
            neighbor_sims = similarities[i]
            neighbor_indices = indices[i]
            
            matches = []
            for j in range(k):
                sim_score = float(neighbor_sims[j])
                if sim_score >= self.threshold:
                    ge_idx = neighbor_indices[j]
                    ge_entity = self.ge_entities[ge_idx]
                    matches.append((ge_entity, sim_score))
            
            if matches:
                self.entity_matches[gt_entity] = matches
                embedding_match_count += 1
            else:
                self.unmatched_entities.append(gt_entity)
        
        print(f"Matching complete:")
        print(f"  Alias matches: {alias_match_count}")
        print(f"  Embedding matches: {embedding_match_count}")
        print(f"  Total matched: {len(self.entity_matches)}/{len(self.gt_entities)}")
        print(f"  Unmatched: {len(self.unmatched_entities)}/{len(self.gt_entities)}")
        
        return self.entity_matches
    
    
    def get_best_match(self, gt_entity: str) -> Optional[Tuple[str, float]]:
        """
        Get the best (highest similarity) match for a Gt entity.
        
        Parameters:
        -----------
        gt_entity : str
            Entity from KG Certificate
            
        Returns:
        --------
        Optional[Tuple[str, float]]
            (matched_entity, similarity_score) or None if no match
        """
        if gt_entity in self.entity_matches:
            return self.entity_matches[gt_entity][0]
        return None
    
    
    def get_all_matches(self, gt_entity: str) -> List[Tuple[str, float]]:
        """
        Get all matches for a Gt entity (handles polysemy).
        
        Parameters:
        -----------
        gt_entity : str
            Entity from KG Certificate
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (matched_entity, similarity_score) tuples
        """
        return self.entity_matches.get(gt_entity, [])
