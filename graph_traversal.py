# graph_traversal.py - Bidirectional BFS for Evidence Path Finding

from collections import deque, defaultdict
import pandas as pd
from typing import List, Tuple, Optional, Dict


class KnowledgeGraphTraversal:
    """
    Implements Bidirectional BFS for finding shortest paths between matched entities
    in the External KG (Ge).
    """
    
    def __init__(self, ge_df: pd.DataFrame, hop_limit: int = 7):
        """
        Initialize the graph traversal system.
        
        Parameters:
        -----------
        ge_df : pd.DataFrame
            External KG dataframe with columns: source_id, subject, predicate, object
        hop_limit : int
            Maximum number of hops to explore
        """
        self.ge_df = ge_df
        self.hop_limit = hop_limit
        self.graph = None
        
        self._build_graph()
    
    
    def _build_graph(self):
        """
        Build adjacency list representation of the External KG (Ge).
        Graph is stored as: {entity: [(neighbor, predicate, direction), ...]}
        where direction indicates the original triple structure.
        """
        self.graph = defaultdict(list)
        
        for _, row in self.ge_df.iterrows():
            subject = str(row['subject'])
            predicate = str(row['predicate'])
            obj = str(row['object'])
            
            # Add edge from subject to object (forward direction)
            self.graph[subject].append((obj, predicate, 'forward'))
            
            # Add edge from object to subject (backward direction - undirected traversal)
            self.graph[obj].append((subject, predicate, 'backward'))
        
        num_nodes = len(self.graph)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        
        print(f"Graph built: {num_nodes} nodes, {num_edges} edges")
    
    
    def reconstruct_path(self,
                        meeting_node: str,
                        parent_fwd: Dict[str, Optional[str]],
                        rel_fwd: Dict[str, Optional[str]],
                        dir_fwd: Dict[str, Optional[str]],
                        parent_bwd: Dict[str, Optional[str]],
                        rel_bwd: Dict[str, Optional[str]],
                        dir_bwd: Dict[str, Optional[str]]) -> List[Tuple[str, str, str]]:
        """
        Reconstruct path as (s, p, o) triples with correct directionality.
        
        Parameters:
        -----------
        meeting_node : str
            The entity where forward and backward searches met
        parent_fwd, rel_fwd, dir_fwd : Dict
            Maps from forward search
        parent_bwd, rel_bwd, dir_bwd : Dict
            Maps from backward search
            
        Returns:
        --------
        List[Tuple[str, str, str]]
            Path as sequence of (subject, predicate, object) triples
        """
        path = []
        stack = []
        
        # Build forward path from start to meeting point
        curr = meeting_node
        while parent_fwd[curr] is not None:
            parent = parent_fwd[curr]
            rel = rel_fwd[curr]
            direction = dir_fwd[curr]
            
            if direction == 'forward':
                stack.append((parent, rel, curr))
            else:  # backward
                stack.append((curr, rel, parent))
            
            curr = parent
        
        # Pop from stack to get correct order from start to meeting point
        while stack:
            path.append(stack.pop())
        
        # Build backward path from meeting point to goal
        curr = meeting_node
        while parent_bwd[curr] is not None:
            next_node = parent_bwd[curr]
            rel = rel_bwd[curr]
            direction = dir_bwd[curr]
            
            if direction == 'forward':
                path.append((curr, rel, next_node))
            else:  # backward
                path.append((next_node, rel, curr))
            
            curr = next_node
        
        return path
    
    
    def bidirectional_bfs(self,
                         start_entity: str,
                         goal_entity: str) -> Optional[List[Tuple[str, str, str]]]:
        """
        Bidirectional BFS shortest path search between two entities.
        
        Parameters:
        -----------
        start_entity : str
            Starting entity
        goal_entity : str
            Goal entity
            
        Returns:
        --------
        Optional[List[Tuple[str, str, str]]]
            Path as sequence of (subject, predicate, object) triples, or None if no path
        """
        # Check if entities exist in graph
        if start_entity not in self.graph:
            return None
        if goal_entity not in self.graph:
            return None
        
        # Special case: start and goal are the same
        if start_entity == goal_entity:
            return []
        
        # Initialize queues: (entity, depth)
        queue_fwd = deque([(start_entity, 0)])
        queue_bwd = deque([(goal_entity, 0)])
        
        # Initialize parent maps
        parent_fwd = {start_entity: None}
        parent_bwd = {goal_entity: None}
        
        # Initialize relation maps
        rel_fwd = {start_entity: None}
        rel_bwd = {goal_entity: None}
        
        # Initialize direction maps
        dir_fwd = {start_entity: None}
        dir_bwd = {goal_entity: None}
        
        # Initialize visited sets
        visited_fwd = {start_entity}
        visited_bwd = {goal_entity}
        
        # BFS loop
        while queue_fwd and queue_bwd:
            
            # Expand from the smaller queue for efficiency
            if len(queue_fwd) <= len(queue_bwd):
                # Forward search
                u, depth = queue_fwd.popleft()
                
                # Check hop limit
                if depth >= self.hop_limit / 2:
                    continue
                
                # Explore neighbors
                for neighbor, rel, direction in self.graph[u]:
                    if neighbor not in visited_fwd:
                        visited_fwd.add(neighbor)
                        parent_fwd[neighbor] = u
                        rel_fwd[neighbor] = rel
                        dir_fwd[neighbor] = direction
                        queue_fwd.append((neighbor, depth + 1))
                        
                        # Check if we've met the backward search
                        if neighbor in visited_bwd:
                            return self.reconstruct_path(
                                neighbor,
                                parent_fwd, rel_fwd, dir_fwd,
                                parent_bwd, rel_bwd, dir_bwd
                            )
            
            else:
                # Backward search
                u, depth = queue_bwd.popleft()
                
                # Check hop limit
                if depth >= self.hop_limit / 2:
                    continue
                
                # Explore neighbors
                for neighbor, rel, direction in self.graph[u]:
                    if neighbor not in visited_bwd:
                        visited_bwd.add(neighbor)
                        parent_bwd[neighbor] = u
                        rel_bwd[neighbor] = rel
                        dir_bwd[neighbor] = direction
                        queue_bwd.append((neighbor, depth + 1))
                        
                        # Check if we've met the forward search
                        if neighbor in visited_fwd:
                            return self.reconstruct_path(
                                neighbor,
                                parent_fwd, rel_fwd, dir_fwd,
                                parent_bwd, rel_bwd, dir_bwd
                            )
        
        # No path found
        return None
    
    
    def find_paths_for_triple(self,
                             gt_triple: Tuple[str, str, str],
                             entity_matches: Dict[str, List[Tuple[str, float]]]) -> Dict:
        """
        Find evidence paths for a single triple from the KG Certificate (Gt).
        
        Parameters:
        -----------
        gt_triple : Tuple[str, str, str]
            A triple (subject, predicate, object) from Gt
        entity_matches : Dict[str, List[Tuple[str, float]]]
            Entity matching results from EntityMatcher
            
        Returns:
        --------
        Dict containing:
            - 'gt_triple': original triple
            - 'matched_entities': (subject, object) matched entities in Ge
            - 'paths': list of paths (handles polysemy)
            - 'status': 'found', 'no_match', or 'no_path'
        """
        gt_subject, gt_predicate, gt_object = gt_triple
        
        # Get matched entities
        subject_matches = entity_matches.get(gt_subject, [])
        object_matches = entity_matches.get(gt_object, [])
        
        if not subject_matches:
            return {
                'gt_triple': gt_triple,
                'matched_entities': None,
                'paths': [],
                'status': 'no_match'
            }
        
        if not object_matches:
            return {
                'gt_triple': gt_triple,
                'matched_entities': None,
                'paths': [],
                'status': 'no_match'
            }
        
        # Try all combinations (handles polysemy)
        all_paths = []
        
        for ge_subject, sub_score in subject_matches:
            for ge_object, obj_score in object_matches:
                
                path = self.bidirectional_bfs(ge_subject, ge_object)
                
                if path is not None:
                    all_paths.append({
                        'start': ge_subject,
                        'end': ge_object,
                        'path': path,
                        'subject_similarity': sub_score,
                        'object_similarity': obj_score
                    })
        
        if all_paths:
            return {
                'gt_triple': gt_triple,
                'matched_entities': (subject_matches[0][0], object_matches[0][0]),
                'paths': all_paths,
                'status': 'found'
            }
        else:
            return {
                'gt_triple': gt_triple,
                'matched_entities': (subject_matches[0][0], object_matches[0][0]),
                'paths': [],
                'status': 'no_path'
            }
    
    
    def find_all_paths(self,
                      gt_df: pd.DataFrame,
                      entity_matches: Dict[str, List[Tuple[str, float]]]) -> List[Dict]:
        """
        Find evidence paths for all triples in the KG Certificate (Gt).
        
        Parameters:
        -----------
        gt_df : pd.DataFrame
            KG Certificate dataframe
        entity_matches : Dict
            Entity matching results from EntityMatcher
            
        Returns:
        --------
        List[Dict]
            List of path results for each triple
        """
        results = []
        
        for idx, row in gt_df.iterrows():
            gt_triple = (str(row['subject']), str(row['predicate']), str(row['object']))
            result = self.find_paths_for_triple(gt_triple, entity_matches)
            results.append(result)
        
        # Summary
        found = sum(1 for r in results if r['status'] == 'found')
        no_match = sum(1 for r in results if r['status'] == 'no_match')
        no_path = sum(1 for r in results if r['status'] == 'no_path')
        
        print(f"Path finding complete:")
        print(f"  Paths found: {found}")
        print(f"  No entity match: {no_match}")
        print(f"  No path found: {no_path}")
        
        return results
