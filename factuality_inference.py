# factuality_inference.py - Hybrid Factuality Inference (LLM-based NLI + Voted NLI)

import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ollama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config


class HybridFactualityInference:
    """
    Factuality inference with two-tier logic using LLM-based NLI or Voted NLI:
    1. Any HIGH confidence refutation → FAKE NEWS (strict)
    2. Otherwise → Weighted voting based on confidence scores
    """
    
    def __init__(self,
                 llm_model: str = None,
                 use_llm_nli: bool = True,
                 fallback_model: str = "roberta-large-mnli",
                 high_confidence_threshold: float = 0.7,
                 min_support_threshold: float = 0.5,
                 vote_scale_factor: int = 10):
        """
        Initialize with hybrid logic and LLM-based or transformer NLI
        
        Parameters:
        -----------
        llm_model : str
            LLM model name for NLI (if use_llm_nli=True)
        use_llm_nli : bool
            Whether to use LLM for NLI (default: True)
        fallback_model : str
            Fallback transformer model if LLM fails
        high_confidence_threshold : float
            Threshold for "definitive" refutation
        min_support_threshold : float
            Minimum confidence to count as support
        vote_scale_factor : int
            Scaling factor for votes
        """
        self.llm_model = llm_model
        self.use_llm_nli = use_llm_nli
        self.high_conf_threshold = high_confidence_threshold
        self.min_support_threshold = min_support_threshold
        self.vote_scale = vote_scale_factor
        
        # Load fallback transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
        
        # Labels for RoBERTa MNLI
        self.labels = ['contradiction', 'neutral', 'entailment']
        
        print(f"Initialized Factuality Inference:")
        print(f"  Mode: {'LLM-based NLI' if use_llm_nli else 'Voted NLI'}")
        if use_llm_nli and llm_model:
            print(f"  LLM Model: {llm_model}")
        print(f"  Fallback Model: {fallback_model}")
        print(f"  High Confidence Threshold: {high_confidence_threshold}")
    
    
    def evaluate_triple_llm(self, hypothesis: str, premise: str, max_retries: int = 3) -> Dict:
        """Evaluate hypothesis against premise using LLM-based NLI"""
        
        user_prompt = f"""Perform Natural Language Inference on the following:

HYPOTHESIS: {hypothesis}
PREMISE: {premise}

Analyze the relationship and return your verdict as JSON."""
        
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": config.NLI_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                # Extract JSON from response
                content = response['message']['content']
                
                # Try to find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Normalize verdict to match expected format
                    verdict_map = {
                        'ENTAILMENT': 'SUPPORTED',
                        'CONTRADICTION': 'REFUTED',
                        'NEUTRAL': 'NOT_ENOUGH_INFO'
                    }
                    
                    verdict = verdict_map.get(result['verdict'], 'NOT_ENOUGH_INFO')
                    confidence = float(result['confidence'])
                    
                    # Create scores dict for compatibility
                    scores = {
                        'contradict': confidence if verdict == 'REFUTED' else 0.0,
                        'neutral': confidence if verdict == 'NOT_ENOUGH_INFO' else 0.0,
                        'entail': confidence if verdict == 'SUPPORTED' else 0.0
                    }
                    
                    return {
                        'verdict': verdict,
                        'confidence': confidence,
                        'scores': scores,
                        'reasoning': result.get('reasoning', ''),
                        'key_factors': result.get('key_factors', []),
                        'method': 'LLM'
                    }
                else:
                    raise ValueError("No JSON found in response")
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Fallback to transformer model
                    return self.evaluate_triple_transformer(hypothesis, premise)
        
        # Fallback
        return self.evaluate_triple_transformer(hypothesis, premise)
    
    
    def evaluate_triple_transformer(self, hypothesis: str, premise: str) -> Dict:
        """Fallback: Evaluate using transformer model"""
        inputs = self.tokenizer(premise, hypothesis,
                               return_tensors="pt",
                               truncation=True,
                               max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        scores = {
            'contradict': float(probs[0]),
            'neutral': float(probs[1]),
            'entail': float(probs[2])
        }
        
        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]
        
        # Determine verdict
        if max_label == 'contradict':
            verdict = 'REFUTED'
        elif max_label == 'entail':
            verdict = 'SUPPORTED'
        else:
            verdict = 'NOT_ENOUGH_INFO'
        
        return {
            'verdict': verdict,
            'confidence': max_score,
            'scores': scores,
            'reasoning': 'Transformer-based inference',
            'key_factors': [],
            'method': 'Transformer'
        }
    
    
    def evaluate_triple(self, hypothesis: str, premise: str) -> Dict:
        """Main evaluation method - routes to LLM or transformer"""
        if self.use_llm_nli and self.llm_model:
            return self.evaluate_triple_llm(hypothesis, premise)
        else:
            return self.evaluate_triple_transformer(hypothesis, premise)
    
    
    def infer_kg_certificate_factuality(self, 
                                       gt_df: pd.DataFrame,
                                       evidence_paths: Dict) -> Dict:
        """
        Infer factuality using HYBRID logic:
        1. Check for high-confidence refutations first (strict)
        2. If none, use weighted voting
        
        Parameters:
        -----------
        gt_df : pd.DataFrame
            KG Certificate dataframe
        evidence_paths : Dict
            Evidence paths for each triple (from graph traversal)
            
        Returns:
        --------
        Dict containing:
            - overall_verdict: 'TRUE NEWS', 'FAKE NEWS', or 'UNVERIFIED'
            - overall_explanation: explanation of verdict
            - triple_results: detailed results for each triple
            - summary: statistics
        """
        triple_results = []
        has_high_conf_refute = False
        
        # Voting accumulators
        total_support_votes = 0
        total_refute_votes = 0
        
        for idx, row in gt_df.iterrows():
            gt_triple = (str(row['subject']), str(row['predicate']), str(row['object']))
            hypothesis = f"{row['subject']} {row['predicate']} {row['object']}"
            
            paths = evidence_paths.get(gt_triple, [])
            
            if not paths:
                triple_results.append({
                    'gt_triple': gt_triple,
                    'hypothesis': hypothesis,
                    'final_verdict': 'NOT_ENOUGH_INFO',
                    'confidence': 0.0,
                    'explanation': 'No evidence paths found',
                    'evidence_results': [],
                    'high_conf_refute': False
                })
                continue
            
            # Evaluate against all evidence paths
            evidence_results = []
            triple_has_high_conf_refute = False
            
            triple_support_votes = 0
            triple_refute_votes = 0
            
            for path_info in paths:
                # Convert path to premise string
                path = path_info['path'] if isinstance(path_info, dict) else path_info
                premise = " → ".join([f"{s} {p} {o}" for s, p, o in path])
                
                result = self.evaluate_triple(hypothesis, premise)
                
                evidence_results.append({
                    'premise': premise[:200],
                    'verdict': result['verdict'],
                    'confidence': result['confidence'],
                    'scores': result['scores'],
                    'reasoning': result.get('reasoning', ''),
                    'method': result.get('method', 'Unknown')
                })
                
                # Check for HIGH CONFIDENCE REFUTATION
                if result['verdict'] == 'REFUTED' and result['confidence'] >= self.high_conf_threshold:
                    triple_has_high_conf_refute = True
                    has_high_conf_refute = True
                
                # Accumulate votes (for use if no high-conf refutations)
                if result['verdict'] == 'REFUTED':
                    votes = int(result['confidence'] * self.vote_scale)
                    triple_refute_votes += votes
                elif result['verdict'] == 'SUPPORTED' and result['confidence'] >= self.min_support_threshold:
                    votes = int(result['confidence'] * self.vote_scale)
                    triple_support_votes += votes
            
            # Determine triple verdict
            if triple_has_high_conf_refute:
                final_verdict = 'REFUTED'
                confidence = max(e['confidence'] for e in evidence_results if e['verdict'] == 'REFUTED')
                explanation = f"High-confidence refutation found (conf={confidence:.3f})"
            elif triple_support_votes > triple_refute_votes:
                final_verdict = 'SUPPORTED'
                total_votes = triple_support_votes + triple_refute_votes
                confidence = triple_support_votes / total_votes if total_votes > 0 else 0
                explanation = f"Vote: {triple_support_votes} support vs {triple_refute_votes} refute"
            elif triple_refute_votes > triple_support_votes:
                final_verdict = 'REFUTED'
                total_votes = triple_support_votes + triple_refute_votes
                confidence = triple_refute_votes / total_votes if total_votes > 0 else 0
                explanation = f"Vote: {triple_refute_votes} refute vs {triple_support_votes} support"
            else:
                final_verdict = 'NOT_ENOUGH_INFO'
                confidence = 0.0
                explanation = "Insufficient or mixed evidence"
            
            total_support_votes += triple_support_votes
            total_refute_votes += triple_refute_votes
            
            triple_results.append({
                'gt_triple': gt_triple,
                'hypothesis': hypothesis,
                'final_verdict': final_verdict,
                'confidence': confidence,
                'explanation': explanation,
                'evidence_results': evidence_results,
                'high_conf_refute': triple_has_high_conf_refute
            })
        
        # OVERALL VERDICT LOGIC
        if has_high_conf_refute:
            overall_verdict = 'FAKE NEWS'
            overall_explanation = 'At least one claim refuted with high confidence'
        elif total_support_votes > total_refute_votes:
            overall_verdict = 'TRUE NEWS'
            overall_explanation = f'Weighted vote: {total_support_votes} support vs {total_refute_votes} refute'
        elif total_refute_votes > total_support_votes:
            overall_verdict = 'FAKE NEWS'
            overall_explanation = f'Weighted vote: {total_refute_votes} refute vs {total_support_votes} support'
        else:
            overall_verdict = 'UNVERIFIED'
            overall_explanation = 'Insufficient evidence to verify claims'
        
        # Summary statistics
        supported = sum(1 for r in triple_results if r['final_verdict'] == 'SUPPORTED')
        refuted = sum(1 for r in triple_results if r['final_verdict'] == 'REFUTED')
        not_enough = sum(1 for r in triple_results if r['final_verdict'] == 'NOT_ENOUGH_INFO')
        
        avg_confidence = np.mean([r['confidence'] for r in triple_results]) if triple_results else 0.0
        
        return {
            'overall_verdict': overall_verdict,
            'overall_explanation': overall_explanation,
            'triple_results': triple_results,
            'summary': {
                'total_triples': len(triple_results),
                'supported': supported,
                'refuted': refuted,
                'not_enough_info': not_enough,
                'avg_confidence': float(avg_confidence),
                'total_support_votes': total_support_votes,
                'total_refute_votes': total_refute_votes,
                'has_high_conf_refute': has_high_conf_refute
            }
        }
