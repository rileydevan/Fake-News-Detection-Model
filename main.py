import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

# Import custom modules
from kg_construction import construct_kg_certificate, clean_tweet_for_search
from evidence_retrieval import EvidenceRetriever
from entity_matching import EntityMatcherHybrid
from graph_traversal import KnowledgeGraphTraversal
from factuality_inference import HybridFactualityInference
from utils import save_results, visualize_kg, create_results_summary
import config

##=======================================
# # 1. Run setup
# ./setup.sh

# # 2. When asked about models, type 'n'

# # 3. Download one model manually
# ollama pull gemma3:12b

# # 4. Launch the app
# ./run.sh
##=======================================

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .verdict-true {
        background: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        border-left: 4px solid #10B981;
    }
    .verdict-fake {
        background: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        border-left: 4px solid #EF4444;
    }
    .verdict-unverified {
        background: #FEF3C7;
        color: #92400E;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        border-left: 4px solid #F59E0B;
    }
    .stage-complete {
        color: #10B981;
        font-weight: 600;
    }
    .stage-processing {
        color: #3B82F6;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cumulative_kg' not in st.session_state:
    st.session_state.cumulative_kg = pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'serpapi_key' not in st.session_state:
    st.session_state.serpapi_key = None

# Sidebar configuration
with st.sidebar:
    # st.image("SimpleLogo.png", width=True)
    # st.markdown("### 🔧 Configuration")
    
    # SerpAPI Key
    serpapi_key = st.text_input(
        "SerpAPI Key",
        type="password",
        value=st.session_state.serpapi_key or "",
        help="Enter your SerpAPI key for evidence retrieval"
    )
    if serpapi_key:
        st.session_state.serpapi_key = serpapi_key
        st.success("✓ API Key configured")
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 Model Configuration")
    
    kg_model = st.selectbox(
        "KG Construction Model",
        config.AVAILABLE_MODELS,
        help="Select the LLM for knowledge graph construction"
    )
    
    inference_method = st.radio(
        "Inference Method",
        ["LLM-based NLI", "Voted NLI"],
        help="Choose the factuality inference approach"
    )
    
    if inference_method == "LLM-based NLI":
        nli_model = st.selectbox(
            "NLI Model",
            config.AVAILABLE_MODELS,
            help="Select the LLM for NLI inference"
        )
    else:
        nli_model = "roberta-large-mnli"
    
    st.markdown("---")
    
    # Advanced Settings Expander
    with st.expander("⚙️ Advanced Settings"):
        
        # Entity Matching Threshold
        entity_label = "Entity similarity threshold"
        if st.session_state.get('similarity_threshold', 0.90) == 0.90:
            entity_label += " <span style='color: #28a745; font-size: 0.85em;'>(optimal)</span>"
        st.markdown(entity_label, unsafe_allow_html=True)
        similarity_threshold = st.slider(
            "entity_slider",
            0.65, 0.95, 0.90, 0.05,
            label_visibility="collapsed",
            help="Minimum cosine similarity for entity matching. Higher = stricter matching.",
            key="similarity_threshold"
        )
        
        # Graph Traversal Hop Limit
        hop_label = "Graph traversal hop limit"
        if st.session_state.get('hop_limit', 5) == 5:
            hop_label += " <span style='color: #28a745; font-size: 0.85em;'>(optimal)</span>"
        st.markdown(hop_label, unsafe_allow_html=True)
        hop_limit = st.slider(
            "hop_slider",
            1, 10, 5, 1,
            label_visibility="collapsed",
            help="Maximum hops in graph traversal. Higher = more paths but slower.",
            key="hop_limit"
        )
        
        
        # High Confidence Threshold
        conf_label = "Refute confidence threshold"
        if st.session_state.get('confidence_threshold', 0.95) == 0.95:
            conf_label += " <span style='color: #28a745; font-size: 0.85em;'>(optimal)</span>"
        st.markdown(conf_label, unsafe_allow_html=True)
        confidence_threshold = st.slider(
            "conf_slider",
            0.6, 1.0, 0.95, 0.05,
            label_visibility="collapsed",
            help="Threshold for definitive refutation. If any evidence exceeds this, claim is marked FAKE.",
            key="confidence_threshold"
        )
        # Minimum Support Threshold
        support_label = "Minimum support threshold"
        if st.session_state.get('min_support_threshold', 0.50) == 0.50:
            support_label += " <span style='color: #28a745; font-size: 0.85em;'>(optimal)</span>"
        st.markdown(support_label, unsafe_allow_html=True)
        min_support_threshold = st.slider(
            "support_slider",
            0.4, 0.7, 0.50, 0.05,
            label_visibility="collapsed",
            help="Minimum confidence for evidence to count as support. Below this = ignored.",
            key="min_support_threshold"
        )
        
        # Number of Search Results
        num_search_results = st.slider(
            "Number of search results",
            1, 10, 5, 1,
            help="Number of results to retrieve from SerpAPI per claim."
        )
        
        # Reset button
        if st.button("🎯 Reset to Optimal Values", use_container_width=True):
            st.session_state['similarity_threshold'] = 0.90
            st.session_state['hop_limit'] = 5
            st.session_state['confidence_threshold'] = 0.95
            st.session_state['min_support_threshold'] = 0.50
            st.rerun()

# Main header
st.markdown('<div class="main-header">🔍 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown("**Powered by Knowledge Graphs and Natural Language Inference**")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Query", "📊 Batch Processing", "📈 Results History", "ℹ️ About"])

with tab1:
    st.markdown("### Single Claim Analysis")
    st.markdown("Enter a news claim (max 280 characters) for fact-checking.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        claim_text = st.text_area(
            "News Claim",
            height=100,
            max_chars=280,
            placeholder="Enter a news claim to verify...",
            help="Enter a short claim or statement to fact-check"
        )
        
    with col2:
        st.metric("Characters", len(claim_text) if claim_text else 0, 
                 delta=f"{280 - len(claim_text) if claim_text else 280} remaining")
    
    use_cumulative = st.checkbox(
        "Use Cumulative Knowledge Graph",
        value=True,
        help="Add evidence to existing KG from previous queries"
    )
    
    if st.button("🔍 Analyse Claim", type="primary", use_container_width=True):
        if not st.session_state.serpapi_key:
            st.error("⚠️ Please enter your SerpAPI key in the sidebar.")
        elif not claim_text or len(claim_text.strip()) == 0:
            st.error("⚠️ Please enter a claim to analyse.")
        else:
            # Processing pipeline
            with st.spinner("Processing claim..."):
                try:
                    # Create progress tracking
                    progress_container = st.container()
                    
                    with progress_container:
                        st.markdown("#### 🔄 Processing Pipeline")
                        stage1 = st.empty()
                        stage2 = st.empty()
                        stage3 = st.empty()
                        stage4 = st.empty()
                        stage5 = st.empty()
                    
                    # Stage 1: Clean and prepare claim
                    stage1.markdown("🔄 **Stage 1:** Preparing claim...")
                    cleaned_text = clean_tweet_for_search(claim_text)
                    claim_id = f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    stage1.markdown("✅ **Stage 1:** <span class='stage-complete'>Claim prepared</span>", unsafe_allow_html=True)
                    
                    # Stage 2: Evidence Retrieval
                    stage2.markdown("🔄 **Stage 2:** Retrieving evidence...")
                    retriever = EvidenceRetriever(
                        api_key=st.session_state.serpapi_key,
                        kg_model=kg_model
                    )
                    evidence_df, search_query = retriever.retrieve_evidence(
                        cleaned_text,
                        num_results=num_search_results
                    )
                    stage2.markdown(f"✅ **Stage 2:** <span class='stage-complete'>Retrieved {len(evidence_df)} evidence triples (Query: '{search_query}')</span>", unsafe_allow_html=True)
                    
                    # Update cumulative KG if enabled
                    if use_cumulative and not st.session_state.cumulative_kg.empty:
                        evidence_df = pd.concat([st.session_state.cumulative_kg, evidence_df], ignore_index=True)
                        evidence_df = evidence_df.drop_duplicates(subset=['subject', 'predicate', 'object'])
                    
                    if use_cumulative:
                        st.session_state.cumulative_kg = evidence_df.copy()
                    
                    # Stage 3: KG Construction
                    stage3.markdown("🔄 **Stage 3:** Building claim knowledge graph...")
                    gt_df = construct_kg_certificate(cleaned_text, claim_id, kg_model)
                    stage3.markdown(f"✅ **Stage 3:** <span class='stage-complete'>Constructed KG with {len(gt_df)} triples</span>", unsafe_allow_html=True)
                    
                    # Stage 4: Entity Matching & Graph Traversal
                    stage4.markdown("🔄 **Stage 4:** Matching entities and finding evidence paths...")
                    
                    matcher = EntityMatcherHybrid(similarity_threshold=similarity_threshold)
                    matcher.load_external_kg(evidence_df)
                    matcher.load_kg_certificate(gt_df)
                    matches = matcher.match_entities()
                    
                    traversal = KnowledgeGraphTraversal(evidence_df, hop_limit=hop_limit)
                    path_results = traversal.find_all_paths(gt_df, matches)
                    
                    paths_found = sum(1 for r in path_results if r['status'] == 'found')
                    stage4.markdown(f"✅ **Stage 4:** <span class='stage-complete'>Matched {len(matches)} entities, found {paths_found} evidence paths</span>", unsafe_allow_html=True)
                    
                    # Extract evidence paths
                    Pi = {}
                    for result in path_results:
                        gt_triple = result['gt_triple']
                        if result['status'] == 'found':
                            Pi[gt_triple] = [p['path'] for p in result['paths']]
                        else:
                            Pi[gt_triple] = []
                    
                    # Stage 5: Factuality Inference
                    stage5.markdown("🔄 **Stage 5:** Inferring factuality...")
                    
                    nli = HybridFactualityInference(
                        llm_model=nli_model if inference_method == "LLM-based NLI" else None,
                        use_llm_nli=(inference_method == "LLM-based NLI"),
                        high_confidence_threshold=confidence_threshold
                    )
                    
                    nli_results = nli.infer_kg_certificate_factuality(gt_df, Pi)
                    stage5.markdown("✅ **Stage 5:** <span class='stage-complete'>Factuality inference complete</span>", unsafe_allow_html=True)
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Verdict Card
                    verdict = nli_results['overall_verdict']
                    confidence = nli_results['summary']['avg_confidence']
                    explanation = nli_results['overall_explanation']
                    
                    verdict_class = {
                        'TRUE NEWS': 'verdict-true',
                        'FAKE NEWS': 'verdict-fake',
                        'UNVERIFIED': 'verdict-unverified'
                    }.get(verdict, 'verdict-unverified')
                    
                    st.markdown(f"""
                    <div class="{verdict_class}">
                        <h3>Verdict: {verdict}</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Explanation:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Claims Extracted", nli_results['summary']['total_triples'])
                    with col2:
                        st.metric("Supported Claims", nli_results['summary']['supported'], 
                                delta="Positive" if nli_results['summary']['supported'] > 0 else None)
                    with col3:
                        st.metric("Refuted Claims", nli_results['summary']['refuted'],
                                delta="Critical" if nli_results['summary']['refuted'] > 0 else None,
                                delta_color="inverse")
                    with col4:
                        st.metric("Avg Confidence", f"{confidence:.2%}")
                    
                    # Detailed Results
                    with st.expander("📋 Detailed Triple Analysis", expanded=True):
                        for i, triple_result in enumerate(nli_results['triple_results'], 1):
                            st.markdown(f"**Claim {i}:** {triple_result['hypothesis']}")
                            
                            verdict_emoji = {
                                'SUPPORTED': '✅',
                                'REFUTED': '❌',
                                'NOT_ENOUGH_INFO': '❓'
                            }.get(triple_result['final_verdict'], '❓')
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"{verdict_emoji} **Verdict:** {triple_result['final_verdict']}")
                                st.markdown(f"**Explanation:** {triple_result['explanation']}")
                            with col2:
                                st.metric("Confidence", f"{triple_result['confidence']:.2%}")
                            
                            if triple_result['evidence_results']:
                                with st.expander(f"View Evidence ({len(triple_result['evidence_results'])} paths)"):
                                    for j, evidence in enumerate(triple_result['evidence_results'], 1):
                                        st.markdown(f"**Evidence Path {j}:**")
                                        st.markdown(f"- {evidence['premise'][:200]}...")
                                        st.markdown(f"- Verdict: {evidence['verdict']} (Confidence: {evidence['confidence']:.2%})")
                            
                            st.markdown("---")
                    
                    # Save to history
                    result_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'claim': claim_text,
                        'verdict': verdict,
                        'confidence': confidence,
                        'num_triples': len(gt_df),
                        'num_evidence_paths': paths_found,
                        'search_query': search_query
                    }
                    st.session_state.processing_history.append(result_entry)
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("### 💾 Export Results")
                    
                    results_json = json.dumps({
                        'claim': claim_text,
                        'claim_id': claim_id,
                        'search_query': search_query,
                        'verdict': verdict,
                        'confidence': confidence,
                        'explanation': explanation,
                        'summary': nli_results['summary'],
                        'triple_results': nli_results['triple_results'],
                        'kg_certificate': gt_df.to_dict(orient='records'),
                        'evidence_kg_size': len(evidence_df)
                    }, indent=2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📄 Download JSON Report",
                            data=results_json,
                            file_name=f"fact_check_{claim_id}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    with col2:
                        csv_data = gt_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download KG Certificate (CSV)",
                            data=csv_data,
                            file_name=f"kg_certificate_{claim_id}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

with tab2:
    st.markdown("### Batch Processing")
    st.markdown("Upload a CSV file with claims in a column named **'text'** for batch fact-checking.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV must contain a 'text' column with claims to verify"
    )
    
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            if 'text' not in df_input.columns:
                st.error("❌ CSV must contain a column named 'text' with the claims.")
            else:
                st.success(f"✅ Loaded {len(df_input)} claims for processing")
                st.dataframe(df_input.head(10), use_container_width=True)
                
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    if not st.session_state.serpapi_key:
                        st.error("⚠️ Please enter your SerpAPI key in the sidebar.")
                    else:
                        # Batch processing
                        results_list = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize components
                        retriever = EvidenceRetriever(
                            api_key=st.session_state.serpapi_key,
                            kg_model=kg_model
                        )
                        
                        # Cumulative evidence KG
                        cumulative_evidence_kg = pd.DataFrame(columns=['source_id', 'subject', 'predicate', 'object'])
                        
                        for idx, row in df_input.iterrows():
                            status_text.text(f"Processing claim {idx + 1}/{len(df_input)}...")
                            
                            try:
                                claim_text = str(row['text'])
                                claim_id = f"batch_claim_{idx}"
                                
                                # Evidence retrieval
                                evidence_df, search_query = retriever.retrieve_evidence(
                                    claim_text,
                                    num_results=num_search_results
                                )
                                
                                # Add to cumulative
                                cumulative_evidence_kg = pd.concat([cumulative_evidence_kg, evidence_df], ignore_index=True)
                                cumulative_evidence_kg = cumulative_evidence_kg.drop_duplicates(subset=['subject', 'predicate', 'object'])
                                
                                # KG construction
                                gt_df = construct_kg_certificate(claim_text, claim_id, kg_model)
                                
                                # Entity matching
                                matcher = EntityMatcherHybrid(similarity_threshold=similarity_threshold)
                                matcher.load_external_kg(cumulative_evidence_kg)
                                matcher.load_kg_certificate(gt_df)
                                matches = matcher.match_entities()
                                
                                # Graph traversal
                                traversal = KnowledgeGraphTraversal(cumulative_evidence_kg, hop_limit=hop_limit)
                                path_results = traversal.find_all_paths(gt_df, matches)
                                
                                # Extract evidence
                                Pi = {}
                                for result in path_results:
                                    gt_triple = result['gt_triple']
                                    if result['status'] == 'found':
                                        Pi[gt_triple] = [p['path'] for p in result['paths']]
                                    else:
                                        Pi[gt_triple] = []
                                
                                # Inference
                                nli = HybridFactualityInference(
                                    llm_model=nli_model if inference_method == "LLM-based NLI" else None,
                                    use_llm_nli=(inference_method == "LLM-based NLI"),
                                    high_confidence_threshold=confidence_threshold
                                )
                                nli_results = nli.infer_kg_certificate_factuality(gt_df, Pi)
                                
                                # Store result
                                results_list.append({
                                    'claim_id': claim_id,
                                    'claim_text': claim_text,
                                    'verdict': nli_results['overall_verdict'],
                                    'confidence': nli_results['summary']['avg_confidence'],
                                    'explanation': nli_results['overall_explanation'],
                                    'num_triples': len(gt_df),
                                    'supported': nli_results['summary']['supported'],
                                    'refuted': nli_results['summary']['refuted'],
                                    'not_enough_info': nli_results['summary']['not_enough_info'],
                                    'search_query': search_query
                                })
                                
                            except Exception as e:
                                results_list.append({
                                    'claim_id': f"batch_claim_{idx}",
                                    'claim_text': claim_text,
                                    'verdict': 'ERROR',
                                    'confidence': 0.0,
                                    'explanation': str(e),
                                    'num_triples': 0,
                                    'supported': 0,
                                    'refuted': 0,
                                    'not_enough_info': 0,
                                    'search_query': ''
                                })
                            
                            progress_bar.progress((idx + 1) / len(df_input))
                        
                        status_text.text("✅ Batch processing complete!")
                        
                        # Display results
                        results_df = pd.DataFrame(results_list)
                        
                        st.markdown("---")
                        st.markdown("## 📊 Batch Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Claims", len(results_df))
                        with col2:
                            true_count = (results_df['verdict'] == 'TRUE NEWS').sum()
                            st.metric("True News", true_count)
                        with col3:
                            fake_count = (results_df['verdict'] == 'FAKE NEWS').sum()
                            st.metric("Fake News", fake_count)
                        with col4:
                            unverified_count = (results_df['verdict'] == 'UNVERIFIED').sum()
                            st.metric("Unverified", unverified_count)
                        
                        # Results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv_download = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            data=csv_download,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

with tab3:
    st.markdown("### Processing History")
    
    if st.session_state.processing_history:
        history_df = pd.DataFrame(st.session_state.processing_history)
        
        st.markdown(f"**Total Queries:** {len(history_df)}")
        
        # Display history
        for idx, entry in enumerate(reversed(st.session_state.processing_history), 1):
            with st.expander(f"Query {len(st.session_state.processing_history) - idx + 1}: {entry['timestamp'][:19]}"):
                st.markdown(f"**Claim:** {entry['claim']}")
                st.markdown(f"**Search Query:** {entry['search_query']}")
                st.markdown(f"**Verdict:** {entry['verdict']}")
                st.markdown(f"**Confidence:** {entry['confidence']:.2%}")
                st.markdown(f"**Triples Extracted:** {entry['num_triples']}")
                st.markdown(f"**Evidence Paths Found:** {entry['num_evidence_paths']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.processing_history = []
            st.rerun()
    else:
        st.info("No processing history yet. Analyse some claims to see them here!")

with tab4:
    st.markdown("### About This System")
    
    st.markdown("""
    ## 🔍 Fake News Detection System
    
    This Decision Support System (DSS) uses advanced Knowledge Graph techniques and Natural Language Inference 
    to verify factual claims in short news texts.
    
    ### 📚 Methodology
    
    The system implements a multi-stage pipeline:
    
    1. **Evidence Retrieval**: Uses SerpAPI to gather relevant information from reliable sources
    2. **Knowledge Graph Construction**: Extracts structured triples from both claims and evidence
    3. **Entity Matching**: Identifies corresponding entities using hybrid semantic similarity
    4. **Graph Traversal**: Finds evidence paths using bidirectional breadth-first search
    5. **Factuality Inference**: Uses either LLM-based or voting NLI to determine veracity
    
    ### 🎯 Key Features
    
    - **Single or Batch Processing**: Analyse individual claims or process multiple claims at once
    - **Cumulative Learning**: Build a growing knowledge graph from previous queries
    - **Flexible Model Selection**: Choose from multiple state-of-the-art LLMs
    - **Multiple Inference Methods**: LLM-based NLI or traditional voted NLI
    - **Transparent Results**: View detailed reasoning and evidence paths
    - **Export Functionality**: Download results in JSON or CSV format
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **LLMs**: Ollama (gpt-oss, gemma3, qwen3, deepseek-r1)
    - **NLI**: RoBERTa-large-MNLI / Custom LLM-based
    - **Embeddings**: Sentence Transformers (all-mpnet-base-v2)
    - **Evidence Retrieval**: SerpAPI
    
    ### 📖 Citation
    
    If you use this system in your research, please cite the original methodology paper.
    
    ### ⚠️ Disclaimer
    
    This system is designed to assist in fact-checking but should not be the sole determinant of factuality. 
    Always verify important information through multiple reliable sources.
    """)
    
    st.markdown("---")
    st.markdown("**Version**: 1.0.0 | **Last Updated**: 2025")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        Powered by Knowledge Graphs & NLI
    </div>
    """,
    unsafe_allow_html=True
)
