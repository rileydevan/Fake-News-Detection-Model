# 🔍 Fake News Detection Decision Support System

A comprehensive Decision Support System (DSS) for detecting fake news using Knowledge Graphs and Natural Language Inference.

## 📋 Overview

This DSS implements a multi-stage fake news detection pipeline:

1. **Evidence Retrieval**: Gathers relevant information from web sources via SerpAPI
2. **Knowledge Graph Construction**: Extracts structured triples from claims and evidence
3. **Entity Matching**: Identifies corresponding entities using hybrid semantic similarity
4. **Graph Traversal**: Finds evidence paths using bidirectional breadth-first search
5. **Factuality Inference**: Determines claim veracity using LLM-based or voted NLI

## 🚀 Features

- ✅ **Single or Batch Processing**: Analyze individual claims or process CSV files
- ✅ **Cumulative Learning**: Build a growing knowledge graph from previous queries
- ✅ **Flexible Model Selection**: Choose from multiple state-of-the-art LLMs
- ✅ **Multiple Inference Methods**: LLM-based NLI or traditional voted NLI
- ✅ **Transparent Results**: View detailed reasoning and evidence paths
- ✅ **Export Functionality**: Download results in JSON or CSV format
- ✅ **Interactive Visualizations**: Explore knowledge graphs visually

## 🛠️ Installation

### Prerequisites

1. **Python 3.9+**
2. **Ollama** (for local LLM inference)
3. **SerpAPI Account** (for evidence retrieval)

### Step 1: Install Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Step 2: Pull Required Models

```bash
ollama pull gpt-oss:20b
ollama pull gemma3:27b
ollama pull qwen3:30b
ollama pull deepseek-r1:14b
ollama pull gemma3:12b
```

### Step 3: Clone Repository

```bash
git clone <repository-url>
cd fake-news-dss
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Get SerpAPI Key

1. Sign up at [SerpAPI](https://serpapi.com/)
2. Get your API key from the dashboard
3. Keep it ready for configuration in the app

## 🎯 Usage

### Start the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### Configuration

1. **Enter SerpAPI Key**: In the sidebar, enter your SerpAPI key
2. **Select Models**: Choose KG construction model and inference method
3. **Adjust Settings**: Optionally configure advanced parameters

### Single Query Mode

1. Navigate to the **"Single Query"** tab
2. Enter a claim (max 280 characters)
3. Choose whether to use cumulative knowledge graph
4. Click **"Analyze Claim"**
5. View results and download reports

### Batch Processing Mode

1. Navigate to the **"Batch Processing"** tab
2. Upload a CSV file with a column named **"text"** containing claims
3. Click **"Process Batch"**
4. Download results when complete

### CSV Format for Batch Processing

Your CSV must have a column named **"text"**:

```csv
text
"Breaking: New vaccine approved by FDA"
"Scientists discover cure for common cold"
"Local mayor announces infrastructure plan"
```

## 📊 Understanding Results

### Verdict Types

- **TRUE NEWS**: Claims are supported by evidence with high confidence
- **FAKE NEWS**: Claims are refuted by evidence (including high-confidence refutations)
- **UNVERIFIED**: Insufficient evidence to verify claims

### Confidence Scores

- **0.90-1.00**: High confidence, strong evidence
- **0.70-0.90**: Moderate confidence, good evidence
- **0.50-0.70**: Low confidence, weak evidence
- **Below 0.50**: Very low confidence, insufficient evidence

### Triple Analysis

Each claim is broken down into subject-predicate-object triples:
- **Supported**: Evidence confirms the claim
- **Refuted**: Evidence contradicts the claim
- **Not Enough Info**: No clear evidence found

## 🔧 Configuration Options

### Model Selection

**KG Construction Models:**
- `gpt-oss:20b` - Recommended for best accuracy
- `gemma3:27b` - Good balance of speed and accuracy
- `qwen3:30b` - High quality triples
- `deepseek-r1:14b` - Fast processing
- `gemma3:12b` - Fastest option

**Inference Methods:**
- **LLM-based NLI**: Uses LLM for nuanced reasoning (more accurate but slower)
- **Voted NLI**: Uses transformer models with voting (faster but less nuanced)

### Advanced Settings

- **Entity Matching Threshold** (0.5-1.0): Minimum similarity for entity matching
- **Graph Traversal Hop Limit** (3-10): Maximum hops in graph search
- **High Confidence Threshold** (0.5-1.0): Threshold for definitive refutation
- **Number of Search Results** (1-10): Results to retrieve from SerpAPI

## 📁 Project Structure

```
fake-news-dss/
├── main.py                      # Main Streamlit application
├── config.py                    # Configuration and constants
├── kg_construction.py           # Knowledge graph construction
├── evidence_retrieval.py        # Evidence retrieval via SerpAPI
├── entity_matching.py           # Hybrid entity matching
├── graph_traversal.py           # Bidirectional BFS traversal
├── factuality_inference.py      # Factuality inference system
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔬 Methodology

### 1. Evidence Retrieval
- Uses LLM to generate optimal search queries from claims
- Queries SerpAPI to gather relevant web sources
- Constructs External Knowledge Graph (Ge) from search results

### 2. Knowledge Graph Certificate
- Extracts triples from input claim using LLM
- Creates structured representation (Gt) of claim components
- Normalizes entities and predicates for matching

### 3. Entity Matching
- **Hybrid Approach**:
  - First tries domain-specific alias matching (e.g., "COVID-19" = "coronavirus")
  - Falls back to semantic similarity using FAISS vector search
- Uses sentence transformers (all-mpnet-base-v2) for embeddings

### 4. Graph Traversal
- **Bidirectional BFS Algorithm**:
  - Searches from both claim entities simultaneously
  - Finds shortest paths between matched entities
  - Returns evidence chains connecting claim components

### 5. Factuality Inference
- **Two-Tier Logic**:
  1. Any high-confidence refutation → FAKE NEWS (strict)
  2. Otherwise → Weighted voting based on confidence scores
- **Inference Methods**:
  - LLM-based NLI: Uses LLM with specialized prompting
  - Voted NLI: Uses RoBERTa-large-MNLI with confidence weighting

## 📈 Performance Tips

### For Faster Processing
1. Use smaller models (gemma3:12b, deepseek-r1:14b)
2. Reduce number of search results
3. Lower hop limit for graph traversal
4. Use Voted NLI instead of LLM-based NLI

### For Better Accuracy
1. Use larger models (gpt-oss:20b, gemma3:27b)
2. Increase number of search results (7-10)
3. Use LLM-based NLI
4. Enable cumulative knowledge graph

### For Cost Optimization
1. Batch process claims to reuse evidence KG
2. Use cumulative mode to reduce API calls
3. Cache results for repeated queries

## 🐛 Troubleshooting

### Ollama Not Found
```bash
# Make sure Ollama is running
ollama serve

# Test if models are available
ollama list
```

### SerpAPI Errors
- **Rate Limit**: Free tier has limits; consider upgrading
- **Invalid Key**: Check key is entered correctly in sidebar
- **No Results**: Try different search terms or increase result count

### Memory Issues
- Reduce batch size
- Use smaller models
- Clear cumulative KG periodically

### Slow Performance
- Check Ollama is using GPU (if available)
- Reduce model size
- Lower number of search results

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@software{fake_news_dss_2025,
  title={Fake News Detection Decision Support System},
  author={[Your Name]},
  year={2025},
  note={Knowledge Graph-based Fake News Detection}
}
```

## ⚠️ Disclaimer

This system is designed to **assist** in fact-checking but should **not be the sole determinant** of factuality. Always verify important information through multiple reliable sources. The system's accuracy depends on:
- Quality of available evidence
- Model selection and configuration
- Domain-specific knowledge coverage

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional domain-specific aliases
- Enhanced NLI prompts
- Alternative evidence sources
- Performance optimizations
- UI/UX improvements

## 📄 License

[Specify your license here]

## 📧 Contact

For questions, issues, or feedback, please [contact information or create an issue].

---

**Built using Streamlit, Ollama, and Knowledge Graphs**
