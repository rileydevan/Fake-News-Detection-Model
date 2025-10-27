# ⚙️ Configuration Guide

Detailed guide to configuring the Fake News Detection DSS for optimal performance.

## 🔧 Basic Configuration

### SerpAPI Key

**Required**: Yes  
**Location**: Sidebar → Configuration  
**Purpose**: Retrieves evidence from Google search results

**How to get:**
1. Visit [https://serpapi.com](https://serpapi.com)
2. Sign up for free account (100 searches/month)
3. Navigate to Dashboard → API Key
4. Copy your key

**Troubleshooting:**
- ❌ "Invalid API key" → Check key is copied correctly
- ❌ "Rate limit exceeded" → Upgrade plan or wait for reset
- ❌ "No results" → Check internet connection

---

## 🤖 Model Configuration

### KG Construction Model

**Location**: Sidebar → Model Configuration  
**Purpose**: Extracts triples from claims and evidence

**Available Models:**

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| `gpt-oss:20b` | 20B | Slow | ⭐⭐⭐⭐⭐ | Best accuracy, research |
| `gemma3:27b` | 27B | Slow | ⭐⭐⭐⭐⭐ | High quality, production |
| `qwen3:30b` | 30B | Slow | ⭐⭐⭐⭐ | Balanced, multilingual |
| `deepseek-r1:14b` | 14B | Fast | ⭐⭐⭐⭐ | Good speed/accuracy balance |
| `gemma3:12b` | 12B | Fastest | ⭐⭐⭐ | Quick testing, demos |

**Recommendation:**
- **Production**: `gpt-oss:20b` or `gemma3:27b`
- **Development**: `deepseek-r1:14b`
- **Quick testing**: `gemma3:12b`

### Inference Method

**Location**: Sidebar → Model Configuration  
**Purpose**: Determines how claims are verified against evidence

#### Option 1: LLM-based NLI

**Pros:**
- ✅ More nuanced understanding
- ✅ Better temporal reasoning
- ✅ Handles numerical tolerance
- ✅ Detailed explanations

**Cons:**
- ❌ Slower processing
- ❌ Higher resource usage
- ❌ Requires good LLM

**Best for:** High-stakes verification, research, detailed analysis

**NLI Model Selection:**
- Same models as KG Construction
- Can use different model than KG construction
- Recommend: `gpt-oss:20b` for best reasoning

#### Option 2: Voted NLI

**Pros:**
- ✅ Fast processing
- ✅ Lower resource usage
- ✅ Deterministic results
- ✅ Proven accuracy (RoBERTa)

**Cons:**
- ❌ Less nuanced
- ❌ May miss subtle contradictions
- ❌ Basic confidence scores

**Best for:** Batch processing, quick verification, resource constraints

**Technical Details:**
- Uses `roberta-large-mnli`
- Pre-trained on Multi-NLI dataset
- 355M parameters

---

## 🎛️ Advanced Settings

### Entity Matching Threshold

**Default**: 0.8  
**Range**: 0.5 - 1.0  
**Purpose**: Minimum similarity for entity matching

**How it works:**
- Entities from claim matched to entities in evidence
- Uses semantic similarity (embeddings)
- Higher threshold = stricter matching

**Tuning Guide:**

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.9-1.0 | Very strict | Exact entity matching needed |
| 0.8-0.9 | Balanced | Default, works well generally |
| 0.7-0.8 | Lenient | Catch more matches, some false positives |
| 0.5-0.7 | Very lenient | Broad matching, noisy results |

**Examples:**
- 0.95: "COVID-19" matches "COVID-19" but not "coronavirus"
- 0.80: "COVID-19" matches "coronavirus", "pandemic"
- 0.65: "COVID-19" matches "virus", "disease"

**Recommendation:**
- Start with 0.8
- Increase to 0.9 if getting irrelevant matches
- Decrease to 0.7 if missing obvious connections

### Graph Traversal Hop Limit

**Default**: 7  
**Range**: 3 - 10  
**Purpose**: Maximum path length in evidence graph

**How it works:**
- System searches for paths between entities
- Each hop = one relationship in the graph
- Longer paths = more tenuous connections

**Tuning Guide:**

| Hop Limit | Search Space | Speed | Use Case |
|-----------|--------------|-------|----------|
| 3-4 | Small | Fast | Direct evidence only |
| 5-7 | Medium | Balanced | Default, good coverage |
| 8-10 | Large | Slow | Comprehensive search |

**Trade-offs:**
- **Higher limit:**
  - ✅ Find more evidence paths
  - ✅ Connect distant entities
  - ❌ Slower processing
  - ❌ May find weak connections

- **Lower limit:**
  - ✅ Faster processing
  - ✅ Only direct evidence
  - ❌ May miss valid connections
  - ❌ More "no path found" results

**Recommendation:**
- Start with 7
- Increase to 9-10 for complex topics
- Decrease to 5 for simple claims

### High Confidence Threshold

**Default**: 0.7  
**Range**: 0.5 - 1.0  
**Purpose**: Minimum confidence for definitive refutation

**How it works:**
- **Critical parameter** for verdict logic
- If ANY evidence refutes claim with confidence ≥ threshold → FAKE NEWS
- Otherwise, use weighted voting

**Tuning Guide:**

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.9-1.0 | Very strict | Only certain refutations count |
| 0.7-0.8 | Balanced | Default, catches most fake news |
| 0.6-0.7 | Lenient | More sensitive to refutations |
| 0.5-0.6 | Very lenient | May over-classify as fake |

**Impact on Results:**
- **Higher (0.9):** Fewer FAKE NEWS verdicts, more UNVERIFIED
- **Lower (0.6):** More FAKE NEWS verdicts, fewer UNVERIFIED

**Recommendation:**
- Start with 0.7
- Increase to 0.8-0.9 if too many false positives
- Decrease to 0.6 if missing obvious fake news

### Minimum Support Threshold

**Default**: 0.5  
**Range**: 0.3 - 0.9  
**Purpose**: Minimum confidence for evidence to count as support

**How it works:**
- Evidence with confidence < threshold ignored in voting
- Prevents weak evidence from influencing verdict
- Only applies to supporting evidence (refuting evidence always counts)

**Tuning Guide:**
- **0.7-0.9**: Only strong support counts
- **0.5-0.7**: Balanced (default)
- **0.3-0.5**: Even weak support counts

**Recommendation:** Keep at 0.5 unless specific needs

### Number of Search Results

**Default**: 5  
**Range**: 1 - 10  
**Purpose**: Number of web results to retrieve per query

**Trade-offs:**

| Count | API Calls | Coverage | Speed | Cost |
|-------|-----------|----------|-------|------|
| 1-3 | Low | Limited | Fast | Low |
| 4-6 | Medium | Good | Balanced | Medium |
| 7-10 | High | Comprehensive | Slow | High |

**Considerations:**
- Each result = 1 API call
- More results = better evidence coverage
- SerpAPI free tier: 100 searches/month
- Batch processing accumulates results

**Recommendation:**
- **Single queries**: 5-7 results
- **Batch processing**: 3-5 results (to stay within limits)
- **High-stakes**: 10 results for maximum coverage

---

## 🎯 Configuration Profiles

Pre-configured settings for common use cases:

### Profile 1: Quick Testing
```
Model: gemma3:12b
Inference: Voted NLI
Entity Threshold: 0.75
Hop Limit: 5
Search Results: 3
```
**Use for:** Quick tests, demos, development

### Profile 2: Balanced (Default)
```
Model: gpt-oss:20b
Inference: LLM-based NLI
Entity Threshold: 0.8
Hop Limit: 7
High Confidence: 0.7
Search Results: 5
```
**Use for:** General purpose, production

### Profile 3: High Accuracy
```
Model: gemma3:27b
Inference: LLM-based NLI
Entity Threshold: 0.85
Hop Limit: 9
High Confidence: 0.75
Search Results: 8
```
**Use for:** Research, critical decisions, high-stakes

### Profile 4: Batch Processing
```
Model: deepseek-r1:14b
Inference: Voted NLI
Entity Threshold: 0.8
Hop Limit: 6
Search Results: 3
```
**Use for:** Processing many claims efficiently

### Profile 5: Comprehensive Analysis
```
Model: gpt-oss:20b
Inference: LLM-based NLI (gpt-oss:20b)
Entity Threshold: 0.8
Hop Limit: 10
High Confidence: 0.7
Search Results: 10
```
**Use for:** In-depth analysis, academic research

---

## 📊 Performance Optimization

### Speed Optimization
1. Use smaller models (gemma3:12b, deepseek-r1:14b)
2. Choose Voted NLI over LLM-based
3. Reduce search results to 3
4. Lower hop limit to 5
5. Disable cumulative KG for single queries

### Accuracy Optimization
1. Use larger models (gpt-oss:20b, gemma3:27b)
2. Choose LLM-based NLI
3. Increase search results to 7-10
4. Increase hop limit to 9-10
5. Enable cumulative KG

### Cost Optimization (API Calls)
1. Reduce search results
2. Enable cumulative KG for multiple queries
3. Batch process related claims together
4. Cache results for repeated queries

### Memory Optimization
1. Use smaller models
2. Process in smaller batches
3. Clear cumulative KG periodically
4. Reduce number of search results

---

## 🔍 Troubleshooting Configuration Issues

### Issue: Too many "UNVERIFIED" results
**Solution:**
- Increase search results (7-10)
- Lower entity matching threshold (0.75)
- Increase hop limit (9-10)
- Lower high confidence threshold (0.65)

### Issue: Too many "FAKE NEWS" results (false positives)
**Solution:**
- Increase high confidence threshold (0.8-0.9)
- Increase entity matching threshold (0.85)
- Use LLM-based NLI for better reasoning
- Increase minimum support threshold (0.6)

### Issue: Slow processing
**Solution:**
- Use smaller model
- Switch to Voted NLI
- Reduce search results (3)
- Lower hop limit (5)

### Issue: Out of memory
**Solution:**
- Use smaller model (gemma3:12b)
- Reduce batch size
- Clear cumulative KG
- Close other applications

### Issue: "No paths found"
**Solution:**
- Increase hop limit (9-10)
- Lower entity matching threshold (0.7)
- Increase search results (7-10)
- Check evidence quality

---

## 💡 Best Practices

1. **Start with defaults** and adjust based on results
2. **Test with known claims** (both true and false) to calibrate
3. **Monitor API usage** when using SerpAPI
4. **Document your configuration** for reproducibility
5. **Use appropriate profile** for your use case
6. **Batch similar claims** to reuse evidence
7. **Clear cumulative KG** when switching topics

---

## 📝 Configuration File (Future)

Configuration can be saved/loaded via `config.py`:

```python
# Example custom configuration
CUSTOM_CONFIG = {
    'kg_model': 'gpt-oss:20b',
    'inference_method': 'llm',
    'nli_model': 'gpt-oss:20b',
    'similarity_threshold': 0.8,
    'hop_limit': 7,
    'high_conf_threshold': 0.7,
    'min_support_threshold': 0.5,
    'num_search_results': 5
}
```

To use: Modify values in `config.py` and restart application.

---

**Questions?** See README.md or create an issue.
