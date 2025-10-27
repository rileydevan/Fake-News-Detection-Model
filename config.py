# config.py - Configuration and Constants

# Available Ollama models
AVAILABLE_MODELS = [
    # Cloud models (if accessible)
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-coder:480b-cloud",
    
    # Local models (downloaded)
    "gpt-oss:120b",
    "gpt-oss:20b",
    "gemma3:27b",
    "gemma3:12b",
    "gemma3:4b",
    "gemma3:1b",
    "deepseek-r1:8b",
    "qwen3-coder:30b",
    "qwen3:30b",
    "qwen3:8b",
    "qwen3:4b",
    "phi4"
]

# Triple extraction system prompt
SYSTEM_PROMPT_TRIPLE_EXTRACTION = """
You extract subject-predicate-object triples from SHORT highlighted snippets (search-result snippets).
Your output MUST be a single Python list, wrapped EXACTLY in <python> and </python> tags.
Return NOTHING else (no explanations, no analysis, no think blocks, no backticks).

Rules (strict):

1) Entities (subjects/objects): keep SHORT and minimal - proper nouns, places, organizations, specific concepts.
   - Remove ALL articles: "the WHO" becomes "WHO", "a vaccine" becomes "vaccine"
   - Use canonical Wikipedia-style names: "COVID-19" not "Covid", "United States" not "U.S."
   - Strip titles unless part of proper name: "President Ramaphosa" becomes "Cyril Ramaphosa"
   - Keep events/concepts minimal: "South Africa's tour to NZ" becomes "New Zealand tour"
   - Typical length: 1-4 words maximum
   - Examples: "Pfizer", "South Africa", "COVID-19", "Aspen", "Johnson & Johnson", "vaccine doses"

2) Coreference resolution (CRITICAL): NEVER output pronouns as entities.
   - "We" must resolve to the actual organization or entity being discussed
   - "It", "This", "That" must resolve to the specific thing referenced
   - "They", "Them" must resolve to the specific group or entities
   - "The country", "The company" must use the actual name
   - If referent is unclear, use the most likely entity from context or "unknown entity"
   - NO EXCEPTIONS: pronouns break entity matching across knowledge graphs

3) Predicates: capture the relationship clearly and completely.
   - Must be self-contained and semantically meaningful
   - MINIMUM LENGTH: predicates must be at least 2 words (never just "from", "with", "to", "has")
   - Include temporal/modal/quantitative markers: "announced on March 5", "would manufacture", "allocated 60% to"
   - Target length: 2-8 words ideal, 10 words maximum
   - Good: "tested positive for", "ruled out of", "secured doses from", "announced plans to"
   - Bad: "from", "with", "has", "is", "to"

4) Direct relationships only:
   - Only create triples where there is a DIRECT relationship between two entities in the snippet
   - Each entity typically relates to 2-3 other entities maximum
   - Don't force connections that aren't explicitly stated

5) Split multi-claim sentences into ATOMIC triples (one relation per claim).
   - "X did Y and Z" becomes two triples: [X, did, Y] and [X, did, Z]

6) Each triple is EXACTLY three non-empty strings: ["Subject", "Predicate", "Object"]

7) Do NOT invent facts or add background knowledge not in the snippet

8) Upper bound: output at most 8 triples per snippet

Format:
<python>
[
 ["entity1", "relation", "entity2"],
 ...
]
</python>

Now extract triples from the following input:
{input}
"""

# LLM-based NLI system prompt
NLI_SYSTEM_PROMPT = """You are a precise fact-checking system performing Natural Language Inference (NLI) to verify claims against evidence from knowledge graphs.

## TASK
Determine if PREMISE (evidence) supports, refutes, or is unrelated to HYPOTHESIS (claim).

## OUTPUT LABELS
- **ENTAILMENT**: Premise confirms the hypothesis
- **CONTRADICTION**: Premise refutes the hypothesis (direct conflict only)
- **NEUTRAL**: Premise is unrelated or insufficient

## CRITICAL RULES FOR ACCURACY

### 1. NUMERICAL TOLERANCE
- **±10% margin**: Values within 10% are ENTAILMENT (e.g., "50" supports "48")
- **±20% margin**: Values within 20% but >10% are NEUTRAL (caution zone)
- **>20% difference**: Clear CONTRADICTION
- Round numbers (100, 1000, 1M) get ±15% margin
- Percentages follow same rules: 45% vs 50% = ENTAILMENT

### 2. TEMPORAL REASONING
- **Same year = ENTAILMENT**: "2024" matches "January 2024" or "early 2024"
- **Adjacent periods = ENTAILMENT**: "late 2023" supports "2023/2024"
- **Season/month mismatch in same year = NEUTRAL**: Don't contradict unless year differs
- **Sequence matters**: "after X" contradicts "before X"
- **Duration tolerance**: "3 weeks" vs "21 days" = ENTAILMENT

### 3. ENTITY MATCHING
- **Aliases = Same entity**: "USA" = "United States" = "America"
- **Titles = Person**: "President Biden" = "Joe Biden"
- **Abbreviations**: "WHO" = "World Health Organization"
- **Partial names**: "Trump" = "Donald Trump"
- Be liberal with entity matching, strict with relationship matching

### 4. SEMANTIC RELATIONSHIPS
- **Synonymous predicates**: "won" = "defeated opponent" = "victorious"
- **Hierarchical relations**: "California" in "USA" supports "USA" claims about "California"
- **Causal chains**: If A→B→C, then A relates to C
- **Negations are critical**: "not X" contradicts "X"

### 5. CONTRADICTION REQUIREMENTS (BE STRICT)
Only mark CONTRADICTION when:
- Direct factual conflict (different values, opposite statements)
- Mutually exclusive claims
- Clear temporal impossibility
**NOT contradiction**:
- Missing information (use NEUTRAL)
- Partial information (use NEUTRAL or ENTAILMENT if consistent)
- Different framing of same fact (use ENTAILMENT)

## OUTPUT FORMAT
```json
{
    "verdict": "ENTAILMENT|CONTRADICTION|NEUTRAL",
    "confidence": 0.85,
    "reasoning": "One brief sentence",
    "key_factors": ["factor1", "factor2", "factor3"]
}
```

## CONFIDENCE GUIDELINES
- 0.90-1.00: Explicit match/conflict, clear evidence
- 0.75-0.90: Strong inference, minor ambiguity
- 0.60-0.75: Moderate confidence, some uncertainty
- Below 0.60: Weak inference, high uncertainty

## FINAL REMINDERS
- **Always output valid JSON**
- **Reasoning must be ONE sentence maximum**
- **Apply numerical tolerance consistently**
- **Temporal reasoning: same year = support**
- **Be strict with CONTRADICTION (only direct conflicts)**
- **When uncertain between NEUTRAL/CONTRADICTION, choose NEUTRAL**"""

# Claim query generation prompt
CLAIM_QUERY_PROMPT = """You are a search query generator. Given a claim, generate a concise, effective search query to find evidence on Google.

Rules:
1. Extract the core factual claim
2. Remove opinion words and filler
3. Include key entities and relationships
4. Keep it under 10 words
5. Focus on verifiable facts

Return ONLY the search query, nothing else.

Claim: {claim}

Search query:"""

# Entity matching aliases (domain-specific for COVID-19)
ENTITY_ALIASES = {
    'covid-19': ['corona', 'coronavirus', 'covid', 'sars-cov-2', 'covid19', 'covid-19 pandemic'],
    'cyril ramaphosa': ['ramaphosa', 'president ramaphosa', 'cyril', 'cr', 'president'],
    'south africa': ['sa', 'rsa', 'south african', 'republic of south africa'],
    'vaccine': ['vaccination', 'jab', 'shot', 'dose', 'vaccinate'],
    'pfizer': ['pfizer-biontech', 'biontech', 'pfizer vaccine'],
    'johnson & johnson': ['j&j', 'jnj', 'johnson and johnson', 'janssen', 'j and j'],
    'astrazeneca': ['az', 'astra zeneca', 'oxford-astrazeneca'],
    'omicron': ['omicron variant', 'b.1.1.529'],
    'delta': ['delta variant', 'b.1.617.2'],
    'lockdown': ['lock down', 'shutdown', 'quarantine'],
    'hydroxychloroquine': ['hcq', 'hydroxy chloroquine'],
    'ivermectin': ['iver mectin'],
}

# Default parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_HOP_LIMIT = 7
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MIN_SUPPORT_THRESHOLD = 0.5
DEFAULT_VOTE_SCALE_FACTOR = 10
DEFAULT_NUM_SEARCH_RESULTS = 5

# Embedding model
EMBEDDING_MODEL = 'all-mpnet-base-v2'

# NLI fallback model
NLI_FALLBACK_MODEL = 'roberta-large-mnli'
