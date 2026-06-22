# Modules API Reference

All architecture modules are in `pyrila.modules`.

## Pipeline Modules

---

### `RILATokenizer`

Converts token IDs to embeddings with positional encoding and layer normalization.

**Constructor:**
```python
from pyrila.modules import RILATokenizer
tokenizer = RILATokenizer(config: RILAConfig)
```

**Parameters:**
- `config.vocab_size` — Vocabulary size for embedding table
- `config.hidden_dim` — Output embedding dimension
- `config.max_sequence_length` — Maximum supported sequence length
- `config.dropout` — Dropout rate

**Forward:**
```python
embeddings = tokenizer(token_ids: torch.Tensor) -> torch.Tensor
# token_ids: (batch, seq_len) with values in [0, vocab_size)
# Returns: (batch, seq_len, hidden_dim) embeddings
```

Raises `SequenceTooLongError` if `seq_len > max_sequence_length`.

**Example:**
```python
from pyrila.modules import RILATokenizer
from pyrila.presets import rila_small

config = rila_small()
tokenizer = RILATokenizer(config)
ids = torch.randint(0, config.vocab_size, (2, 64))
embeddings = tokenizer(ids)  # (2, 64, 64)
```

---

### `ContextCellBuilder`

Partitions token sequences into fixed-size Context Cells with positional metadata and structural references to adjacent cells.

**Constructor:**
```python
from pyrila.modules import ContextCellBuilder
builder = ContextCellBuilder(config: RILAConfig)
```

**Parameters:**
- `config.cell_size` — Number of tokens per cell
- `config.hidden_dim` — Token embedding dimension

**Forward:**
```python
cells = builder(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> List[List[ContextCellData]]
# embeddings: (batch, seq_len, hidden_dim)
# attention_mask: (batch, seq_len), 1=valid, 0=padding
# Returns: List of lists of ContextCellData, one per batch element
```

Each `ContextCellData` contains: `tokens`, `position`, `length`, `prev_ref`, `next_ref`, `metadata`.

**Example:**
```python
builder = ContextCellBuilder(config)
embeddings = torch.randn(2, 128, config.hidden_dim)
mask = torch.ones(2, 128, dtype=torch.long)
batch_cells = builder(embeddings, mask)
# batch_cells[0] has ceil(128/cell_size) cells
```

---

### `CellEncoder`

Encodes Context Cells into 4D representations: `CCE = [S, R, T, I]` where each component is an L2-normalized vector of dimension `hidden_dim`.

**Constructor:**
```python
from pyrila.modules import CellEncoder
encoder = CellEncoder(config: RILAConfig)
```

**Parameters:**
- `config.hidden_dim` — Dimension of each sub-vector (S, R, T, I)
- `config.num_heads` — Attention heads for internal transformer
- `config.dropout` — Dropout rate

**Forward:**
```python
cce = encoder(cell_tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor
# cell_tokens: (batch, cell_size, hidden_dim)
# lengths: (batch,) actual token counts
# Returns: (batch, 4 * hidden_dim) CCE vectors
```

**Batch encoding:**
```python
cce = encoder.encode_batch(cells: List[ContextCellData]) -> torch.Tensor
# Returns: (num_cells, 4 * hidden_dim)
```

**Example:**
```python
encoder = CellEncoder(config)
tokens = torch.randn(4, config.cell_size, config.hidden_dim)
lengths = torch.full((4,), config.cell_size, dtype=torch.long)
cce = encoder(tokens, lengths)  # (4, 4*hidden_dim)
```

---

### `RecursiveContextIndex`

Dynamic affinity graph over encoded Context Cells. Builds graph `G = (V, E)` where V = cells and E = learned affinity edges, enabling fast retrieval without processing the full context.

**Constructor:**
```python
from pyrila.modules import RecursiveContextIndex
index = RecursiveContextIndex(config: RILAConfig)
```

**Parameters:**
- `config.cell_encoding_dim` — Dimension of cell encodings (4 * hidden_dim)
- `config.hidden_dim` — Hidden dimension for affinity network
- `config.index_affinity_threshold` — Minimum affinity for edge creation
- `config.max_index_connections` — Maximum edges per node

**Methods:**
```python
index.build_index(cell_encodings: torch.Tensor) -> None
# cell_encodings: (num_cells, cell_encoding_dim)

scores, indices = index.query(query_vector: torch.Tensor, top_k: int) -> Tuple[Tensor, Tensor]
# query_vector: (cell_encoding_dim,)
# Returns: (k,) scores and indices

neighbors = index.expand_from(cell_indices: torch.Tensor, hops: int = 1) -> torch.Tensor
# Returns: neighboring cell indices
```

Raises `IndexNotBuiltError` if `query()` or `expand_from()` is called before `build_index()`.

**Example:**
```python
index = RecursiveContextIndex(config)
encodings = torch.randn(8, config.cell_encoding_dim)
index.build_index(encodings)

query = torch.randn(config.cell_encoding_dim)
scores, indices = index.query(query, top_k=3)
neighbors = index.expand_from(indices, hops=1)
```

---

### `RecursiveContextExplorer`

Iterative retrieval with soft top-k (training, differentiable) and hard top-k (inference, exact). Progressively discovers relevant Context Cells using query projection, scoring, and graph-based expansion.

**Constructor:**
```python
from pyrila.modules import RecursiveContextExplorer
rce = RecursiveContextExplorer(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Input cognitive state dimension
- `config.cell_encoding_dim` — Cell encoding dimension
- `config.hidden_dim` — Hidden layer dimension
- `config.initial_top_k` — Initial cells to retrieve
- `config.max_retrieval_expansions` — Maximum expansion iterations
- `config.retrieval_score_threshold` — Sufficiency threshold
- `config.use_soft_topk` — Enable differentiable selection (training)
- `config.soft_topk_temperature` — Temperature for soft selection

**Forward:**
```python
result = rce(cognitive_state: torch.Tensor, index: RecursiveContextIndex) -> RetrievalResult
# cognitive_state: (cognitive_dim,)
# Returns: RetrievalResult with cell_indices, cell_encodings, relevance_scores, num_retrievals
```

Raises `IndexNotBuiltError` if the index hasn't been built.

**Example:**
```python
rce = RecursiveContextExplorer(config)
index = RecursiveContextIndex(config)
index.build_index(cell_encodings)

cog_state = torch.randn(config.cognitive_dim)
result = rce(cog_state, index)
print(f"Retrieved {result.cell_indices.numel()} cells in {result.num_retrievals} steps")
```

---

### `WorkingContext`

Aggregates retrieved cells into a cognitive-dimension representation using multi-head attention weighted by relevance scores. Within a reasoning session the working context only grows (monotonic).

**Constructor:**
```python
from pyrila.modules import WorkingContext
wc = WorkingContext(config: RILAConfig)
```

**Parameters:**
- `config.cell_encoding_dim` — Input cell encoding dimension
- `config.cognitive_dim` — Output dimension
- `config.num_heads` — Attention heads
- `config.attention_dropout` — Attention dropout

**Forward:**
```python
context = wc(cell_encodings: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor
# cell_encodings: (batch, num_retrieved, cell_encoding_dim)
# relevance_scores: (batch, num_retrieved) in [0, 1]
# Returns: (batch, cognitive_dim)
```

**Reset:**
```python
wc.reset()  # Clear stored cells for a new session
```

**Example:**
```python
wc = WorkingContext(config)
encodings = torch.randn(2, 4, config.cell_encoding_dim)
scores = torch.rand(2, 4)
context = wc(encodings, scores)  # (2, cognitive_dim)
```

---

### `KnowledgeExtractor`

Extracts Knowledge Units from Context Cells via cross-attention. Each cell produces `max_knowledge_units_per_cell` KU slots with activation scores in [0, 1].

**Constructor:**
```python
from pyrila.modules import KnowledgeExtractor
ke = KnowledgeExtractor(config: RILAConfig)
```

**Parameters:**
- `config.hidden_dim` — Input token dimension
- `config.knowledge_dim` — Output KU dimension
- `config.max_knowledge_units_per_cell` — Number of extraction slots
- `config.num_heads` — Cross-attention heads
- `config.dropout` — Dropout rate

**Forward:**
```python
kus, activations = ke(cell_tokens: torch.Tensor, lengths: torch.Tensor)
# cell_tokens: (batch, cell_size, hidden_dim)
# lengths: (batch,) actual token counts
# Returns: kus (batch, max_ku, knowledge_dim), activations (batch, max_ku)
```

**Example:**
```python
ke = KnowledgeExtractor(config)
tokens = torch.randn(2, config.cell_size, config.hidden_dim)
lengths = torch.full((2,), config.cell_size, dtype=torch.long)
kus, acts = ke(tokens, lengths)
```

---

### `RelevanceGate`

Filters Knowledge Units by learned relevance score `r_i = σ(f(KU_i))`. Applies threshold-based gating with a fallback guaranteeing at least one unit passes per batch element.

**Constructor:**
```python
from pyrila.modules import RelevanceGate
gate = RelevanceGate(config: RILAConfig)
```

**Parameters:**
- `config.knowledge_dim` — KU dimension
- `config.relevance_gate_threshold` — Gate threshold in [0, 1]

**Forward:**
```python
gated_units, gate_scores = gate(knowledge_units: torch.Tensor)
# knowledge_units: (batch, num_units, knowledge_dim)
# Returns: gated_units (same shape, scaled by relevance), gate_scores (batch, num_units)
```

**Example:**
```python
gate = RelevanceGate(config)
kus = torch.randn(2, 4, config.knowledge_dim)
gated, scores = gate(kus)
```

---

### `KnowledgeGraphBuilder`

Builds dynamic graph `KG = (N, E)` from Knowledge Units with typed relations and GNN message passing.

**Constructor:**
```python
from pyrila.modules import KnowledgeGraphBuilder
builder = KnowledgeGraphBuilder(config: RILAConfig, relation_threshold=0.5, gnn_iterations=3)
```

**Parameters:**
- `config.knowledge_dim` — KU dimension
- `relation_threshold` — Minimum relation strength for edge creation
- `gnn_iterations` — Number of message passing rounds

**Forward:**
```python
graph_state, adjacency = builder(knowledge_units: torch.Tensor, activations: torch.Tensor)
# knowledge_units: (batch, num_ku, knowledge_dim)
# activations: (batch, num_ku) activation scores
# Returns: graph_state (batch, num_ku, knowledge_dim), adjacency (batch, num_ku, num_ku)
```

**Example:**
```python
builder = KnowledgeGraphBuilder(config)
kus = torch.randn(2, 4, config.knowledge_dim)
acts = torch.rand(2, 4)
state, adj = builder(kus, acts)
```

---

### `CoreLanguageProcessor`

Main cognitive engine: transforms Working Context into a stable Cognitive State through iterative knowledge integration with convergence detection and stability constraints.

**Constructor:**
```python
from pyrila.modules import CoreLanguageProcessor
clp = CoreLanguageProcessor(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Cognitive state dimension
- `config.knowledge_dim` — Knowledge integration dimension
- `config.max_cognitive_iterations` — Maximum refinement iterations
- `config.convergence_threshold` — Convergence detection threshold
- `config.cognitive_stability_lambda` — Max state change norm per iteration

**Forward:**
```python
cognitive_state, metadata = clp(working_context: torch.Tensor, initial_state=None)
# working_context: (batch, cognitive_dim)
# initial_state: Optional previous state
# Returns: cognitive_state (batch, cognitive_dim), metadata dict
```

Metadata contains: `iterations`, `convergence_score`, `converged`, `memory_state`.

**Example:**
```python
clp = CoreLanguageProcessor(config)
wc = torch.randn(2, config.cognitive_dim)
state, meta = clp(wc)
print(f"Converged in {meta['iterations']} iterations: {meta['converged']}")
```

---

### `HypothesisGenerator`

Generates and scores candidate response hypotheses from cognitive state across four criteria (coherence, evidence, query compatibility, consistency).

**Constructor:**
```python
from pyrila.modules import HypothesisGenerator
gen = HypothesisGenerator(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Input/output dimension
- `config.max_hypotheses` — Number of candidates to generate

**Forward:**
```python
winner, all_hypotheses, scores = gen(cognitive_state, query=None, evidence=None)
# cognitive_state: (batch, cognitive_dim)
# Returns: winner (batch, cognitive_dim), all_hypotheses (batch, max_hyp, cognitive_dim), scores (batch, max_hyp, 4)
```

---

### `EvidenceTracker`

Tracks which Knowledge Units support each hypothesis.

**Constructor:**
```python
from pyrila.modules import EvidenceTracker
tracker = EvidenceTracker(config: RILAConfig)
```

**Forward:**
```python
evidence_map, valid_mask = tracker(hypotheses: torch.Tensor, knowledge_units: torch.Tensor)
# hypotheses: (batch, max_hyp, cognitive_dim)
# knowledge_units: (batch, num_ku, knowledge_dim)
# Returns: evidence_map (batch, max_hyp, num_ku), valid_mask (batch, max_hyp)
```

---

### `PreOutputGenerator`

Produces a candidate response embedding for verification, an evidence summary, and a cognitive state snapshot.

**Constructor:**
```python
from pyrila.modules import PreOutputGenerator
gen = PreOutputGenerator(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Input/output dimension
- `config.knowledge_dim` — Evidence projection dimension

**Forward:**
```python
candidate, evidence_summary, snapshot = gen(cognitive_state, evidence=None)
# cognitive_state: (batch, cognitive_dim)
# evidence: Optional (batch, num_evidence, knowledge_dim)
# Returns: candidate (batch, cognitive_dim), evidence_summary (batch, cognitive_dim//2), snapshot (batch, cognitive_dim)
```

**Example:**
```python
gen = PreOutputGenerator(config)
state = torch.randn(2, config.cognitive_dim)
candidate, evidence, snapshot = gen(state)
```

---

### `RecursiveVerificationEngine`

Multi-dimensional confidence evaluation across four dimensions: logical, contextual, semantic, and consistency. Produces `c_t = V(query, WC, s_t, p_t) ∈ [0, 1]`.

**Constructor:**
```python
from pyrila.modules import RecursiveVerificationEngine
rve = RecursiveVerificationEngine(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Input dimension (×4 for concatenated input)
- `config.confidence_threshold` — Acceptance threshold τ

**Forward:**
```python
confidence, accept = rve(query, working_context, cognitive_state, pre_output)
# All inputs: (batch, cognitive_dim)
# Returns: confidence (batch, 1) in [0,1], accept bool
```

Accept is True when confidence >= threshold for all batch items.

**Example:**
```python
rve = RecursiveVerificationEngine(config)
q = wc = s = p = torch.randn(2, config.cognitive_dim)
confidence, accept = rve(q, wc, s, p)
print(f"Confidence: {confidence.mean():.4f}, Accept: {accept}")
```

---

### `RecursiveReasoningLoop`

Iterative reasoning loop with budget control. Orchestrates cycles of failure analysis, query refinement, retrieval, context expansion, CLP re-run, and verification until confidence is met or budget is exhausted.

**Constructor:**
```python
from pyrila.modules import RecursiveReasoningLoop
loop = RecursiveReasoningLoop(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — State dimension
- `config.default_budget` — Default reasoning cycles
- `config.max_budget` — Maximum budget cap
- `config.confidence_threshold` — Target confidence

**Forward:**
```python
final_pre_output, final_confidence, budget_state = loop(
    query, initial_cognitive_state, initial_pre_output, initial_confidence,
    rce, index, working_context_module, clp, pre_output_gen, rve, budget=None,
)
```

---

### `ReasoningBudgetController`

Budget allocation and tracking for reasoning cycles. Uses a learned predictor to estimate budget from cognitive state when no explicit budget is provided.

**Constructor:**
```python
from pyrila.modules import ReasoningBudgetController
controller = ReasoningBudgetController(config: RILAConfig)
```

**Methods:**
```python
budget_state = controller.create_budget(budget=None, cognitive_state=None) -> BudgetState
should_go = controller.should_continue(budget_state, confidence) -> bool
budget_state = controller.step(budget_state, confidence, pre_output) -> BudgetState
```

**Example:**
```python
controller = ReasoningBudgetController(config)
state = controller.create_budget(budget=5)
print(f"Max cycles: {state.max_cycles}, Remaining: {state.remaining}")
```

---

### `FinalDecoder`

Autoregressive transformer decoder for text generation. Supports teacher forcing (training) and configurable sampling (inference) with temperature, top-k, and top-p.

**Constructor:**
```python
from pyrila.modules import FinalDecoder
decoder = FinalDecoder(config: RILAConfig)
```

**Parameters:**
- `config.cognitive_dim` — Input cognitive state dimension
- `config.hidden_dim` — Decoder hidden dimension
- `config.vocab_size` — Output vocabulary size
- `config.num_heads` — Decoder attention heads
- `config.num_layers` — Decoder layers (uses num_layers // 2)
- `config.max_sequence_length` — Maximum generation length
- `config.dropout` — Dropout rate

**Forward (training):**
```python
decoder.train()
logits = decoder(cognitive_state, target_tokens=targets)
# cognitive_state: (batch, cognitive_dim)
# target_tokens: (batch, target_len)
# Returns: (batch, target_len, vocab_size)
```

Raises `TeacherForcingError` if training mode is active and `target_tokens` is None.

**Generate (inference):**
```python
generated = decoder.generate(cognitive_state, max_length=512, temperature=1.0, top_k=50, top_p=0.9)
# Returns: (batch, generated_len) token IDs
```

**Example:**
```python
decoder = FinalDecoder(config)
state = torch.randn(2, config.cognitive_dim)

# Training
decoder.train()
targets = torch.randint(0, config.vocab_size, (2, 16))
logits = decoder(state, target_tokens=targets)  # (2, 16, vocab_size)

# Inference
generated = decoder.generate(state, max_length=50, temperature=0.8)
```

---

## Import Example

```python
from pyrila.modules import (
    RILATokenizer,
    ContextCellBuilder,
    CellEncoder,
    RecursiveContextIndex,
    RecursiveContextExplorer,
    WorkingContext,
    KnowledgeExtractor,
    RelevanceGate,
    KnowledgeGraphBuilder,
    CoreLanguageProcessor,
    HypothesisGenerator,
    EvidenceTracker,
    PreOutputGenerator,
    RecursiveVerificationEngine,
    RecursiveReasoningLoop,
    ReasoningBudgetController,
    FinalDecoder,
)
```

See also: [Model](model.md), [Config](config.md)
