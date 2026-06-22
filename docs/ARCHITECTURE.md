# RILA Architecture

## Overview

RILA (Recursive Indexed Language Architecture) is a novel language model architecture that replaces the monolithic attention-over-all-tokens approach with a structured pipeline of contextual indexing, adaptive retrieval, recursive reasoning, and autonomous verification.

The key insight: instead of attending over the entire input at every layer, RILA organizes input into **Context Cells**, builds a dynamic **affinity graph** for efficient retrieval, and uses a **recursive reasoning loop** that iteratively refines its understanding until a confidence threshold is met.

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RILA PROCESSING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Token IDs                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │  Tokenizer  │  token_ids → embeddings (batch, seq_len, D)               │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │ Cell Builder │  embeddings → List[ContextCellData]                       │
│  └──────┬───────┘  (partitions into fixed-size cells)                       │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │ Cell Encoder │  cells → CCE vectors (num_cells, 4D)                      │
│  └──────┬───────┘  CCE = [Semantic, Relation, Structural, Importance]       │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────┐                                                    │
│  │ Recursive Context   │  CCE → Affinity Graph G=(V,E)                      │
│  │ Index               │  (dynamic graph with learned edges)                │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │ Recursive Context   │  cognitive_state + index → retrieved cells         │
│  │ Explorer (RCE)      │  (iterative retrieval with expansion)              │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────┐                                                       │
│  │ Working Context  │  retrieved cells → aggregated context (batch, H)      │
│  └─────────┬────────┘  (attention-weighted aggregation)                     │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────┐                                                │
│  │ Core Language Processor │  WC → cognitive state s_t (batch, H)           │
│  │ (CLP)                   │  (iterative GRU refinement + convergence)      │
│  └─────────┬───────────────┘                                                │
│            │                                                                │
│            ▼                                                                │
│  ┌──────────────────────┐                                                   │
│  │ Pre-Output Generator │  s_t → candidate response p_t                    │
│  └─────────┬────────────┘                                                   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────┐                                            │
│  │ Recursive Verification      │  c_t = V(query, WC, s_t, p_t)             │
│  │ Engine (RVE)                │  confidence ∈ [0, 1]                       │
│  └─────────┬───────────────────┘                                            │
│            │                                                                │
│            ▼                                                                │
│    ┌───────────────┐                                                        │
│    │ c_t >= τ ?    │──── Yes ──→ Accept p_t                                 │
│    └───────┬───────┘                  │                                     │
│            │ No                       │                                     │
│            ▼                          │                                     │
│    ┌───────────────────┐              │                                     │
│    │ Reasoning Loop    │              │                                     │
│    │ (budget-limited)  │              │                                     │
│    │ - Failure analysis│              │                                     │
│    │ - Query refinement│              │                                     │
│    │ - Re-retrieval    │              │                                     │
│    │ - CLP re-run      │              │                                     │
│    │ - Re-verify       │              │                                     │
│    └───────┬───────────┘              │                                     │
│            │                          │                                     │
│            ▼                          ▼                                     │
│  ┌──────────────────┐                                                       │
│  │  Final Decoder   │  cognitive_state → output tokens                      │
│  │  (Transformer)   │  (teacher forcing / autoregressive sampling)          │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How Context Cells Work

Traditional transformers process all tokens simultaneously with O(n²) attention. RILA instead partitions the input into **Context Cells** — fixed-size blocks of C tokens (default 512).

Each cell is encoded into a 4-dimensional representation:

```
CCE_i = concat(S_i, R_i, T_i, I_i) ∈ R^(4D)
```

Where:
- **S** (Semantic vector): Captures the meaning and content of the cell
- **R** (Relation vector): Encodes relationships to other cells
- **T** (Structural vector): Preserves positional and structural information
- **I** (Importance vector): Indicates relevance to potential queries

This encoding is produced by a shared 2-layer Transformer followed by mean pooling and four projection heads (each L2-normalized).

**Advantages:**
- Reduces the O(n²) attention to O(n/C)² for graph operations
- Enables selective retrieval — only relevant cells are processed deeply
- Scales linearly with context length for the initial encoding pass

## How the Recursive Context Explorer Retrieves Information

The RCE operates in two modes:

**Training mode (soft top-k):** Uses differentiable softmax-weighted selection for gradient flow:
```
weights = softmax(scores / temperature)
soft_selection = Σ weights_i · cell_encodings_i
```

**Inference mode (hard top-k):** Exact selection of highest-scoring cells.

**Retrieval process:**
1. Project cognitive state into query space: `q = W_q · s_t`
2. Score all cells: `score_i = σ(f([q; CCE_i]))`
3. Select top-k initial cells
4. Check sufficiency: `sufficient = σ(g([s_t; mean(retrieved)]))`
5. If insufficient, expand via graph neighbors (1-hop BFS on affinity graph)
6. Repeat expansion up to `max_retrieval_expansions` times
7. Return unique retrieved cells with relevance scores

## How the Core Language Processor Builds Cognitive State

The CLP transforms the Working Context into a stable Cognitive State through iterative refinement:

```
s_{t+1} = Ψ(s_t, KG)
```

Concretely, each iteration:

1. **Knowledge integration**: Transform working context into knowledge-dimension input
2. **GRU update**: `s_{t+1} = GRU(knowledge_input + state_to_knowledge(s_t), s_t)`
3. **Stability enforcement**: `||s_{t+1} - s_t|| ≤ λ` (clamp delta norm)
4. **Memory update**: LSTM cell tracks long-range dependencies
5. **Convergence check**: `score = σ(f([s_t; s_{t+1}]))`

The loop terminates when:
- Convergence score exceeds threshold (typically 0.9), OR
- Maximum iterations reached (returns best state found)

## How the Verification Engine Decides Whether to Reason More

The RVE evaluates pre-output quality across four independent dimensions:

```
c_t = V(query, WC, s_t, p_t)
```

Where the confidence is computed as:

1. **Concatenate inputs**: `x = [query; WC; s_t; p_t]` → (4 × cognitive_dim)
2. **Four evaluator heads** (each MLP → Sigmoid):
   - Logical coherence: Is the reasoning valid?
   - Contextual support: Is it grounded in retrieved context?
   - Semantic consistency: Does it match the query semantics?
   - Self-consistency: Is it internally coherent?
3. **Aggregation**: `confidence = σ(W · [logical; contextual; semantic; consistency])`

**Decision rule:**
- If `confidence >= τ` (default 0.8): **Accept** the pre-output
- If `confidence < τ` AND budget remains: **Trigger reasoning loop**
- If `confidence < τ` AND budget exhausted: **Accept best pre-output found**

## The Reasoning Loop

When the RVE rejects a pre-output, the system enters a recursive reasoning loop:

```
for cycle in range(budget):
    analysis = FailureAnalyzer(s_t, confidence)
    q_refined = GRU(analysis, q_current)          # Refine query
    new_cells = RCE(q_refined, index)             # Re-retrieve
    WC_expanded = WorkingContext(accumulated)      # Expand context
    s_t = CLP(WC_expanded, s_t)                   # Re-process
    p_t = PreOutput(s_t)                          # New candidate
    c_t = RVE(q_refined, WC_expanded, s_t, p_t)  # Re-verify
    if c_t >= τ: break                            # Accept
```

Key properties:
- **Monotonic context growth**: Working context only accumulates (never forgets)
- **Budget-limited**: Hard cap prevents infinite loops
- **Best-so-far tracking**: If budget exhausts, returns highest-confidence output

## The Loss Function

RILA uses a composite loss with four components:

```
L = L_answer + λ₁·L_retrieval + λ₂·L_confidence + λ₃·L_budget
```

### Components

| Loss | Formula | Purpose |
|------|---------|---------|
| L_answer | CrossEntropy(logits, targets) | Standard language modeling |
| L_retrieval | α · R (cells retrieved) | Penalize excessive retrieval |
| L_confidence | BCE(predicted_conf, correctness) | Calibrate confidence |
| L_budget | β · B_used (cycles consumed) | Encourage efficiency |

### Uncertainty Weighting (Kendall et al., 2018)

Instead of fixed λ values, RILA supports learned uncertainty-weighted loss:

```
L = L_answer + Σ_i [(1/2σ²_i)·L_i + (1/2)·log(σ²_i)]
```

Each auxiliary task learns its own `log(σ²)` parameter, automatically balancing loss magnitudes during training. Initial effective weight is 1.0 for all tasks.

## Mathematical Formulas

### Cognitive State Evolution
```
s_{t+1} = Ψ(s_t, KG)
||s_{t+1} - s_t|| ≤ λ    (stability constraint)
```

### Verification Function
```
c_t = V(query, WC, s_t, p_t) ∈ [0, 1]
Accept ⟺ c_t ≥ τ
```

### Affinity Score
```
A(i, j) = σ(f([CCE_i; CCE_j]))
E = {(i,j) : A(i,j) > threshold, degree(i) ≤ max_connections}
```

### Retrieval Score
```
relevance(q, CCE_i) = σ(g([q; CCE_i]))
```

### Convergence Detection
```
conv_score(s_t, s_{t+1}) = σ(h([s_t; s_{t+1}]))
Converged ⟺ conv_score ≥ θ
```

### Budget Control
```
B_used ≤ B_max
continue = (c_t < τ) ∧ (B_used < B_max)
```

## Scale Invariance

The architecture is structurally identical from 2M to 1.3B+ parameters. Only numeric hyperparameters change:

| Config | Params | hidden_dim | cognitive_dim | layers | heads |
|--------|--------|-----------|---------------|--------|-------|
| small  | ~2M    | 64        | 128           | 2      | 4     |
| base   | ~125M  | 768       | 1024          | 12     | 12    |
| large  | ~350M  | 1024      | 1536          | 24     | 16    |
| xl     | ~1.3B  | 2048      | 2560          | 36     | 32    |
