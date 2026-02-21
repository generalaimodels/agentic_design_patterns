

# Chapter 15: Resource-Aware Optimization

---

## 15.1 Definition and Formal Framework

### 15.1.1 What is Resource-Aware Optimization in Agentic Systems

Resource-Aware Optimization (RAO) is the disciplined engineering and mathematical practice of maximizing the task-completion quality of an autonomous agent while satisfying hard or soft constraints on the finite computational, financial, and temporal resources the agent consumes during inference. Unlike classical machine learning optimization—which focuses predominantly on minimizing a loss function during training—RAO operates at **inference time** across multi-step, multi-model agentic pipelines where every action (LLM call, tool invocation, retrieval query, reasoning step) incurs a measurable cost.

**Why RAO is Non-Trivial in Agentic Systems:**

An agentic system is not a single forward pass through a neural network. It is a dynamically unfolding computation graph where:

1. **The number of steps is variable.** An agent solving a complex task may invoke 3 LLM calls or 30, depending on task difficulty and the agent's planning strategy.
2. **Each step compounds resource consumption.** A ReAct-style agent that reasons for $k$ steps with an LLM costing $c$ per call incurs $\mathcal{O}(k \cdot c)$ cost, and $k$ is itself a random variable dependent on problem structure.
3. **Quality is non-monotonic in resource expenditure.** Adding more reasoning steps can improve quality up to a point, after which hallucination, context overflow, or compounding errors degrade performance.
4. **Resources are heterogeneous.** An agent simultaneously consumes tokens (billed per-token), wall-clock latency (user-facing SLA), GPU memory (infrastructure constraint), API credits (budget ceiling), and network bandwidth (retrieval bottleneck).

Formally, an agentic system $\mathcal{A}$ operates as a policy $\pi$ that, given a task $x$ and an evolving state $s_t$, selects actions $a_t$ from an action space $\mathcal{A}_{\text{act}}$ (which includes LLM calls, tool uses, retrieval queries, and termination). The trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$ defines the agent's execution. RAO seeks to optimize $\pi$ such that:

$$
\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\tau \sim \pi}[\mathcal{Q}(\tau)] \quad \text{subject to} \quad \mathbb{E}_{\tau \sim \pi}[\mathcal{C}_r(\tau)] \leq B_r \quad \forall \, r \in \mathcal{R}
$$

where $\mathcal{R}$ is the set of resource types, $\mathcal{C}_r(\tau)$ is the consumption of resource $r$ along trajectory $\tau$, and $B_r$ is the budget for resource $r$.

**Key Distinction from Classical Optimization:** In standard ML, we optimize parameters $\theta$ during training. In RAO for agents, we optimize the **behavioral policy**—which model to call, how much context to include, when to cache, when to terminate—during **inference**, often in real-time.

---

### 15.1.2 Resource-Constrained Optimization: Formal Statement

The canonical formulation of resource-aware optimization is:

$$
\max_{\pi} \; \mathcal{Q}(\pi) \quad \text{s.t.} \quad \mathcal{C}(\pi) \leq B
$$

where:
- $\pi : \mathcal{S} \times \mathcal{H} \rightarrow \Delta(\mathcal{A}_{\text{act}})$ is the agent policy mapping state $s$ and history $\mathcal{H}$ to a distribution over actions.
- $\mathcal{Q}(\pi) = \mathbb{E}_{x \sim \mathcal{D}}[\mathcal{Q}(\pi, x)]$ is the expected quality over the task distribution $\mathcal{D}$.
- $\mathcal{C}(\pi) = \mathbb{E}_{x \sim \mathcal{D}}[\mathcal{C}(\pi, x)]$ is the expected resource consumption.
- $B \in \mathbb{R}^{|\mathcal{R}|}_{+}$ is the multi-dimensional budget vector.

**Lagrangian Relaxation.** We convert the constrained problem to an unconstrained one via Lagrange multipliers:

$$
\mathcal{L}(\pi, \lambda) = \mathcal{Q}(\pi) - \sum_{r \in \mathcal{R}} \lambda_r \left(\mathcal{C}_r(\pi) - B_r\right)
$$

The dual problem is:

$$
\min_{\lambda \geq 0} \max_{\pi} \; \mathcal{L}(\pi, \lambda)
$$

The multiplier $\lambda_r$ has a direct economic interpretation: it is the **shadow price** of resource $r$, representing the marginal quality gain achievable by relaxing budget $B_r$ by one unit. At optimality, KKT complementary slackness yields:

$$
\lambda_r^* \cdot \left(\mathcal{C}_r(\pi^*) - B_r\right) = 0 \quad \forall \, r \in \mathcal{R}
$$

This means that either the budget for resource $r$ is fully consumed ($\mathcal{C}_r(\pi^*) = B_r$), or the multiplier is zero ($\lambda_r^* = 0$, meaning additional budget would not improve quality).

**Multi-Objective Formulation.** In practice, quality itself is multi-dimensional (accuracy, coherence, safety, groundedness). We generalize:

$$
\max_{\pi} \; \mathbf{w}^\top \mathbf{Q}(\pi) \quad \text{s.t.} \quad \mathbf{C}(\pi) \preceq \mathbf{B}
$$

where $\mathbf{Q}(\pi) \in \mathbb{R}^m$ is a vector of quality metrics, $\mathbf{w} \in \Delta^m$ is a weighting over quality dimensions, $\mathbf{C}(\pi) \in \mathbb{R}^{|\mathcal{R}|}$ is the resource consumption vector, and $\preceq$ denotes element-wise inequality.

**Stochastic and Per-Query Constraints.** The expectation-based formulation allows occasional budget violations on individual queries. If hard per-query constraints are required:

$$
\max_{\pi} \; \mathbb{E}[\mathcal{Q}(\pi, x)] \quad \text{s.t.} \quad \mathcal{C}_r(\pi, x) \leq B_r \quad \forall \, x, \, \forall \, r
$$

This is strictly harder and requires worst-case reasoning. A practical relaxation uses chance constraints:

$$
\Pr\left[\mathcal{C}_r(\pi, x) > B_r\right] \leq \epsilon_r \quad \forall \, r
$$

allowing budget exceedance with probability at most $\epsilon_r$.

---

### 15.1.3 Resources: Tokens, Latency, API Cost, Compute, Memory

Each resource dimension has distinct measurement units, cost structures, and optimization levers.

| Resource | Unit | Measurement Method | Cost Structure | Primary Bottleneck |
|---|---|---|---|---|
| **Tokens** | Count (integer) | Tokenizer output length | Per-token pricing (input/output) | Context window limits |
| **Latency** | Milliseconds | Wall-clock timing | User experience / SLA penalties | Sequential LLM calls |
| **API Cost** | Currency (USD) | Provider billing | Linear in token count; model-dependent rates | Multi-call agent loops |
| **Compute** | FLOPs / GPU-seconds | Hardware counters | Infrastructure amortization | Self-hosted inference |
| **Memory** | Bytes (GB) | GPU VRAM / system RAM monitoring | Hardware capacity ceiling | KV-cache growth, batch size |

**Tokens** are the atomic currency of LLM-based agents. For a transformer with vocabulary $\mathcal{V}$ and tokenizer $\mathcal{T}$, the token count of a string $s$ is $|\mathcal{T}(s)|$. Costs are asymmetric: output tokens are typically $2\text{-}4\times$ more expensive than input tokens because output generation requires autoregressive sequential computation, while input processing can be parallelized via prefill.

**Latency** decomposes into:

$$
L_{\text{total}} = L_{\text{prefill}} + L_{\text{decode}} + L_{\text{network}} + L_{\text{tool}} + L_{\text{queue}}
$$

where:
- $L_{\text{prefill}} = \mathcal{O}(n_{\text{input}}^2 \cdot d / P)$ for attention over $n_{\text{input}}$ tokens with model dimension $d$ and parallelism $P$.
- $L_{\text{decode}} = \mathcal{O}(n_{\text{output}} \cdot n_{\text{ctx}} \cdot d / P)$ scales linearly with output tokens and current context length.
- $L_{\text{network}}$ is API round-trip time.
- $L_{\text{tool}}$ is latency of external tool invocations (database queries, web searches).
- $L_{\text{queue}}$ is queuing delay under load.

For a multi-step agent with $k$ sequential LLM calls:

$$
L_{\text{agent}} = \sum_{i=1}^{k} L_{\text{call},i} + \sum_{j=1}^{m} L_{\text{tool},j}
$$

Critically, latency is **non-parallelizable** across sequential reasoning steps—each step depends on the output of the previous one.

**API Cost** follows the standard pricing model:

$$
\mathcal{C}_{\text{call}} = c_{\text{in}} \cdot n_{\text{in}} + c_{\text{out}} \cdot n_{\text{out}}
$$

For example, with GPT-4o at $c_{\text{in}} = \$2.50/\text{MTok}$, $c_{\text{out}} = \$10.00/\text{MTok}$, a single call with 4K input and 1K output tokens costs:

$$
\mathcal{C} = 2.50 \times \frac{4000}{10^6} + 10.00 \times \frac{1000}{10^6} = \$0.01 + \$0.01 = \$0.02
$$

An agent loop of 10 such calls costs $\$0.20$ per query. At 100K queries/day, the daily cost reaches **$\$20{,}000$**.

**Compute (FLOPs)** for a transformer forward pass is approximately:

$$
\text{FLOPs}_{\text{forward}} \approx 2 \cdot P_{\text{model}} \cdot n_{\text{tokens}}
$$

where $P_{\text{model}}$ is the parameter count. For a 70B parameter model processing 4K tokens:

$$
\text{FLOPs} \approx 2 \times 70 \times 10^9 \times 4000 = 5.6 \times 10^{14} \text{ FLOPs}
$$

**Memory** is dominated by model weights and the KV-cache. For a model with $L$ layers, $H$ attention heads, head dimension $d_h$, and context length $n$:

$$
\text{KV-cache memory} = 2 \cdot L \cdot H \cdot d_h \cdot n \cdot b \cdot \text{sizeof(dtype)}
$$

where $b$ is batch size and the factor 2 accounts for both key and value tensors. For Llama-3 70B ($L=80$, $H=64$, $d_h=128$) at FP16 with $n=8192$ and $b=1$:

$$
\text{KV-cache} = 2 \times 80 \times 64 \times 128 \times 8192 \times 2 \text{ bytes} = 21.5 \text{ GB}
$$

This is **in addition to** the ~140 GB required for the model weights in FP16.

---

## 15.2 Token Budget Management

### 15.2.1 Token Counting and Estimation

Accurate token counting is the foundation of all token budget management. Tokens are not characters, not words, and not bytes—they are subword units determined by the specific tokenizer (BPE, SentencePiece, Unigram) associated with each model.

**Exact Token Counting.** Given a tokenizer $\mathcal{T}$ and input string $s$:

$$
n_{\text{tokens}} = |\mathcal{T}.\text{encode}(s)|
$$

Different models produce different token counts for identical text. For the string "Resource-aware optimization":

| Tokenizer | Tokens | Count |
|---|---|---|
| GPT-4 (cl100k_base) | ["Resource", "-aware", " optimization"] | 3 |
| Llama-3 | ["Resource", "-", "aware", " optimization"] | 4 |
| BERT (WordPiece) | ["resource", "-", "aware", " opt", "##imi", "##zation"] | 6 |

**Token Estimation Without Tokenizer Access.** When pre-counting is impractical (e.g., estimating cost before constructing the full prompt), use empirical heuristics:

$$
\hat{n}_{\text{tokens}} \approx \frac{|\text{chars}(s)|}{\alpha}
$$

where $\alpha$ is the characters-per-token ratio. For English text with modern BPE tokenizers, $\alpha \approx 3.5\text{-}4.0$. For code, $\alpha \approx 2.5\text{-}3.0$ (more whitespace and special characters). For non-Latin scripts (Chinese, Japanese, Korean), $\alpha \approx 1.5\text{-}2.0$.

**Token Budget Accounting for Agents.** A single agent turn involves multiple token streams:

$$
n_{\text{total}} = \underbrace{n_{\text{system}}}_{\text{system prompt}} + \underbrace{n_{\text{history}}}_{\text{conversation history}} + \underbrace{n_{\text{retrieval}}}_{\text{retrieved documents}} + \underbrace{n_{\text{tools}}}_{\text{tool definitions}} + \underbrace{n_{\text{user}}}_{\text{user query}} + \underbrace{n_{\text{output}}}_{\text{generated response}}
$$

Over $k$ agent steps, the cumulative input token cost grows because the history accumulates:

$$
N_{\text{input}}^{(k)} = \sum_{t=1}^{k} \left(n_{\text{system}} + \sum_{i=1}^{t-1} (n_{\text{query},i} + n_{\text{response},i}) + n_{\text{query},t}\right)
$$

This is $\mathcal{O}(k^2)$ in the number of steps if history is fully replayed each turn—a critical cost amplifier in multi-step agents.

```python
import tiktoken

class TokenBudgetManager:
    def __init__(self, model: str, budget: int):
        self.enc = tiktoken.encoding_for_model(model)
        self.budget = budget
        self.consumed = 0
    
    def count(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def estimate_multi_turn_cost(
        self, system_prompt: str, turns: list[tuple[str, str]]
    ) -> int:
        """Compute exact cumulative input tokens across k turns."""
        sys_tokens = self.count(system_prompt)
        total = 0
        history_tokens = 0
        for query, response in turns:
            q_tok = self.count(query)
            r_tok = self.count(response)
            # Each turn re-sends system + full history + new query
            turn_input = sys_tokens + history_tokens + q_tok
            total += turn_input + r_tok  # input + output billing
            history_tokens += q_tok + r_tok
        return total
    
    def can_afford(self, text: str) -> bool:
        return self.consumed + self.count(text) <= self.budget
    
    def consume(self, text: str) -> int:
        n = self.count(text)
        self.consumed += n
        return n
```

---

### 15.2.2 Token Allocation Across Agent Components

Given a finite token budget $B_{\text{tok}}$ for a single agent invocation within a context window of size $W$ (e.g., 128K for GPT-4o, 200K for Claude 3.5), the allocation problem is:

$$
\max_{n_{\text{sys}}, n_{\text{hist}}, n_{\text{ret}}, n_{\text{tool}}, n_{\text{out}}} \; \mathcal{Q}(n_{\text{sys}}, n_{\text{hist}}, n_{\text{ret}}, n_{\text{tool}}, n_{\text{out}})
$$

$$
\text{s.t.} \quad n_{\text{sys}} + n_{\text{hist}} + n_{\text{ret}} + n_{\text{tool}} + n_{\text{out}} \leq \min(B_{\text{tok}}, W)
$$

$$
n_i \geq n_i^{\min} \quad \forall \, i \in \{\text{sys}, \text{hist}, \text{ret}, \text{tool}, \text{out}\}
$$

**Priority-Based Allocation Strategy.** Components have different marginal value curves. Empirically:

1. **System prompt** ($n_{\text{sys}}$): Near-constant. Contains role definitions, behavioral constraints, output format specifications. Typically 200–1000 tokens. Marginal value drops sharply beyond the minimum necessary specification.

2. **Tool definitions** ($n_{\text{tool}}$): Semi-constant. Scales with the number of available tools. Each tool schema (name, description, parameters, examples) consumes 100–300 tokens. For $T$ tools: $n_{\text{tool}} \approx 150T$. **Optimization lever:** Dynamically select only relevant tools per query instead of providing all $T$.

3. **User query** ($n_{\text{user}}$): Fixed input. Cannot be compressed without risking task misunderstanding.

4. **Retrieved context** ($n_{\text{ret}}$): Highest variability and highest optimization potential. Retrieved documents are often redundant, noisy, or partially relevant. Empirical studies show that retrieval quality follows a concave utility curve—the first few relevant chunks provide most of the value.

5. **Conversation history** ($n_{\text{hist}}$): Grows linearly with conversation length. Older turns contribute diminishing value for most tasks. Recency-weighted compression is effective.

6. **Output** ($n_{\text{out}}$): Must be reserved. If the context window is consumed by input, no tokens remain for generation. Reserve rule:

$$
n_{\text{out}}^{\text{reserved}} = \min\left(n_{\text{out}}^{\max}, W - n_{\text{input}}\right)
$$

**Dynamic Allocation Algorithm:**

```python
def allocate_tokens(
    window_size: int,
    budget: int,
    system_tokens: int,
    user_tokens: int,
    tool_schemas: list[dict],
    retrieved_chunks: list[str],
    history_turns: list[tuple[str, str]],
    min_output_reserve: int = 1024,
    tokenizer = None,
) -> dict:
    effective_limit = min(window_size, budget)
    
    # Fixed allocations
    fixed = system_tokens + user_tokens
    
    # Dynamic tool selection (only include relevant tools)
    tool_tokens = sum(tokenizer.count(str(t)) for t in tool_schemas)
    
    # Reserve output
    remaining = effective_limit - fixed - tool_tokens - min_output_reserve
    
    # Allocate remaining between history and retrieval
    # Empirical split: 40% history, 60% retrieval (task-dependent)
    history_budget = int(remaining * 0.4)
    retrieval_budget = int(remaining * 0.6)
    
    # Truncate history (keep most recent turns)
    allocated_history = truncate_from_oldest(history_turns, history_budget, tokenizer)
    
    # Rank and truncate retrieved chunks by relevance score
    allocated_retrieval = select_top_chunks(retrieved_chunks, retrieval_budget, tokenizer)
    
    return {
        "system": system_tokens,
        "user": user_tokens,
        "tools": tool_tokens,
        "history": allocated_history,
        "retrieval": allocated_retrieval,
        "output_reserve": min_output_reserve,
    }
```

---

### 15.2.3 Prompt Compression Techniques

Prompt compression reduces token count while preserving information content relevant to task completion. This is formalized as an information-theoretic optimization:

$$
\min_{s'} \; |\mathcal{T}(s')| \quad \text{s.t.} \quad I(s'; y) \geq I(s; y) - \epsilon
$$

where $s$ is the original prompt, $s'$ is the compressed version, $y$ is the desired output, and $I(\cdot; \cdot)$ denotes mutual information. The constraint ensures that the compressed prompt retains at least $(1-\epsilon)$ fraction of the task-relevant information.

#### LLMLingua and Selective Context Compression

**LLMLingua** (Jiang et al., 2023) operates on the principle that tokens with **low perplexity** under a reference language model contribute less task-relevant information and can be removed with minimal quality degradation.

**Algorithm:**

1. Compute token-level perplexity using a small reference model $M_{\text{ref}}$ (e.g., GPT-2 or LLaMA-7B):

$$
\text{ppl}(x_i | x_{<i}) = \exp\left(-\log p_{M_{\text{ref}}}(x_i | x_{<i})\right)
$$

2. Rank tokens by perplexity. Tokens with **high perplexity** are more informative (surprising, thus carry more information content per Shannon's theory).

3. Remove low-perplexity tokens (predictable, redundant) until reaching the target compression ratio $\rho$:

$$
|s'| = \rho \cdot |s|, \quad \rho \in (0, 1)
$$

4. Apply **budget controller** to distribute compression non-uniformly across prompt components (instructions get less compression than retrieved documents).

**LLMLingua-2** improves upon this with a **data distillation** approach: a small classifier is trained on (original, compressed) pairs labeled by GPT-4, learning token-level importance directly for the downstream task rather than relying on generic perplexity.

**Formal Compression Bound:** If the prompt has entropy $H(s)$ and the task requires mutual information $I(s; y)$, then the theoretical minimum compressed length is:

$$
|s'|_{\min} \geq \frac{I(s; y)}{\log |\mathcal{V}|}
$$

In practice, compression ratios of $2\times$–$5\times$ are achievable with $<5\%$ quality degradation on typical retrieval-augmented generation tasks.

```python
class LLMLinguaCompressor:
    def __init__(self, ref_model, tokenizer, target_ratio: float = 0.5):
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.target_ratio = target_ratio
    
    def compute_token_importance(self, text: str) -> list[tuple[str, float]]:
        tokens = self.tokenizer.encode(text)
        importances = []
        
        for i, token_id in enumerate(tokens):
            context = tokens[:i]
            # Compute negative log-likelihood (higher = more important)
            with torch.no_grad():
                logits = self.ref_model(torch.tensor([context + [token_id]]))[0]
                log_prob = -F.cross_entropy(
                    logits[0, -1:], torch.tensor([token_id])
                ).item()
            importances.append((token_id, -log_prob))  # NLL as importance
        
        return importances
    
    def compress(self, text: str) -> str:
        token_importances = self.compute_token_importance(text)
        target_count = int(len(token_importances) * self.target_ratio)
        
        # Keep top-k most important tokens (highest NLL = most surprising)
        sorted_by_importance = sorted(
            enumerate(token_importances), 
            key=lambda x: x[1][1], 
            reverse=True
        )
        keep_indices = sorted([idx for idx, _ in sorted_by_importance[:target_count]])
        
        kept_tokens = [token_importances[i][0] for i in keep_indices]
        return self.tokenizer.decode(kept_tokens)
```

#### Extractive Summarization of Context

For retrieved documents, extractive summarization selects the most relevant sentences or passages:

$$
S^* = \arg\max_{S \subseteq \mathcal{D}, |S| \leq k} \; \text{Relevance}(S, q) + \lambda \cdot \text{Coverage}(S) - \mu \cdot \text{Redundancy}(S)
$$

where:
- $\text{Relevance}(S, q) = \sum_{s \in S} \cos(\mathbf{e}_s, \mathbf{e}_q)$ measures semantic similarity to the query.
- $\text{Coverage}(S) = |\bigcup_{s \in S} \text{entities}(s)|$ ensures broad information coverage.
- $\text{Redundancy}(S) = \sum_{i \neq j} \cos(\mathbf{e}_{s_i}, \mathbf{e}_{s_j})$ penalizes overlapping content.

This is a submodular optimization problem (when Coverage is submodular), solvable with greedy algorithms achieving a $(1 - 1/e)$-approximation guarantee.

**Practical Extractive Pipeline:**

```python
def extractive_compress(
    documents: list[str], 
    query: str, 
    token_budget: int,
    embedder,
    tokenizer,
) -> str:
    # Split documents into sentences
    sentences = []
    for doc in documents:
        sentences.extend(sent_tokenize(doc))
    
    # Embed query and sentences
    q_emb = embedder.encode(query)
    s_embs = embedder.encode(sentences)
    
    # Score by relevance
    scores = cosine_similarity([q_emb], s_embs)[0]
    
    # Greedy selection with MMR (Maximal Marginal Relevance)
    selected = []
    selected_embs = []
    remaining_budget = token_budget
    
    for _ in range(len(sentences)):
        best_idx, best_score = -1, -float('inf')
        for i, (sent, score) in enumerate(zip(sentences, scores)):
            if i in [s[0] for s in selected]:
                continue
            sent_tokens = tokenizer.count(sent)
            if sent_tokens > remaining_budget:
                continue
            # MMR: balance relevance and diversity
            if selected_embs:
                max_sim = max(cosine_similarity([s_embs[i]], selected_embs)[0])
            else:
                max_sim = 0
            mmr_score = 0.7 * score - 0.3 * max_sim
            if mmr_score > best_score:
                best_idx, best_score = i, mmr_score
        
        if best_idx == -1:
            break
        selected.append((best_idx, sentences[best_idx]))
        selected_embs.append(s_embs[best_idx])
        remaining_budget -= tokenizer.count(sentences[best_idx])
    
    # Return sentences in original order
    selected.sort(key=lambda x: x[0])
    return " ".join(s for _, s in selected)
```

---

### 15.2.4 Response Length Control

Output tokens are the most expensive resource (higher per-token pricing, sequential generation latency). Controlling response length without degrading quality requires structured techniques.

**Approaches:**

1. **Explicit length instructions in the system prompt:**

```
Respond in at most 3 sentences. Be concise and direct.
```

Effectiveness varies by model. Instruction-following adherence for length constraints is typically 60–80% without further enforcement.

2. **`max_tokens` parameter:** Hard truncation at the API level. Risk: mid-sentence truncation produces incoherent output. Set $n_{\text{max}} = \lceil \hat{n}_{\text{needed}} \cdot 1.2 \rceil$ with a 20% safety margin.

3. **Structured output formats:** JSON schemas with constrained fields naturally limit verbosity:

```json
{
  "answer": "string (max 100 chars)",
  "confidence": "float",
  "sources": ["string (max 3 items)"]
}
```

4. **Logit bias for early stopping:** Increase the logit of the EOS (end-of-sequence) token as generation progresses:

$$
\text{logit}_{\text{EOS}}^{(t)} = \text{logit}_{\text{EOS}}^{(t),\text{orig}} + \beta \cdot \max\left(0, \frac{t - t_{\text{target}}}{t_{\text{target}}}\right)
$$

where $t_{\text{target}}$ is the desired output length. The bias $\beta$ ramps up after the target length, gently encouraging termination.

5. **Two-pass generation:** First generate with a cheap model to estimate required length, then generate with the target model using a calibrated `max_tokens`. This amortizes the estimation cost.

---

## 15.3 Latency Optimization

### 15.3.1 Latency Profiling of Agent Pipelines

Before optimizing latency, one must measure and decompose it precisely. An agent pipeline's end-to-end latency is:

$$
L_{\text{e2e}} = \sum_{i=1}^{N_{\text{serial}}} L_i + \max_{j \in \text{parallel}} L_j
$$

where serial steps are summed and parallel steps contribute only their maximum.

**Decomposition of a Single LLM Call:**

$$
L_{\text{LLM}} = L_{\text{network}} + L_{\text{queue}} + L_{\text{prefill}} + L_{\text{decode}} + L_{\text{post}}
$$

| Component | Typical Range | Scaling Behavior | Optimization Lever |
|---|---|---|---|
| $L_{\text{network}}$ | 10–100 ms | Fixed per call | Co-location, edge deployment |
| $L_{\text{queue}}$ | 0–5000 ms | Load-dependent | Auto-scaling, priority queues |
| $L_{\text{prefill}}$ | 50–500 ms | $\mathcal{O}(n_{\text{in}}^2)$ or $\mathcal{O}(n_{\text{in}})$ with linear attention | Input compression, KV-cache reuse |
| $L_{\text{decode}}$ | 100–10000 ms | $\mathcal{O}(n_{\text{out}})$ per token | Speculative decoding, shorter outputs |
| $L_{\text{post}}$ | 1–10 ms | Fixed | Minimal overhead |

**Profiling Implementation:**

```python
import time
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class LatencyProfile:
    component: str
    start: float = 0.0
    end: float = 0.0
    children: list = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000

class AgentProfiler:
    def __init__(self):
        self.traces: list[LatencyProfile] = []
        self._stack: list[LatencyProfile] = []
    
    @contextmanager
    def trace(self, component: str):
        profile = LatencyProfile(component=component, start=time.perf_counter())
        if self._stack:
            self._stack[-1].children.append(profile)
        else:
            self.traces.append(profile)
        self._stack.append(profile)
        try:
            yield profile
        finally:
            profile.end = time.perf_counter()
            self._stack.pop()
    
    def report(self) -> dict:
        def flatten(profile, depth=0):
            result = {"component": "  " * depth + profile.component, 
                      "ms": round(profile.duration_ms, 2)}
            lines = [result]
            for child in profile.children:
                lines.extend(flatten(child, depth + 1))
            return lines
        
        all_lines = []
        for trace in self.traces:
            all_lines.extend(flatten(trace))
        return all_lines

# Usage in an agent loop
profiler = AgentProfiler()

async def agent_step(query: str):
    with profiler.trace("agent_step"):
        with profiler.trace("retrieval"):
            docs = await retrieve(query)
        with profiler.trace("prompt_construction"):
            prompt = build_prompt(query, docs)
        with profiler.trace("llm_call"):
            response = await llm.generate(prompt)
        with profiler.trace("tool_execution"):
            result = await execute_tools(response)
    return result
```

**Critical Insight: Amdahl's Law for Agent Pipelines.**

If a fraction $f$ of end-to-end latency is in the LLM calls and $(1-f)$ is in retrieval and tools, then the maximum speedup from optimizing only LLM inference is:

$$
S_{\max} = \frac{1}{(1-f) + f/s_{\text{LLM}}}
$$

where $s_{\text{LLM}}$ is the LLM speedup factor. If LLM calls constitute 60% of latency and we achieve a $3\times$ speedup, the overall speedup is only:

$$
S = \frac{1}{0.4 + 0.6/3} = \frac{1}{0.6} = 1.67\times
$$

This mandates **holistic** pipeline optimization, not single-component focus.

---

### 15.3.2 Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive generation by using a small, fast **draft model** $M_q$ to propose $\gamma$ tokens in parallel, which are then **verified** by the large **target model** $M_p$ in a single forward pass.

**Algorithm:**

1. Draft model $M_q$ generates $\gamma$ candidate tokens $(x_1, x_2, \ldots, x_\gamma)$ autoregressively (fast, since $M_q$ is small).
2. Target model $M_p$ evaluates all $\gamma$ tokens in a single forward pass (parallelized prefill), computing $p(x_i | x_{<i})$ for each.
3. Accept/reject each token using a modified rejection sampling scheme:

$$
x_i \text{ is accepted with probability } \min\left(1, \frac{p(x_i | x_{<i})}{q(x_i | x_{<i})}\right)
$$

where $p$ is the target model's distribution and $q$ is the draft model's distribution.

4. If token $x_i$ is rejected at position $i$, sample a correction token from the **adjusted distribution**:

$$
p'(x) = \text{norm}\left(\max\left(0, p(x | x_{<i}) - q(x | x_{<i})\right)\right)
$$

5. All tokens after the first rejection are discarded.

**Key Property:** The output distribution is **mathematically identical** to sampling from $M_p$ alone. Speculative decoding introduces zero quality degradation—it is a pure latency optimization.

**Expected Speedup:** If the acceptance rate per token is $\alpha$ (probability that the draft matches the target), the expected number of tokens generated per target model forward pass is:

$$
\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

The speedup ratio compared to standard autoregressive decoding is:

$$
S = \frac{\mathbb{E}[\text{tokens per step}]}{c \cdot \gamma + 1}
$$

where $c = L_{\text{draft}} / L_{\text{target}}$ is the cost ratio of draft vs. target inference. For typical values ($\alpha = 0.7$, $\gamma = 5$, $c = 0.1$):

$$
\mathbb{E}[\text{tokens}] = \frac{1 - 0.7^6}{1 - 0.7} = \frac{1 - 0.118}{0.3} = 2.94
$$

$$
S = \frac{2.94}{0.1 \times 5 + 1} = \frac{2.94}{1.5} = 1.96\times
$$

**Practical Implementation Considerations:**

- **Draft model selection:** Use a model from the same family (e.g., Llama-3-8B drafting for Llama-3-70B) to maximize token acceptance rate due to shared training distribution.
- **Adaptive $\gamma$:** Increase speculation length $\gamma$ when acceptance rates are high, decrease when low. This maximizes throughput adaptively.
- **Self-drafting:** Some architectures use early exit from the target model's own layers as the draft, eliminating the need for a separate model.

```python
def speculative_decode(
    target_model, draft_model, prompt_ids, max_tokens, gamma=5
):
    generated = list(prompt_ids)
    
    for _ in range(max_tokens // gamma + 1):
        # Step 1: Draft gamma tokens with small model
        draft_tokens = []
        draft_probs = []
        draft_input = list(generated)
        for _ in range(gamma):
            q_logits = draft_model(torch.tensor([draft_input]))
            q_dist = F.softmax(q_logits[0, -1], dim=-1)
            token = torch.multinomial(q_dist, 1).item()
            draft_tokens.append(token)
            draft_probs.append(q_dist[token].item())
            draft_input.append(token)
        
        # Step 2: Verify all gamma tokens with target model in one pass
        verify_input = generated + draft_tokens
        p_logits = target_model(torch.tensor([verify_input]))
        
        # Step 3: Accept/reject
        accepted = 0
        for i in range(gamma):
            pos = len(generated) + i - 1
            p_dist = F.softmax(p_logits[0, pos], dim=-1)
            p_prob = p_dist[draft_tokens[i]].item()
            q_prob = draft_probs[i]
            
            if random.random() < min(1.0, p_prob / q_prob):
                generated.append(draft_tokens[i])
                accepted += 1
            else:
                # Sample correction token from adjusted distribution
                adjusted = torch.clamp(p_dist - q_dist_full, min=0)
                adjusted = adjusted / adjusted.sum()
                correction = torch.multinomial(adjusted, 1).item()
                generated.append(correction)
                break
        
        if len(generated) - len(prompt_ids) >= max_tokens:
            break
    
    return generated[len(prompt_ids):]
```

---

### 15.3.3 Caching Strategies

Caching is the single most impactful latency optimization for agentic systems, where queries are frequently repetitive or semantically similar.

#### Semantic Caching

$$
\text{cache\_hit}(q) = \exists \, q' \in \mathcal{C} : \text{sim}(q, q') \geq \tau
$$

where $\mathcal{C}$ is the cache of previously answered queries, $\text{sim}$ is a similarity function (typically cosine similarity over dense embeddings), and $\tau$ is the acceptance threshold.

**Architecture:**

1. **Encode:** Embed incoming query $q$ using a dense encoder $E$: $\mathbf{e}_q = E(q)$.
2. **Search:** Find nearest neighbor in cache: $q^* = \arg\max_{q' \in \mathcal{C}} \cos(\mathbf{e}_q, \mathbf{e}_{q'})$.
3. **Threshold:** If $\cos(\mathbf{e}_q, \mathbf{e}_{q^*}) \geq \tau$, return cached response $R(q^*)$.
4. **Miss:** Otherwise, invoke the LLM, cache the result: $\mathcal{C} \leftarrow \mathcal{C} \cup \{(q, \mathbf{e}_q, R(q))\}$.

**Threshold Calibration.** The threshold $\tau$ controls the precision-recall tradeoff:

$$
\text{Precision}(\tau) = P[\text{response is correct} \mid \text{cache hit at threshold } \tau]
$$

$$
\text{Recall}(\tau) = P[\text{cache hit} \mid \text{query is answerable from cache}]
$$

Higher $\tau$ increases precision (fewer incorrect cache hits) but reduces recall (more cache misses). Calibrate $\tau$ on a held-out query set by plotting the precision-recall curve and selecting the operating point that satisfies the quality SLA.

**Typical values:** $\tau \in [0.90, 0.98]$ for factual Q&A; $\tau \in [0.95, 0.99]$ for code generation (where minor semantic differences yield incorrect outputs).

```python
import numpy as np
from collections import OrderedDict

class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.95, max_size: int = 10000):
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.cache = OrderedDict()  # (embedding, response) pairs
        self.embeddings = []  # for batch similarity computation
        self.keys = []
    
    def get(self, query: str) -> tuple[str | None, float]:
        q_emb = self.embedder.encode(query)
        
        if not self.embeddings:
            return None, 0.0
        
        # Compute similarities
        sims = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        if best_sim >= self.threshold:
            key = self.keys[best_idx]
            self.cache.move_to_end(key)  # LRU update
            return self.cache[key]["response"], best_sim
        
        return None, best_sim
    
    def put(self, query: str, response: str):
        q_emb = self.embedder.encode(query)
        
        if len(self.cache) >= self.max_size:
            # Evict LRU
            oldest_key = next(iter(self.cache))
            idx = self.keys.index(oldest_key)
            self.keys.pop(idx)
            self.embeddings.pop(idx)
            self.cache.pop(oldest_key)
        
        self.cache[query] = {"response": response, "embedding": q_emb}
        self.keys.append(query)
        self.embeddings.append(q_emb)
```

#### Exact Match Caching

For deterministic queries (e.g., tool calls with identical parameters, structured API requests), exact match caching is simpler and faster:

$$
\text{cache\_hit}(q) = q \in \mathcal{C}
$$

Implementation uses hash-based lookup ($\mathcal{O}(1)$ amortized). The cache key must be **canonicalized** to handle semantically equivalent but syntactically different queries:

```python
import hashlib, json

def canonical_key(query: str, params: dict) -> str:
    """Canonical cache key: normalize whitespace, sort params."""
    normalized_query = " ".join(query.lower().split())
    sorted_params = json.dumps(params, sort_keys=True)
    raw = f"{normalized_query}|{sorted_params}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

#### KV-Cache Reuse Across Turns

In multi-turn conversations, the KV-cache from previous turns can be **reused** to avoid recomputing attention for the shared prefix.

**Principle:** If turns $1$ through $t-1$ have already been processed, the KV-cache stores the key and value tensors for all $n_1 + n_2 + \cdots + n_{t-1}$ tokens. At turn $t$, only the new tokens $n_t$ need to go through the full attention computation, while the cached prefix is reused.

**Latency Savings:**

$$
L_{\text{with\_KV\_reuse}} = L_{\text{prefill}}(n_t) + L_{\text{decode}}(n_{\text{out},t})
$$

$$
L_{\text{without\_KV\_reuse}} = L_{\text{prefill}}\left(\sum_{i=1}^{t} n_i\right) + L_{\text{decode}}(n_{\text{out},t})
$$

For attention with complexity $\mathcal{O}(n^2 d)$, the savings on prefill are:

$$
\text{Savings ratio} = 1 - \frac{n_t^2}{\left(\sum_{i=1}^{t} n_i\right)^2}
$$

For a conversation at turn 10 where each turn is 500 tokens and the new query is 500 tokens:

$$
\text{Savings} = 1 - \frac{500^2}{5000^2} = 1 - 0.01 = 99\%
$$

**Implementation Challenges:**
- KV-cache must be stored in GPU memory between turns, consuming $\mathcal{O}(L \cdot H \cdot d_h \cdot n_{\text{prefix}})$ memory.
- Cache invalidation: if the system prompt changes, the entire cache is invalidated.
- **PagedAttention** (vLLM) manages KV-cache memory like virtual memory pages, allowing efficient sharing across requests with common prefixes.

---

### 15.3.4 Model Selection by Latency Requirements

Different latency SLAs demand different model choices. The selection problem is:

$$
M^* = \arg\max_{M \in \mathcal{M}} \; \mathcal{Q}(M) \quad \text{s.t.} \quad L_{p_{99}}(M) \leq L_{\text{SLA}}
$$

where $L_{p_{99}}(M)$ is the 99th percentile latency of model $M$.

**Latency Tiers:**

| Latency SLA | Model Class | Example Models | Use Case |
|---|---|---|---|
| $< 100$ ms | Small specialized | DistilBERT, TinyLlama, ONNX-optimized | Classification, routing |
| $100$–$500$ ms | Medium | Llama-3-8B, Mistral-7B, GPT-4o-mini | Simple generation, tool selection |
| $500$ ms–$2$ s | Large | Llama-3-70B, GPT-4o, Claude 3.5 Sonnet | Complex reasoning |
| $2$–$30$ s | Frontier | GPT-4, Claude 3 Opus, o1 | Multi-step reasoning, code gen |
| $> 30$ s | Ensemble/chain | Multi-model pipelines | Research, batch processing |

**Latency-Aware Model Router:**

```python
class LatencyAwareRouter:
    def __init__(self, models: dict[str, dict]):
        # models: {name: {"client": ..., "p99_latency_ms": ..., "quality_score": ...}}
        self.models = models
    
    def select(self, latency_budget_ms: float, min_quality: float = 0.0) -> str:
        eligible = [
            (name, info) for name, info in self.models.items()
            if info["p99_latency_ms"] <= latency_budget_ms
            and info["quality_score"] >= min_quality
        ]
        if not eligible:
            # Fallback to fastest model
            return min(self.models, key=lambda m: self.models[m]["p99_latency_ms"])
        
        # Among eligible, pick highest quality
        return max(eligible, key=lambda x: x[1]["quality_score"])[0]
```

---

### 15.3.5 Streaming for Perceived Latency Reduction

Streaming does not reduce actual latency but dramatically improves **perceived latency** (time to first meaningful output). The user begins reading while the model is still generating.

**Time-to-First-Token (TTFT)** is the key metric:

$$
\text{TTFT} = L_{\text{network}} + L_{\text{queue}} + L_{\text{prefill}} + L_{\text{1st\_decode}}
$$

For GPT-4o, typical TTFT is 200–500 ms, while total generation for a 500-token response takes 3–5 seconds. Streaming exposes the first token in 200 ms instead of making the user wait 5 seconds.

**Inter-Token Latency (ITL):** The time between consecutive tokens. For a smooth streaming experience, ITL should be $< 50$ ms (approximating human reading speed of ~250 words/minute $\approx$ 330 tokens/minute $\approx$ 180 ms/token).

**Streaming in Agent Pipelines:** Streaming is more complex for agents because:

1. **Tool calls must complete before streaming the final answer.** The agent may need to execute a database query mid-generation.
2. **Multi-step reasoning produces intermediate outputs** that should not be shown to the user.
3. **Solution:** Stream the **final synthesis step** while buffering intermediate reasoning.

```python
async def stream_agent_response(query: str, client):
    # Phase 1: Non-streaming reasoning (buffered)
    reasoning_steps = []
    for step in range(MAX_STEPS):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=build_messages(query, reasoning_steps),
            stream=False,  # Buffer intermediate steps
        )
        action = parse_action(response)
        if action.type == "final_answer":
            break
        result = await execute_tool(action)
        reasoning_steps.append((action, result))
    
    # Phase 2: Stream the final synthesis
    final_messages = build_final_messages(query, reasoning_steps)
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=final_messages,
        stream=True,  # Stream to user
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

---

## 15.4 Cost Optimization

### 15.4.1 Cost Modeling for Agentic Workflows

The total cost of an agentic workflow is:

$$
\mathcal{C}_{\text{total}} = \sum_{\text{calls}} \left(c_{\text{input}} \cdot n_{\text{input}} + c_{\text{output}} \cdot n_{\text{output}}\right)
$$

For a multi-step agent with $k$ steps, each accumulating history:

$$
\mathcal{C}_{\text{agent}}(k) = \sum_{t=1}^{k} \left[c_{\text{in}} \cdot \left(n_{\text{sys}} + \sum_{i=1}^{t-1}(n_{\text{q},i} + n_{\text{r},i}) + n_{\text{q},t}\right) + c_{\text{out}} \cdot n_{\text{r},t}\right]
$$

Expanding the input cost:

$$
\mathcal{C}_{\text{input}}(k) = c_{\text{in}} \cdot \left[k \cdot n_{\text{sys}} + \sum_{t=1}^{k}\sum_{i=1}^{t-1}(n_{\text{q},i} + n_{\text{r},i}) + \sum_{t=1}^{k} n_{\text{q},t}\right]
$$

The double summation is the dominant term, growing as $\mathcal{O}(k^2)$:

$$
\sum_{t=1}^{k}\sum_{i=1}^{t-1}(n_{\text{q},i} + n_{\text{r},i}) = \sum_{i=1}^{k-1}(k - i)(n_{\text{q},i} + n_{\text{r},i})
$$

If average turn length is $\bar{n}$, this is approximately $\frac{k(k-1)}{2} \cdot \bar{n}$, confirming quadratic cost scaling in the number of agent steps.

**Cost Model Implementation:**

```python
@dataclass
class CostModel:
    input_cost_per_mtok: float  # USD per million input tokens
    output_cost_per_mtok: float  # USD per million output tokens
    
    def call_cost(self, n_input: int, n_output: int) -> float:
        return (
            self.input_cost_per_mtok * n_input / 1e6
            + self.output_cost_per_mtok * n_output / 1e6
        )
    
    def agent_trajectory_cost(
        self, system_tokens: int, turns: list[tuple[int, int]]
    ) -> float:
        """Compute total cost for k-step agent trajectory."""
        total = 0.0
        history_tokens = 0
        for query_tokens, response_tokens in turns:
            input_tokens = system_tokens + history_tokens + query_tokens
            total += self.call_cost(input_tokens, response_tokens)
            history_tokens += query_tokens + response_tokens
        return total

# Example: GPT-4o pricing
gpt4o_cost = CostModel(input_cost_per_mtok=2.50, output_cost_per_mtok=10.00)

# 10-step agent, 300-token system prompt, ~500 tokens per turn
turns = [(200, 300)] * 10
cost = gpt4o_cost.agent_trajectory_cost(300, turns)
# This computes the escalating cost accurately
```

**Cost Projections at Scale:**

For $Q$ queries/day, average $k$ steps per query:

$$
\mathcal{C}_{\text{daily}} = Q \cdot \mathcal{C}_{\text{agent}}(k)
$$

| Queries/Day | Avg Steps | Avg Cost/Query (GPT-4o) | Daily Cost | Monthly Cost |
|---|---|---|---|---|
| 1,000 | 3 | $\$0.03$ | $\$30$ | $\$900$ |
| 10,000 | 5 | $\$0.08$ | $\$800$ | $\$24,000$ |
| 100,000 | 8 | $\$0.20$ | $\$20,000$ | $\$600,000$ |
| 1,000,000 | 10 | $\$0.50$ | $\$500,000$ | $\$15,000,000$ |

These projections make the case for aggressive cost optimization.

---

### 15.4.2 Model Cascade: Cheap Model → Expensive Model on Failure

A model cascade routes queries through increasingly capable (and expensive) models, only escalating when cheaper models fail or express low confidence.

**Formal Framework:** Given models $M_1, M_2, \ldots, M_K$ ordered by cost ($c_1 < c_2 < \cdots < c_K$) and quality ($q_1 \leq q_2 \leq \cdots \leq q_K$):

$$
M^*(x) = M_i \quad \text{where} \quad i = \min\{j : \text{conf}(M_j, x) \geq \theta_j \text{ or } j = K\}
$$

The expected cost is:

$$
\mathbb{E}[\mathcal{C}] = \sum_{i=1}^{K} P(\text{reach } M_i) \cdot c_i
$$

where:

$$
P(\text{reach } M_i) = \prod_{j=1}^{i-1} P(\text{conf}(M_j, x) < \theta_j)
$$

**Confidence Estimation Methods:**

1. **Logit-based confidence:** For classification-like outputs:

$$
\text{conf}(M, x) = \max_y \, p_M(y | x) = \max_y \, \frac{\exp(z_y)}{\sum_{y'} \exp(z_{y'})}
$$

2. **Self-reported confidence:** Prompt the model to assess its own confidence:

```
After answering, rate your confidence from 0.0 to 1.0. 
If below 0.7, flag for escalation.
```

3. **Consistency-based confidence:** Generate $n$ samples and measure agreement:

$$
\text{conf}(M, x) = \frac{|\text{most common answer}|}{n}
$$

This is the principle behind self-consistency (Wang et al., 2023). High agreement indicates high confidence.

4. **Verifier-based confidence:** A separate, small, fine-tuned model evaluates the response quality:

$$
\text{conf}(M, x) = V(x, M(x)) \in [0, 1]
$$

```python
class ModelCascade:
    def __init__(self, stages: list[dict]):
        """
        stages: [
            {"model": "gpt-4o-mini", "threshold": 0.85, "cost_per_mtok": (0.15, 0.60)},
            {"model": "gpt-4o",      "threshold": 0.70, "cost_per_mtok": (2.50, 10.0)},
            {"model": "o1",          "threshold": 0.00, "cost_per_mtok": (15.0, 60.0)},
        ]
        """
        self.stages = stages
        self.metrics = {"stage_counts": [0] * len(stages), "total_cost": 0.0}
    
    async def run(self, query: str, messages: list[dict]) -> dict:
        for i, stage in enumerate(self.stages):
            self.metrics["stage_counts"][i] += 1
            
            response = await call_model(
                stage["model"], messages,
                temperature=0.3,  # Lower temp for consistency
                n=3 if i < len(self.stages) - 1 else 1,  # Multi-sample for confidence
            )
            
            if i < len(self.stages) - 1:
                confidence = self._estimate_confidence(response)
                if confidence >= stage["threshold"]:
                    return {"response": response.choices[0].message.content,
                            "model": stage["model"], "stage": i, 
                            "confidence": confidence}
            else:
                # Final stage: always accept
                return {"response": response.choices[0].message.content,
                        "model": stage["model"], "stage": i, 
                        "confidence": 1.0}
    
    def _estimate_confidence(self, response) -> float:
        """Consistency-based confidence: agreement among n samples."""
        answers = [normalize(c.message.content) for c in response.choices]
        from collections import Counter
        counts = Counter(answers)
        return counts.most_common(1)[0][1] / len(answers)
```

**Cost Savings Analysis:** If 70% of queries are answered by $M_1$ (cheap) and only 30% escalate to $M_2$:

$$
\mathcal{C}_{\text{cascade}} = 0.7 \cdot c_1 + 0.3 \cdot (c_1 + c_2) = c_1 + 0.3 \cdot c_2
$$

Compared to always using $M_2$:

$$
\text{Savings} = 1 - \frac{c_1 + 0.3 \cdot c_2}{c_2} = 1 - \frac{c_1}{c_2} - 0.3
$$

If $c_1/c_2 = 0.05$ (GPT-4o-mini vs. GPT-4o): savings $= 1 - 0.05 - 0.3 = 65\%$.

---

### 15.4.3 Batch Processing for Cost Efficiency

Batch processing amortizes overhead costs and enables throughput-optimized inference.

**API-Level Batching:** Many providers offer batch APIs with 50% cost discounts for non-real-time processing. OpenAI's Batch API processes requests within a 24-hour window at half the standard price.

**Prompt Batching:** Multiple independent sub-tasks are packed into a single prompt:

```
Process the following 5 items. Return results as a JSON array.
Item 1: [text1]
Item 2: [text2]
...
Item 5: [text5]
```

**Cost analysis:** If each item requires 200 input tokens individually (plus 500 tokens system prompt), processing 5 items separately costs:

$$
\mathcal{C}_{\text{separate}} = 5 \times (c_{\text{in}} \cdot 700 + c_{\text{out}} \cdot 100) = 5c_{\text{in}} \cdot 700 + 5c_{\text{out}} \cdot 100
$$

Batching in one prompt:

$$
\mathcal{C}_{\text{batch}} = c_{\text{in}} \cdot (500 + 5 \times 200) + c_{\text{out}} \cdot 500 = c_{\text{in}} \cdot 1500 + c_{\text{out}} \cdot 500
$$

Savings on input tokens: $3500 \to 1500$ (57% reduction) because the system prompt is shared.

**Caveats:**
- Quality can degrade with too many items per batch (attention dilution).
- Error isolation: one malformed item can corrupt the entire batch output.
- Use structured output (JSON mode) to ensure parseable batch results.

---

### 15.4.4 Self-Hosting vs. API Cost Analysis

The decision between self-hosting and API usage is a fundamental infrastructure economics problem.

**Total Cost of Ownership (TCO) for Self-Hosting:**

$$
\mathcal{C}_{\text{self}} = \underbrace{\mathcal{C}_{\text{hardware}}}_{\text{GPU lease/purchase}} + \underbrace{\mathcal{C}_{\text{ops}}}_{\text{engineering labor}} + \underbrace{\mathcal{C}_{\text{infra}}}_{\text{networking, storage}} + \underbrace{\mathcal{C}_{\text{energy}}}_{\text{power \& cooling}}
$$

**Breakeven Analysis:** Let $r_{\text{API}}$ be the per-token API rate and $\mathcal{C}_{\text{self,fixed}}$ the monthly self-hosting cost. Breakeven occurs at:

$$
N_{\text{breakeven}} = \frac{\mathcal{C}_{\text{self,fixed}}}{r_{\text{API}}}
$$

**Concrete Example:** Self-hosting Llama-3-70B on 2×A100 80GB GPUs:

| Component | Monthly Cost |
|---|---|
| 2× A100 80GB (cloud lease) | $\$6,000$ |
| Engineering overhead (0.25 FTE) | $\$5,000$ |
| Networking, storage, monitoring | $\$1,000$ |
| **Total** | **$\$12,000/month$** |

Self-hosted throughput: ~30 tokens/second per A100 for 70B FP16 → ~60 tokens/s total → ~155M tokens/month (at 100% utilization).

API equivalent (GPT-4o at $\$2.50$ input, $\$10.00$ output per MTok, assuming 50/50 split):

$$
r_{\text{avg}} = \frac{2.50 + 10.00}{2} = \$6.25 \text{ per MTok}
$$

$$
\mathcal{C}_{\text{API}} = 155 \times 6.25 = \$968.75/\text{month}
$$

**In this scenario, the API is 12× cheaper!** Self-hosting only becomes cost-effective when:
1. Utilization is extremely high (>90%).
2. Token volume exceeds millions per day.
3. Data privacy requirements preclude external APIs.
4. Latency requirements demand dedicated infrastructure.
5. The self-hosted model (e.g., a fine-tuned open-source model) provides quality parity with the API model.

**Decision Matrix:**

| Factor | Favors API | Favors Self-Hosting |
|---|---|---|
| Volume | $< 100$M tokens/month | $> 1$B tokens/month |
| Utilization | Variable, bursty | Sustained, predictable |
| Data sensitivity | Non-sensitive | Regulated (HIPAA, GDPR) |
| Model customization | Standard models | Fine-tuned, custom |
| Engineering resources | Limited ML ops team | Dedicated infra team |
| Latency requirement | Standard (200ms+) | Ultra-low ($<50$ ms) |

---

### 15.4.5 Dynamic Model Routing Based on Task Complexity

Dynamic routing assigns each query to the most cost-efficient model capable of answering it correctly, based on a real-time complexity estimate.

**Complexity Estimation:** Given a query $x$, estimate its difficulty $d(x) \in [0, 1]$:

$$
d(x) = f_\theta(x) \quad \text{where } f_\theta \text{ is a trained classifier/regressor}
$$

**Training the Router:**

1. Collect a dataset of $(x, y_{\text{correct}})$ pairs.
2. Run each query through all models $M_1, \ldots, M_K$.
3. Label each query with the **cheapest model that answers correctly:**

$$
\ell(x) = \arg\min_i \{c_i : M_i(x) = y_{\text{correct}}\}
$$

4. Train a small classifier $f_\theta$ (e.g., distilled BERT or logistic regression on embeddings) to predict $\ell(x)$.

**Routing Policy:**

$$
M^*(x) = \begin{cases}
M_1 & \text{if } d(x) < \delta_1 \\
M_2 & \text{if } \delta_1 \leq d(x) < \delta_2 \\
\vdots \\
M_K & \text{if } d(x) \geq \delta_{K-1}
\end{cases}
$$

Thresholds $\delta_i$ are optimized on a validation set to maximize quality subject to budget constraints.

**Feature Engineering for Routing:**

```python
def extract_routing_features(query: str) -> dict:
    return {
        "length": len(query),
        "num_sentences": len(sent_tokenize(query)),
        "avg_word_length": np.mean([len(w) for w in query.split()]),
        "has_code": bool(re.search(r'```|def |class |import ', query)),
        "has_math": bool(re.search(r'\\frac|\\sum|equation|integral', query)),
        "question_type": classify_question_type(query),  # factual/reasoning/creative
        "domain_embedding": embedder.encode(query),  # dense features
        "num_constraints": count_constraints(query),
        "requires_multi_step": detect_multi_step(query),
    }
```

**Routing with Learned Quality Predictors:**

A more sophisticated approach trains a quality predictor $\hat{q}_i(x)$ for each model $M_i$, predicting the expected quality of $M_i$ on query $x$. The router then solves:

$$
M^*(x) = \arg\min_{i} \; c_i \quad \text{s.t.} \quad \hat{q}_i(x) \geq q_{\text{min}}
$$

This is the cheapest model predicted to achieve acceptable quality.

```python
class DynamicRouter:
    def __init__(self, models: list[dict], quality_predictors: dict):
        self.models = sorted(models, key=lambda m: m["cost"])
        self.quality_predictors = quality_predictors  # {model_name: predictor}
    
    def route(self, query: str, min_quality: float = 0.8) -> str:
        features = extract_routing_features(query)
        
        for model in self.models:  # Cheapest first
            predicted_quality = self.quality_predictors[model["name"]].predict(features)
            if predicted_quality >= min_quality:
                return model["name"]
        
        # Fallback to most expensive model
        return self.models[-1]["name"]
```

---

## 15.5 Compute and Infrastructure Optimization

### 15.5.1 GPU Memory Management for Self-Hosted Models

GPU memory is the primary constraint for self-hosted LLM inference. The total GPU memory consumption is:

$$
\text{Mem}_{\text{total}} = \text{Mem}_{\text{weights}} + \text{Mem}_{\text{KV-cache}} + \text{Mem}_{\text{activations}} + \text{Mem}_{\text{overhead}}
$$

**Model Weights:**

$$
\text{Mem}_{\text{weights}} = P_{\text{model}} \times \text{sizeof(dtype)}
$$

| Model | Parameters | FP32 | FP16 | INT8 | INT4 |
|---|---|---|---|---|---|
| Llama-3-8B | 8B | 32 GB | 16 GB | 8 GB | 4 GB |
| Llama-3-70B | 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| Llama-3-405B | 405B | 1620 GB | 810 GB | 405 GB | 203 GB |

**KV-Cache (derived in §15.1.3):**

$$
\text{Mem}_{\text{KV}} = 2 \cdot L \cdot n_{\text{kv\_heads}} \cdot d_h \cdot n_{\text{seq}} \cdot b \cdot \text{sizeof(dtype)}
$$

Note: models using Grouped Query Attention (GQA) have $n_{\text{kv\_heads}} < n_{\text{heads}}$, significantly reducing KV-cache size. Llama-3-70B uses 8 KV heads (vs. 64 attention heads), yielding an $8\times$ KV-cache reduction compared to standard multi-head attention.

**Memory Management Strategies:**

1. **PagedAttention (vLLM):** Manages KV-cache as non-contiguous memory pages, eliminating fragmentation. Memory utilization improves from ~50% (contiguous allocation) to ~95%.

2. **Continuous Batching:** Instead of padding all sequences to the same length in a batch, process sequences as they complete. New requests fill slots vacated by finished sequences.

3. **Prefix Sharing:** When multiple requests share a common prefix (e.g., same system prompt), store the KV-cache for the prefix once and share across requests.

```python
# vLLM deployment with memory optimization
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,          # Split across 4 GPUs
    gpu_memory_utilization=0.90,     # Use 90% of GPU memory
    max_model_len=8192,              # Limit context to manage KV-cache
    enable_prefix_caching=True,      # Reuse common prefixes
    swap_space=16,                   # GB of CPU swap for overflow
    enforce_eager=False,             # Use CUDA graphs for speed
)

sampling = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    top_p=0.9,
)
```

---

### 15.5.2 Quantization for Inference Efficiency

Quantization reduces model precision from FP16/BF16 to lower bit-widths, directly reducing memory consumption and increasing inference throughput.

**Formal Definition:** Quantization maps a floating-point tensor $\mathbf{W} \in \mathbb{R}^{m \times n}$ to a discrete representation:

$$
\hat{W}_{ij} = s \cdot \text{clamp}\left(\left\lfloor \frac{W_{ij}}{s} \right\rceil + z, \, 0, \, 2^b - 1\right)
$$

where $s$ is the scale factor, $z$ is the zero-point, $b$ is the bit-width, and $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

**Scale factor computation (symmetric quantization):**

$$
s = \frac{\max(|\mathbf{W}|)}{2^{b-1} - 1}
$$

**Quantization Methods Comparison:**

| Method | Bits | Memory Reduction | Quality Loss | Calibration Required | Speed Improvement |
|---|---|---|---|---|---|
| FP16 (baseline) | 16 | 1× | 0% | No | 1× |
| GPTQ | 4 | 4× | 1–3% | Yes (128 samples) | ~2× |
| AWQ | 4 | 4× | 0.5–2% | Yes | ~2× |
| GGUF (llama.cpp) | 2–8 | 2–8× | Varies | No | CPU-optimized |
| SqueezeLLM | 3 | 5.3× | 1–2% | Yes | ~2.5× |
| QuIP# | 2 | 8× | 3–8% | Yes | ~3× |

**AWQ (Activation-Aware Weight Quantization):** Identifies that a small fraction (~1%) of weights are significantly more important because they process channels with **large activation magnitudes**. These salient weights are preserved at higher precision (or equivalently, the corresponding channels are scaled up before quantization and scaled back during inference):

$$
\mathbf{Q}(\mathbf{W} \cdot \text{diag}(\mathbf{s})) \cdot \text{diag}(\mathbf{s})^{-1} \cdot \mathbf{x} \approx \mathbf{W} \cdot \mathbf{x}
$$

where $\mathbf{s}$ is a per-channel scaling vector derived from activation statistics.

**Practical Quantization Pipeline:**

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

# Quantize with AWQ
quant_config = {
    "zero_point": True,
    "q_group_size": 128,    # Quantize in groups of 128 weights
    "w_bit": 4,             # 4-bit quantization
    "version": "GEMM",      # Use GEMM kernel for inference
}
model.quantize(tokenizer, quant_config=quant_config)

# Save: 70B model from 140GB → 35GB
model.save_quantized("llama-3-70b-awq-4bit")
```

**Quality Validation Protocol:** After quantization, evaluate on a comprehensive benchmark suite:

$$
\Delta_{\text{quality}} = \frac{\text{Score}_{\text{FP16}} - \text{Score}_{\text{quantized}}}{\text{Score}_{\text{FP16}}} \times 100\%
$$

Accept quantization if $\Delta_{\text{quality}} < \epsilon_{\text{tolerance}}$ (typically 2–3%) across all critical benchmarks.

---

### 15.5.3 Model Parallelism and Distributed Inference

When a model does not fit on a single GPU, parallelism strategies distribute computation across multiple devices.

**Tensor Parallelism (TP):** Splits individual weight matrices across GPUs. For a linear layer $\mathbf{Y} = \mathbf{XW}$ where $\mathbf{W} \in \mathbb{R}^{d \times d}$:

Column-parallel: $\mathbf{W} = [\mathbf{W}_1 | \mathbf{W}_2 | \cdots | \mathbf{W}_P]$

$$
\mathbf{Y}_i = \mathbf{X} \mathbf{W}_i \quad \text{(computed on GPU } i\text{)}
$$

All-gather to reconstruct $\mathbf{Y} = [\mathbf{Y}_1 | \mathbf{Y}_2 | \cdots | \mathbf{Y}_P]$.

**Communication cost per layer:** $\mathcal{O}(b \cdot n \cdot d / P)$ for the all-reduce/all-gather operation. This becomes the bottleneck: inter-GPU bandwidth (NVLink: 900 GB/s; PCIe: 64 GB/s) determines the maximum useful $P$.

**Pipeline Parallelism (PP):** Assigns entire layers to different GPUs. For $L$ layers and $P$ GPUs, each GPU handles $L/P$ layers. The forward pass is a pipeline:

$$
\text{GPU}_1: \text{Layers } 1\text{-}20 \rightarrow \text{GPU}_2: \text{Layers } 21\text{-}40 \rightarrow \cdots
$$

**Pipeline bubble:** With $P$ pipeline stages and micro-batch count $M$, the bubble fraction (idle time) is:

$$
\text{Bubble fraction} = \frac{P - 1}{M + P - 1}
$$

For $P=4$, $M=8$: bubble = $3/11 = 27\%$ idle time. Increase $M$ to reduce the bubble.

**Expert Parallelism (EP):** For Mixture-of-Experts (MoE) models, different experts reside on different GPUs:

$$
\text{GPU}_i \text{ hosts experts } \{E_{iK+1}, \ldots, E_{(i+1)K}\}
$$

Tokens are routed to the appropriate GPU based on the gating function.

**Practical Configuration for Llama-3-70B:**

```bash
# 4× A100 80GB with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

Memory breakdown with TP=4:
- Model weights: $140 \text{ GB} / 4 = 35 \text{ GB/GPU}$
- KV-cache: $\sim 21 \text{ GB} / 4 = 5.3 \text{ GB/GPU}$ (at full 8K context)
- Activations + overhead: $\sim 5 \text{ GB/GPU}$
- Total per GPU: $\sim 45 \text{ GB}$ — fits in 80 GB with room for batching.

---

### 15.5.4 Auto-Scaling Agent Infrastructure

Auto-scaling dynamically adjusts compute resources based on demand, balancing cost (no over-provisioning) and latency (no under-provisioning).

**Scaling Metrics:**

$$
\text{Scale-out trigger: } \quad \frac{\text{queue depth}}{N_{\text{instances}}} > \theta_{\text{up}} \quad \text{or} \quad \text{GPU utilization} > u_{\text{up}}
$$

$$
\text{Scale-in trigger: } \quad \frac{\text{queue depth}}{N_{\text{instances}}} < \theta_{\text{down}} \quad \text{and} \quad \text{GPU utilization} < u_{\text{down}}
$$

**Predictive Scaling:** Use historical traffic patterns to pre-scale:

$$
\hat{N}(t + \Delta t) = f_{\text{forecast}}\left(N(t), N(t-1), \ldots, N(t-k); \text{day-of-week}, \text{hour}\right)
$$

Time-series models (ARIMA, Prophet, or lightweight neural forecasters) predict demand 15–30 minutes ahead, allowing GPU instances to warm up (model loading takes 2–5 minutes for large models).

**Scaling Architecture:**

```yaml
# Kubernetes HPA (Horizontal Pod Autoscaler) for vLLM
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: pending_requests
        target:
          type: AverageValue
          averageValue: "10"  # Scale when avg pending > 10
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "80"  # Scale when GPU util > 80%
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
        - type: Pods
          value: 2
          periodSeconds: 120
```

**Cold Start Mitigation:** GPU model loading is the primary cold-start bottleneck. Strategies:

1. **Keep-warm pools:** Maintain minimum replicas that are always loaded.
2. **Model caching on local SSD:** Pre-cache model weights on instance storage for faster loading ($\sim$30 seconds from NVMe vs. $\sim$5 minutes from network storage).
3. **Speculative pre-warming:** Start loading models when predictive scaling anticipates demand increase.

---

### 15.5.5 Edge Deployment Considerations

Edge deployment brings inference to user devices or edge servers, eliminating network latency and API costs but imposing severe resource constraints.

**Resource Constraints on Edge:**

| Device Class | Memory | Compute | Power | Example |
|---|---|---|---|---|
| Smartphone | 4–8 GB RAM | 10–30 TOPS (NPU) | 5–10W | iPhone 16, Pixel 9 |
| Laptop | 16–64 GB RAM | 50–200 TOPS (GPU/NPU) | 35–100W | M3 MacBook, RTX laptop |
| Edge server | 32–128 GB RAM | 100–1000 TOPS | 200–500W | NVIDIA Jetson AGX, Intel NUC |

**Model Selection for Edge:**

$$
M_{\text{edge}}^* = \arg\max_{M} \; \mathcal{Q}(M) \quad \text{s.t.} \quad \text{Mem}(M) \leq R_{\text{mem}}, \; \text{Latency}(M) \leq L_{\text{max}}
$$

**Edge-Optimized Models:**

| Model | Parameters | Quantized Size | TTFT on M3 | Quality (MMLU) |
|---|---|---|---|---|
| Phi-3-mini | 3.8B | 2.1 GB (Q4) | ~200 ms | 68.8 |
| Llama-3.2-3B | 3B | 1.8 GB (Q4) | ~150 ms | 63.4 |
| Gemma-2-2B | 2B | 1.2 GB (Q4) | ~100 ms | 51.3 |
| Qwen2.5-1.5B | 1.5B | 0.9 GB (Q4) | ~80 ms | 56.5 |

**Edge Inference Frameworks:**

- **llama.cpp:** C/C++ implementation with extensive quantization support (Q2–Q8). Runs on CPU, Metal (Apple), CUDA, Vulkan.
- **MLX:** Apple-optimized framework for Apple Silicon. Native unified memory eliminates CPU-GPU transfers.
- **ONNX Runtime Mobile:** Cross-platform, optimized for mobile NPUs.
- **ExecuTorch:** PyTorch's edge deployment framework targeting mobile and embedded.

**Hybrid Edge-Cloud Architecture:**

```python
class HybridEdgeCloudAgent:
    def __init__(self, edge_model, cloud_client, complexity_threshold=0.6):
        self.edge_model = edge_model  # Local small model
        self.cloud_client = cloud_client  # API client
        self.threshold = complexity_threshold
        self.complexity_estimator = load_complexity_model()
    
    async def process(self, query: str) -> str:
        complexity = self.complexity_estimator.predict(query)
        
        if complexity < self.threshold:
            # Process locally: zero latency, zero cost
            return self.edge_model.generate(query)
        else:
            # Escalate to cloud
            response = await self.cloud_client.generate(query)
            return response
```

This architecture routes simple queries (classification, entity extraction, short Q&A) to the edge model and complex queries (multi-step reasoning, code generation, long-form synthesis) to cloud APIs.

---

## 15.6 Quality-Resource Tradeoff Analysis

### 15.6.1 Pareto Frontier of Quality vs. Cost

The Pareto frontier defines the set of configurations where no improvement in quality is possible without increasing cost, and no cost reduction is achievable without degrading quality.

**Formal Definition:** A configuration $c_i = (q_i, r_i)$ (quality, resource) **Pareto-dominates** $c_j$ if $q_i \geq q_j$ and $r_i \leq r_j$ with at least one strict inequality. The Pareto frontier $\mathcal{P}$ is:

$$
\mathcal{P} = \{c_i : \nexists \, c_j \text{ that Pareto-dominates } c_i\}
$$

**Constructing the Pareto Frontier for Agent Configurations:**

1. **Enumerate configurations** by varying:
   - Model choice (GPT-4o-mini, GPT-4o, Claude 3.5 Sonnet, Llama-3-70B)
   - Number of reasoning steps (1, 3, 5, 10)
   - Retrieval depth (0, 3, 5, 10 chunks)
   - Prompt compression ratio (1.0, 0.75, 0.5)
   - Temperature (0.0, 0.3, 0.7)
   - Self-consistency samples (1, 3, 5)

2. **Evaluate each configuration** on a benchmark:
   - Quality: task accuracy, F1, ROUGE, human preference scores
   - Cost: total tokens consumed × pricing

3. **Compute the frontier:**

```python
def compute_pareto_frontier(
    configs: list[dict]  # [{"name": ..., "quality": ..., "cost": ...}, ...]
) -> list[dict]:
    """Return Pareto-optimal configurations."""
    # Sort by cost ascending
    sorted_configs = sorted(configs, key=lambda c: c["cost"])
    
    frontier = []
    max_quality = -float('inf')
    
    for config in sorted_configs:
        if config["quality"] > max_quality:
            frontier.append(config)
            max_quality = config["quality"]
    
    return frontier

def plot_pareto(configs, frontier):
    import matplotlib.pyplot as plt
    
    costs = [c["cost"] for c in configs]
    qualities = [c["quality"] for c in configs]
    
    f_costs = [c["cost"] for c in frontier]
    f_qualities = [c["quality"] for c in frontier]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(costs, qualities, alpha=0.3, label="All configurations")
    plt.plot(f_costs, f_qualities, 'r-o', linewidth=2, label="Pareto frontier")
    plt.xlabel("Cost per query (USD)")
    plt.ylabel("Quality (accuracy)")
    plt.title("Pareto Frontier: Quality vs. Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Typical Pareto Frontier Shape:**

The frontier is typically concave, reflecting diminishing returns:

```
Quality
  1.0 |                              ●──── Frontier
      |                         ●
  0.9 |                    ●
      |               ●
  0.8 |          ●
      |     ●
  0.7 |  ●
      | ●
  0.6 |●
      |________________________
      $0.001  $0.01   $0.1   $1.0
              Cost per query
```

**Marginal quality per unit cost** decreases along the frontier: moving from $\$0.001$ to $\$0.01$ might gain 15% quality, but moving from $\$0.10$ to $\$1.00$ gains only 3%.

---

### 15.6.2 Diminishing Returns Analysis

Diminishing returns characterize the relationship between resource investment and quality gain. Formally, the quality function $\mathcal{Q}(r)$ as a function of resource $r$ is **concave**:

$$
\frac{\partial^2 \mathcal{Q}}{\partial r^2} < 0
$$

This means:

$$
\mathcal{Q}(r + \Delta r) - \mathcal{Q}(r) > \mathcal{Q}(r + 2\Delta r) - \mathcal{Q}(r + \Delta r)
$$

**Empirical Diminishing Returns Curves:**

1. **Reasoning steps vs. quality:** For a ReAct agent on HotpotQA:

| Steps | Accuracy | Marginal Gain | Cost |
|---|---|---|---|
| 1 | 52% | — | $\$0.01$ |
| 3 | 71% | +19% | $\$0.04$ |
| 5 | 78% | +7% | $\$0.09$ |
| 8 | 81% | +3% | $\$0.18$ |
| 12 | 82% | +1% | $\$0.35$ |

2. **Retrieved chunks vs. quality:**

| Chunks | F1 Score | Marginal Gain |
|---|---|---|
| 0 | 0.45 | — |
| 1 | 0.62 | +0.17 |
| 3 | 0.73 | +0.11 |
| 5 | 0.77 | +0.04 |
| 10 | 0.78 | +0.01 |

3. **Self-consistency samples vs. accuracy:**

$$
\text{Accuracy}(n) \approx A_{\infty} - (A_{\infty} - A_1) \cdot e^{-\lambda(n-1)}
$$

where $A_{\infty}$ is the asymptotic accuracy, $A_1$ is single-sample accuracy, and $\lambda$ controls convergence rate.

**Optimal Stopping:** The marginal value of an additional unit of resource $r$ is:

$$
\text{MV}(r) = \frac{\partial \mathcal{Q}}{\partial r}
$$

The marginal cost is $\text{MC}(r) = c_r$ (constant for linear pricing). The optimal resource allocation satisfies:

$$
\text{MV}(r^*) = \lambda \cdot \text{MC}(r^*)
$$

where $\lambda$ is the Lagrange multiplier from the budget constraint. Operationally: **stop investing in a resource when its marginal quality gain per dollar falls below the shadow price of the budget.**

```python
def find_optimal_allocation(
    quality_fn,    # quality_fn(r) -> quality
    cost_fn,       # cost_fn(r) -> cost  
    budget: float,
    resource_range: np.ndarray,
) -> float:
    """Find resource level that maximizes quality within budget."""
    # Compute marginal quality / marginal cost
    qualities = np.array([quality_fn(r) for r in resource_range])
    costs = np.array([cost_fn(r) for r in resource_range])
    
    # Filter feasible points
    feasible = resource_range[costs <= budget]
    feasible_qualities = qualities[costs <= budget]
    
    if len(feasible) == 0:
        return resource_range[0]  # Minimum resource
    
    optimal_idx = np.argmax(feasible_qualities)
    return feasible[optimal_idx]
```

---

### 15.6.3 Budget-Aware Agent Design Patterns

These are architectural patterns that embed resource awareness into the agent's decision-making process.

**Pattern 1: Budget-Conditioned Policy**

The agent policy $\pi$ receives the remaining budget as an input:

$$
a_t = \pi(s_t, h_t, B_{\text{remaining},t})
$$

As the budget depletes, the agent shifts to cheaper actions (simpler models, fewer retrieval calls, terser outputs). Implementation:

```python
class BudgetAwareAgent:
    def __init__(self, budget: float, cost_model: CostModel):
        self.budget = budget
        self.remaining = budget
        self.cost_model = cost_model
    
    def select_model(self) -> str:
        budget_fraction = self.remaining / self.budget
        if budget_fraction > 0.5:
            return "gpt-4o"        # Full budget: use best model
        elif budget_fraction > 0.2:
            return "gpt-4o-mini"   # Mid budget: use efficient model
        else:
            return "gpt-3.5-turbo" # Low budget: use cheapest
    
    def should_retrieve(self) -> bool:
        return self.remaining > self.budget * 0.3  # Skip retrieval if low budget
    
    def max_reasoning_steps(self) -> int:
        budget_fraction = self.remaining / self.budget
        return max(1, int(budget_fraction * 10))
    
    def step(self, query: str) -> str:
        model = self.select_model()
        
        context = ""
        if self.should_retrieve():
            context = self.retrieve(query)
            self.remaining -= self.cost_model.retrieval_cost()
        
        response = self.call_llm(model, query, context)
        call_cost = self.cost_model.call_cost(
            count_tokens(query + context), count_tokens(response)
        )
        self.remaining -= call_cost
        
        return response
```

**Pattern 2: Anytime Agent**

An anytime agent produces progressively better answers as more resources are consumed. It can be interrupted at any time and will return the best answer available:

$$
\mathcal{Q}(t_1) \leq \mathcal{Q}(t_2) \quad \forall \, t_1 < t_2
$$

Implementation: maintain a running "best answer" that is refined iteratively.

```python
class AnytimeAgent:
    def __init__(self, models: list[str], time_budget_s: float):
        self.models = models  # Ordered cheap → expensive
        self.time_budget = time_budget_s
        self.best_answer = None
        self.best_confidence = 0.0
    
    async def solve(self, query: str) -> str:
        start = time.time()
        
        for model in self.models:
            elapsed = time.time() - start
            if elapsed >= self.time_budget:
                break
            
            remaining = self.time_budget - elapsed
            try:
                response = await asyncio.wait_for(
                    self.call_llm(model, query),
                    timeout=remaining
                )
                confidence = self.estimate_confidence(response)
                
                if confidence > self.best_confidence:
                    self.best_answer = response
                    self.best_confidence = confidence
                    
                    if confidence > 0.95:
                        break  # Confident enough, stop early
            except asyncio.TimeoutError:
                break
        
        return self.best_answer
```

**Pattern 3: Resource-Aware Planning (Meta-Reasoning)**

The agent explicitly reasons about which actions are worth their resource cost before executing them. This is **meta-reasoning** or **rational meta-cognition**:

$$
a^* = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}[\text{VOI}(a)] - \mathcal{C}(a)
$$

where $\text{VOI}(a)$ is the **Value of Information** of action $a$—the expected quality improvement from executing $a$.

```python
class MetaReasoningAgent:
    def plan_actions(self, query: str, remaining_budget: float) -> list[dict]:
        candidate_actions = [
            {"type": "retrieve", "cost": 0.002, "expected_voi": 0.15},
            {"type": "reason_step", "cost": 0.01, "expected_voi": 0.08},
            {"type": "web_search", "cost": 0.005, "expected_voi": 0.12},
            {"type": "verify", "cost": 0.015, "expected_voi": 0.05},
            {"type": "use_expensive_model", "cost": 0.05, "expected_voi": 0.20},
        ]
        
        # Select actions with positive net value within budget
        selected = []
        budget = remaining_budget
        
        # Sort by VOI/cost ratio (efficiency)
        ranked = sorted(
            candidate_actions,
            key=lambda a: a["expected_voi"] / a["cost"],
            reverse=True
        )
        
        for action in ranked:
            if action["cost"] <= budget and action["expected_voi"] > action["cost"]:
                selected.append(action)
                budget -= action["cost"]
        
        return selected
```

---

### 15.6.4 Resource Monitoring Dashboards

Production agent systems require real-time monitoring of resource consumption, quality metrics, and efficiency ratios.

**Key Metrics to Monitor:**

1. **Cost Metrics:**
   - Cost per query (P50, P95, P99)
   - Daily/weekly cost trend
   - Cost breakdown by model, component, and user segment
   - Cost anomaly detection (sudden spikes)

2. **Latency Metrics:**
   - End-to-end latency (P50, P95, P99)
   - TTFT (time to first token)
   - Latency breakdown by pipeline stage
   - Queue depth over time

3. **Token Metrics:**
   - Input/output token ratio
   - Token waste ratio (tokens in truncated/unused context)
   - Context window utilization

4. **Quality Metrics:**
   - Task success rate
   - User satisfaction (thumbs up/down)
   - Cascade escalation rate
   - Cache hit rate and cache quality (false positive rate)

5. **Efficiency Ratios:**

$$
\text{Quality per Dollar} = \frac{\mathcal{Q}}{\mathcal{C}_{\text{total}}}
$$

$$
\text{Quality per Latency} = \frac{\mathcal{Q}}{L_{\text{e2e}}}
$$

$$
\text{Cache Efficiency} = \frac{\text{cache hits} \times \text{avg API cost saved}}{\text{cache infrastructure cost}}
$$

**Implementation with Prometheus + Grafana:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Cost metrics
COST_PER_QUERY = Histogram(
    'agent_cost_per_query_usd', 
    'Cost per agent query in USD',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)
TOTAL_COST = Counter('agent_total_cost_usd', 'Cumulative cost', ['model'])
TOKEN_USAGE = Counter('agent_tokens_total', 'Token usage', ['type', 'model'])

# Latency metrics
E2E_LATENCY = Histogram(
    'agent_e2e_latency_seconds',
    'End-to-end agent latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
STAGE_LATENCY = Histogram(
    'agent_stage_latency_seconds',
    'Per-stage latency',
    ['stage']  # retrieval, llm_call, tool_execution, etc.
)
TTFT = Histogram('agent_ttft_seconds', 'Time to first token')

# Quality metrics
TASK_SUCCESS = Counter('agent_task_success_total', 'Successful task completions')
CASCADE_ESCALATION = Counter('agent_cascade_escalation', 'Cascade escalations', ['from_model', 'to_model'])
CACHE_HIT = Counter('agent_cache_hits_total', 'Cache hits', ['cache_type'])

# Efficiency gauges
QUALITY_PER_DOLLAR = Gauge('agent_quality_per_dollar', 'Rolling quality per dollar')

class MonitoredAgent:
    async def process(self, query: str) -> str:
        start = time.time()
        
        # Check cache
        cached = self.cache.get(query)
        if cached:
            CACHE_HIT.labels(cache_type="semantic").inc()
            return cached
        
        # Process
        with STAGE_LATENCY.labels(stage="retrieval").time():
            docs = await self.retrieve(query)
        
        model = self.router.select(query)
        with STAGE_LATENCY.labels(stage="llm_call").time():
            response, usage = await self.call_llm(model, query, docs)
        
        # Record metrics
        cost = self.cost_model.call_cost(usage.input_tokens, usage.output_tokens)
        COST_PER_QUERY.observe(cost)
        TOTAL_COST.labels(model=model).inc(cost)
        TOKEN_USAGE.labels(type="input", model=model).inc(usage.input_tokens)
        TOKEN_USAGE.labels(type="output", model=model).inc(usage.output_tokens)
        E2E_LATENCY.observe(time.time() - start)
        
        return response
```

**Alert Rules:**

```yaml
# Prometheus alert rules
groups:
  - name: agent_resource_alerts
    rules:
      - alert: HighCostPerQuery
        expr: histogram_quantile(0.95, agent_cost_per_query_usd_bucket) > 0.50
        for: 10m
        annotations:
          summary: "P95 cost per query exceeds $0.50"
      
      - alert: HighLatency
        expr: histogram_quantile(0.99, agent_e2e_latency_seconds_bucket) > 10
        for: 5m
        annotations:
          summary: "P99 latency exceeds 10 seconds"
      
      - alert: LowCacheHitRate
        expr: rate(agent_cache_hits_total[1h]) / rate(agent_queries_total[1h]) < 0.1
        for: 30m
        annotations:
          summary: "Cache hit rate below 10%"
      
      - alert: CostBudgetExceeded
        expr: increase(agent_total_cost_usd[24h]) > 5000
        annotations:
          summary: "Daily cost exceeds $5000 budget"
```

**Dashboard Layout:**

```
┌─────────────────────────────────┬──────────────────────────────┐
│  COST OVERVIEW                  │  LATENCY DISTRIBUTION        │
│  Daily: $1,247 / $5,000 budget  │  P50: 1.2s  P95: 4.8s      │
│  ████████░░░░░░ 25%             │  P99: 12.3s  TTFT: 340ms   │
│  Trend: ↓ 8% vs yesterday      │  [histogram chart]           │
├─────────────────────────────────┼──────────────────────────────┤
│  MODEL DISTRIBUTION             │  TOKEN EFFICIENCY            │
│  GPT-4o-mini: 72% of queries   │  Avg input: 3,200 tokens     │
│  GPT-4o: 23%                    │  Avg output: 450 tokens      │
│  o1: 5%                         │  Context utilization: 34%    │
│  [pie chart]                    │  Cache hit rate: 23%         │
├─────────────────────────────────┼──────────────────────────────┤
│  QUALITY METRICS                │  COST BREAKDOWN BY COMPONENT │
│  Success rate: 94.2%            │  LLM calls: 78%             │
│  User satisfaction: 4.3/5       │  Retrieval: 12%              │
│  Cascade rate: 28%              │  Tool execution: 7%          │
│  [time series chart]            │  Infrastructure: 3%          │
└─────────────────────────────────┴──────────────────────────────┘
```

---

**Chapter Summary: Core Principles of Resource-Aware Optimization**

| Principle | Formalization | Key Technique |
|---|---|---|
| Budget-constrained maximization | $\max_\pi \mathcal{Q}(\pi) \; \text{s.t.} \; \mathcal{C}(\pi) \leq B$ | Lagrangian relaxation, shadow prices |
| Token economy | $\mathcal{O}(k^2)$ cost growth in multi-step agents | Prompt compression, history summarization |
| Latency decomposition | $L = L_{\text{prefill}} + L_{\text{decode}} + L_{\text{tool}}$ | Speculative decoding, KV-cache reuse |
| Cost cascading | Route cheap $\to$ expensive by confidence | Learned routers, consistency-based confidence |
| Compute efficiency | Memory = weights + KV-cache + activations | Quantization (AWQ/GPTQ), tensor parallelism |
| Pareto optimality | No quality gain without cost increase on frontier | Multi-config evaluation, diminishing returns analysis |
| Production observability | Real-time resource consumption tracking | Prometheus metrics, cost alerts, efficiency dashboards |

The fundamental insight of resource-aware optimization is that **intelligence is a budget allocation problem.** The best agentic systems are not those that always use the most powerful model—they are those that allocate the right amount of computation to each query, each step, and each component, achieving maximum quality per unit of resource consumed.