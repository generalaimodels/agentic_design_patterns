

# Chapter 16: Reasoning Techniques

---

## 16.1 Definition and Formal Framework

### 16.1.1 What is Reasoning in LLM-Based Agents

**Reasoning** in LLM-based agents is the computational process of constructing, traversing, and validating chains of intermediate logical propositions to derive conclusions that are not directly retrievable from parametric memory or immediate context. It is the mechanism by which an agent bridges the gap between an input query $x$ and a correct output $y$ when the mapping $x \mapsto y$ requires non-trivial inferential steps.

**Formal Definition.** Let $\mathcal{X}$ be the space of input queries, $\mathcal{Y}$ the space of outputs, and $\mathcal{R}$ the space of reasoning traces. Reasoning is a function:

$$
\mathcal{F}: \mathcal{X} \rightarrow \mathcal{R}^* \times \mathcal{Y}
$$

where $\mathcal{R}^* = \bigcup_{n=0}^{\infty} \mathcal{R}^n$ denotes variable-length sequences of reasoning steps. The agent produces a trace $\mathbf{r} = (r_1, r_2, \ldots, r_n) \in \mathcal{R}^*$ before emitting a final answer $y \in \mathcal{Y}$.

**Why Reasoning is Necessary.** Standard LLM inference computes $P(y \mid x)$ in a single autoregressive pass. This works for associative recall (System 1) but fails for tasks requiring compositional generalization—problems where the answer depends on combining multiple facts, applying multi-step logical rules, or performing arithmetic operations that were never seen in that exact combination during training. Reasoning decomposes the intractable direct mapping into a sequence of tractable intermediate steps.

**Reasoning vs. Retrieval vs. Generation.**

| Capability | Mechanism | Example |
|-----------|-----------|---------|
| **Retrieval** | Look up stored fact from parametric memory | "What is the capital of France?" → "Paris" |
| **Generation** | Produce novel text by sampling from learned distribution | "Write a poem about autumn" |
| **Reasoning** | Construct logical chain bridging premises to conclusion | "If all A are B, and all B are C, is every A a C?" |

Reasoning is distinguished by its requirement for **compositionality** (combining sub-results), **systematicity** (applying rules consistently to novel inputs), and **verifiability** (each intermediate step can be independently checked for correctness).

**The Reasoning Bottleneck in Agents.** In agentic systems, reasoning determines:
1. **Planning quality:** Which tools to call, in what order, with what arguments
2. **Error recovery:** Detecting when an intermediate result is incorrect and backtracking
3. **State tracking:** Maintaining coherent beliefs across multi-turn interactions
4. **Goal decomposition:** Breaking complex objectives into achievable sub-goals

Without effective reasoning, agents degenerate into brittle pattern-matchers that fail on any task distribution shift from their training data.

---

### 16.1.2 Reasoning as Sequential Inference

The fundamental mathematical insight behind modern LLM reasoning is that injecting intermediate reasoning tokens into the generative process increases the expressiveness of the computation performed by a fixed-depth Transformer.

**Standard Direct Inference:**

$$
P(y \mid x) = \prod_{j=1}^{|y|} P(y_j \mid y_{<j}, x)
$$

This computes the answer in a single forward pass per token. The computational depth is bounded by the number of Transformer layers $L$, limiting the complexity class of functions the model can represent (bounded-depth threshold circuits, $\text{TC}^0$).

**Reasoning-Augmented Inference:**

$$
P(y \mid x) = \sum_{\mathbf{r} \in \mathcal{R}^*} P(\mathbf{r} \mid x) \cdot P(y \mid \mathbf{r}, x)
$$

By marginalizing over reasoning traces $\mathbf{r}$, the model accesses a richer function class. In practice, we do not marginalize—instead, we generate a single trace (or a small set) and condition the answer on it:

$$
P(y \mid x) \approx P(\mathbf{r}^* \mid x) \cdot P(y \mid \mathbf{r}^*, x)
$$

where $\mathbf{r}^*$ is the generated reasoning trace. The sequential factorization of the trace is:

$$
P(\mathbf{r} \mid x) = \prod_{i=1}^{n} P(r_i \mid r_{<i}, x)
$$

Therefore, the complete reasoning-augmented generative process is:

$$
P(y \mid x) = \prod_{i=1}^{n} P(r_i \mid r_{1}, \ldots, r_{i-1}, x) \cdot P(y \mid r_{1}, r_{2}, \ldots, r_n, x)
$$

**Computational Complexity Implications.** Each reasoning token $r_i$ effectively adds one additional layer of computation. A Transformer with $L$ layers producing $n$ reasoning tokens achieves effective depth $O(L \cdot n)$. This is critical:

- Without chain-of-thought: The model performs $O(L)$ serial computation steps. For a standard Transformer with $L \approx 32\text{–}80$ layers, this limits the model to problems solvable in constant parallel depth.
- With $n$ reasoning tokens: The model performs $O(L \cdot n)$ serial steps, enabling it to simulate computations requiring polynomial time.

**Theoretical Result (Feng et al., 2023; Merrill & Sabharwal, 2024).** A constant-depth, constant-precision Transformer cannot solve inherently sequential problems (e.g., composing $n$ permutations, evaluating Boolean formulas of depth $n$) without chain-of-thought. With $O(n)$ intermediate tokens, the same Transformer becomes Turing-complete up to the sequence length bound.

**Information-Theoretic Perspective.** Let $H(Y \mid X)$ be the conditional entropy of the answer given the question. Chain-of-thought reduces this entropy incrementally:

$$
H(Y \mid X) \geq H(Y \mid R_1, X) \geq H(Y \mid R_1, R_2, X) \geq \cdots \geq H(Y \mid R_{1:n}, X)
$$

Each reasoning step $R_i$ provides information about $Y$ that reduces the uncertainty remaining. The data processing inequality ensures that no step can increase uncertainty—but poorly chosen reasoning steps may provide negligible information gain, wasting tokens without reducing $H(Y \mid R_{1:i}, X)$.

**Optimal Reasoning Length.** The optimal number of reasoning steps $n^*$ balances information gain against cost:

$$
n^* = \arg\min_{n} \left[ H(Y \mid R_{1:n}, X) + \lambda \cdot n \right]
$$

where $\lambda$ is the per-token cost. This mirrors the rate-distortion tradeoff in information theory.

---

### 16.1.3 System 1 vs. System 2 Thinking in AI Agents

Drawing from Kahneman's dual-process theory, LLM agent reasoning can be categorized into two operational modes:

**System 1 (Fast, Intuitive, Pattern-Matching):**

- Single forward pass through the model
- No explicit reasoning trace generated
- Relies on statistical associations learned during pretraining
- Low latency, low token consumption
- Sufficient for: factual recall, simple classification, template-based generation
- Failure mode: confidently produces plausible-sounding but incorrect answers on tasks requiring multi-step inference

**Formal characterization of System 1:**

$$
y_{\text{S1}} = \arg\max_y P_\theta(y \mid x) \quad \text{(direct decoding, no intermediate tokens)}
$$

**System 2 (Slow, Deliberate, Analytical):**

- Generates explicit reasoning trace before answering
- Explores multiple solution paths (branching, backtracking)
- Verifies intermediate conclusions
- Higher latency and token consumption
- Required for: mathematical proof, multi-hop reasoning, planning under uncertainty
- Failure mode: overthinking simple problems, compounding errors in long chains

**Formal characterization of System 2:**

$$
y_{\text{S2}} = \arg\max_y P_\theta(y \mid \mathbf{r}^*, x), \quad \mathbf{r}^* = \text{Search}(\mathcal{R}, x, \text{eval})
$$

where $\text{Search}$ denotes an explicit search process over the reasoning space $\mathcal{R}$ guided by an evaluation function $\text{eval}$.

**Adaptive Mode Selection.** An optimal agent dynamically selects the reasoning mode based on estimated task complexity:

$$
\text{mode}(x) = \begin{cases}
\text{System 1} & \text{if } \hat{d}(x) \leq \delta \\
\text{System 2} & \text{if } \hat{d}(x) > \delta
\end{cases}
$$

where $\hat{d}(x)$ is an estimated difficulty score (computed via a lightweight classifier, perplexity of the query, or historical performance on similar queries) and $\delta$ is a calibrated threshold.

**Modern Instantiation.** This is precisely the architecture behind OpenAI's o1/o3 models and DeepSeek-R1: the model is trained to produce extended "thinking" tokens (System 2) before the answer, with the training process (typically GRPO or expert iteration) teaching the model when and how much to think.

| Property | System 1 | System 2 |
|----------|----------|----------|
| Token overhead | $O(1)$ output tokens | $O(n)$ reasoning tokens |
| Latency | $\sim$1s | $\sim$10–120s |
| Accuracy on hard tasks | Low (30–50%) | High (70–95%) |
| Cost | Low | 10–100× higher |
| Models | GPT-4o, Claude Sonnet | o1, o3, DeepSeek-R1, QwQ |

---

### 16.1.4 Taxonomy: Deductive, Inductive, Abductive, Analogical Reasoning

**Deductive Reasoning** (General → Specific)

Applies universal rules to specific instances to derive logically certain conclusions.

$$
\frac{P_1: \forall x.\; A(x) \Rightarrow B(x), \quad P_2: A(c)}{C: B(c)} \quad \text{(Modus Ponens)}
$$

- **LLM Implementation:** Given premises in the prompt, the model applies logical rules to derive conclusions.
- **Strengths:** Conclusions are guaranteed correct if premises are true and rules are applied validly.
- **Weakness in LLMs:** Models frequently fail at pure deductive reasoning, especially with negation, contraposition, and chains longer than 5–7 steps. This is because Transformers learn soft statistical patterns, not hard logical rules.
- **Example task:** "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"

**Inductive Reasoning** (Specific → General)

Generalizes from observed instances to universal patterns.

$$
\frac{O_1: A(c_1) \wedge B(c_1), \ldots, O_n: A(c_n) \wedge B(c_n)}{H: \forall x.\; A(x) \Rightarrow B(x)} \quad \text{(with probability, not certainty)}
$$

- **LLM Implementation:** In-context learning is fundamentally inductive reasoning—the model observes few-shot examples and infers the underlying pattern.
- **Bayesian formulation:**

$$
P(H \mid O_{1:n}) = \frac{P(O_{1:n} \mid H) \cdot P(H)}{P(O_{1:n})}
$$

- **Weakness in LLMs:** Susceptible to spurious correlations in examples; may infer the wrong generalization from ambiguous evidence.

**Abductive Reasoning** (Observation → Best Explanation)

Infers the most likely explanation for observed data (inference to the best explanation).

$$
\frac{O: B(c), \quad H: A(c) \Rightarrow B(c)}{E: A(c) \text{ is the best explanation for } B(c)}
$$

More formally:

$$
H^* = \arg\max_H P(H \mid O) = \arg\max_H P(O \mid H) \cdot P(H)
$$

- **LLM Implementation:** Diagnostic agents perform abductive reasoning—given symptoms (observations), infer the disease (explanation). Debugging agents observe error messages and abductively infer the root cause.
- **Key challenge:** Multiple hypotheses may explain the same observations. The agent must evaluate competing explanations and rank by plausibility.

**Analogical Reasoning** (Source Domain → Target Domain)

Transfers structural relationships from a familiar domain to an unfamiliar one.

$$
\frac{S: R(a, b) \text{ in domain } \mathcal{D}_1}{T: R(a', b') \text{ in domain } \mathcal{D}_2} \quad \text{where } a \leftrightarrow a', b \leftrightarrow b'
$$

- **Formal structure:** Let $\phi: \mathcal{D}_1 \rightarrow \mathcal{D}_2$ be a structural mapping. If $R(a, b)$ holds in $\mathcal{D}_1$, then $R(\phi(a), \phi(b))$ is hypothesized to hold in $\mathcal{D}_2$.
- **LLM Implementation:** Few-shot prompting is implicit analogical reasoning—the model maps structural patterns from examples to the test input.
- **Weakness in LLMs:** Models often transfer surface-level features rather than deep structural relations, a phenomenon called the "surface form competition" bias.

**Comparison Matrix:**

| Reasoning Type | Direction | Certainty | LLM Capability | Key Failure Mode |
|---------------|-----------|-----------|----------------|-----------------|
| Deductive | General → Specific | Logically certain | Moderate (fails on long chains) | Negation errors, invalid rule application |
| Inductive | Specific → General | Probabilistic | Strong (in-context learning) | Spurious generalization |
| Abductive | Observation → Explanation | Probabilistic | Moderate | Multiple competing hypotheses |
| Analogical | Domain₁ → Domain₂ | Heuristic | Variable | Surface vs. structural transfer |

---

## 16.2 Chain-of-Thought (CoT) Reasoning

### 16.2.1 Standard CoT Prompting

**Chain-of-Thought prompting** (Wei et al., 2022) is the foundational technique that elicits intermediate reasoning steps from an LLM by providing exemplars that demonstrate step-by-step problem solving.

**Mechanism.** Instead of mapping input directly to output, CoT prompting appends a reasoning trace $\mathbf{r}$ between input $x$ and output $y$:

$$
\text{Standard: } x \rightarrow y
$$

$$
\text{CoT: } x \rightarrow r_1 \rightarrow r_2 \rightarrow \cdots \rightarrow r_n \rightarrow y
$$

The LLM generates $\mathbf{r}$ and $y$ as a single autoregressive sequence. The reasoning steps $r_i$ are expressed in natural language and serve dual purposes:
1. **Computational scaffolding:** Each step performs a sub-computation whose result is available in context for subsequent steps.
2. **Attention guidance:** Intermediate tokens create attention anchors that help the model retrieve and combine relevant information.

**Mathematical Model.** The probability of the correct answer increases with CoT because:

$$
P(y \mid x, \mathbf{r}) \geq P(y \mid x) \quad \text{when } \mathbf{r} \text{ is a valid reasoning trace}
$$

This holds because the reasoning trace $\mathbf{r}$ provides additional conditioning information that is correlated with the correct answer. From a mutual information perspective:

$$
I(Y; R \mid X) > 0 \quad \Rightarrow \quad H(Y \mid X, R) < H(Y \mid X)
$$

**Exemplar Format:**

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. He bought 2 cans × 3 balls/can = 6 balls.
   Total = 5 + 6 = 11 tennis balls.
   The answer is 11.
```

**Empirical Impact (Wei et al., 2022):**

| Benchmark | Standard Prompting | CoT Prompting | Relative Gain |
|-----------|-------------------|---------------|---------------|
| GSM8K | 17.9% | 57.1% | +219% |
| SVAMP | 68.9% | 79.0% | +15% |
| AQUA | 30.3% | 52.0% | +72% |
| StrategyQA | 65.4% | 73.0% | +12% |

**Scale Dependence.** CoT provides negligible benefit for models smaller than ~10B parameters and can even hurt performance by introducing irrelevant tokens. The benefit emerges sharply above ~60B parameters (PaLM-540B, GPT-4 scale), suggesting that effective CoT requires sufficient parametric capacity to execute each reasoning step correctly.

$$
\Delta\mathcal{Q}_{\text{CoT}}(P) = \mathcal{Q}_{\text{CoT}}(P) - \mathcal{Q}_{\text{standard}}(P) \approx \begin{cases}
\leq 0 & \text{if } P < P_{\text{threshold}} \\
\text{positive, increasing} & \text{if } P \geq P_{\text{threshold}}
\end{cases}
$$

where $P$ is the model's parameter count and $P_{\text{threshold}} \approx 10\text{B–}60\text{B}$ depending on the task.

---

### 16.2.2 Zero-Shot CoT ("Let's think step by step")

**Key Innovation (Kojima et al., 2022).** Appending the single phrase "Let's think step by step" to any query elicits chain-of-thought reasoning without requiring hand-crafted exemplars.

**Two-Stage Process:**

**Stage 1 — Reasoning Extraction:**

$$
\hat{\mathbf{r}} = \text{LLM}(x \oplus \text{"Let's think step by step."})
$$

**Stage 2 — Answer Extraction:**

$$
\hat{y} = \text{LLM}(x \oplus \hat{\mathbf{r}} \oplus \text{"Therefore, the answer is"})
$$

The second stage is crucial: without explicit answer extraction, the model may generate a correct reasoning trace but fail to produce a parseable final answer.

**Why It Works.** The phrase "Let's think step by step" activates a latent reasoning mode in the model's generation distribution. During pretraining, the model has encountered millions of step-by-step explanations (textbooks, tutorials, Stack Overflow answers). The trigger phrase biases the model's conditional distribution toward generating structured reasoning traces:

$$
P(\text{reasoning trace} \mid x, \text{"Let's think step by step"}) \gg P(\text{reasoning trace} \mid x)
$$

**Variants and Improvements:**

| Trigger Phrase | GSM8K Accuracy | Notes |
|---------------|----------------|-------|
| (no trigger) | 17.9% | Direct answer |
| "Let's think step by step" | 40.7% | Original zero-shot CoT |
| "Let's work this out step by step to be sure we have the right answer" | 43.2% | Slightly improved |
| "Take a deep breath and work on this problem step-by-step" | 44.6% | (Yang et al., 2023) |
| "Think carefully, show all work" | 42.1% | Alternative phrasing |

**Limitations.**

1. **Inconsistent reasoning quality:** Without exemplars guiding the format, the model may produce disorganized or incomplete reasoning.
2. **Verbosity:** The model often generates unnecessary steps, increasing token cost.
3. **Error propagation:** A wrong intermediate step corrupts all subsequent reasoning with no mechanism for self-correction.

---

### 16.2.3 Few-Shot CoT with Exemplar Chains

**Method.** Provide $k$ exemplars, each consisting of a question, a detailed reasoning chain, and the final answer. The model then generates a reasoning chain for the test query by analogy.

**Prompt Structure:**

$$
\text{prompt} = \bigoplus_{j=1}^{k} (x_j, \mathbf{r}_j, y_j) \oplus x_{\text{test}}
$$

**Exemplar Design Principles:**

1. **Diversity of reasoning patterns.** Include exemplars requiring different reasoning types (arithmetic, logical deduction, multi-hop retrieval) to demonstrate the model's reasoning flexibility.

2. **Complexity matching.** Exemplar difficulty should approximate the test distribution. Excessively simple exemplars fail to elicit deep reasoning; excessively complex exemplars confuse the model.

3. **Step granularity calibration.** Each step should perform exactly one atomic operation (one arithmetic calculation, one logical inference, one fact retrieval). Mixing multiple operations in a single step degrades accuracy.

4. **Error-free chains.** Any error in exemplar chains teaches the model to replicate that error pattern.

**Formal Exemplar Selection.** Given a pool of candidate exemplars $\mathcal{E} = \{(x_j, \mathbf{r}_j, y_j)\}_{j=1}^{N}$, select $k$ exemplars to maximize expected test performance:

$$
\mathcal{E}^* = \arg\max_{\mathcal{S} \subseteq \mathcal{E}, |\mathcal{S}|=k} \; \mathbb{E}_{x_{\text{test}} \sim \mathcal{D}_{\text{test}}} \left[ \mathcal{Q}\left(\text{LLM}(\mathcal{S} \oplus x_{\text{test}})\right) \right]
$$

Practical selection heuristics:
- **Similarity-based:** Select exemplars with highest embedding similarity to the test query.
- **Diversity-based:** Select exemplars that cover the maximum number of reasoning skills (use clustering on exemplar embeddings, select one per cluster).
- **Complexity-stratified:** Include a mix of easy, medium, and hard exemplars.

**Implementation:**

```python
def few_shot_cot_prompt(exemplars: list[dict], test_query: str) -> str:
    prompt = ""
    for ex in exemplars:
        prompt += f"Question: {ex['question']}\n"
        prompt += f"Reasoning:\n{ex['chain']}\n"
        prompt += f"Answer: {ex['answer']}\n\n"
    prompt += f"Question: {test_query}\n"
    prompt += "Reasoning:\n"
    return prompt
```

---

### 16.2.4 Auto-CoT: Automatic Chain Generation

**Problem.** Manual creation of CoT exemplars is labor-intensive, error-prone, and does not scale across diverse task domains.

**Auto-CoT (Zhang et al., 2023).** Automatically generates diverse, high-quality exemplar chains through a two-phase pipeline:

**Phase 1: Question Clustering.**

Cluster training questions into $k$ groups using sentence embeddings:

$$
\{G_1, G_2, \ldots, G_k\} = \text{KMeans}\left(\{\text{emb}(x_j)\}_{j=1}^{N}, k\right)
$$

Select the question closest to each cluster centroid as the representative:

$$
x_i^* = \arg\min_{x \in G_i} \| \text{emb}(x) - \mu_i \|_2
$$

This ensures exemplar diversity—one exemplar per reasoning pattern.

**Phase 2: Zero-Shot Chain Generation.**

For each representative question $x_i^*$, generate a reasoning chain using zero-shot CoT:

$$
\mathbf{r}_i = \text{LLM}(x_i^* \oplus \text{"Let's think step by step"})
$$

Apply heuristic filters to reject low-quality chains:
- Reject if the chain is too short (fewer than 3 steps)
- Reject if the chain is too long (more than 20 steps—likely looping)
- Reject if the final answer format is unparseable
- Optionally verify the answer against a known ground truth

**Algorithm:**

```python
def auto_cot(questions: list[str], k: int, model) -> list[dict]:
    # Phase 1: Cluster questions
    embeddings = [embed(q) for q in questions]
    clusters = KMeans(n_clusters=k).fit(embeddings)
    
    exemplars = []
    for i in range(k):
        # Select representative question
        cluster_questions = [q for q, label in zip(questions, clusters.labels_) if label == i]
        cluster_embeddings = [e for e, label in zip(embeddings, clusters.labels_) if label == i]
        centroid = clusters.cluster_centers_[i]
        
        distances = [np.linalg.norm(e - centroid) for e in cluster_embeddings]
        representative = cluster_questions[np.argmin(distances)]
        
        # Phase 2: Generate chain
        chain = model.generate(representative + "\nLet's think step by step.")
        answer = extract_answer(chain)
        
        # Quality filter
        steps = chain.split("\n")
        if 3 <= len(steps) <= 20 and answer is not None:
            exemplars.append({
                "question": representative,
                "chain": chain,
                "answer": answer
            })
    
    return exemplars
```

**Performance.** Auto-CoT matches or slightly underperforms manually crafted CoT on most benchmarks (within 1–3% accuracy) while requiring zero human annotation effort.

---

### 16.2.5 Faithful CoT: Ensuring Reasoning Reflects Actual Computation

**The Faithfulness Problem.** LLMs can produce reasoning traces that appear logically sound but do not reflect the actual computational process the model uses to arrive at its answer. The model may:
1. **Post-hoc rationalize:** Generate a plausible explanation after already deciding the answer (the "motivated reasoning" failure).
2. **Skip steps:** Arrive at the correct answer through a shortcut not reflected in the trace.
3. **Produce inconsistent chains:** The stated reasoning implies answer $A$, but the model outputs answer $B$.

**Formal Definition of Faithfulness.** A reasoning trace $\mathbf{r}$ is faithful to the model's computation if:

$$
\text{Faithful}(\mathbf{r}, x, y) \iff P(y \mid x, \mathbf{r}) \gg P(y \mid x, \mathbf{r}') \quad \forall \, \mathbf{r}' \text{ implying } y' \neq y
$$

And additionally, perturbing a critical step in $\mathbf{r}$ changes the model's answer:

$$
\text{Counterfactual faithfulness: } r_i \rightarrow \tilde{r}_i \Rightarrow y \rightarrow \tilde{y} \neq y
$$

If replacing a reasoning step does not change the answer, the model was not actually using that step—it is decoration, not computation.

**Faithful CoT (Lyu et al., 2023).** Forces faithfulness by coupling natural language reasoning with executable symbolic programs:

**Architecture:**

```
Input x → LLM generates:
    1. Natural language plan (human-readable)
    2. Symbolic program (executable code)
→ Execute program → Deterministic answer
```

$$
y = \text{Execute}\left(\text{Program}(\mathbf{r})\right) \quad \text{not} \quad y = \text{LLM}(\mathbf{r})
$$

The answer is derived from program execution, not from the LLM's next-token prediction after the reasoning trace. This guarantees that the reasoning (as encoded in the program) is causally responsible for the answer.

**Example:**

```
Question: A store had 42 coloring books. They sold 27 and then got 
          a shipment of 20 more. How many do they have now?

Natural Language Plan:
  Step 1: Start with 42 coloring books
  Step 2: Subtract the 27 that were sold: 42 - 27 = 15
  Step 3: Add the 20 from the new shipment: 15 + 20 = 35

Symbolic Program:
  initial = 42
  after_sale = initial - 27
  after_shipment = after_sale + 20
  answer = after_shipment  # = 35
```

**Measuring Faithfulness — Perturbation Tests:**

1. **Early Answering Test:** Truncate the reasoning trace at step $i$ and check if the model still produces the same answer. If yes for early truncations, the later steps are unfaithful.

$$
\text{faithfulness\_score} = \min \{i : P(y \mid x, r_{1:i}) > 0.9 \cdot P(y \mid x, r_{1:n})\}
$$

A high score (close to $n$) indicates the model needs most steps. A low score indicates early steps suffice—later steps are rationalization.

2. **Corruption Test:** Inject an incorrect intermediate result and check if the model's answer changes:

$$
\Delta y = \mathbb{1}[\text{LLM}(x, r_1, \ldots, \tilde{r}_i, \ldots, r_n) \neq y]
$$

If $\Delta y = 0$ (answer unchanged despite corrupted step), the model ignored that step—unfaithful reasoning.

3. **Paraphrase Consistency:** Rephrase the reasoning trace while preserving logical content. A faithful model should produce the same answer.

---

## 16.3 Advanced Structured Reasoning

### 16.3.1 Tree of Thoughts (ToT)

**Motivation.** Standard CoT generates a single linear chain. If any step is wrong, the entire chain fails with no recovery mechanism. Tree of Thoughts (Yao et al., 2024) generalizes CoT from a chain to a tree, enabling exploration of multiple reasoning paths with backtracking.

**Formal Framework.** Define the reasoning process as a search over a tree $\mathcal{T} = (V, E)$:

- **Root node:** $v_0 = x$ (the input query)
- **Internal nodes:** Partial reasoning states $s = (r_1, r_2, \ldots, r_i)$
- **Edges:** Thought extensions—adding one reasoning step $r_{i+1}$
- **Leaf nodes:** Complete reasoning traces leading to candidate answers
- **Evaluation function:** $V(s) \in [0, 1]$ scoring the promise of partial state $s$

The search objective is:

$$
\mathbf{r}^* = \arg\max_{\mathbf{r} \in \text{Leaves}(\mathcal{T})} V(\mathbf{r})
$$

**Components in Detail:**

**1. Thought Decomposition.** The problem is decomposed into steps where each step is a "thought"—a coherent unit of reasoning. The granularity of thoughts is task-dependent:

| Task | Thought Granularity | Example |
|------|---------------------|---------|
| Creative writing | Paragraph-level plan | "The story opens with..." |
| Math problem | One equation/computation | "Substituting $x = 3$..." |
| Game playing | One move | "Place X in center" |
| Planning | One sub-goal | "First, gather requirements" |

**2. Thought Generation.** At each node $s$, generate $b$ candidate next thoughts (branching factor):

$$
\{r_{i+1}^{(1)}, r_{i+1}^{(2)}, \ldots, r_{i+1}^{(b)}\} = \text{Sample}^b\left(P_\theta(r_{i+1} \mid s, x)\right)
$$

Two generation strategies:
- **Independent sampling:** Sample $b$ completions independently. Higher diversity, but may produce redundant thoughts.
- **Sequential proposal:** Generate thoughts one at a time, conditioning each on previous proposals to ensure diversity.

**3. Evaluation Function $V(s)$.** Scores the quality/promise of a partial reasoning state. Implemented via:

- **Value prompting:** Ask the LLM to evaluate the state:

```
Given the current reasoning state, rate the likelihood of reaching 
the correct answer on a scale of 1-10.
State: {s}
Rating:
```

$$
V(s) = \frac{\text{LLM\_rating}(s)}{10}
$$

- **Classification prompting:** Ask the LLM to classify the state as "sure", "maybe", or "impossible":

$$
V(s) = \begin{cases} 1.0 & \text{if "sure"} \\ 0.5 & \text{if "maybe"} \\ 0.0 & \text{if "impossible"} \end{cases}
$$

- **Simulation (rollout):** Complete the reasoning from state $s$ using greedy decoding and evaluate the final answer quality.

**4. Search Algorithms.**

**Breadth-First Search (BFS):**

```python
def tot_bfs(problem, branching_factor, max_depth, beam_width):
    current_states = [State(problem)]
    
    for depth in range(max_depth):
        candidates = []
        for state in current_states:
            # Generate b candidate next thoughts
            thoughts = generate_thoughts(state, branching_factor)
            for thought in thoughts:
                new_state = state.extend(thought)
                score = evaluate(new_state)
                candidates.append((new_state, score))
        
        # Keep top-k states (beam search)
        candidates.sort(key=lambda x: x[1], reverse=True)
        current_states = [s for s, _ in candidates[:beam_width]]
    
    # Return best final state
    return max(current_states, key=evaluate)
```

**Depth-First Search (DFS) with Backtracking:**

```python
def tot_dfs(problem, branching_factor, max_depth, threshold):
    def dfs(state, depth):
        if depth == max_depth:
            return state, evaluate(state)
        
        thoughts = generate_thoughts(state, branching_factor)
        best_state, best_score = None, -float('inf')
        
        for thought in thoughts:
            new_state = state.extend(thought)
            score = evaluate(new_state)
            
            if score < threshold:
                continue  # Prune unpromising branches
            
            final_state, final_score = dfs(new_state, depth + 1)
            if final_score > best_score:
                best_state, best_score = final_state, final_score
        
        return best_state, best_score
    
    return dfs(State(problem), 0)
```

**Pruning Heuristics:**

1. **Score threshold:** Prune nodes with $V(s) < \tau_{\text{prune}}$.
2. **Diversity pruning:** If two sibling nodes are too similar (cosine similarity $> 0.95$), keep only the higher-scoring one.
3. **Depth-adaptive threshold:** Tighten the pruning threshold as depth increases to prevent exponential explosion:

$$
\tau(d) = \tau_0 + \alpha \cdot d
$$

4. **Confidence-based early termination:** If any node achieves $V(s) > \tau_{\text{confident}}$ (e.g., 0.95), immediately commit to that branch.

**Cost Analysis.**

Let $b$ = branching factor, $d$ = depth, $w$ = beam width:

| Search Strategy | LLM Calls (Generation) | LLM Calls (Evaluation) | Total |
|----------------|----------------------|----------------------|-------|
| BFS (beam=$w$) | $w \cdot b \cdot d$ | $w \cdot b \cdot d$ | $O(wbd)$ |
| DFS (full) | $b^d$ | $b^d$ | $O(b^d)$ |
| DFS (pruned) | $\ll b^d$ | $\ll b^d$ | Problem-dependent |

Typical settings: $b = 3\text{–}5$, $d = 3\text{–}5$, $w = 3\text{–}5$ → 30–75 LLM calls per problem. This is 30–75× more expensive than standard CoT but can solve problems that CoT cannot.

---

### 16.3.2 Graph of Thoughts (GoT)

**Motivation.** Tree of Thoughts restricts reasoning to a tree structure—each node has exactly one parent. Real reasoning often requires **merging insights from multiple branches** (aggregation), **refining previous thoughts** (loops), and **non-linear dependencies**. Graph of Thoughts (Besta et al., 2024) generalizes the reasoning structure from a tree to a directed acyclic graph (DAG) or even a general directed graph.

**Formal Framework.** The reasoning structure is a directed graph $\mathcal{G} = (V, E)$ where:

- $V$ is the set of thought nodes (partial reasoning states)
- $E \subseteq V \times V$ represents dependencies between thoughts
- A node $v$ can have multiple parents (merge/aggregation)
- Cycles are permitted for iterative refinement

**Graph Operations:**

| Operation | Description | Formal Definition |
|-----------|-------------|-------------------|
| **Generate** | Create new thought nodes from an existing node | $v' = \text{LLM}(v, \text{gen\_prompt})$ |
| **Aggregate** | Merge insights from multiple nodes into one | $v_{\text{agg}} = \text{LLM}(v_1, v_2, \ldots, v_k, \text{agg\_prompt})$ |
| **Refine** | Improve an existing thought node | $v' = \text{LLM}(v, \text{feedback}, \text{refine\_prompt})$ |
| **Score** | Evaluate a thought node's quality | $s(v) = \text{LLM}(v, \text{eval\_prompt}) \in [0, 1]$ |

**Key Advantage over ToT: Aggregation.** Consider sorting a list. ToT would explore different sorting strategies as independent branches. GoT can:
1. Split the list into sub-lists (generate)
2. Sort each sub-list independently in parallel (generate)
3. Merge sorted sub-lists (aggregate)—this operation has no analog in tree structures

**Formal Aggregation:**

$$
v_{\text{merged}} = f_{\text{agg}}(v_1, v_2, \ldots, v_k) = \text{LLM}\left(\bigoplus_{i=1}^{k} v_i, \text{"Synthesize these partial solutions into a single coherent solution."}\right)
$$

**Graph Controller Architecture:**

```python
class GraphOfThoughts:
    def __init__(self, model, evaluator):
        self.model = model
        self.evaluator = evaluator
        self.graph = nx.DiGraph()
        self.node_counter = 0
    
    def generate(self, parent_id: int, n: int = 3) -> list[int]:
        parent_content = self.graph.nodes[parent_id]['content']
        children_ids = []
        for _ in range(n):
            thought = self.model.generate(
                f"Given: {parent_content}\nGenerate the next reasoning step:"
            )
            child_id = self._add_node(thought, parents=[parent_id])
            children_ids.append(child_id)
        return children_ids
    
    def aggregate(self, node_ids: list[int]) -> int:
        contents = [self.graph.nodes[nid]['content'] for nid in node_ids]
        merged = self.model.generate(
            f"Synthesize these partial results into one:\n" +
            "\n".join(f"- {c}" for c in contents)
        )
        return self._add_node(merged, parents=node_ids)
    
    def refine(self, node_id: int, feedback: str) -> int:
        content = self.graph.nodes[node_id]['content']
        refined = self.model.generate(
            f"Original: {content}\nFeedback: {feedback}\nImproved version:"
        )
        return self._add_node(refined, parents=[node_id])
    
    def score(self, node_id: int) -> float:
        content = self.graph.nodes[node_id]['content']
        return self.evaluator.evaluate(content)
    
    def _add_node(self, content: str, parents: list[int]) -> int:
        nid = self.node_counter
        self.node_counter += 1
        self.graph.add_node(nid, content=content, score=None)
        for pid in parents:
            self.graph.add_edge(pid, nid)
        return nid
```

**Non-Linear Reasoning Topologies.** GoT supports reasoning patterns impossible in linear or tree structures:

```
Linear (CoT):       A → B → C → D

Tree (ToT):         A → B₁ → C₁
                      → B₂ → C₂

Graph (GoT):        A → B₁ ↘
                      → B₂ → D (aggregation) → E → F (refinement of E)
                      → B₃ ↗                    ↑______|
```

**Volume of Thought (VoT) Metric.** GoT introduces a metric for reasoning complexity:

$$
\text{VoT}(\mathcal{G}) = |V| + |E| - |\text{connected\_components}(\mathcal{G})|
$$

GoT achieves higher VoT than CoT or ToT while maintaining accuracy, indicating richer reasoning structure per problem.

---

### 16.3.3 Algorithm of Thoughts (AoT)

**Core Idea (Sel et al., 2023).** Rather than having the LLM explore a search tree externally (as in ToT), AoT instructs the LLM to simulate the search algorithm *within a single generation*—performing in-context algorithmic exploration.

**Key Insight.** LLMs have been pretrained on algorithmic descriptions (pseudocode, textbook examples of DFS/BFS). AoT exploits this by prompting the model to execute the algorithm step by step, maintaining the search state within the context window.

**Prompt Structure:**

```
Solve the following problem using depth-first search with backtracking.

Problem: [problem description]

Search Process:
Step 1: Try approach A...
  → Evaluation: This leads to contradiction because...
  → Backtrack.

Step 2: Try approach B...
  → Sub-step 2.1: ...
  → Sub-step 2.2: This looks promising...
  → Continue.

Step 3: ...
```

**Formal Model.** The LLM generates a trace that encodes an implicit search tree:

$$
\mathbf{r}_{\text{AoT}} = (s_1, e_1, d_1, s_2, e_2, d_2, \ldots)
$$

where $s_i$ is a search state, $e_i$ is its evaluation, and $d_i \in \{\text{continue}, \text{backtrack}\}$ is the search decision.

**Advantages over ToT:**
- **Single LLM call** instead of $O(wbd)$ calls (dramatically lower cost)
- **Shared context:** The model can learn from failed branches within the same generation
- **No external orchestration:** No search controller code needed

**Limitations:**
- **Context window bound:** The implicit search tree must fit within the context window
- **Reliability:** The model may fail to faithfully simulate the algorithm, especially for deep trees
- **No true parallelism:** Unlike ToT's parallel LLM calls, AoT is sequential

---

### 16.3.4 Skeleton-of-Thought (SoT)

**Core Idea (Ning et al., 2024).** Reduce end-to-end latency by first generating a skeleton (outline) of the answer, then expanding each point in parallel.

**Two-Phase Process:**

**Phase 1: Skeleton Generation (Sequential, Fast).**

$$
\text{skeleton} = \text{LLM}(x, \text{"Provide a concise skeleton outline of your answer."})
$$

Output: A list of $k$ key points: $[p_1, p_2, \ldots, p_k]$.

**Phase 2: Parallel Expansion.**

$$
\text{expanded}_i = \text{LLM}(x, p_i, \text{"Expand this point in detail."}) \quad \forall \, i \in \{1, \ldots, k\} \quad \text{(in parallel)}
$$

**Final answer:** Concatenation of all expanded points.

**Latency Analysis:**

$$
L_{\text{CoT}} = L_{\text{generate}}(n_{\text{total}}) \approx \frac{n_{\text{total}}}{R_{\text{decode}}}
$$

$$
L_{\text{SoT}} = L_{\text{skeleton}} + \max_{i} L_{\text{expand}_i} = \frac{n_{\text{skeleton}}}{R_{\text{decode}}} + \frac{\max_i n_{\text{expand}_i}}{R_{\text{decode}}}
$$

Since $n_{\text{skeleton}} + \max_i n_{\text{expand}_i} \ll n_{\text{total}} = n_{\text{skeleton}} + \sum_i n_{\text{expand}_i}$:

$$
\text{Speedup} \approx \frac{\sum_i n_{\text{expand}_i}}{\max_i n_{\text{expand}_i}} \approx k \quad \text{(for balanced expansions)}
$$

With $k = 5$ key points, SoT achieves approximately $2\text{–}3\times$ speedup (not a full $5\times$ due to skeleton generation overhead and unbalanced expansion lengths).

**Applicability.** SoT works well for tasks with naturally decomposable outputs (essay writing, report generation, multi-aspect analysis). It is **not suitable** for tasks requiring sequential logical dependencies (math proofs, step-by-step calculations).

---

### 16.3.5 Thread of Thought

**Core Idea.** For complex queries that span multiple interrelated aspects, Thread of Thought (ThoT) instructs the model to systematically address each aspect in a dedicated thread, maintaining explicit awareness of cross-thread dependencies.

**Prompt Pattern:**

```
Walk me through this in a systematic thread-by-thread manner. 
For each thread, clearly state what aspect you're addressing and 
how it connects to other threads.

Thread 1 [Aspect A]: ...
Thread 2 [Aspect B]: ...
Cross-thread synthesis: ...
```

**Formal Model.** Let the query $x$ decompose into $m$ aspects $\{a_1, \ldots, a_m\}$. Each thread $t_i$ reasons about aspect $a_i$ independently:

$$
t_i = \text{LLM}(x, a_i, \text{"Reason about aspect } a_i \text{"})
$$

A synthesis step integrates all threads:

$$
y = \text{LLM}(t_1, t_2, \ldots, t_m, \text{"Synthesize threads into final answer"})
$$

**Distinction from SoT.** SoT parallelizes *generation* for latency; ThoT structures *reasoning* for quality on multi-faceted problems. ThoT threads may be generated sequentially (each informed by previous threads) or in parallel (if aspects are independent).

---

## 16.4 Iterative and Test-Time Reasoning

### 16.4.1 Test-Time Compute Scaling

**Key Insight.** Model quality can be improved not only by investing more compute during training (scaling model parameters, training data, or training FLOPs) but also by investing more compute during inference. This is the principle of **test-time compute scaling**.

**Formal Framework.** Model quality is a function of both training compute $c_{\text{train}}$ and inference compute $c_{\text{inference}}$:

$$
\mathcal{Q} = f(c_{\text{train}}, c_{\text{inference}})
$$

**Training-time scaling (Chinchilla law):**

$$
\mathcal{Q}_{\text{train}}(c) \approx a - \frac{b}{c^{\alpha}} \quad \text{with } \alpha \approx 0.05\text{–}0.1
$$

This yields diminishing returns: doubling training compute improves quality by ~4–7%.

**Inference-time scaling:**

$$
\mathcal{Q}_{\text{inference}}(c) \approx \mathcal{Q}_0 + \beta \cdot \log(c_{\text{inference}})
$$

where $c_{\text{inference}}$ is measured in tokens generated, number of samples, or number of search steps.

**Critical Result (Snell et al., 2024).** For a fixed total compute budget $C_{\text{total}}$, there exists an optimal allocation between training and inference:

$$
c_{\text{train}}^*, c_{\text{inference}}^* = \arg\max_{c_t + c_i \cdot N_{\text{queries}} \leq C_{\text{total}}} f(c_t, c_i)
$$

In many regimes, a smaller model with more inference-time compute outperforms a larger model with less inference-time compute at the same total budget.

**Inference-Time Compute Strategies:**

| Strategy | Compute Multiplier | Quality Gain | Mechanism |
|----------|-------------------|--------------|-----------|
| Standard (1 sample) | 1× | Baseline | Single forward pass |
| Chain-of-thought | 1.5–3× | +10–30% | More tokens generated |
| Self-consistency ($k$ samples) | $k\times$ | +5–15% | Majority voting |
| Best-of-$N$ + verifier | $N\times$ + verifier cost | +10–25% | Selection |
| ToT/GoT | 30–100× | +15–40% | Structured search |
| MCTS-guided reasoning | 50–500× | +20–50% | Monte Carlo Tree Search |

**Scaling Curve (Empirical, Math Tasks):**

$$
\text{Accuracy}(N) \approx A_{\max} - (A_{\max} - A_1) \cdot N^{-\gamma}
$$

where $N$ is the number of samples/search steps and $\gamma \approx 0.3\text{–}0.5$ for math reasoning tasks. The accuracy improves as a power law with diminishing returns.

---

### 16.4.2 Self-Consistency (Majority Voting Over Multiple CoT Paths)

**Method (Wang et al., 2023).** Generate $k$ independent reasoning chains using temperature sampling ($T > 0$), extract the final answer from each chain, and select the answer that appears most frequently.

**Formal Definition:**

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^{k} \mathbb{1}[g(\mathbf{r}_i) = y]
$$

where:
- $\mathbf{r}_i \sim P_\theta(\mathbf{r} \mid x, T)$ is the $i$-th reasoning chain sampled with temperature $T$
- $g(\mathbf{r}_i)$ extracts the final answer from chain $\mathbf{r}_i$
- $\mathbb{1}[\cdot]$ is the indicator function

**Why It Works.** If the model has probability $p > 0.5$ of producing a correct reasoning chain, then by the law of large numbers, the majority vote converges to the correct answer as $k \rightarrow \infty$:

$$
P(\hat{y} = y^*) = \sum_{j=\lceil k/2 \rceil}^{k} \binom{k}{j} p^j (1-p)^{k-j} \xrightarrow{k \to \infty} 1 \quad \text{for } p > 0.5
$$

The error rate decreases exponentially in $k$:

$$
P(\hat{y} \neq y^*) \leq \exp\left(-2k\left(p - \frac{1}{2}\right)^2\right) \quad \text{(Hoeffding's inequality)}
$$

**Optimal Sampling Temperature.** The temperature $T$ controls the diversity-quality tradeoff:
- $T$ too low → All chains are identical (no diversity, equivalent to single sample)
- $T$ too high → Chains are low-quality (errors dominate)

Optimal $T$ typically lies in $[0.5, 0.9]$ for reasoning tasks. Formally:

$$
T^* = \arg\max_T \; P\left(\arg\max_y \sum_{i=1}^{k} \mathbb{1}[g(\mathbf{r}_i^{(T)}) = y] = y^*\right)
$$

**Weighted Self-Consistency.** Weight each chain by its log-likelihood:

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^{k} w_i \cdot \mathbb{1}[g(\mathbf{r}_i) = y], \quad w_i = \exp\left(\frac{1}{|\mathbf{r}_i|} \sum_{j} \log P(r_{i,j} \mid r_{i,<j}, x)\right)
$$

This gives higher weight to chains that the model itself considers more likely—a built-in quality filter.

**Implementation:**

```python
def self_consistency(model, prompt: str, k: int = 10, temperature: float = 0.7) -> str:
    answers = []
    for _ in range(k):
        response = model.generate(prompt, temperature=temperature)
        answer = extract_final_answer(response)
        answers.append(answer)
    
    # Majority voting
    answer_counts = Counter(answers)
    return answer_counts.most_common(1)[0][0]
```

**Empirical Results:**

| Benchmark | CoT (greedy) | SC (k=10) | SC (k=40) | Improvement |
|-----------|-------------|-----------|-----------|-------------|
| GSM8K (PaLM-540B) | 56.5% | 74.4% | 78.0% | +38% |
| SVAMP | 79.0% | 86.6% | 88.5% | +12% |
| ARC-Challenge | 85.2% | 90.1% | 91.4% | +7% |

---

### 16.4.3 Best-of-N Sampling with Verifier

**Method.** Generate $N$ candidate solutions independently, then use a trained verifier model to select the best one:

$$
\hat{y} = \arg\max_{i \in \{1, \ldots, N\}} V(\mathbf{r}_i, x)
$$

where $V: \mathcal{R}^* \times \mathcal{X} \rightarrow \mathbb{R}$ is the verifier scoring function.

**Distinction from Self-Consistency.** Self-consistency selects by majority vote (no quality model). Best-of-$N$ uses a learned verifier to discriminate between correct and incorrect solutions, even if the incorrect answer is the majority.

**Verifier Training.** The verifier is trained on pairs of (problem, solution, correctness\_label):

$$
\mathcal{L}_{\text{verifier}} = -\mathbb{E}_{(x, \mathbf{r}, c)} \left[ c \cdot \log V(\mathbf{r}, x) + (1-c) \cdot \log(1 - V(\mathbf{r}, x)) \right]
$$

where $c \in \{0, 1\}$ is the correctness label. Training data is generated by:
1. Sampling many solutions from the generator for each training problem
2. Automatically checking correctness against ground truth
3. Training a classifier (often a fine-tuned LLM) to predict correctness

**Scaling Behavior.** The probability that the best solution among $N$ samples is correct:

$$
P(\text{at least one correct in } N) = 1 - (1 - p)^N
$$

where $p$ is the per-sample correctness probability. For $p = 0.3$ and $N = 64$:

$$
P = 1 - 0.7^{64} \approx 1 - 10^{-10} \approx 1.0
$$

However, the verifier must successfully identify the correct solution from $N$ candidates. Verifier accuracy $v$ gives:

$$
P(\text{correct selection}) \approx \left(1 - (1-p)^N\right) \cdot v
$$

**Cost-Optimal $N$.** Given per-sample cost $c_{\text{gen}}$ and verifier cost $c_{\text{ver}}$:

$$
\text{Total cost} = N \cdot c_{\text{gen}} + c_{\text{ver}}
$$

The optimal $N$ maximizes quality per dollar:

$$
N^* = \arg\max_N \frac{\mathcal{Q}(N)}{N \cdot c_{\text{gen}} + c_{\text{ver}}}
$$

Typically $N \in [8, 256]$ depending on task difficulty and verifier quality.

---

### 16.4.4 Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)

**The Reward Model Distinction.** Both PRMs and ORMs provide signals to guide reasoning, but they differ fundamentally in what they evaluate.

**Outcome Reward Model (ORM).** Scores only the final answer:

$$
\text{ORM}(x, \mathbf{r}, y) = P(\text{correct} \mid x, y)
$$

The ORM ignores the reasoning trace entirely—it only cares whether the answer $y$ is correct.

**Process Reward Model (PRM).** Scores each intermediate reasoning step:

$$
\text{PRM}(x, r_1, r_2, \ldots, r_t) = P(\text{step } r_t \text{ is correct} \mid x, r_1, \ldots, r_{t-1})
$$

The overall score of a reasoning trace is the product of per-step scores:

$$
\text{PRM\_score}(\mathbf{r}) = \prod_{t=1}^{n} \text{PRM}(x, r_{1:t})
$$

Or equivalently, the minimum per-step score (bottleneck scoring):

$$
\text{PRM\_score}(\mathbf{r}) = \min_{t \in \{1,\ldots,n\}} \text{PRM}(x, r_{1:t})
$$

**Comparison:**

| Property | ORM | PRM |
|----------|-----|-----|
| **Supervision granularity** | Final answer only | Each reasoning step |
| **Training data cost** | Low (auto-check answers) | High (human step-level labels or synthetic) |
| **Error localization** | Cannot identify which step is wrong | Pinpoints the first error step |
| **Credit assignment** | Sparse (entire chain gets same reward) | Dense (each step rewarded individually) |
| **Search guidance** | Poor (must complete full trace to evaluate) | Strong (prune bad branches early) |
| **Empirical performance** | Good for simple tasks | Superior for complex multi-step reasoning |

**PRM for Search Guidance.** During beam search or tree search, PRM enables step-level pruning:

$$
\text{Prune step } r_{t+1} \text{ if } \text{PRM}(x, r_{1:t}, r_{t+1}) < \tau_{\text{step}}
$$

This prevents the search from wasting compute exploring subtrees rooted at incorrect steps.

**PRM Training (Lightman et al., 2023).** Training data consists of (problem, step, correctness) triples:

**Human-labeled approach:**
1. Generate solutions step by step
2. Human annotators label each step as correct/incorrect/neutral
3. Train a classifier (fine-tuned LLM) on these labels

**Synthetic approach (Math-Shepherd, Wang et al., 2024):**
1. Generate solution up to step $t$
2. Complete the solution from step $t$ using the generator $M$ times
3. If any completion reaches the correct answer, label step $t$ as correct
4. Otherwise, label step $t$ as incorrect

$$
\text{label}(r_t) = \begin{cases}
\text{correct} & \text{if } \exists \; i \in \{1,\ldots,M\} : g(\mathbf{r}_{>t}^{(i)}) = y^* \\
\text{incorrect} & \text{otherwise}
\end{cases}
$$

This is a Monte Carlo estimate of step correctness—a step is correct if the correct answer remains reachable from that step.

**Formal Connection to Advantage Functions in RL.** The PRM's per-step score is analogous to the advantage function $A(s_t, a_t)$ in reinforcement learning:

$$
\text{PRM}(x, r_{1:t}) \approx A(s_t, r_t) = Q(s_t, r_t) - V(s_t)
$$

where $Q(s_t, r_t)$ is the expected reward from taking step $r_t$ in state $s_t = (x, r_{1:t-1})$, and $V(s_t)$ is the value of state $s_t$. Positive PRM scores indicate steps that increase the probability of reaching the correct answer; negative scores indicate steps that decrease it.

**Key Result (Lightman et al., 2023).** On MATH benchmark with GPT-4 as the generator:

| Verifier | Best-of-100 Accuracy |
|----------|---------------------|
| No verifier (majority vote) | 72.4% |
| ORM | 76.1% |
| PRM | **78.2%** |

PRM's step-level guidance provides a consistent advantage, especially on problems requiring more reasoning steps.

---

### 16.4.5 Iterative Refinement with Feedback

**Method.** Instead of generating a single answer, the agent generates an initial solution, receives feedback (from a critic model, a tool, or self-evaluation), and iteratively improves the solution.

**Formal Framework.** Iterative refinement is a fixed-point iteration:

$$
\mathbf{r}^{(t+1)} = \text{Refine}\left(\mathbf{r}^{(t)}, \text{Feedback}(\mathbf{r}^{(t)}, x)\right)
$$

converging (ideally) to a fixed point $\mathbf{r}^* = \text{Refine}(\mathbf{r}^*, \text{Feedback}(\mathbf{r}^*, x))$ or stopping when quality exceeds a threshold.

**Quality Trajectory:**

$$
\mathcal{Q}(\mathbf{r}^{(0)}) \leq \mathcal{Q}(\mathbf{r}^{(1)}) \leq \cdots \leq \mathcal{Q}(\mathbf{r}^{(T)})
$$

This monotonic improvement is **not guaranteed** in practice. Empirically, refinement improves quality for 1–3 iterations, then plateaus or degrades (the model may "refine" correct parts into incorrect ones).

**Feedback Sources:**

| Source | Type | Reliability | Example |
|--------|------|------------|---------|
| Self-critique (same LLM) | Natural language feedback | Low–Medium | "Re-examine step 3 for errors" |
| External verifier | Binary or scored | High | Unit test passes/fails |
| Code execution | Deterministic | Very high | Runtime errors, incorrect output |
| Human feedback | Natural language | Highest | "The calculation in line 2 is wrong" |
| Reward model | Scalar score | Medium–High | PRM score drops at step 4 |

**Self-Refine (Madaan et al., 2023):**

```python
def self_refine(model, problem: str, max_iterations: int = 3) -> str:
    # Initial generation
    solution = model.generate(f"Solve: {problem}")
    
    for i in range(max_iterations):
        # Self-critique
        feedback = model.generate(
            f"Problem: {problem}\n"
            f"Solution: {solution}\n"
            f"Identify any errors or areas for improvement:"
        )
        
        # Check if the model thinks the solution is already correct
        if "no errors" in feedback.lower() or "correct" in feedback.lower():
            break
        
        # Refine based on feedback
        solution = model.generate(
            f"Problem: {problem}\n"
            f"Previous solution: {solution}\n"
            f"Feedback: {feedback}\n"
            f"Provide an improved solution:"
        )
    
    return solution
```

**Critical Limitation (Huang et al., 2024).** Recent work shows that LLMs cannot reliably self-correct their reasoning without external feedback. Self-refinement works primarily when:
1. The model can verify (e.g., via code execution) whether its answer is correct
2. The initial error is a surface-level mistake (typo, formatting) rather than a deep reasoning error
3. An external signal (tool output, unit test result) provides grounded feedback

Without external grounding, self-refinement often degrades performance—the model "corrects" correct answers into incorrect ones.

---

## 16.5 Reasoning with External Verification

### 16.5.1 Code-Based Verification (Write Code to Check Answer)

**Method.** The agent generates both a natural language solution and executable code to verify its answer. Discrepancies between the analytical solution and the code output trigger re-reasoning.

**Architecture:**

$$
x \xrightarrow{\text{LLM}} (\mathbf{r}_{\text{NL}}, y_{\text{NL}}, \text{code}_{\text{verify}}) \xrightarrow{\text{Execute}} y_{\text{code}}
$$

$$
\text{If } y_{\text{NL}} \neq y_{\text{code}} \Rightarrow \text{Re-reason with discrepancy as feedback}
$$

**Implementation (PAL - Program-Aided Language Models, Gao et al., 2023):**

```python
def pal_solve(model, problem: str) -> str:
    # Generate code solution
    code = model.generate(
        f"Write Python code to solve this problem. "
        f"Print the final answer.\n\nProblem: {problem}\n\nCode:"
    )
    
    # Execute in sandboxed environment
    try:
        result = execute_sandboxed(code, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        # Fallback: re-generate with error feedback
        code_v2 = model.generate(
            f"Problem: {problem}\n"
            f"Previous code:\n{code}\n"
            f"Error: {str(e)}\n"
            f"Fixed code:"
        )
        result = execute_sandboxed(code_v2, timeout=10)
        return result.stdout.strip()
```

**Dual Verification Pattern:**

```python
def dual_verify(model, problem: str) -> dict:
    # Path 1: Natural language reasoning
    nl_solution = model.generate(f"Solve step by step: {problem}")
    nl_answer = extract_answer(nl_solution)
    
    # Path 2: Code solution
    code = model.generate(f"Write Python code to solve: {problem}")
    code_answer = execute_sandboxed(code)
    
    if nl_answer == code_answer:
        return {"answer": nl_answer, "confidence": "high", "method": "dual-verified"}
    else:
        # Reconciliation: use code answer (more reliable for computational tasks)
        return {"answer": code_answer, "confidence": "medium", 
                "discrepancy": f"NL={nl_answer}, Code={code_answer}"}
```

**When Code Verification is Effective:**

| Task Type | Code Verifiable? | Reliability |
|-----------|-----------------|-------------|
| Arithmetic/algebra | Yes | Very high |
| Data analysis | Yes | Very high |
| Logical puzzles | Partially | Medium (encoding logic is error-prone) |
| Reading comprehension | No | N/A |
| Creative writing | No | N/A |
| Physics with numerical answer | Yes | High |

---

### 16.5.2 Symbolic Reasoning Integration

**Method.** Translate the reasoning problem (or sub-problem) into a formal symbolic representation and use a symbolic solver to compute the answer.

**Architecture:**

$$
x \xrightarrow{\text{LLM (formalization)}} \phi(x) \xrightarrow{\text{Symbolic Solver}} y_{\text{symbolic}} \xrightarrow{\text{LLM (interpretation)}} y_{\text{NL}}
$$

where $\phi(x)$ is the formal representation (first-order logic, SAT formula, linear program, etc.).

**Symbolic Backends:**

| Backend | Representation | Capabilities |
|---------|---------------|-------------|
| Z3 (SMT Solver) | First-order logic + theories | Satisfiability, optimization, constraint solving |
| SymPy | Symbolic algebra | Equation solving, calculus, simplification |
| Prolog | Horn clauses | Logical inference, constraint logic programming |
| CPLEX/Gurobi | Linear/integer programs | Optimization |
| Lean 4 / Coq | Type theory | Formal theorem proving |

**Example: LLM + Z3 for Logic Puzzles:**

```python
from z3 import *

def symbolic_solve(model, problem: str) -> str:
    # LLM translates natural language to Z3 constraints
    z3_code = model.generate(
        f"Translate this logic puzzle into Z3 Python code.\n"
        f"Puzzle: {problem}\n"
        f"Z3 Code:"
    )
    
    # Execute Z3 solver
    result = execute_sandboxed(z3_code)
    
    # LLM interprets symbolic result
    interpretation = model.generate(
        f"Puzzle: {problem}\n"
        f"Solver output: {result}\n"
        f"Express the solution in natural language:"
    )
    
    return interpretation
```

**Strengths:**
- Symbolic solvers provide **provably correct** answers within their domain
- Eliminates arithmetic and logical errors endemic to LLM reasoning
- Handles constraint satisfaction problems that are intractable for chain-of-thought

**Weaknesses:**
- The **formalization step** is error-prone—the LLM may incorrectly translate the problem into formal notation
- Limited to problems that can be expressed in the symbolic backend's language
- The LLM must understand the symbolic language well enough to generate valid programs

---

### 16.5.3 Formal Verification Backends

**Method.** Use formal verification systems (Lean 4, Coq, Isabelle) to verify mathematical proofs generated by LLMs.

**Architecture:**

$$
x \xrightarrow{\text{LLM}} \text{proof sketch} \xrightarrow{\text{LLM (formalize)}} \text{Lean 4 proof} \xrightarrow{\text{Lean 4 kernel}} \begin{cases} \checkmark & \text{proof valid} \\ \times & \text{proof invalid, error at step } t \end{cases}
$$

**Key Systems:**

| System | Application | Notable Result |
|--------|-------------|----------------|
| AlphaProof (DeepMind, 2024) | IMO-level math proofs in Lean | Solved 4/6 IMO 2024 problems |
| DeepSeek-Prover | Lean 4 proof generation | State-of-the-art on miniF2F benchmark |
| LEGO-Prover | Modular proof construction | Growing library of verified lemmas |

**Feedback Loop:**

```
LLM generates proof attempt → Lean 4 checks → 
  If valid: accept
  If invalid: extract error message → LLM repairs proof → Lean 4 checks → ...
```

The error messages from the formal verifier provide precise, actionable feedback—far superior to self-critique.

$$
\text{Success rate} \approx 1 - (1 - p_{\text{single}})^{N_{\text{attempts}}}
$$

where $p_{\text{single}}$ is the probability a single proof attempt is correct. With $p_{\text{single}} = 0.05$ and $N = 100$ attempts with error-guided repair, success rates can reach 60–80% on competition-level problems.

---

### 16.5.4 Tool-Assisted Reasoning (Calculator, Interpreter)

**Principle.** Outsource computational sub-tasks where LLMs are unreliable (arithmetic, precise string operations, date calculations) to deterministic tools, while the LLM handles natural language understanding, planning, and interpretation.

**Tool Integration Pattern:**

$$
\mathbf{r} = r_1^{\text{NL}} \rightarrow \underbrace{r_2^{\text{tool call}}}_{\text{calculator(47 × 83)}} \rightarrow \underbrace{r_3^{\text{tool result}}}_{3901} \rightarrow r_4^{\text{NL}} \rightarrow \cdots
$$

**Common Tools for Reasoning:**

| Tool | Purpose | Error Rate (LLM alone) | Error Rate (with tool) |
|------|---------|----------------------|----------------------|
| Calculator | Arithmetic | 5–15% | ~0% |
| Python interpreter | Complex computation | 3–10% | <1% |
| Calendar | Date arithmetic | 15–30% | ~0% |
| Unit converter | Unit conversions | 10–20% | ~0% |
| Web search | Factual verification | Varies | Varies (depends on source) |
| Database | Structured data queries | N/A (requires tool) | Low |

**Toolformer Pattern (Schick et al., 2023).** The model learns to decide when to call a tool within the reasoning process, inserting API calls at positions where they maximize accuracy:

$$
r_t = \begin{cases}
\text{NL reasoning step} & \text{if } \text{LLM confidence} > \theta \\
\text{tool\_call(args)} \rightarrow \text{result} & \text{if } \text{LLM confidence} \leq \theta
\end{cases}
$$

**Implementation with Function Calling:**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g., '47 * 83 + 12'"}
                },
                "required": ["expression"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": problem}],
    tools=tools,
    tool_choice="auto"
)
```

---

## 16.6 Multi-Step and Multi-Hop Reasoning

### 16.6.1 Decomposition-Based Multi-Hop: Question → Sub-Questions → Answers → Synthesis

**Definition.** Multi-hop reasoning requires combining information from multiple independent sources or performing multiple sequential inference steps where each step's output feeds into the next.

**Formal Structure.** A multi-hop question $x$ decomposes into sub-questions $\{q_1, q_2, \ldots, q_m\}$ with a dependency structure encoded as a DAG:

$$
x \xrightarrow{\text{decompose}} (q_1, q_2, \ldots, q_m, \text{DAG}) \xrightarrow{\text{solve each}} (a_1, a_2, \ldots, a_m) \xrightarrow{\text{synthesize}} y
$$

The dependency DAG specifies which sub-question answers are needed to answer other sub-questions:

$$
q_i \text{ depends on } \{a_j : (j, i) \in E_{\text{DAG}}\}
$$

**Example:**

```
Question: "What is the population of the country where the Eiffel Tower is located?"

Decomposition:
  q1: "Where is the Eiffel Tower located?" → a1: "Paris, France"
  q2: "What is the population of France?" (depends on a1) → a2: "67.75 million"

Synthesis: "The population of the country where the Eiffel Tower is located 
           (France) is approximately 67.75 million."
```

**Decomposition Strategies:**

**Strategy 1: Sequential Decomposition.**

```python
def sequential_decompose(model, question: str) -> str:
    # Generate sub-questions
    decomposition = model.generate(
        f"Break this question into simpler sub-questions that can be "
        f"answered independently. List them in order of dependency.\n"
        f"Question: {question}\nSub-questions:"
    )
    
    sub_questions = parse_sub_questions(decomposition)
    answers = {}
    
    for sq in sub_questions:
        # Substitute previously answered sub-question results
        resolved_sq = substitute_answers(sq, answers)
        answer = model.generate(f"Answer concisely: {resolved_sq}")
        answers[sq.id] = answer
    
    # Final synthesis
    final = model.generate(
        f"Original question: {question}\n"
        f"Sub-answers: {answers}\n"
        f"Synthesize a final answer:"
    )
    return final
```

**Strategy 2: IRCoT (Interleaving Retrieval with CoT, Trivedi et al., 2023).**

Interleave reasoning steps with retrieval calls:

$$
r_1 \rightarrow \text{retrieve}(r_1) \rightarrow r_2 \rightarrow \text{retrieve}(r_2) \rightarrow \cdots \rightarrow y
$$

Each reasoning step identifies what information is needed next, triggers retrieval, and uses the retrieved information to generate the next reasoning step.

---

### 16.6.2 Least-to-Most Prompting

**Method (Zhou et al., 2023).** Two-stage prompting that first decomposes the problem into increasingly complex sub-problems, then solves them bottom-up from easiest to hardest.

**Stage 1: Decomposition.**

$$
x \xrightarrow{\text{LLM}} (q_1, q_2, \ldots, q_m) \quad \text{where } q_1 \text{ is easiest, } q_m = x
$$

**Stage 2: Sequential Solution.**

$$
a_1 = \text{LLM}(q_1)
$$
$$
a_2 = \text{LLM}(q_2 \mid q_1, a_1)
$$
$$
\vdots
$$
$$
a_m = \text{LLM}(q_m \mid q_1, a_1, q_2, a_2, \ldots, q_{m-1}, a_{m-1})
$$

Each sub-problem is solved with all previous sub-answers in context, building from simple to complex.

**Key Advantage.** Least-to-most excels on tasks requiring compositional generalization—solving problems more complex than any training example by decomposing them into familiar sub-problems.

**Example (SCAN Benchmark, Length Generalization):**

```
Problem: "jump around left thrice and walk opposite right twice"

Decomposition:
  q1: What does "jump around left" mean?
  q2: What does "jump around left thrice" mean? (uses a1)
  q3: What does "walk opposite right" mean?
  q4: What does "walk opposite right twice" mean? (uses a3)
  q5: What does the full command mean? (uses a2, a4)
```

**Formal Advantage.** Standard CoT with $k$-shot examples can solve problems of complexity at most $c_{\max} = \max_{j \leq k} c(x_j)$ (the maximum complexity in the exemplars). Least-to-most can solve problems of complexity:

$$
c_{\text{L2M}} = \sum_{j=1}^{m} c(q_j) \gg c_{\max}
$$

because each sub-problem has complexity $c(q_j) \leq c_{\max}$, and compositional solutions to higher-complexity problems are constructed from solved sub-problems.

---

### 16.6.3 Backward Chaining

**Method.** Start from the goal (desired conclusion) and work backward to identify what premises or sub-goals must be established to prove the goal. This is the reverse of forward chaining (which starts from known facts and derives conclusions).

**Formal Definition.** Given a goal $G$ and a knowledge base $\mathcal{KB}$:

$$
\text{BackwardChain}(G, \mathcal{KB}) = \begin{cases}
\text{True} & \text{if } G \in \mathcal{KB} \\
\text{True} & \text{if } \exists \text{ rule } (P_1 \wedge \cdots \wedge P_k \Rightarrow G) \in \mathcal{KB} \\
& \quad \text{and } \forall i.\; \text{BackwardChain}(P_i, \mathcal{KB}) = \text{True} \\
\text{False} & \text{otherwise}
\end{cases}
$$

**Implementation in LLM Agents:**

```python
def backward_chain(model, goal: str, known_facts: list[str], depth: int = 5) -> dict:
    if depth == 0:
        return {"proved": False, "reason": "Depth limit reached"}
    
    # Check if goal is directly known
    if any(is_entailed(fact, goal) for fact in known_facts):
        return {"proved": True, "reason": f"Directly known: {goal}"}
    
    # Ask LLM: what would need to be true for the goal to hold?
    sub_goals = model.generate(
        f"To prove: {goal}\n"
        f"Known facts: {known_facts}\n"
        f"What intermediate facts would need to be established? List them:"
    )
    
    sub_goal_list = parse_list(sub_goals)
    
    # Recursively prove each sub-goal
    for sg in sub_goal_list:
        result = backward_chain(model, sg, known_facts, depth - 1)
        if not result["proved"]:
            return {"proved": False, "reason": f"Could not prove sub-goal: {sg}"}
    
    return {"proved": True, "sub_goals": sub_goal_list}
```

**When to Use Backward Chaining:**

| Scenario | Forward Chaining | Backward Chaining |
|----------|-----------------|-------------------|
| Small knowledge base, specific goal | Inefficient (explores irrelevant facts) | Efficient (goal-directed) |
| Large knowledge base, broad query | Efficient | May have exponential branching |
| Verification tasks | Less natural | Natural ("what do we need to verify?") |
| Planning | Can explore aimlessly | Efficient (starts from goal state) |

---

### 16.6.4 Causal Reasoning Chains

**Definition.** Causal reasoning identifies cause-effect relationships and uses them to predict outcomes, explain observations, or plan interventions.

**Formal Framework (Pearl's Causal Hierarchy):**

| Level | Query Type | Formal | LLM Capability |
|-------|-----------|--------|-----------------|
| **Association** (Level 1) | $P(Y \mid X)$ | Observational | Strong (pretrained on correlations) |
| **Intervention** (Level 2) | $P(Y \mid \text{do}(X))$ | Interventional | Weak (confuses correlation with causation) |
| **Counterfactual** (Level 3) | $P(Y_{X=x'} \mid X=x, Y=y)$ | Counterfactual | Very weak |

**Intervention vs. Observation.** The key distinction:

$$
P(Y \mid X = x) \neq P(Y \mid \text{do}(X = x))
$$

unless there are no confounders. LLMs often fail on interventional queries because their training data consists entirely of observational correlations.

**Example Failure:**

```
Observation: "Countries that consume more chocolate win more Nobel prizes."
LLM (Level 1): "Eating more chocolate leads to more Nobel prizes." ← Confounded
Correct (Level 2): "do(chocolate consumption = high)" would not increase Nobel prizes.
                   Both are caused by wealth/education (confounder).
```

**Causal Chain Prompting.** Explicitly prompt the model to reason about causal mechanisms:

```
Identify the causal chain for this scenario:
1. What is the direct cause?
2. What mechanisms link cause to effect?
3. Are there confounding variables?
4. Would intervening on the proposed cause change the effect?
```

**Structural Causal Model Integration:**

$$
x \xrightarrow{\text{LLM}} \text{SCM}(V, U, F) \xrightarrow{\text{do-calculus}} P(Y \mid \text{do}(X))
$$

where $V$ = endogenous variables, $U$ = exogenous variables, $F$ = structural equations. The LLM translates the problem into an SCM, and formal do-calculus computes interventional queries.

---

## 16.7 Reasoning Evaluation

### 16.7.1 Reasoning Trace Quality Assessment

**Multi-Dimensional Assessment Framework:**

A reasoning trace $\mathbf{r} = (r_1, \ldots, r_n)$ is evaluated across multiple quality dimensions:

| Dimension | Definition | Measurement |
|-----------|-----------|-------------|
| **Correctness** | Each step follows logically from premises | Per-step verification (PRM or human) |
| **Completeness** | No critical steps are omitted | Coverage of necessary sub-goals |
| **Relevance** | All steps contribute to the solution | Counterfactual step deletion test |
| **Coherence** | Steps flow logically in sequence | Inter-step consistency checks |
| **Granularity** | Steps are at appropriate level of detail | Neither too coarse nor too fine |
| **Efficiency** | Minimal steps to reach the correct answer | $n$ compared to optimal trace length |

**Formal Quality Score:**

$$
\mathcal{Q}_{\text{trace}}(\mathbf{r}) = \prod_{d \in \mathcal{D}} q_d(\mathbf{r})^{w_d}
$$

where $\mathcal{D} = \{\text{correctness, completeness, relevance, coherence}\}$ and $w_d$ are importance weights summing to 1.

**Automated Assessment Methods:**

1. **Step-level PRM scoring:**

$$
q_{\text{correctness}}(\mathbf{r}) = \prod_{t=1}^{n} \text{PRM}(r_{1:t})
$$

2. **LLM-as-Judge:**

```python
def assess_trace(model, problem: str, trace: str) -> dict:
    assessment = model.generate(
        f"Evaluate this reasoning trace for the given problem.\n"
        f"Problem: {problem}\n"
        f"Trace: {trace}\n\n"
        f"Score each dimension (1-5):\n"
        f"- Correctness: Is each step logically valid?\n"
        f"- Completeness: Are all necessary steps present?\n"
        f"- Relevance: Does every step contribute?\n"
        f"- Coherence: Do steps flow logically?\n"
        f"Provide scores and justification."
    )
    return parse_assessment(assessment)
```

3. **Counterfactual Deletion Test.** For each step $r_i$, remove it and check if the answer changes:

$$
\text{Relevance}(r_i) = \mathbb{1}\left[\text{LLM}(x, r_{1:i-1}, r_{i+1:n}) \neq y\right]
$$

Steps where removal changes the answer are relevant; steps where removal does not change the answer are superfluous.

---

### 16.7.2 Faithfulness of Explanations

**The Core Question:** Does the reasoning trace $\mathbf{r}$ actually represent the model's computational process, or is it a post-hoc rationalization?

**Faithfulness Metrics:**

**1. Counterfactual Sensitivity (CS).**

$$
\text{CS}(\mathbf{r}, x) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\left[\text{LLM}(x, r_1, \ldots, \tilde{r}_i, \ldots, r_n) \neq y\right]
$$

where $\tilde{r}_i$ is a corrupted version of step $r_i$ (e.g., changing a number, negating a conclusion). High CS means the model is actually using the reasoning steps (faithful). Low CS means the model ignores the reasoning trace (unfaithful).

**2. Early Answering Test (EAT).**

$$
\text{EAT}(\mathbf{r}, x) = \min \{i : P(y \mid x, r_{1:i}) > 0.9\}
$$

If the model can produce the correct answer after very few steps, the remaining steps are likely unfaithful decoration.

**3. Biconditional Consistency.**

$$
\text{BC}(\mathbf{r}, x) = P\left(y_{\text{trace}} = y_{\text{direct}} \mid x\right)
$$

where $y_{\text{trace}}$ is the answer implied by the reasoning trace and $y_{\text{direct}}$ is the model's answer without the trace. If these frequently disagree, the model is not following its own reasoning.

**4. Atanasova et al. (2023) Framework.** Measure faithfulness through:
- **Sufficiency:** If the trace alone (without the original input) is sufficient to derive the answer
- **Comprehensiveness:** If removing the trace degrades performance

$$
\text{Sufficiency} = P(y \mid \mathbf{r}) - P(y \mid \text{random baseline})
$$

$$
\text{Comprehensiveness} = P(y \mid x, \mathbf{r}) - P(y \mid x)
$$

High sufficiency + high comprehensiveness = faithful reasoning.

---

### 16.7.3 Benchmarks: GSM8K, MATH, ARC, BBH, GPQA

**Comprehensive Benchmark Overview:**

**GSM8K (Grade School Math 8K).**
- **Domain:** Grade school math word problems
- **Size:** 8,500 problems (7,500 train, 1,000 test)
- **Difficulty:** 2–8 reasoning steps
- **Format:** Natural language question → numerical answer
- **Metric:** Exact match accuracy
- **Current SOTA:** >95% (GPT-4o, o1)
- **Significance:** Standard benchmark for basic mathematical reasoning. Approaching saturation for frontier models.

**MATH (Hendrycks et al., 2021).**
- **Domain:** Competition mathematics (AMC, AIME, Olympiad level)
- **Size:** 12,500 problems across 7 subjects (algebra, geometry, number theory, etc.)
- **Difficulty levels:** 1–5 (level 5 = Olympiad difficulty)
- **Format:** Problem → LaTeX-formatted solution
- **Metric:** Exact match of final answer (after normalization)
- **Current SOTA:** ~90% (o1, DeepSeek-R1-0528)
- **Significance:** Tests deep mathematical reasoning, multi-step proof construction, and creative problem-solving.

**ARC (AI2 Reasoning Challenge).**
- **Domain:** Grade school science (multiple choice)
- **Splits:** ARC-Easy (5,197 questions) and ARC-Challenge (2,590 questions)
- **Difficulty:** ARC-Challenge contains questions that simple retrieval/co-occurrence methods fail on
- **Format:** Multiple choice (A/B/C/D)
- **Current SOTA:** >95% (frontier models)
- **Significance:** Tests scientific commonsense reasoning and the ability to combine factual knowledge with inference.

**BBH (BIG-Bench Hard).**
- **Domain:** 23 challenging tasks from BIG-Bench where prior LMs performed below average human raters
- **Tasks include:** Boolean expressions, causal judgment, date understanding, disambiguation QA, dyck languages, formal fallacies, geometric shapes, hyperbaton, logical deduction, movie recommendation, multistep arithmetic, navigate, object counting, penguins in a table, reasoning about colored objects, ruin names, salient translation, snarks, sports understanding, temporal sequences, tracking shuffled objects, web of lies, word sorting
- **Format:** Various (classification, generation, multiple choice)
- **Current SOTA:** ~85–95% with CoT (frontier models)
- **Significance:** Diverse reasoning tasks that probe different cognitive abilities.

**GPQA (Graduate-Level Google-Proof Q&A, Rein et al., 2024).**
- **Domain:** Expert-level questions in physics, chemistry, biology
- **Size:** 448 questions (GPQA Diamond: 198 hardest)
- **Difficulty:** PhD-level; questions designed to be unanswerable by non-experts even with internet access
- **Expert accuracy:** ~65% (domain experts), ~34% (non-expert PhDs)
- **Format:** Multiple choice (4 options)
- **Current SOTA:** ~60–70% (o1, Claude 3.5 Opus)
- **Significance:** Tests genuine expert-level reasoning beyond pattern matching. Far from saturation.

**Benchmark Difficulty Spectrum:**

$$
\text{GSM8K} \ll \text{ARC-Challenge} < \text{BBH} < \text{MATH (Level 5)} < \text{GPQA Diamond}
$$

**Benchmark Selection Guide for Reasoning Evaluation:**

| If Testing... | Use Benchmark |
|---------------|---------------|
| Basic arithmetic reasoning | GSM8K |
| Multi-step mathematical reasoning | MATH |
| Scientific commonsense | ARC |
| Diverse cognitive reasoning | BBH |
| Expert-level domain reasoning | GPQA |
| Code generation + reasoning | HumanEval, MBPP, SWE-bench |
| Formal mathematical proving | miniF2F, ProofNet |

---

### 16.7.4 Process-Level vs. Outcome-Level Evaluation

**Outcome-Level Evaluation (OLE).**

Evaluates only the correctness of the final answer:

$$
\text{OLE}(\mathbf{r}, x) = \mathbb{1}[g(\mathbf{r}) = y^*]
$$

where $g(\mathbf{r})$ extracts the final answer from the reasoning trace and $y^*$ is the ground truth.

**Advantages:**
- Simple to implement (just check the answer)
- Requires only answer-level ground truth
- Correlates directly with task utility

**Limitations:**
- Cannot distinguish correct reasoning from lucky guessing
- Cannot identify *where* reasoning went wrong (no diagnostic value)
- Rewards unfaithful reasoning (right answer, wrong process)
- Fails to credit partially correct reasoning that arrives at a wrong answer due to a single computational error

**Process-Level Evaluation (PLE).**

Evaluates each reasoning step independently:

$$
\text{PLE}(\mathbf{r}, x) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{step } r_i \text{ is valid given } r_{1:i-1} \text{ and } x]
$$

**Step Validity Criteria:**

1. **Logical validity:** $r_i$ follows logically from $r_{1:i-1}$ and $x$
2. **Factual correctness:** Any factual claims in $r_i$ are true
3. **Computational correctness:** Any calculations in $r_i$ are correct
4. **Relevance:** $r_i$ contributes to solving the problem

**Implementation via Automated Step Verification:**

```python
def process_level_evaluate(model, problem: str, trace: str, ground_truth: str) -> dict:
    steps = parse_reasoning_steps(trace)
    step_scores = []
    
    for i, step in enumerate(steps):
        # Check logical validity
        validity = model.generate(
            f"Problem: {problem}\n"
            f"Previous steps: {steps[:i]}\n"
            f"Current step: {step}\n"
            f"Is this step logically valid? (yes/no/partially)\n"
            f"If no, explain the error."
        )
        
        step_scores.append({
            "step": i + 1,
            "content": step,
            "valid": parse_validity(validity),
            "explanation": validity
        })
    
    # Identify first error step
    first_error = next((s for s in step_scores if not s["valid"]), None)
    
    return {
        "outcome_correct": extract_answer(trace) == ground_truth,
        "process_score": sum(s["valid"] for s in step_scores) / len(step_scores),
        "first_error_step": first_error,
        "step_details": step_scores
    }
```

**Composite Evaluation (Best Practice).** Combine outcome and process evaluation:

$$
\text{Score}(\mathbf{r}, x) = \alpha \cdot \text{OLE}(\mathbf{r}, x) + (1 - \alpha) \cdot \text{PLE}(\mathbf{r}, x)
$$

with $\alpha \in [0.3, 0.7]$ depending on whether the application prioritizes correctness of final answer vs. quality of reasoning process.

**Error Taxonomy from Process Evaluation:**

| Error Type | Description | Frequency (Empirical) |
|-----------|-------------|----------------------|
| Arithmetic error | Wrong calculation | 15–25% of errors |
| Logical fallacy | Invalid inference step | 20–30% |
| Missing step | Critical step omitted | 10–15% |
| Hallucinated fact | Incorrect factual claim used as premise | 10–20% |
| Goal drift | Reasoning diverges from the question | 5–10% |
| Premature conclusion | Correct reasoning abandoned too early | 5–10% |
| Circular reasoning | Step $r_i$ assumes what it's trying to prove | 3–5% |

**Diagnostic Value.** Process-level evaluation enables targeted improvement:
- High arithmetic errors → integrate calculator tool
- High logical fallacy rate → improve CoT exemplars with explicit logical structure
- High hallucinated fact rate → integrate retrieval augmentation
- High goal drift → add explicit goal-checking prompts between reasoning steps

---

## Summary: Reasoning Technique Selection Guide

| Task Characteristics | Recommended Technique | Cost | Quality |
|---------------------|----------------------|------|---------|
| Simple, familiar | Direct prompting (System 1) | 1× | Baseline |
| Moderate, well-structured | Zero-shot or few-shot CoT | 1.5–3× | +15–30% |
| Complex, single-path | Few-shot CoT + self-consistency | 10–40× | +25–40% |
| Complex, multiple solution paths | Tree of Thoughts | 30–100× | +30–50% |
| Requires synthesis of partial results | Graph of Thoughts | 20–80× | +25–45% |
| Mathematical/computational | PAL / Code verification | 2–5× | +20–40% |
| Requires proof-level rigor | Formal verification (Lean) | 50–500× | Provably correct |
| Multi-hop, information integration | Least-to-Most / IRCoT | 5–15× | +20–35% |
| Iterative quality improvement | Self-Refine + external feedback | 3–10× | +10–20% |

**The Fundamental Tradeoff:**

$$
\mathcal{Q}(\text{reasoning}) = g\left(c_{\text{inference}}\right) \quad \text{with } g'(c) > 0, \; g''(c) < 0
$$

Quality is a concave, monotonically increasing function of inference-time compute. The practitioner's task is to find the operating point on this curve that optimizes quality per unit cost for their specific application constraints.