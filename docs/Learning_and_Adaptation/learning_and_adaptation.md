

# Chapter 8: Learning and Adaptation

---

## 8.1 Definition and Formal Framework

### 8.1.1 What is Learning in the Context of Agentic Systems

Learning in agentic systems denotes the systematic process by which an autonomous agent modifies its internal representations, decision policies, or external knowledge stores in response to interaction with environments, users, or its own reflective processes—such that future task performance improves under a formally specified objective function.

**Formal Definition.** An agentic system $\mathcal{A}$ is defined as a tuple:

$$
\mathcal{A} = \langle \mathcal{S}, \mathcal{A}ct, \mathcal{O}, \mathcal{T}, \pi_\theta, \mathcal{M}, \mathcal{L} \rangle
$$

where:
- $\mathcal{S}$ is the state space (environment + internal agent state)
- $\mathcal{A}ct$ is the action space (tool calls, API invocations, text generation, code execution)
- $\mathcal{O}$ is the observation space (user messages, tool outputs, environment feedback)
- $\mathcal{T}: \mathcal{S} \times \mathcal{A}ct \rightarrow \Delta(\mathcal{S})$ is the stochastic transition function
- $\pi_\theta: \mathcal{S} \rightarrow \Delta(\mathcal{A}ct)$ is the parameterized policy
- $\mathcal{M}$ is the agent's memory system (episodic, semantic, procedural)
- $\mathcal{L}$ is the learning algorithm that updates $\theta$ or $\mathcal{M}$

**Learning** is then defined as any mapping:

$$
\mathcal{L}: (\theta_t, \mathcal{M}_t, \mathcal{E}_t) \mapsto (\theta_{t+1}, \mathcal{M}_{t+1})
$$

where $\mathcal{E}_t = \{(s_i, a_i, o_i, r_i)\}_{i=1}^{t}$ is the accumulated experience trajectory up to time $t$, and the update satisfies the **performance improvement criterion**:

$$
\mathbb{E}_{\tau \sim \pi_{\theta_{t+1}}}[R(\tau)] \geq \mathbb{E}_{\tau \sim \pi_{\theta_t}}[R(\tau)]
$$

for cumulative reward $R(\tau) = \sum_{k=0}^{T} \gamma^k r_k$ with discount factor $\gamma \in [0,1]$.

**Critical Distinction from Classical ML.** Classical machine learning systems learn during a training phase and deploy a frozen model. Agentic learning is fundamentally different across several axes:

| Dimension | Classical ML | Agentic Learning |
|---|---|---|
| **Temporality** | Train → Deploy (static) | Continuous loop (dynamic) |
| **Granularity** | Batch updates over datasets | Per-interaction or per-episode |
| **Modality** | Single learning signal (loss) | Multi-signal: rewards, verbal feedback, tool outputs, self-reflection |
| **Scope** | Parameters only | Parameters + memory + prompts + tool configurations |
| **Agency** | Passive (fed data) | Active (chooses what to learn, when, and how) |

**Taxonomy of Learning Channels in Agentic Systems.** An agent can learn through multiple simultaneous channels:

1. **Parametric Learning**: Direct modification of neural network weights $\theta$ via gradient-based optimization
2. **In-Context Learning (ICL)**: Implicit Bayesian inference within the forward pass, conditioned on demonstrations in the context window
3. **Memory-Augmented Learning**: Writing to and retrieving from external memory stores $\mathcal{M}$
4. **Prompt-Level Adaptation**: Modifying the instruction template or few-shot exemplars fed to the base model
5. **Tool-Configuration Learning**: Adjusting which tools are available, their invocation patterns, and parameter defaults
6. **Policy-Structure Learning**: Modifying the decision graph, workflow topology, or planning strategy

**The Learning Objective for Agents.** Unlike single-task supervised learning with a fixed loss $\ell(f_\theta(x), y)$, agentic learning must optimize a **multi-objective, non-stationary** function:

$$
J(\theta, \mathcal{M}) = \underbrace{\mathbb{E}_{\text{task} \sim P(\text{task})}[\text{Success}(\pi_{\theta, \mathcal{M}}, \text{task})]}_{\text{task completion}} + \lambda_1 \underbrace{\mathcal{H}(\pi_\theta)}_{\text{exploration}} - \lambda_2 \underbrace{C(\pi_\theta)}_{\text{cost}} - \lambda_3 \underbrace{V(\pi_\theta)}_{\text{safety violations}}
$$

where $\mathcal{H}(\pi_\theta)$ is an entropy bonus encouraging exploration, $C(\pi_\theta)$ captures computational/monetary cost, and $V(\pi_\theta)$ penalizes safety constraint violations.

---

### 8.1.2 Adaptation as Parameter/Strategy Update

The fundamental update equation for agentic adaptation generalizes gradient descent to encompass all adaptation modalities:

$$
\theta_{t+1} = \theta_t + \eta \cdot \Delta(\text{experience}_t)
$$

This seemingly simple equation encodes profound generality. We unpack each component rigorously.

**The Parameter Vector $\theta$.** In agentic systems, $\theta$ is a **generalized state vector** that extends beyond neural network weights:

$$
\theta = [\theta_{\text{weights}}, \theta_{\text{prompt}}, \theta_{\text{memory}}, \theta_{\text{strategy}}, \theta_{\text{tools}}]
$$

- $\theta_{\text{weights}} \in \mathbb{R}^d$: Neural network parameters (billions of dimensions for LLMs)
- $\theta_{\text{prompt}} \in \mathcal{V}^*$: Prompt template (variable-length token sequence over vocabulary $\mathcal{V}$)
- $\theta_{\text{memory}} \in \mathcal{M}$: Contents of episodic/semantic memory stores
- $\theta_{\text{strategy}} \in \mathcal{G}$: Planning graph, workflow structure, or decision tree
- $\theta_{\text{tools}} \in 2^{\mathcal{T}} \times \mathcal{C}$: Active tool set and configuration parameters

**The Experience Signal $\Delta(\text{experience}_t)$.** The update direction is derived from heterogeneous experience:

$$
\Delta(\text{experience}_t) = f_{\text{aggregate}}\Big(\underbrace{\nabla_\theta \ell_t}_{\text{gradient}},\; \underbrace{r_t}_{\text{reward}},\; \underbrace{v_t}_{\text{verbal feedback}},\; \underbrace{o_t}_{\text{observation}},\; \underbrace{\rho_t}_{\text{self-reflection}}\Big)
$$

Each signal type induces a different adaptation mechanism:

**Type 1: Gradient-Based Parametric Update (Weight Modification)**

$$
\theta_{\text{weights}}^{(t+1)} = \theta_{\text{weights}}^{(t)} - \eta \nabla_{\theta} \mathcal{L}(\theta; \mathcal{D}_t)
$$

where $\mathcal{L}$ is a task-specific loss computed over a mini-batch $\mathcal{D}_t$. For agents using reinforcement learning, this becomes a policy gradient:

$$
\Delta_{\text{PG}} = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^{\pi}(s_t, a_t)\right]
$$

with advantage function $A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$.

**Type 2: Memory Update (Non-Parametric)**

$$
\mathcal{M}_{t+1} = \mathcal{M}_t \cup \{(\text{key}_t, \text{value}_t, \text{metadata}_t)\} \setminus \text{Evict}(\mathcal{M}_t)
$$

where $\text{Evict}(\cdot)$ removes stale or low-utility entries based on recency, frequency, or relevance scoring.

**Type 3: Prompt Update (Discrete Optimization)**

$$
\theta_{\text{prompt}}^{(t+1)} = \arg\max_{p \in \mathcal{P}} \sum_{(x,y) \in \mathcal{D}_{\text{val}}} \text{Score}(\text{LLM}(p, x), y)
$$

where $\mathcal{P}$ is a candidate prompt set generated via mutation, paraphrasing, or gradient-guided search.

**Type 4: Strategy Update (Structural)**

$$
\theta_{\text{strategy}}^{(t+1)} = \text{Reflect}(\theta_{\text{strategy}}^{(t)}, \text{trajectory}_t, \text{outcome}_t)
$$

This is typically implemented via the agent's own reasoning: the LLM analyzes its past trajectory and proposes structural modifications to its planning procedure.

**The Learning Rate $\eta$.** In the agentic context, $\eta$ is not a scalar but a **modality-specific adaptation rate tensor**:

$$
\eta = \text{diag}(\eta_{\text{weights}}, \eta_{\text{prompt}}, \eta_{\text{memory}}, \eta_{\text{strategy}})
$$

- $\eta_{\text{weights}}$: Typically $10^{-5}$ to $10^{-3}$ (slow, high-confidence updates)
- $\eta_{\text{prompt}}$: Discrete; one update per optimization cycle
- $\eta_{\text{memory}}$: Near-instantaneous (single experience can be stored)
- $\eta_{\text{strategy}}$: Episode-level (updated after task completion/failure)

This hierarchy reflects a **timescale separation principle**: fast adaptation mechanisms (memory, context) handle immediate novelty, while slow mechanisms (weight updates) consolidate robust long-term knowledge.

---

### 8.1.3 Online vs. Offline Learning in Agents

This distinction is fundamental to deployment architecture and determines the agent's ability to handle distributional shift.

**Offline Learning.** The agent learns from a static, pre-collected dataset $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^{N}$ without further environment interaction:

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\mathcal{L}(\theta; s, a, r, s')\right]
$$

**Key Properties:**
- No exploration risk (no unsafe actions during learning)
- Subject to **distributional shift**: the learned policy $\pi_\theta$ may encounter states absent from $\mathcal{D}$
- Requires **offline RL corrections** (e.g., Conservative Q-Learning, Decision Transformer)

**The Offline RL Problem.** Standard Q-learning applied to offline data suffers from **overestimation bias** on out-of-distribution (OOD) actions:

$$
Q_{\text{offline}}(s, a) = r + \gamma \max_{a'} Q(s', a') \quad \text{(overestimates for unseen } a' \text{)}
$$

Conservative Q-Learning (CQL) addresses this by adding a regularizer:

$$
\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q_\theta(s,a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q_\theta(s,a)]\right] + \frac{1}{2}\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - \hat{\mathcal{B}}^\pi Q_{\hat{\theta}}(s,a))^2\right]
$$

where $\hat{\pi}_\beta$ is the behavioral policy that generated $\mathcal{D}$, and $\hat{\mathcal{B}}^\pi$ is the empirical Bellman operator.

**Online Learning.** The agent interacts with the environment and updates its policy in real-time:

$$
\text{For } t = 1, 2, \ldots: \quad s_t \xrightarrow{\pi_\theta} a_t \xrightarrow{\mathcal{T}} (r_t, s_{t+1}) \xrightarrow{\mathcal{L}} \theta_{t+1}
$$

**Key Properties:**
- Handles non-stationarity naturally
- Faces the **exploration-exploitation tradeoff**
- Risk of catastrophic actions during exploration
- Requires safety constraints: $\pi_\theta \in \Pi_{\text{safe}}$

**Hybrid Paradigm: Online-Offline Learning for Agents.** Modern agentic systems employ a hybrid approach:

```
┌──────────────────────────────────────────────────┐
│            Hybrid Learning Architecture          │
├──────────────────────────────────────────────────┤
│                                                  │
│  Phase 1 (Offline): Pre-train on static corpora  │
│    θ₀ = PreTrain(D_corpus)                       │
│                                                  │
│  Phase 2 (Offline): Fine-tune on curated tasks   │
│    θ₁ = FineTune(θ₀, D_task)                     │
│                                                  │
│  Phase 3 (Offline): RLHF alignment               │
│    θ₂ = RLHF(θ₁, D_human_feedback)              │
│                                                  │
│  Phase 4 (Online): Deploy with ICL + Memory      │
│    At inference: adapt via context + retrieval    │
│                                                  │
│  Phase 5 (Online): Collect interaction data       │
│    D_new = {trajectories from deployment}         │
│                                                  │
│  Phase 6 (Offline): Periodic re-training          │
│    θ₃ = FineTune(θ₂, D_task ∪ D_new)            │
│                                                  │
│  → Repeat Phases 4–6                             │
└──────────────────────────────────────────────────┘
```

**Regret Analysis.** For online learning, we quantify performance via **regret**—the cumulative difference between optimal and actual performance:

$$
\text{Regret}(T) = \sum_{t=1}^{T} \left[V^*(s_t) - V^{\pi_{\theta_t}}(s_t)\right]
$$

where $V^*$ is the optimal value function. Sublinear regret $\text{Regret}(T) = o(T)$ implies the agent converges to optimal behavior. For UCB-style exploration in agentic settings:

$$
a_t = \arg\max_a \left[\hat{Q}(s_t, a) + c\sqrt{\frac{\ln t}{N_t(s_t, a)}}\right]
$$

where $N_t(s_t, a)$ counts how many times action $a$ has been taken in state $s_t$, yielding $\text{Regret}(T) = O(\sqrt{T \ln T})$.

---

### 8.1.4 Distinction: Weight Updates vs. In-Context Learning vs. Prompt Adaptation

These three adaptation mechanisms operate at fundamentally different levels of the computational stack, with distinct mathematical properties, timescales, and capacity bounds.

**Axis 1: Weight Updates (Parametric Learning)**

Modification of neural network parameters through backpropagation:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)
$$

*Mathematical Characterization:*
- **Capacity**: Bounded by parameter count $|\theta|$; can encode $O(|\theta|)$ bits of information (information-theoretic limit)
- **Persistence**: Permanent until overwritten by subsequent training
- **Computational Cost**: $O(|\theta| \cdot |\mathcal{B}|)$ per update (forward + backward pass)
- **Latency**: Hours to weeks for full fine-tuning; minutes for LoRA/adapter methods
- **Expressivity**: Can learn arbitrary computable functions (universal approximation)
- **Risk**: Catastrophic forgetting, mode collapse, alignment degradation

*Formal Expressivity Result.* For a transformer with $L$ layers, $H$ heads, and embedding dimension $d$, the function class representable by weight updates is:

$$
\mathcal{F}_{\text{weight}} = \{f: \mathcal{V}^* \rightarrow \Delta(\mathcal{V}) \mid f \text{ is computable by } \text{TF}(L, H, d, \theta)\}
$$

This class grows with $|\theta|$ and, for sufficient depth, can approximate any Turing-computable sequence-to-sequence function (per the transformer universality theorem).

**Axis 2: In-Context Learning (Non-Parametric, Implicit)**

The model processes demonstrations within the context window without modifying weights:

$$
\hat{y} = \text{LLM}_\theta\left([(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k), x_{\text{query}}]\right)
$$

*Mathematical Characterization:*
- **Capacity**: Bounded by context length $C$; can utilize $O(C)$ tokens of information
- **Persistence**: Ephemeral—lost when context window resets
- **Computational Cost**: $O(C^2 \cdot d)$ per inference (quadratic attention)
- **Latency**: Instantaneous (single forward pass)
- **Expressivity**: Can implement linear regression, gradient descent, and simple algorithms *implicitly within the forward pass*
- **Risk**: Recency bias, distraction by irrelevant examples, context overflow

*Theoretical Result (Akyürek et al., 2023; Von Oswald et al., 2023).* Transformers performing ICL on linear regression tasks implicitly implement gradient descent. Specifically, for input-output pairs $(x_i, y_i)$ where $y_i = w^* \cdot x_i + \epsilon_i$, the transformer's output after processing $k$ demonstrations approximates:

$$
\hat{w}_{\text{ICL}} \approx \hat{w}_{\text{GD}}^{(L)} = \left(I - (I - \eta X^\top X)^L\right)(X^\top X)^{-1} X^\top Y
$$

where $L$ is the number of transformer layers, analogous to $L$ steps of gradient descent with learning rate $\eta$.

**Axis 3: Prompt Adaptation (Structured Input Modification)**

Systematic modification of the instruction/template without weight changes:

$$
\text{prompt}^* = \arg\max_{\text{prompt} \in \mathcal{P}} \mathbb{E}_{(x,y) \sim \mathcal{D}}[\text{Metric}(\text{LLM}_\theta(\text{prompt}, x), y)]
$$

*Mathematical Characterization:*
- **Capacity**: Bounded by prompt token budget $B$; typically $B \ll C$
- **Persistence**: Persistent across interactions (stored externally as text)
- **Computational Cost**: Optimization requires $O(|\mathcal{P}| \cdot |\mathcal{D}|)$ LLM calls
- **Latency**: Minutes to hours for optimization; instantaneous for deployment
- **Expressivity**: Constrained to behaviors the frozen model can exhibit given appropriate instructions
- **Risk**: Brittleness to phrasing, limited to model's existing competencies

**Comparative Analysis Table:**

| Property | Weight Updates | In-Context Learning | Prompt Adaptation |
|---|---|---|---|
| **Update target** | $\theta \in \mathbb{R}^d$ | Activations $h \in \mathbb{R}^{C \times d}$ | Template $p \in \mathcal{V}^B$ |
| **Learning signal** | $\nabla_\theta \mathcal{L}$ | Implicit from demonstrations | Metric on validation set |
| **Timescale** | Hours–weeks | Milliseconds | Minutes–hours |
| **Persistence** | Permanent | Ephemeral | Persistent (external) |
| **Capacity** | $O(|\theta|)$ bits | $O(C \cdot \log|\mathcal{V}|)$ bits | $O(B \cdot \log|\mathcal{V}|)$ bits |
| **Forgetting risk** | Catastrophic | None (stateless) | None (external storage) |
| **Novel capabilities** | Yes | Limited to latent | No (frozen model) |
| **Cost** | GPU hours | Inference cost | LLM API calls |

**Interaction Effects.** These mechanisms are not independent—they compose multiplicatively:

$$
\text{Performance}(\theta, p, \mathcal{C}) \neq \text{Performance}(\theta, \emptyset, \emptyset) + \text{Performance}(\theta_0, p, \emptyset) + \text{Performance}(\theta_0, \emptyset, \mathcal{C})
$$

where $\mathcal{C}$ denotes in-context demonstrations and $\theta_0$ denotes the base (pre-fine-tuned) model. Well-tuned weights amplify the effectiveness of both ICL and prompt optimization, creating **super-additive** gains.

---

## 8.2 In-Context Learning (ICL)

### 8.2.1 Few-Shot Learning as Bayesian Inference

In-context learning can be formalized as **implicit Bayesian inference**, where the transformer, having been pre-trained on diverse tasks, maintains an implicit prior over tasks and performs posterior inference given demonstrations.

**Bayesian Formulation.** Let $\mathcal{T}$ denote the space of possible tasks, where each task $\tau \in \mathcal{T}$ is associated with a conditional distribution $p(y|x, \tau)$. The pre-training distribution induces a prior $p(\tau)$ over tasks. Given $k$ demonstrations $\mathcal{D}_k = \{(x_i, y_i)\}_{i=1}^{k}$, the Bayesian posterior over tasks is:

$$
p(\tau | \mathcal{D}_k) = \frac{p(\mathcal{D}_k | \tau) \cdot p(\tau)}{p(\mathcal{D}_k)} = \frac{p(\tau) \prod_{i=1}^{k} p(y_i | x_i, \tau)}{\int_{\mathcal{T}} p(\tau') \prod_{i=1}^{k} p(y_i | x_i, \tau') \, d\tau'}
$$

The **predictive distribution** for a new query $x_{k+1}$ is:

$$
p(y_{k+1} | x_{k+1}, \mathcal{D}_k) = \int_{\mathcal{T}} p(y_{k+1} | x_{k+1}, \tau) \cdot p(\tau | \mathcal{D}_k) \, d\tau = \mathbb{E}_{\tau \sim p(\tau | \mathcal{D}_k)}[p(y_{k+1} | x_{k+1}, \tau)]
$$

**Claim (Xie et al., 2022).** A transformer pre-trained on sequences generated from a mixture of Hidden Markov Models (HMMs) performs ICL that is equivalent to Bayesian prediction. Formally, if the pre-training data is generated as:

$$
\tau \sim p(\tau), \quad (x_1, y_1, x_2, y_2, \ldots) \sim p(\cdot | \tau)
$$

then the pre-trained transformer's output approximates:

$$
\text{TF}_\theta(x_{k+1} | \mathcal{D}_k) \approx p(y_{k+1} | x_{k+1}, \mathcal{D}_k) = \sum_\tau p(\tau | \mathcal{D}_k) p(y_{k+1} | x_{k+1}, \tau)
$$

**Evidence Accumulation Dynamics.** As more demonstrations are provided, the posterior concentrates around the true task $\tau^*$. The rate of concentration depends on the **KL divergence** between tasks:

$$
D_{\text{KL}}(p(\cdot | x, \tau^*) \| p(\cdot | x, \tau)) > \epsilon \quad \forall \tau \neq \tau^*
$$

implies that the posterior mass on $\tau^*$ grows exponentially:

$$
p(\tau^* | \mathcal{D}_k) \geq 1 - |\mathcal{T}| \cdot \exp(-k \cdot \epsilon)
$$

This provides a **sample complexity bound** for ICL: the number of demonstrations required for reliable task identification scales as $O(\frac{1}{\epsilon}\log|\mathcal{T}|)$.

**Practical Implications for Agent Design:**

1. **Demonstration Diversity**: Examples should be chosen to maximize the KL divergence between the target task and confounding tasks
2. **Prior Alignment**: The model's pre-training distribution should cover the task family; ICL fails when the task is too far from the pre-training support
3. **Demonstration Ordering**: Due to positional encoding effects, demonstration order matters—empirically, random ordering with the most informative examples last tends to perform best

**Limitations of the Bayesian View.** The Bayesian interpretation has known boundary conditions:

- **Capacity Ceiling**: For tasks requiring $> C$ tokens of demonstration, ICL necessarily truncates the posterior computation
- **Approximation Error**: The transformer only *approximates* Bayesian inference; the approximation quality degrades for complex task families
- **Inductive Bias Mismatch**: If the pre-training distribution poorly covers the target task family, the implicit prior $p(\tau)$ assigns negligible mass, causing ICL failure regardless of demonstration quality

---

### 8.2.2 Dynamic Example Selection

Static few-shot prompting uses a fixed set of demonstrations for all queries. **Dynamic example selection** adapts the demonstration set per query to maximize informativeness.

**Problem Formulation.** Given a demonstration pool $\mathcal{D}_{\text{pool}} = \{(x_i, y_i)\}_{i=1}^{N}$, a query $x_q$, and a budget of $k$ demonstrations, select:

$$
\mathcal{D}_k^* = \arg\max_{\mathcal{D}_k \subset \mathcal{D}_{\text{pool}}, |\mathcal{D}_k| = k} \; \text{Score}(x_q, \mathcal{D}_k)
$$

**Selection Strategies:**

**Strategy 1: Similarity-Based Retrieval.** Select demonstrations most similar to the query in embedding space:

$$
\mathcal{D}_k^* = \text{Top-}k_{(x_i, y_i) \in \mathcal{D}_{\text{pool}}} \; \text{sim}(\phi(x_q), \phi(x_i))
$$

where $\phi(\cdot)$ is an embedding function and $\text{sim}(\cdot, \cdot)$ is cosine similarity:

$$
\text{sim}(u, v) = \frac{u^\top v}{\|u\| \cdot \|v\|}
$$

**Strategy 2: Diversity-Augmented Selection.** Prevent redundancy using **Determinantal Point Processes (DPPs)**:

$$
P(\mathcal{D}_k) \propto \det(L_{\mathcal{D}_k})
$$

where $L$ is a kernel matrix with entries $L_{ij} = q_i \cdot \text{sim}(\phi(x_i), \phi(x_j)) \cdot q_j$, and $q_i = \text{sim}(\phi(x_q), \phi(x_i))$ captures relevance. The DPP naturally balances relevance (diagonal entries) with diversity (off-diagonal repulsion).

**Strategy 3: Information-Theoretic Selection.** Maximize the mutual information between demonstrations and the correct output:

$$
\mathcal{D}_k^* = \arg\max_{\mathcal{D}_k} I(Y_q; \mathcal{D}_k | x_q) = \arg\max_{\mathcal{D}_k} \left[H(Y_q | x_q) - H(Y_q | x_q, \mathcal{D}_k)\right]
$$

Since $H(Y_q | x_q)$ is constant with respect to $\mathcal{D}_k$, this reduces to minimizing the conditional entropy $H(Y_q | x_q, \mathcal{D}_k)$, which we approximate via the model's own predictive uncertainty:

$$
H(Y_q | x_q, \mathcal{D}_k) \approx -\sum_{y} \text{LLM}_\theta(y | \mathcal{D}_k, x_q) \log \text{LLM}_\theta(y | \mathcal{D}_k, x_q)
$$

**Strategy 4: Reinforcement-Learned Selection.** Train a retriever model $\pi_\phi$ to select demonstrations that maximize downstream task performance:

$$
\phi^* = \arg\max_\phi \mathbb{E}_{x_q \sim P(x)}\left[\text{Reward}\left(\text{LLM}_\theta(\pi_\phi(x_q, \mathcal{D}_{\text{pool}}), x_q)\right)\right]
$$

This can be trained with REINFORCE:

$$
\nabla_\phi J(\phi) = \mathbb{E}\left[\nabla_\phi \log \pi_\phi(\mathcal{D}_k | x_q) \cdot R(\mathcal{D}_k, x_q)\right]
$$

**Implementation Architecture for Dynamic Selection:**

```
Query x_q
    │
    ▼
┌──────────────┐
│  Embedding   │ ──→ φ(x_q) ∈ ℝ^d
│   Model      │
└──────────────┘
    │
    ▼
┌──────────────┐
│ Vector Index │ ──→ Candidate set C_m (m >> k)
│  (FAISS/     │     via approximate nearest neighbor
│   Annoy)     │
└──────────────┘
    │
    ▼
┌──────────────┐
│  Re-Ranker   │ ──→ Score each candidate on relevance,
│  + DPP       │     diversity, difficulty
│  Sampler     │
└──────────────┘
    │
    ▼
┌──────────────┐
│  Order       │ ──→ Arrange selected k examples for
│  Optimizer   │     optimal position effects
└──────────────┘
    │
    ▼
  D_k* = [(x_{σ(1)}, y_{σ(1)}), ..., (x_{σ(k)}, y_{σ(k)})]
```

---

### 8.2.3 Task Inference from Context

The agent must infer *what task* is being requested from the provided demonstrations, even when the task is never explicitly named.

**Formal Framework.** Given demonstrations $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{k}$, the agent computes a posterior over tasks:

$$
P(\text{task} | \text{demonstrations}) \propto \prod_{i=1}^{k} P(y_i | x_i, \text{task}) \cdot P(\text{task})
$$

This is the **Bayesian task identification** equation. The prior $P(\text{task})$ is implicitly encoded in the model's pre-training; the likelihood $P(y_i | x_i, \text{task})$ measures how well each task explains the observed demonstrations.

**Decomposition of Task Inference.** Task identification requires inferring multiple latent variables simultaneously:

$$
\text{task} = (\underbrace{f}_{\text{input-output mapping}}, \underbrace{\mathcal{X}}_{\text{input domain}}, \underbrace{\mathcal{Y}}_{\text{output format}}, \underbrace{c}_{\text{constraints}})
$$

The agent must infer:
1. **The function $f$**: What transformation maps inputs to outputs?
2. **The domain $\mathcal{X}$**: What type of inputs are expected?
3. **The format $\mathcal{Y}$**: What structure should outputs have?
4. **The constraints $c$**: What implicit rules govern the mapping?

**Task Ambiguity and Resolution.** When demonstrations are insufficient to uniquely identify the task, the posterior $P(\text{task} | \mathcal{D})$ has high entropy. The agent can resolve ambiguity by:

1. **Active querying**: Request additional demonstrations for discriminative inputs
2. **Posterior sampling**: Sample a task from the posterior and generate outputs accordingly
3. **Hedging**: Produce outputs that score well under multiple plausible tasks

The **information gain** from an additional demonstration $(x_{k+1}, y_{k+1})$ is:

$$
\text{IG}(x_{k+1}) = H[\text{task} | \mathcal{D}_k] - \mathbb{E}_{y_{k+1}}[H[\text{task} | \mathcal{D}_k \cup \{(x_{k+1}, y_{k+1})\}]]
$$

An agent practicing **active learning** selects:

$$
x_{k+1}^* = \arg\max_x \text{IG}(x)
$$

**Multi-Task Interference.** When the demonstration set contains examples from multiple tasks (either intentionally or due to noise), the posterior becomes multi-modal:

$$
P(\text{task} | \mathcal{D}) = \sum_{j=1}^{J} w_j \cdot \delta(\text{task} - \tau_j) \quad \text{with} \quad w_j \propto P(\tau_j) \prod_{i \in \mathcal{I}_j} P(y_i | x_i, \tau_j)
$$

where $\mathcal{I}_j$ is the subset of demonstrations consistent with task $\tau_j$. The agent must either identify the dominant task or recognize the mixture.

---

### 8.2.4 Limits of ICL: Context Length, Recency Bias

Despite its power, ICL has fundamental limitations that constrain agent capability.

**Limit 1: Context Window Capacity.** The transformer's context window $C$ (measured in tokens) imposes a hard information bottleneck:

$$
I_{\text{ICL}} \leq C \cdot \log_2 |\mathcal{V}| \quad \text{bits}
$$

For a model with $C = 128{,}000$ tokens and vocabulary $|\mathcal{V}| = 100{,}000$, this is approximately $128{,}000 \times 17 \approx 2.2 \times 10^6$ bits. While seemingly large, this is dwarfed by the information in weight space ($|\theta| \times 16$ bits for fp16 $\gg 10^{10}$ bits for a 7B model).

**Practical consequences:**
- An agent cannot learn a complex task requiring more demonstrations than fit in context
- Long-horizon tasks with extensive histories overflow the window
- Multi-step reasoning chains compound token usage

**Limit 2: Recency Bias.** Transformers exhibit a systematic bias toward information appearing later in the context. For causal (decoder-only) models, the attention pattern creates a **primacy-recency curve**:

$$
\text{Influence}(i) \propto \begin{cases} \alpha_{\text{primacy}} \cdot \exp(-\lambda_1 \cdot i) & \text{for small } i \\ \alpha_{\text{recency}} \cdot \exp(-\lambda_2 \cdot (k - i)) & \text{for } i \text{ near } k \end{cases}
$$

where position $i$ ranges from 1 (first demonstration) to $k$ (last demonstration). This creates a **U-shaped influence curve** where middle demonstrations are least influential—the "lost in the middle" phenomenon (Liu et al., 2023).

**Formal analysis.** The attention score between query position $q$ and key position $i$ in a causal transformer is:

$$
\alpha_{q,i} = \frac{\exp\left(\frac{(W_Q h_q)^\top (W_K h_i)}{\sqrt{d_k}} + b_{q-i}\right)}{\sum_{j \leq q} \exp\left(\frac{(W_Q h_q)^\top (W_K h_j)}{\sqrt{d_k}} + b_{q-j}\right)}
$$

where $b_{q-i}$ is a positional bias term (from ALiBi, RoPE, or learned positional encodings). For RoPE, this term decays with distance, creating an implicit recency preference.

**Limit 3: Sensitivity to Demonstration Format and Order.** ICL performance is surprisingly brittle to surface-level variations:

$$
\text{Var}_{\sigma \in S_k}[\text{Accuracy}(\text{LLM}_\theta(\sigma(\mathcal{D}_k), x_q))] \gg 0
$$

where $\sigma$ ranges over all $k!$ permutations of the demonstration order. Empirically, accuracy can swing by 20–30 percentage points depending on ordering.

**Limit 4: Inability to Learn Genuinely Novel Computations.** ICL can only activate capabilities latent in the pre-trained weights. If a task requires a computation not represented in the model's function class $\mathcal{F}_\theta$, no number of demonstrations will elicit correct behavior:

$$
\forall \mathcal{D}_k: \quad f^* \notin \mathcal{F}_\theta \implies P(\text{LLM}_\theta(\mathcal{D}_k, x_q) = f^*(x_q)) \approx 0
$$

**Limit 5: Quadratic Computational Scaling.** Each additional demonstration adds tokens to the context, increasing inference cost quadratically:

$$
\text{FLOPs} = O((|\text{prompt}| + k \cdot \bar{\ell}_{\text{demo}} + |x_q|)^2 \cdot d \cdot L)
$$

where $\bar{\ell}_{\text{demo}}$ is the average demonstration length.

**Mitigation Strategies Summary:**

| Limitation | Mitigation |
|---|---|
| Context overflow | Retrieval-augmented context curation |
| Recency bias | Position-aware demonstration ordering |
| Format sensitivity | Standardized, validated templates |
| Novelty ceiling | Weight updates for genuinely new capabilities |
| Quadratic cost | Sparse attention, demonstration compression |

---

## 8.3 Prompt-Level Adaptation

### 8.3.1 Automatic Prompt Optimization (APO)

Automatic Prompt Optimization treats the prompt as a learnable parameter and optimizes it against a task-specific objective using search algorithms.

**Problem Statement.** Given a frozen language model $\text{LLM}_\theta$, a validation dataset $\mathcal{D}_{\text{val}} = \{(x_i, y_i)\}_{i=1}^{n}$, and a metric function $M: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}$, find:

$$
p^* = \arg\max_{p \in \mathcal{P}} \frac{1}{n} \sum_{i=1}^{n} M\left(\text{LLM}_\theta(p \oplus x_i), y_i\right)
$$

where $\oplus$ denotes concatenation and $\mathcal{P}$ is the (combinatorially vast) space of natural language prompts.

**Challenge: Non-Differentiable, Discrete Search Space.** Unlike continuous weight optimization, prompts are discrete token sequences. The search space $\mathcal{P} = \mathcal{V}^{\leq B}$ has size $\sum_{l=1}^{B} |\mathcal{V}|^l \approx |\mathcal{V}|^B$, which is astronomically large for any reasonable prompt length $B$.

**Method 1: Gradient-Guided Discrete Search (AutoPrompt, Shin et al., 2020).**

For models with accessible embeddings, we can use gradient information to guide token selection. Define the prompt as a sequence of learnable token embeddings $e_1, \ldots, e_B$:

$$
\tilde{p} = [e_1, e_2, \ldots, e_B] \in \mathbb{R}^{B \times d}
$$

Compute the gradient of the task loss with respect to each prompt token embedding:

$$
g_j = \nabla_{e_j} \mathcal{L}(\theta; \tilde{p}, \mathcal{D}_{\text{val}})
$$

Then select replacement tokens using the **projected gradient** onto the vocabulary:

$$
e_j^{\text{new}} = \arg\min_{v \in \mathcal{V}} \; (E_v - e_j + \eta \cdot g_j)^\top (E_v - e_j + \eta \cdot g_j)
$$

where $E_v$ is the embedding of vocabulary token $v$. This is a first-order Taylor approximation of the optimal token swap.

**Method 2: LLM-Based Prompt Generation and Refinement (APE, Zhou et al., 2023).**

Use a separate LLM (the "meta-prompter") to generate and refine prompts:

```
Step 1: Generate candidates
  For i = 1..N:
    p_i = MetaLLM("Generate an instruction for the following task:
                    Input: {x_1} → Output: {y_1}
                    Input: {x_2} → Output: {y_2}
                    ...")

Step 2: Evaluate candidates
  For each p_i:
    score_i = (1/|D_val|) Σ_j M(LLM_θ(p_i ⊕ x_j), y_j)

Step 3: Refine top candidates
  For top-K prompts p_{(1)}, ..., p_{(K)}:
    p'_i = MetaLLM("Improve this instruction: {p_{(i)}}
                     It failed on: {failure_cases}
                     Suggested improvements: ...")

Step 4: Repeat Steps 2-3 until convergence
```

**Method 3: Evolutionary Optimization (EvoPrompt, Guo et al., 2024).**

Apply evolutionary algorithms to the prompt space:

$$
\text{Population}_0 = \{p_1^{(0)}, \ldots, p_M^{(0)}\} \quad \text{(random initialization)}
$$

Each generation performs:
- **Selection**: Choose parents proportional to fitness $\text{Score}(p_i)$
- **Crossover**: Combine segments of two parent prompts
- **Mutation**: Apply LLM-based paraphrasing or token substitution
- **Evaluation**: Score offspring on $\mathcal{D}_{\text{val}}$

$$
p_{\text{child}} = \text{Crossover}(p_{\text{parent}_1}, p_{\text{parent}_2}) + \text{Mutation}(\epsilon)
$$

$$
\text{Population}_{g+1} = \text{Top-}M\left(\text{Population}_g \cup \{p_{\text{child}_1}, \ldots, p_{\text{child}_K}\}\right)
$$

Convergence is typically observed within 10–50 generations.

**Method 4: Bayesian Optimization over Prompt Space.**

Model the objective function $f(p) = \mathbb{E}[\text{Score}(p)]$ with a Gaussian Process (GP) surrogate over a prompt embedding space:

$$
f(p) \sim \mathcal{GP}(m(z_p), k(z_p, z_{p'}))
$$

where $z_p = \text{Encoder}(p)$ maps prompts to a continuous representation. The acquisition function (e.g., Expected Improvement) guides the search:

$$
p_{\text{next}} = \arg\max_p \text{EI}(p) = \arg\max_p \mathbb{E}\left[\max(f(p) - f^+, 0)\right]
$$

where $f^+ = \max_{p \in \text{observed}} f(p)$.

---

### 8.3.2 DSPy-Style Prompt Compilation

DSPy (Khattab et al., 2023) introduces a **programming model** for LLM pipelines that separates program logic from prompt engineering, enabling automatic compilation of prompts.

**Core Abstraction.** A DSPy program defines:
- **Signatures**: Input-output specifications (e.g., `question -> answer`)
- **Modules**: Composable LLM operations (e.g., `ChainOfThought`, `Retrieve`, `Predict`)
- **Metrics**: Task-specific evaluation functions

The **compiler** (called a **teleprompter**) optimizes prompts to maximize the metric:

$$
\text{prompt}^* = \arg\max_{\text{prompt}} \mathbb{E}_{(x, y) \sim \mathcal{D}}[M(\text{prompt}, x, y)]
$$

**Formal Compilation Pipeline:**

**Step 1: Program Specification.**

```python
class RAGAgent(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate(context=context, question=question)
        return answer
```

**Step 2: Metric Definition.**

$$
M(p, x, y) = \mathbb{1}[\text{LLM}_\theta(p, x) = y] \quad \text{(exact match)}
$$

or more generally:

$$
M(p, x, y) = F_1(\text{LLM}_\theta(p, x), y) \quad \text{(token-level F1)}
$$

**Step 3: Teleprompter Optimization.**

The teleprompter (e.g., `BootstrapFewShot`, `MIPRO`, `COPRO`) generates optimized demonstrations and instructions:

**BootstrapFewShot:** Generates demonstrations by running the program on training examples and selecting successful traces:

$$
\mathcal{D}_{\text{demos}} = \{(x_i, \text{trace}_i, y_i) : M(\text{trace}_i, y_i) \geq \tau\}_{i=1}^{k}
$$

**MIPRO (Multi-prompt Instruction PRoposal Optimizer):**
1. Generate candidate instructions using a meta-LLM
2. Generate candidate demonstration sets via bootstrapping
3. Use Bayesian optimization to jointly search over instruction × demonstration combinations:

$$
(\text{instr}^*, \mathcal{D}_{\text{demos}}^*) = \arg\max_{(\text{instr}, \mathcal{D})} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{val}}}[M(\text{instr}, \mathcal{D}, x, y)]
$$

**Compilation Correctness Guarantee.** Under certain conditions, DSPy compilation preserves program semantics:

$$
\forall x: \quad \text{Program}(x) \text{ succeeds } \implies \text{CompiledProgram}(x) \text{ succeeds with probability } \geq 1 - \delta
$$

where $\delta$ decreases with the quality of the training set and the expressiveness of the prompt space.

**Key Mathematical Insight.** DSPy transforms the prompt optimization problem from an unstructured search over natural language to a **structured combinatorial optimization**:

$$
\text{Unstructured}: \max_{p \in \mathcal{V}^*} f(p) \quad \longrightarrow \quad \text{Structured}: \max_{(\text{instr}, \text{demos}, \text{format}) \in \mathcal{I} \times \mathcal{D}^k \times \mathcal{F}} f(\text{instr}, \text{demos}, \text{format})
$$

This decomposition dramatically reduces the effective search space by exploiting the modular structure of the program.

---

### 8.3.3 Meta-Prompting: Learning Which Prompts Work

Meta-prompting operates at a higher level of abstraction: rather than optimizing a specific prompt, it learns a **strategy for generating effective prompts** across task families.

**Formal Definition.** A meta-prompt $\mu$ is a function that maps task descriptions to optimized prompts:

$$
\mu: \mathcal{T}_{\text{desc}} \rightarrow \mathcal{P}^*
$$

such that for any new task description $t \in \mathcal{T}_{\text{desc}}$:

$$
\mu(t) \approx \arg\max_{p \in \mathcal{P}} \mathbb{E}_{(x,y) \sim \mathcal{D}_t}[M(\text{LLM}_\theta(p, x), y)]
$$

**Meta-Learning Formulation.** Using the MAML (Model-Agnostic Meta-Learning) framework adapted for prompts:

$$
\mu^* = \arg\min_\mu \sum_{\text{task } t} \mathcal{L}_t\left(\mu(t)\right)
$$

where $\mathcal{L}_t(p) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_t}[M(\text{LLM}_\theta(p, x), y)]$.

**Implementation: Prompt Strategy Library.** The meta-prompting system maintains a library of prompting strategies with associated performance metadata:

$$
\mathcal{S}_{\text{meta}} = \{(\text{strategy}_j, \text{domain}_j, \text{performance}_j)\}_{j=1}^{K}
$$

Given a new task, the system:

1. **Classifies** the task into a domain: $\hat{d} = \text{Classify}(t)$
2. **Retrieves** relevant strategies: $\mathcal{S}_{\text{relevant}} = \{s \in \mathcal{S}_{\text{meta}} : \text{sim}(d_s, \hat{d}) > \tau\}$
3. **Adapts** the best strategy: $p^* = \text{Adapt}(\text{best}(\mathcal{S}_{\text{relevant}}), t)$
4. **Evaluates** and updates the library: $\mathcal{S}_{\text{meta}} \leftarrow \mathcal{S}_{\text{meta}} \cup \{(\text{strategy}_{\text{new}}, \hat{d}, \text{score})\}$

**Categories of Meta-Prompting Strategies:**

| Strategy | Description | Best For |
|---|---|---|
| Chain-of-Thought | Step-by-step reasoning | Math, logic |
| Tree-of-Thought | Branching exploration | Planning, puzzles |
| Self-Consistency | Multiple samples + majority vote | Factual QA |
| ReAct | Interleaved reasoning + action | Tool use |
| Structured Output | JSON/XML formatting constraints | Data extraction |
| Role-Playing | Persona assignment | Creative, domain-specific |

---

### 8.3.4 Prompt Versioning and Regression Testing

As agents evolve, prompt modifications must be managed with the rigor of software engineering to prevent regressions.

**Prompt Version Control System.** Each prompt is treated as a versioned artifact:

$$
p_v = (v, \text{content}, \text{metadata}, \text{hash}, \text{performance})
$$

where:
- $v \in \mathbb{N}$: Version number
- $\text{content} \in \mathcal{V}^*$: The prompt text
- $\text{metadata}$: Author, timestamp, rationale for change
- $\text{hash} = \text{SHA256}(\text{content})$: Content hash for deduplication
- $\text{performance} \in \mathbb{R}^m$: Performance vector across $m$ benchmark dimensions

**Regression Testing Framework.** Before deploying a new prompt $p_{v+1}$, validate that it does not degrade performance on previously passing test cases:

$$
\text{Deploy}(p_{v+1}) \iff \forall b \in \mathcal{B}: \quad M_b(p_{v+1}) \geq M_b(p_v) - \epsilon_b
$$

where $\mathcal{B}$ is the benchmark suite and $\epsilon_b$ is the acceptable regression tolerance per benchmark.

**Statistical Significance Testing.** For stochastic LLM outputs, use paired hypothesis tests to determine if performance differences are real:

$$
H_0: \mu(M(p_{v+1})) = \mu(M(p_v)) \quad \text{vs.} \quad H_1: \mu(M(p_{v+1})) \neq \mu(M(p_v))
$$

Using a paired $t$-test on $n$ test examples:

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad \bar{d} = \frac{1}{n}\sum_{i=1}^n [M_i(p_{v+1}) - M_i(p_v)], \quad s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}
$$

Reject $H_0$ if $|t| > t_{\alpha/2, n-1}$. For large $n$, the bootstrap confidence interval is preferred for non-normal metric distributions.

**A/B Testing in Production.** Deploy both prompt versions simultaneously to a fraction of traffic:

$$
\text{Traffic split}: \quad P(\text{use } p_{v+1}) = \alpha, \quad P(\text{use } p_v) = 1 - \alpha
$$

Monitor key metrics (success rate, latency, user satisfaction) and use sequential hypothesis testing (e.g., mSPRT) for early stopping:

$$
\Lambda_n = \prod_{i=1}^{n} \frac{f(x_i | H_1)}{f(x_i | H_0)} \quad \text{Stop if } \Lambda_n > B \text{ or } \Lambda_n < 1/B
$$

---

## 8.4 Experience-Based Learning

### 8.4.1 Success/Failure Memory Banks

Agents accumulate experience through task execution and store structured records for future retrieval.

**Memory Bank Architecture.** A success/failure memory bank is a queryable database:

$$
\mathcal{M}_{\pm} = \{(q_i, a_i, o_i, r_i, \text{label}_i, \text{analysis}_i)\}_{i=1}^{N}
$$

where:
- $q_i$: The task/query
- $a_i$: The sequence of actions taken (trajectory)
- $o_i$: The observations received
- $r_i \in \mathbb{R}$: The reward/score received
- $\text{label}_i \in \{+, -\}$: Success or failure classification
- $\text{analysis}_i$: Post-hoc analysis of why the approach succeeded/failed

**Storage Formalization.** After each task execution, the agent performs:

$$
\text{Store}(\mathcal{M}_{\pm}, \text{experience}_t) = \begin{cases}
\mathcal{M}_{+} \leftarrow \mathcal{M}_{+} \cup \{\text{experience}_t\} & \text{if } r_t \geq \tau_{+} \\
\mathcal{M}_{-} \leftarrow \mathcal{M}_{-} \cup \{\text{experience}_t\} & \text{if } r_t < \tau_{-} \\
\text{discard} & \text{otherwise}
\end{cases}
$$

The thresholds $\tau_+$ and $\tau_-$ filter out mediocre experiences, keeping only highly informative extremes.

**Retrieval for Decision Making.** When facing a new task $q_{\text{new}}$, the agent retrieves relevant memories:

$$
\mathcal{R}_{+} = \text{Top-}k^{+}\left(\{m \in \mathcal{M}_{+} : \text{sim}(\phi(q_{\text{new}}), \phi(q_m)) > \tau_{\text{rel}}\}\right)
$$

$$
\mathcal{R}_{-} = \text{Top-}k^{-}\left(\{m \in \mathcal{M}_{-} : \text{sim}(\phi(q_{\text{new}}), \phi(q_m)) > \tau_{\text{rel}}\}\right)
$$

These are injected into the prompt as contrastive examples:

```
Previous successful approach for similar task:
[Retrieved success memory with analysis]

Previous failed approach to AVOID:
[Retrieved failure memory with root cause analysis]

Current task: [q_new]
```

**Value of Negative Examples.** Information-theoretically, failure memories can be more informative than successes. The information gain from a failure is:

$$
I_{\text{failure}} = -\log P(\text{failure} | \text{strategy}) \cdot \mathbb{1}[\text{failure generalizes to current task}]
$$

Rare failures (low $P(\text{failure})$) carry high information content, helping the agent avoid catastrophic error modes.

---

### 8.4.2 Learning from Trajectory Data

Trajectory data captures the sequential decision-making process, enabling richer learning than outcome-only feedback.

**Trajectory Definition.** A trajectory $\tau$ is an ordered sequence:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)
$$

For LLM agents, each state $s_t$ is the accumulated context (conversation history + tool outputs), each action $a_t$ is a generated text/tool call, and each reward $r_t$ may be sparse (only at episode end) or dense (at each step).

**Trajectory Quality Decomposition.** The value of a trajectory can be decomposed into per-step contributions:

$$
V(\tau) = \sum_{t=0}^{T} \gamma^t r_t = \sum_{t=0}^{T} \gamma^t \left[Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) + V^{\pi}(s_t)\right] = \sum_{t=0}^{T} \gamma^t A^{\pi}(s_t, a_t) + V^{\pi}(s_0)
$$

The **advantage** $A^{\pi}(s_t, a_t)$ isolates the contribution of each action above/below the expected baseline, enabling **credit assignment**—identifying which specific actions in a trajectory caused success or failure.

**Learning Methods from Trajectories:**

**Method 1: Behavioral Cloning (BC).** Directly imitate successful trajectories via supervised learning:

$$
\theta^* = \arg\min_\theta \sum_{\tau \in \mathcal{D}_+} \sum_{t=0}^{T_\tau} -\log \pi_\theta(a_t | s_t)
$$

This suffers from **compounding error**: small per-step deviations accumulate over long horizons, causing the agent to drift into states not covered by the training data:

$$
\text{Error}_{\text{BC}}(T) = O(\epsilon_{\text{per-step}} \cdot T^2)
$$

**Method 2: DAgger (Dataset Aggregation).** Iteratively collect expert corrections on the agent's own trajectory:

$$
\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(s, \pi^*(s)) : s \in \text{Trajectory}(\pi_{\theta_i})\}
$$

This achieves linear error scaling: $\text{Error}_{\text{DAgger}}(T) = O(\epsilon_{\text{per-step}} \cdot T)$.

**Method 3: Decision Transformer.** Cast trajectory learning as sequence modeling. A decision transformer conditions on **desired return** $\hat{R}_t$:

$$
a_t = \text{DT}_\theta(\hat{R}_t, s_t, \hat{R}_{t-1}, s_{t-1}, a_{t-1}, \ldots)
$$

The training objective is:

$$
\mathcal{L}_{\text{DT}} = \sum_{\tau \in \mathcal{D}} \sum_{t} -\log \pi_\theta(a_t | \hat{R}_t, s_t, \hat{R}_{t-1}, \ldots, s_0)
$$

At inference, setting $\hat{R}_0$ to a high target value biases the model toward high-reward trajectories.

**Method 4: Trajectory Ranking via Direct Preference Optimization (DPO).** Given pairs of trajectories $(\tau^w, \tau^l)$ where $\tau^w$ is preferred:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(\tau^w, \tau^l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(\tau^w)}{\pi_{\text{ref}}(\tau^w)} - \beta \log \frac{\pi_\theta(\tau^l)}{\pi_{\text{ref}}(\tau^l)}\right)\right]
$$

where $\sigma$ is the sigmoid function and $\beta$ is a temperature parameter controlling deviation from the reference policy $\pi_{\text{ref}}$.

---

### 8.4.3 Strategy Libraries: Reusable Plans and Procedures

Strategy libraries abstract successful trajectories into **reusable, parameterized plans** that can be instantiated for new tasks.

**Formal Definition.** A strategy library is:

$$
\mathcal{L}_{\text{strategy}} = \{(\text{name}_i, \text{preconditions}_i, \text{procedure}_i, \text{postconditions}_i, \text{stats}_i)\}_{i=1}^{K}
$$

Each strategy $\sigma_i$ is a tuple:
- $\text{name}_i$: Human-readable identifier
- $\text{preconditions}_i: \mathcal{S} \rightarrow \{0, 1\}$: Boolean function determining applicability
- $\text{procedure}_i: \mathcal{S} \rightarrow \mathcal{A}ct^*$: Parameterized action sequence
- $\text{postconditions}_i: \mathcal{S} \rightarrow \{0, 1\}$: Expected outcome verification
- $\text{stats}_i = (n_{\text{use}}, n_{\text{success}}, \bar{r}, \bar{t})$: Usage statistics

**Strategy Selection as Multi-Armed Bandit.** When multiple strategies are applicable, the agent faces an exploration-exploitation tradeoff. Using Thompson Sampling:

$$
\text{For each applicable } \sigma_i: \quad \hat{p}_i \sim \text{Beta}(n_{\text{success}_i} + 1, n_{\text{fail}_i} + 1)
$$

$$
\sigma^* = \arg\max_i \hat{p}_i
$$

This automatically balances trying well-known strategies with exploring less-tested ones.

**Strategy Composition.** Complex tasks require composing multiple strategies. A **hierarchical task network (HTN)** decomposes a high-level goal into strategy invocations:

$$
\text{Goal} \xrightarrow{\text{decompose}} [\sigma_{i_1}(\text{params}_1), \sigma_{i_2}(\text{params}_2), \ldots, \sigma_{i_m}(\text{params}_m)]
$$

with dependencies encoded as a directed acyclic graph (DAG):

$$
G = (V, E), \quad V = \{\sigma_{i_1}, \ldots, \sigma_{i_m}\}, \quad (u, v) \in E \iff \sigma_u \text{ must complete before } \sigma_v
$$

---

### 8.4.4 Skill Acquisition and Generalization

Skills are self-contained behavioral modules that can be composed, transferred, and refined over the agent's lifetime.

**Skill Formalization.** A skill $s$ is a **options framework** construct (Sutton, Precup, Singh, 1999):

$$
s = \langle \mathcal{I}_s, \pi_s, \beta_s \rangle
$$

where:
- $\mathcal{I}_s \subseteq \mathcal{S}$: The **initiation set** (states where the skill can be invoked)
- $\pi_s: \mathcal{S} \rightarrow \Delta(\mathcal{A}ct)$: The skill's internal policy
- $\beta_s: \mathcal{S} \rightarrow [0, 1]$: The **termination condition** (probability of ending the skill)

**Skill Acquisition Process:**

1. **Discovery**: Identify recurring subproblems across tasks via trajectory clustering
2. **Extraction**: Distill a skill policy from successful trajectory segments
3. **Parameterization**: Abstract the skill to handle variations
4. **Verification**: Test the skill on held-out instances
5. **Registration**: Add to the skill library with metadata

**Generalization via Skill Abstraction.** A skill generalizes when its parameters are abstracted from specific values to types:

$$
\text{Concrete}: \quad \text{SearchGoogle("machine learning papers 2024")}
$$

$$
\text{Abstract}: \quad \text{WebSearch}(\text{query}: \text{string}, \text{source}: \text{SearchEngine})
$$

The abstraction level is determined by the **generalization-specificity tradeoff**:

$$
\text{Utility}(\text{abstraction level } \alpha) = \underbrace{|\{t : s_\alpha \text{ applicable to task } t\}|}_{\text{applicability (↑ with α)}} \times \underbrace{P(\text{success} | s_\alpha)}_{\text{success rate (↓ with α)}}
$$

The optimal abstraction level $\alpha^*$ maximizes expected utility across the task distribution.

---

### 8.4.5 Voyager-Style Skill Library

Voyager (Wang et al., 2023) introduced a paradigm where an LLM agent in Minecraft autonomously discovers, codes, verifies, and stores reusable skills.

**Skill Library Definition:**

$$
\mathcal{S} = \{(s_i, \text{code}_i, \text{description}_i)\}_{i=1}^{n}
$$

where:
- $s_i$: Skill name/identifier
- $\text{code}_i$: Executable code implementing the skill
- $\text{description}_i$: Natural language description for retrieval

**Voyager Pipeline Formalization:**

```
┌────────────────────────────────────────────────────────┐
│                  Voyager Skill Lifecycle                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. CURRICULUM GENERATOR                               │
│     g_t = LLM("Given current state s_t and skills S,  │
│                suggest next exploration goal")          │
│                                                        │
│  2. CODE GENERATOR                                     │
│     code_t = LLM("Write JavaScript code to achieve     │
│                    goal g_t using available skills S    │
│                    and Minecraft API")                  │
│                                                        │
│  3. ENVIRONMENT EXECUTION                              │
│     result_t = Execute(code_t, environment)            │
│                                                        │
│  4. SELF-VERIFICATION                                  │
│     success_t = LLM("Did the code achieve g_t?        │
│                       State before: s_t                │
│                       State after: s_{t+1}             │
│                       Error log: errors_t")            │
│                                                        │
│  5. ITERATIVE REFINEMENT (if failed)                   │
│     For attempt = 1..max_retries:                      │
│       code_t = LLM("Fix the code given errors:         │
│                      {errors_t}")                      │
│       result_t = Execute(code_t, environment)          │
│       if success: break                                │
│                                                        │
│  6. SKILL STORAGE (if verified)                        │
│     S ← S ∪ {(name_t, code_t, description_t)}         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Retrieval Mechanism.** When a new task requires existing skills, the agent retrieves relevant ones via embedding similarity:

$$
\mathcal{S}_{\text{relevant}} = \text{Top-}k\left(\{s_i \in \mathcal{S} : \text{sim}(\phi(\text{task description}), \phi(\text{description}_i)) > \tau\}\right)
$$

Retrieved skills are injected into the code generation prompt, enabling **compositional skill reuse**:

```javascript
// Retrieved skill: mineBlock(bot, blockName, count)
// Retrieved skill: craftItem(bot, itemName, count)
// New task: Build a wooden pickaxe
async function buildWoodenPickaxe(bot) {
    await mineBlock(bot, "oak_log", 1);   // reused
    await craftItem(bot, "oak_planks", 4); // reused
    await craftItem(bot, "stick", 2);      // reused
    await craftItem(bot, "wooden_pickaxe", 1); // reused
}
```

**Skill Library Growth Dynamics.** The library grows approximately logarithmically with agent experience, as early skills are reused in later compositions:

$$
|\mathcal{S}(t)| \approx \alpha \cdot \log(1 + \beta \cdot t) + \gamma
$$

where $\alpha$ controls the growth rate, $\beta$ scales with task diversity, and $\gamma$ represents seed skills. This sub-linear growth reflects **skill compositionality**: new capabilities are increasingly composed from existing skills rather than created from scratch.

**Formal Compositionality Property:**

$$
P(\text{new task solvable} | |\mathcal{S}| = n) = 1 - \prod_{j=1}^{n} (1 - c_j)
$$

where $c_j$ is the probability that skill $j$ contributes to solving the task. As the library grows, the probability of having relevant building blocks approaches 1—provided the skill distribution covers the task distribution.

---

## 8.5 Reinforcement Learning for Agents

### 8.5.1 Reward Modeling for Agentic Tasks

Reward modeling is the process of constructing a reward function $R: \mathcal{S} \times \mathcal{A}ct \rightarrow \mathbb{R}$ that accurately reflects the desirability of agent behavior.

**Challenge: Reward Specification for Complex Agent Behavior.** Unlike simple RL environments (Atari, robotics), agentic tasks have multi-dimensional, often conflicting objectives:

$$
R(\tau) = \underbrace{w_1 \cdot R_{\text{task}}(\tau)}_{\text{task completion}} + \underbrace{w_2 \cdot R_{\text{quality}}(\tau)}_{\text{output quality}} + \underbrace{w_3 \cdot R_{\text{efficiency}}(\tau)}_{\text{resource usage}} + \underbrace{w_4 \cdot R_{\text{safety}}(\tau)}_{\text{safety compliance}} + \underbrace{w_5 \cdot R_{\text{user}}(\tau)}_{\text{user satisfaction}}
$$

**Learned Reward Models.** When the true reward is difficult to specify programmatically, we train a reward model $R_\psi$ from human preferences:

**Bradley-Terry Model.** Given pairs of trajectories $(\tau^A, \tau^B)$ with human preference labels:

$$
P(\tau^A \succ \tau^B) = \sigma(R_\psi(\tau^A) - R_\psi(\tau^B)) = \frac{\exp(R_\psi(\tau^A))}{\exp(R_\psi(\tau^A)) + \exp(R_\psi(\tau^B))}
$$

The reward model is trained via maximum likelihood:

$$
\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(\tau^w, \tau^l) \sim \mathcal{D}_{\text{pref}}}\left[\log \sigma(R_\psi(\tau^w) - R_\psi(\tau^l))\right]
$$

where $\tau^w$ is the preferred trajectory and $\tau^l$ is the dispreferred one.

**Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs).**

*ORM*: Provides reward only at the end of a trajectory:

$$
R_{\text{ORM}}(\tau) = R(s_T) \quad \text{(single scalar for entire trajectory)}
$$

*PRM*: Provides reward at each intermediate step:

$$
R_{\text{PRM}}(\tau) = \sum_{t=0}^{T} r_t, \quad r_t = R_\psi(s_t, a_t) \quad \text{(per-step feedback)}
$$

PRMs enable finer-grained credit assignment but require more expensive annotation. The advantage is formalized as:

$$
\text{Var}[\nabla_\theta J_{\text{PRM}}] \leq \text{Var}[\nabla_\theta J_{\text{ORM}}]
$$

Lower gradient variance leads to more stable and faster optimization.

**Reward Hacking.** A critical failure mode where the agent exploits loopholes in the reward model:

$$
\pi_{\text{hacked}} = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}[R_\psi(\tau)] \quad \text{where} \quad R_\psi(\tau) \gg R_{\text{true}}(\tau)
$$

The agent finds trajectories that score highly under $R_\psi$ but violate the *intended* reward. Mitigation strategies:

1. **KL regularization**: Penalize deviation from a reference policy:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R_\psi(\tau)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

2. **Reward model ensembles**: Use the minimum reward across multiple models:
$$
R_{\text{ensemble}}(\tau) = \min_{j \in [M]} R_{\psi_j}(\tau)
$$

3. **Constrained optimization**: Enforce hard constraints:
$$
\max_\theta \mathbb{E}[R(\tau)] \quad \text{s.t.} \quad P(\text{violation} | \pi_\theta) \leq \delta
$$

---

### 8.5.2 Policy Gradient Methods for Agent Behavior

Policy gradient methods directly optimize the policy parameters $\theta$ by estimating the gradient of the expected return.

**The Policy Gradient Theorem (Sutton et al., 2000).**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau)\right]
$$

**Derivation.** Starting from the objective:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P(\tau | \theta) R(\tau) \, d\tau
$$

Taking the gradient:

$$
\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau | \theta) R(\tau) \, d\tau = \int P(\tau | \theta) \nabla_\theta \log P(\tau | \theta) R(\tau) \, d\tau
$$

using the log-derivative trick $\nabla_\theta P(\tau | \theta) = P(\tau | \theta) \nabla_\theta \log P(\tau | \theta)$.

Since $P(\tau | \theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) P(s_{t+1} | s_t, a_t)$, and the transition probabilities are independent of $\theta$:

$$
\nabla_\theta \log P(\tau | \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

**Variance Reduction Techniques:**

**Baseline Subtraction.** Replace $R(\tau)$ with the advantage $A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^{\pi}(s_t, a_t)\right]
$$

This does not change the expected gradient but reduces variance:

$$
\text{Var}[\nabla_\theta J \text{ with advantage}] \leq \text{Var}[\nabla_\theta J \text{ with } R(\tau)]
$$

**Generalized Advantage Estimation (GAE, Schulman et al., 2016).** Smoothly interpolates between bias and variance:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error, $\gamma$ is the discount factor, and $\lambda \in [0, 1]$ controls the bias-variance tradeoff.

**Proximal Policy Optimization (PPO, Schulman et al., 2017).** The dominant algorithm for LLM-based agent training:

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio. The clipping prevents destructively large policy updates.

**Application to LLM Agents.** For an LLM agent generating a response token-by-token, the policy is:

$$
\pi_\theta(a_t | s_t) = P_\theta(\text{token}_t | \text{tokens}_{<t}, \text{context})
$$

The action space is the vocabulary $\mathcal{V}$, and the trajectory is the complete generated text. The reward can be:
- **Sparse**: A single reward at the end of generation (task success/failure)
- **Dense**: Per-token rewards from a process reward model
- **Shaped**: Intermediate rewards for reaching subgoals

---

### 8.5.3 RLHF for Agent Alignment

Reinforcement Learning from Human Feedback (RLHF) aligns agent behavior with human preferences, going beyond task completion to capture nuanced quality dimensions.

**Three-Phase RLHF Pipeline:**

**Phase 1: Supervised Fine-Tuning (SFT).**

$$
\theta_{\text{SFT}} = \arg\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{demo}}}[-\log P_\theta(y | x)]
$$

This produces a policy capable of the target behavior but not optimized for quality.

**Phase 2: Reward Model Training.**

Collect comparison data $\mathcal{D}_{\text{pref}} = \{(x_i, y_i^w, y_i^l)\}$ where $y_i^w$ is preferred over $y_i^l$:

$$
\psi^* = \arg\min_\psi \mathbb{E}_{(x, y^w, y^l) \sim \mathcal{D}_{\text{pref}}}\left[-\log \sigma\left(R_\psi(x, y^w) - R_\psi(x, y^l)\right)\right]
$$

**Phase 3: RL Optimization.**

Optimize the policy against the learned reward model with KL regularization:

$$
\theta^* = \arg\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[R_\psi(x, y) - \beta D_{\text{KL}}(\pi_\theta(\cdot | x) \| \pi_{\text{ref}}(\cdot | x))\right]
$$

The KL term prevents the policy from drifting too far from the SFT model, which would cause reward hacking.

**Closed-Form Solution (DPO, Rafailov et al., 2023).** The optimal policy under the RLHF objective has a closed form:

$$
\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{1}{\beta} R(x, y)\right)
$$

This insight leads to **Direct Preference Optimization**, which bypasses explicit reward modeling:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y^w, y^l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y^w | x)}{\pi_{\text{ref}}(y^w | x)} - \beta \log \frac{\pi_\theta(y^l | x)}{\pi_{\text{ref}}(y^l | x)}\right)\right]
$$

**Agent-Specific RLHF Considerations:**

1. **Multi-turn alignment**: Agent trajectories span multiple turns; preferences must cover entire episodes, not just single responses
2. **Tool-use alignment**: The reward model must evaluate the *appropriateness* of tool calls, not just final answers
3. **Safety alignment**: Agent actions have real-world consequences; safety constraints must be non-negotiable:

$$
\pi^* = \arg\max_\theta J(\theta) \quad \text{s.t.} \quad \forall s: \pi_\theta(a_{\text{unsafe}} | s) < \delta_{\text{safety}}
$$

4. **Compositional rewards**: Different trajectory components (reasoning, tool use, communication) may need separate reward dimensions

---

### 8.5.4 Online RL from Environment Feedback

In deployment, agents receive real-time feedback from their environment—API responses, user reactions, task completion signals—that can drive online policy improvement.

**Online RL Loop for Agents:**

$$
\text{For each interaction } t: \quad s_t \xrightarrow{\pi_\theta} a_t \xrightarrow{\text{env}} (o_t, r_t) \xrightarrow{\text{buffer}} \mathcal{B} \xrightarrow[\text{periodically}]{\text{update}} \theta_{t+1}
$$

**Experience Replay Buffer.** Online RL for agents uses a replay buffer $\mathcal{B}$ with prioritized sampling:

$$
P(\text{sample } i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}, \quad p_i = |\delta_i| + \epsilon
$$

where $\delta_i$ is the TD error and $\alpha$ controls prioritization strength. High-error experiences are replayed more frequently.

**Environment Reward Sources for Agents:**

| Source | Signal Type | Latency | Reliability |
|---|---|---|---|
| Task completion | Binary (0/1) | End of episode | High |
| API response codes | Categorical (200/400/500) | Immediate | High |
| User explicit feedback | Scalar (rating) | Variable | Medium |
| User implicit feedback | Binary (continued/abandoned) | Variable | Low |
| Automated checks | Binary (test pass/fail) | Immediate | High |
| Execution time | Continuous | Immediate | High |

**Safety-Constrained Online RL.** The agent must explore safely using **constrained MDPs**:

$$
\max_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[R(\tau)] \quad \text{s.t.} \quad \forall i: \mathbb{E}_{\pi_\theta}[C_i(\tau)] \leq d_i
$$

where $C_i(\tau)$ are cost functions measuring different safety dimensions and $d_i$ are corresponding thresholds. Lagrangian relaxation converts this to:

$$
\min_{\lambda \geq 0} \max_\theta \left[\mathbb{E}_{\pi_\theta}[R(\tau)] - \sum_i \lambda_i (\mathbb{E}_{\pi_\theta}[C_i(\tau)] - d_i)\right]
$$

---

### 8.5.5 Verbal Reinforcement Learning (Reflexion)

Reflexion (Shinn et al., 2023) introduces **verbal reinforcement learning**, where the "reward" is natural language feedback that the agent uses for self-improvement without weight updates.

**Key Insight.** Instead of scalar rewards, the agent receives (or generates) textual analysis of its performance, which is stored in memory and used to condition future behavior.

**Formal Framework.** Define a verbal RL episode:

$$
\text{For trial } k = 1, 2, \ldots:
$$

1. **Act**: Generate trajectory $\tau_k \sim \pi_\theta(\cdot | s_0, \mathcal{M}_{k-1})$ conditioned on memory
2. **Evaluate**: Obtain outcome $o_k = \text{Eval}(\tau_k)$ (pass/fail + error details)
3. **Reflect**: Generate reflection $\rho_k = \text{LLM}_\theta(\text{``Analyze failure: ''} + \tau_k + o_k)$
4. **Store**: $\mathcal{M}_k = \mathcal{M}_{k-1} \cup \{\rho_k\}$
5. **Retry**: Use updated memory in next trial

**The Reflection as Gradient Analogue.** The reflection $\rho_k$ serves as a natural language "gradient signal":

$$
\underbrace{\rho_k}_{\text{verbal gradient}} \leftrightarrow \underbrace{\nabla_\theta \mathcal{L}}_{\text{numerical gradient}}
$$

| Numerical Gradient | Verbal Reflection |
|---|---|
| Direction in parameter space | Description of what to change |
| Magnitude of the gradient | Severity of the failure |
| Per-parameter decomposition | Specific critique per step |
| Requires differentiable loss | Requires only pass/fail + observation |

**Mathematical Analogy.** Verbal RL can be viewed as performing optimization in a **semantic function space** rather than a numerical parameter space:

$$
\pi_{k+1}(\cdot | s) = \pi_\theta(\cdot | s, \text{prompt} + \mathcal{M}_k) \approx \pi_\theta(\cdot | s, \text{prompt} + \mathcal{M}_{k-1}) + \underbrace{\eta \cdot \text{Semantic}(\rho_k)}_{\text{functional update}}
$$

where $\text{Semantic}(\rho_k)$ is the behavioral change induced by adding reflection $\rho_k$ to the context.

**Convergence Behavior.** Empirically, Reflexion converges within 3–5 trials for many coding and reasoning tasks:

$$
P(\text{success at trial } k) \approx 1 - (1 - p_{\text{base}}) \cdot \prod_{j=1}^{k-1} (1 - q_j)
$$

where $p_{\text{base}}$ is the base success rate and $q_j$ is the probability of the $j$-th reflection fixing the remaining errors.

**Limitations:**
- **No weight modification**: Cannot learn truly novel capabilities beyond the model's latent capacity
- **Memory accumulation**: Reflections consume context window, eventually causing overflow
- **Reflection quality**: Depends on the model's self-diagnosis ability; can generate misleading reflections
- **Local optima**: Verbal gradients may not escape fundamental strategy failures

---

## 8.6 Continual and Lifelong Learning

### 8.6.1 Catastrophic Forgetting in Agent Context

When an agent's weights are updated to improve performance on new tasks, performance on previously mastered tasks degrades—this is **catastrophic forgetting**.

**Formal Definition.** Let $\theta_k$ be the parameters after training on tasks $\mathcal{T}_1, \ldots, \mathcal{T}_k$. Catastrophic forgetting occurs when:

$$
\mathcal{L}_{\mathcal{T}_j}(\theta_k) \gg \mathcal{L}_{\mathcal{T}_j}(\theta_j) \quad \text{for } j \ll k
$$

The performance on early task $\mathcal{T}_j$ measured after training on $k$ tasks is much worse than immediately after training on $\mathcal{T}_j$.

**Quantification: Backward Transfer.** The backward transfer metric measures forgetting:

$$
\text{BWT} = \frac{1}{K-1} \sum_{j=1}^{K-1} \left[\text{Perf}(\theta_K, \mathcal{T}_j) - \text{Perf}(\theta_j, \mathcal{T}_j)\right]
$$

$\text{BWT} < 0$ indicates forgetting; $\text{BWT} > 0$ indicates backward knowledge transfer (rare).

**Root Cause Analysis.** Forgetting occurs because gradient descent on $\mathcal{T}_k$ moves $\theta$ through parameter space without regard for the loss landscape of previous tasks. Formally, the gradient for the new task projects onto directions that increase loss on old tasks:

$$
\nabla_\theta \mathcal{L}_{\mathcal{T}_k}(\theta)^\top \nabla_\theta \mathcal{L}_{\mathcal{T}_j}(\theta) < 0 \quad \text{(conflicting gradients)}
$$

**Mitigation Strategies for Agents:**

**Strategy 1: Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017).** Add a quadratic penalty that discourages changing parameters important for previous tasks:

$$
\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\mathcal{T}_k}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{k-1,i}^*)^2
$$

where $F_i = \mathbb{E}\left[\left(\frac{\partial \log P(y|x, \theta)}{\partial \theta_i}\right)^2\right]$ is the diagonal of the Fisher Information Matrix, measuring the importance of parameter $\theta_i$ for previous tasks.

**Strategy 2: Progressive Networks.** Add new capacity for each task while freezing old parameters:

$$
\theta_k = [\underbrace{\theta_1^{\text{frozen}}}_{\text{Task 1}}, \underbrace{\theta_2^{\text{frozen}}}_{\text{Task 2}}, \ldots, \underbrace{\theta_k^{\text{trainable}}}_{\text{Task k}}, \underbrace{W_k^{\text{lateral}}}_{\text{cross-connections}}]
$$

**Strategy 3: Experience Replay.** Maintain a buffer of examples from previous tasks and interleave them during training:

$$
\mathcal{L}_{\text{replay}}(\theta) = \mathcal{L}_{\mathcal{T}_k}(\theta) + \alpha \sum_{j=1}^{k-1} \mathcal{L}_{\mathcal{T}_j}(\theta; \mathcal{B}_j)
$$

where $\mathcal{B}_j$ is a small replay buffer from task $j$.

**Strategy 4: LoRA Adapters per Task.** Use separate low-rank adapters for each task family:

$$
W_{\text{task}_k} = W_0 + B_k A_k, \quad B_k \in \mathbb{R}^{d \times r}, A_k \in \mathbb{R}^{r \times d}, \quad r \ll d
$$

At inference, the agent selects the appropriate adapter based on task classification.

---

### 8.6.2 Knowledge Accumulation Over Time

Agents must accumulate knowledge that persists and compounds across interactions, creating an ever-growing foundation for improved performance.

**Knowledge Types and Storage:**

$$
\mathcal{K} = \underbrace{\mathcal{K}_{\text{factual}}}_{\text{world knowledge}} \cup \underbrace{\mathcal{K}_{\text{procedural}}}_{\text{how-to knowledge}} \cup \underbrace{\mathcal{K}_{\text{episodic}}}_{\text{experience memories}} \cup \underbrace{\mathcal{K}_{\text{meta}}}_{\text{self-knowledge}}
$$

**Knowledge Consolidation Pipeline:**

$$
\text{Raw Experience} \xrightarrow{\text{Extract}} \text{Knowledge Candidates} \xrightarrow{\text{Validate}} \text{Verified Knowledge} \xrightarrow{\text{Integrate}} \mathcal{K} \xrightarrow{\text{Compress}} \text{Consolidated KB}
$$

1. **Extraction**: After each interaction, identify generalizable knowledge:
$$
k_{\text{new}} = \text{LLM}(\text{``What general principle can be extracted from this experience?''} + \tau_t)
$$

2. **Validation**: Check consistency with existing knowledge:
$$
\text{Valid}(k_{\text{new}}) = \begin{cases} 1 & \text{if } \neg \exists k \in \mathcal{K}: \text{Contradicts}(k_{\text{new}}, k) \\ 0.5 & \text{if conflict detected, requires resolution} \\ 0 & \text{if provably false} \end{cases}
$$

3. **Integration**: Merge with existing knowledge, resolving conflicts via **evidence weighting**:
$$
\text{Confidence}(k) = \frac{\sum_{e \in \text{Evidence}(k)} w_e \cdot \text{support}(e, k)}{|\text{Evidence}(k)|}
$$

4. **Compression**: Periodically consolidate redundant knowledge entries:
$$
\mathcal{K}_{\text{compressed}} = \arg\min_{|\mathcal{K}'| \leq B} D_{\text{KL}}(P_{\mathcal{K}} \| P_{\mathcal{K}'})
$$

---

### 8.6.3 Curriculum Learning for Agent Skills

Curriculum learning presents tasks in an order optimized for learning efficiency, starting from simple tasks and progressively increasing difficulty.

**Formal Curriculum Design.** A curriculum $\mathcal{C}$ is an ordering function over the task space:

$$
\mathcal{C}: \mathcal{T} \rightarrow \mathbb{R}^+ \quad \text{(difficulty score)}
$$

The agent trains on tasks in order of increasing difficulty:

$$
\text{Training sequence}: \mathcal{T}_{\sigma(1)}, \mathcal{T}_{\sigma(2)}, \ldots \quad \text{where } \mathcal{C}(\mathcal{T}_{\sigma(i)}) \leq \mathcal{C}(\mathcal{T}_{\sigma(i+1)})
$$

**Difficulty Metrics:**

1. **Horizon length**: Number of steps required: $\mathcal{C}_{\text{horizon}}(\mathcal{T}) = T_{\mathcal{T}}$
2. **Tool complexity**: Number and nesting depth of tool calls required
3. **Reasoning depth**: Number of logical inference steps
4. **Compositionality**: Number of sub-skills required
5. **Ambiguity**: Entropy of the optimal action distribution: $\mathcal{C}_{\text{ambiguity}}(\mathcal{T}) = H[\pi^*(\cdot | s_0)]$

**Automatic Curriculum Generation.** The agent self-generates its curriculum:

$$
\mathcal{T}_{\text{next}} = \arg\max_{\mathcal{T}} \underbrace{\text{Learning Progress}(\mathcal{T})}_{\text{maximize improvement rate}}
$$

where learning progress is the rate of performance improvement:

$$
\text{LP}(\mathcal{T}) = \frac{\partial}{\partial t} \text{Perf}(\theta_t, \mathcal{T}) \approx \text{Perf}(\theta_t, \mathcal{T}) - \text{Perf}(\theta_{t-\Delta}, \mathcal{T})
$$

Tasks where the agent is currently improving fastest are prioritized—these are tasks at the **zone of proximal development** (neither too easy nor too hard).

**Formal Result (Bengio et al., 2009).** Under mild assumptions, curriculum learning converges faster than random task ordering:

$$
T_{\text{curriculum}} \leq T_{\text{random}} \cdot \frac{H(\mathcal{C})}{H(\text{uniform})}
$$

where $H(\mathcal{C})$ is the entropy of the curriculum distribution and $H(\text{uniform})$ is the entropy of uniform task sampling. The ratio $< 1$ when the curriculum concentrates on informative tasks.

---

### 8.6.4 Transfer Learning Across Tasks and Domains

Transfer learning enables agents to leverage knowledge from one task/domain to accelerate learning in another.

**Formal Transfer Framework.** Given a source domain $\mathcal{D}_S$ with task $\mathcal{T}_S$ and a target domain $\mathcal{D}_T$ with task $\mathcal{T}_T$, transfer learning aims to improve the target learning function $f_T$ using knowledge from $\mathcal{D}_S$ and $\mathcal{T}_S$:

$$
\text{Perf}(f_T \text{ with transfer}) > \text{Perf}(f_T \text{ without transfer})
$$

**Transfer Taxonomy for Agents:**

| Transfer Type | What Transfers | Example |
|---|---|---|
| **Feature transfer** | Learned representations $\phi(\cdot)$ | Pre-trained embeddings for new domain |
| **Policy transfer** | Action distributions $\pi(\cdot|s)$ | Customer service → technical support |
| **Skill transfer** | Reusable behavioral modules | Web search skill → research agent |
| **Strategy transfer** | High-level planning structures | Debugging strategy → general troubleshooting |
| **Reward transfer** | Evaluation criteria | Quality standards across domains |

**Negative Transfer.** Transfer can *hurt* performance when source and target are sufficiently different:

$$
\text{Negative transfer} \iff D(\mathcal{D}_S, \mathcal{D}_T) > \tau_{\text{transfer}} \quad \text{where } D \text{ is a domain divergence measure}
$$

The **domain adaptation bound** (Ben-David et al., 2010) formalizes this:

$$
\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*
$$

where $\epsilon_T(h)$ and $\epsilon_S(h)$ are target and source errors, $d_{\mathcal{H}\Delta\mathcal{H}}$ is the $\mathcal{H}$-divergence between domains, and $\lambda^*$ is the optimal combined error.

---

## 8.7 Self-Improvement and Bootstrapping

### 8.7.1 Self-Play for Capability Enhancement

Self-play enables an agent to improve by competing against or collaborating with copies of itself, generating increasingly challenging training data.

**Classical Self-Play (Silver et al., 2017).** The agent plays against previous versions of itself:

$$
\pi_{k+1} = \text{Improve}(\pi_k, \text{games}(\pi_k \text{ vs } \pi_k))
$$

**Convergence to Nash Equilibrium.** In two-player zero-sum games, iterated self-play with sufficient exploration converges to the minimax optimal policy:

$$
\pi^* = \arg\max_\pi \min_{\pi'} \mathbb{E}[\text{Payoff}(\pi, \pi')]
$$

**Adaptation for LLM Agents.** Self-play extends beyond games to:

1. **Debate**: Two LLM instances argue opposing positions, improving argumentation quality
$$
\text{quality}_{k+1} = \text{Judge}(\text{Argument}(\pi_k^A, \text{position}), \text{Argument}(\pi_k^B, \neg\text{position}))
$$

2. **Red-teaming**: One instance tries to elicit failures, the other tries to be robust
$$
\pi_{\text{attacker}}^{k+1} = \arg\max_\pi P(\text{failure}(\pi_{\text{defender}}^k, \text{attack} \sim \pi))
$$
$$
\pi_{\text{defender}}^{k+1} = \arg\min_\pi P(\text{failure}(\pi, \text{attack} \sim \pi_{\text{attacker}}^{k+1}))
$$

3. **Question generation and answering**: One instance generates questions, another answers, a third evaluates
$$
\mathcal{D}_{k+1} = \{(q, a, s) : q \sim \pi_{\text{gen}}^k, a \sim \pi_{\text{ans}}^k(q), s = \text{Eval}(q, a)\}
$$

---

### 8.7.2 Synthetic Data Generation for Self-Training

The agent generates its own training data, filters for quality, and trains on the filtered set.

**Self-Training Loop (STaR, Zelikman et al., 2022):**

$$
\text{For iteration } k = 1, 2, \ldots:
$$

1. **Generate**: Sample solutions for training problems:
$$
\hat{y}_i^{(k)} \sim \pi_{\theta_k}(\cdot | x_i) \quad \text{for each } x_i \in \mathcal{D}_{\text{train}}
$$

2. **Filter**: Keep only correct solutions:
$$
\mathcal{D}_k^+ = \{(x_i, \hat{y}_i^{(k)}) : \text{Verify}(\hat{y}_i^{(k)}, y_i^*) = \text{True}\}
$$

3. **Rationalize** (STaR-specific): For problems where the model failed, provide the answer and ask for a rationalization:
$$
\hat{y}_i^{\text{rat}} \sim \pi_{\theta_k}(\cdot | x_i, \text{hint}: y_i^*) \quad \text{for } x_i \notin \mathcal{D}_k^+
$$

4. **Train**: Fine-tune on the combined dataset:
$$
\theta_{k+1} = \text{FineTune}(\theta_k, \mathcal{D}_k^+ \cup \mathcal{D}_k^{\text{rat}})
$$

**Quality Filtering Mechanisms:**

- **Execution-based verification**: Run generated code, check test cases
- **Consistency filtering**: Generate $n$ samples, keep those that appear in $\geq m$ of $n$ outputs:
$$
\text{Keep}(\hat{y}) \iff |\{j : \hat{y}^{(j)} \approx \hat{y}\}| \geq m
$$
- **Reward model filtering**: Keep samples scoring above a threshold:
$$
\mathcal{D}^+ = \{(x, \hat{y}) : R_\psi(x, \hat{y}) \geq \tau\}
$$

**Formal Analysis of Self-Training Convergence.** Self-training converges when the model's pass rate exceeds a critical threshold. Let $p_k$ be the probability that $\pi_{\theta_k}$ generates a correct solution. Then:

$$
p_{k+1} = p_k + (1 - p_k) \cdot \alpha_k
$$

where $\alpha_k$ is the fraction of previously unsolvable problems that become solvable after training on iteration $k$'s data. Convergence to $p_\infty < 1$ occurs when $\alpha_k \rightarrow 0$, i.e., when the model's ability to generate novel correct solutions from filtered self-examples plateaus.

---

### 8.7.3 Distillation of Agent Trajectories

Distill a smaller, faster student agent from the trajectories of a larger, more capable teacher agent.

**Trajectory Distillation Framework.** Given a teacher policy $\pi_T$ and a student policy $\pi_S$:

$$
\theta_S^* = \arg\min_{\theta_S} \mathbb{E}_{\tau \sim \pi_T}\left[\sum_{t} D_{\text{KL}}\left(\pi_T(\cdot | s_t) \| \pi_{\theta_S}(\cdot | s_t)\right)\right]
$$

**Distillation Variants:**

**Variant 1: Action-Level Distillation.** Match the teacher's action distribution at each state:

$$
\mathcal{L}_{\text{action}} = \sum_t \text{KL}(\pi_T(\cdot | s_t) \| \pi_S(\cdot | s_t))
$$

**Variant 2: Trajectory-Level Distillation.** Match the teacher's full trajectory distribution:

$$
\mathcal{L}_{\text{traj}} = D_{\text{KL}}(P_T(\tau) \| P_S(\tau))
$$

**Variant 3: Outcome-Conditioned Distillation.** Train the student only on successful teacher trajectories:

$$
\mathcal{L}_{\text{outcome}} = \mathbb{E}_{\tau \sim \pi_T, R(\tau) > \tau_{\text{success}}}\left[-\sum_t \log \pi_S(a_t | s_t)\right]
$$

**Variant 4: Rationale Distillation.** Distill not just actions but the teacher's reasoning process:

$$
\mathcal{L}_{\text{rationale}} = \sum_t \left[-\log P_S(\text{reasoning}_t | s_t) - \log P_S(a_t | s_t, \text{reasoning}_t)\right]
$$

This preserves the chain-of-thought structure, enabling the student to generalize beyond the specific trajectories seen.

**Compression-Performance Tradeoff.** The distillation loss introduces a bias-variance tradeoff governed by the model capacity ratio:

$$
\text{Perf}(\pi_S) \leq \text{Perf}(\pi_T) - \underbrace{\epsilon_{\text{approx}}(|\theta_S|, |\theta_T|)}_{\text{capacity gap}} - \underbrace{\epsilon_{\text{est}}(|\mathcal{D}_{\text{distill}}|)}_{\text{finite data}}
$$

where $\epsilon_{\text{approx}}$ decreases as $|\theta_S| \rightarrow |\theta_T|$ and $\epsilon_{\text{est}}$ decreases with more distillation data.

---

### 8.7.4 Limits of Self-Improvement

Self-improvement has fundamental theoretical and practical limits that prevent unbounded capability growth.

**Limit 1: Fixed-Point Convergence.** Any self-improvement process that doesn't introduce external information converges to a fixed point:

$$
\exists \theta^*: \quad \mathcal{L}(\theta^*, \mathcal{D}(\pi_{\theta^*})) = \theta^* \quad \text{(no further update occurs)}
$$

**Proof sketch.** The self-improvement mapping $T: \theta \mapsto \theta'$ where $\theta' = \text{Train}(\theta, \text{Data}(\pi_\theta))$ is a contraction mapping on a compact parameter space (under mild regularity conditions). By the Banach fixed-point theorem, repeated application converges to a unique fixed point.

**Limit 2: Verification Bottleneck.** Self-improvement requires the ability to verify whether generated outputs are correct. But verification can be as hard as generation:

$$
\text{For problems in NP} \setminus P: \quad \text{Verification is easy (polynomial)}
$$

$$
\text{For problems beyond NP}: \quad \text{Verification is as hard as generation}
$$

For tasks where the agent cannot reliably verify its outputs (e.g., open-ended creative writing, complex scientific reasoning), self-training degenerates because the filter $\mathcal{D}^+$ admits incorrect examples.

**Limit 3: Mode Collapse.** Iterated self-training narrows the output distribution:

$$
H[\pi_{\theta_{k+1}}(\cdot | x)] \leq H[\pi_{\theta_k}(\cdot | x)] \quad \text{(entropy decreases monotonically)}
$$

The agent loses diversity, converging on a narrow set of strategies and losing the ability to explore alternatives.

**Limit 4: Error Amplification.** Imperfect filtering allows some incorrect examples into the training set. Over iterations, these errors compound:

$$
\epsilon_k = \epsilon_0 + \sum_{j=1}^{k} (1 - \text{FilterAccuracy}_j) \cdot \text{NewErrors}_j
$$

If the filtering accuracy is below a critical threshold, errors accumulate and performance degrades—a phenomenon called **model collapse** (Shumailov et al., 2024).

**The Information-Theoretic Ceiling.** Self-improvement cannot create information not present in the original model and its environment:

$$
I(\theta_k; \text{ground truth}) \leq I(\theta_0; \text{ground truth}) + \sum_{j=1}^{k} I(\text{env feedback}_j; \text{ground truth})
$$

Without external information sources (new data, human feedback, environment interaction), the agent's knowledge is bounded by what was encoded during pre-training.

**Escaping Self-Improvement Limits.** The only ways to surpass these limits are:

1. **External data**: Introduce genuinely new information from the environment or humans
2. **Architectural changes**: Increase model capacity ($|\theta|$) to represent more complex functions
3. **Algorithmic improvements**: Better search, reasoning, or verification procedures
4. **Tool augmentation**: Access to tools (calculators, databases, simulators) that provide exact computation beyond the model's capacity

$$
\text{Effective capability} = \underbrace{f(\theta)}_{\text{model capacity}} + \underbrace{g(\mathcal{T}_{\text{tools}})}_{\text{tool augmentation}} + \underbrace{h(\mathcal{E}_{\text{external}})}_{\text{external information}}
$$

The practical implication is clear: **sustainable agent improvement requires a continuous flow of high-quality external signal**—whether from human feedback, environment interaction, or curated data—not merely recursive self-application.

---

**Chapter Summary.** Learning and adaptation in agentic systems operate across multiple timescales and modalities—from instantaneous in-context learning through prompt optimization to slow parametric updates. The mathematical frameworks of Bayesian inference, policy optimization, and information theory provide rigorous foundations for understanding the capabilities and limitations of each mechanism. The frontier challenge remains: designing agents that learn efficiently, accumulate knowledge without forgetting, and improve reliably without the failure modes of reward hacking, mode collapse, or self-referential error amplification.