

# Chapter 1: Routing

---

## 1.1 Definition and Formal Framework

### 1.1.1 What is Routing in Agentic Systems

Routing in agentic systems is the **computational decision process** that determines which downstream agent, model, tool, or processing pipeline should handle a given input at each step of execution. It is the control-plane mechanism that governs the flow of information and task delegation within a multi-component AI system.

In a monolithic LLM deployment, every query is served by the same model endpoint. In contrast, an **agentic architecture** comprises heterogeneous components—specialized language models, retrieval systems, code executors, vision modules, external APIs, and human-in-the-loop interfaces. Routing is the intelligence layer that mediates between an incoming request and the appropriate executor.

**Core Properties of Routing in Agentic Systems:**

| Property | Description |
|---|---|
| **Selectivity** | The router must discriminate among $n$ candidate paths based on input characteristics |
| **Context-Sensitivity** | Routing decisions depend on the full conversational context, not just the current token |
| **State-Awareness** | The system state (memory, tool availability, prior actions) conditions the routing decision |
| **Composability** | Routing decisions can be chained, nested, and composed into complex execution graphs |
| **Fallibility** | Routing is a prediction problem; misrouting must be detected and corrected |

Routing is fundamentally distinct from simple dispatching. A dispatcher forwards requests based on predetermined, static rules (e.g., round-robin). A **router** in an agentic system performs **inference**—it reasons about the nature of the input, the capabilities of available agents, and the current system constraints to produce an optimal assignment.

**Why Routing Matters:**

1. **Capability Specialization**: No single model excels at every task. A code-generation specialist outperforms a general-purpose LLM on programming tasks; a medical reasoning model outperforms both on clinical queries. Routing enables the system to leverage the right specialist.

2. **Cost Efficiency**: Large frontier models (e.g., GPT-4, Claude Opus) are expensive. Routing simple queries to smaller, cheaper models while reserving large models for complex tasks reduces cost by $10\text{-}100\times$ without meaningful quality degradation.

3. **Latency Optimization**: Different paths have different latency profiles. Routing enables the system to meet latency SLAs by selecting faster paths for time-sensitive queries.

4. **Safety and Compliance**: Certain queries require specific guardrails, content filters, or human review. Routing ensures sensitive inputs reach appropriate handling pipelines.

5. **Scalability**: Routing distributes load across heterogeneous resources, preventing bottlenecks and enabling horizontal scaling of agentic systems.

---

### 1.1.2 Routing as a Decision Function

We formalize routing as a **decision function** mapping from the input space to the action space:

$$
R: (q, C, S) \rightarrow a_i
$$

where:

- $q \in \mathcal{Q}$ is the **query** — the current user input, task description, or intermediate result requiring processing. Formally, $q$ can be a natural language string, a structured object, or a multimodal input (text + image + audio).

- $C \in \mathcal{C}$ is the **context** — the accumulated conversational history, retrieved documents, tool outputs, and any other information that conditions the interpretation of $q$. We define:

$$
C = \{c_1, c_2, \ldots, c_t\}
$$

where $c_j$ represents the $j$-th context element (prior turn, retrieved passage, tool result, etc.), and $t$ is the context horizon.

- $S \in \mathcal{S}$ is the **system state** — the current operational state of the agentic system, including:

$$
S = (\mathcal{A}_{\text{avail}}, \mathcal{M}_{\text{mem}}, \mathcal{B}_{\text{budget}}, \mathcal{L}_{\text{load}}, \mathcal{P}_{\text{policy}})
$$

  - $\mathcal{A}_{\text{avail}} \subseteq \mathcal{A}$: the set of currently available agents/models (some may be offline, rate-limited, or degraded)
  - $\mathcal{M}_{\text{mem}}$: the working memory state (scratchpad contents, variable bindings)
  - $\mathcal{B}_{\text{budget}}$: remaining compute/cost budget
  - $\mathcal{L}_{\text{load}}$: current load on each agent/model endpoint
  - $\mathcal{P}_{\text{policy}}$: routing policies, safety constraints, and compliance rules

- $a_i \in \mathcal{A} = \{a_1, a_2, \ldots, a_n\}$ is the **selected agent or path** — the chosen downstream component that will process $q$.

**Probabilistic Formulation:**

In practice, routing is rarely a hard deterministic function. We model it as a **probabilistic decision**:

$$
P(a_i \mid q, C, S) = \frac{\exp(f_\theta(q, C, S, a_i))}{\sum_{j=1}^{n} \exp(f_\theta(q, C, S, a_j))}
$$

where $f_\theta: \mathcal{Q} \times \mathcal{C} \times \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ is a **scoring function** parameterized by $\theta$ that assigns a real-valued score to each candidate agent given the input triple $(q, C, S)$.

The routing decision is then:

$$
a^* = \arg\max_{a_i \in \mathcal{A}} P(a_i \mid q, C, S)
$$

**Multi-Objective Routing:**

In realistic systems, routing must optimize multiple objectives simultaneously:

$$
a^* = \arg\max_{a_i \in \mathcal{A}} \left[ \lambda_1 \cdot \text{Quality}(a_i, q) + \lambda_2 \cdot \text{Speed}(a_i) - \lambda_3 \cdot \text{Cost}(a_i) + \lambda_4 \cdot \text{Safety}(a_i, q) \right]
$$

where $\lambda_1, \lambda_2, \lambda_3, \lambda_4 \geq 0$ are weighting coefficients reflecting system priorities, and each term is a normalized scoring function for the respective objective.

**Sequential Routing (Multi-Step):**

For multi-step agentic workflows, routing becomes a **sequential decision process**. At each step $t$:

$$
a_t^* = R(q_t, C_t, S_t)
$$

where $q_t$ is the intermediate query at step $t$, $C_t = C_{t-1} \cup \{\text{output}(a_{t-1}^*)\}$ is the updated context, and $S_t$ is the updated system state. This formulation connects routing to **Markov Decision Processes (MDPs)**, which we explore in Section 1.2.4.

---

### 1.1.3 Distinction from Traditional Load Balancing

Routing in agentic systems is fundamentally different from traditional load balancing in distributed systems. The following table delineates these distinctions:

| Dimension | Traditional Load Balancing | Agentic Routing |
|---|---|---|
| **Decision Basis** | Server health, queue depth, round-robin, hash-based | Semantic content of the query, task type, required capabilities |
| **Homogeneity Assumption** | All backend servers are functionally equivalent | Agents are heterogeneous with distinct capabilities |
| **Input Analysis** | None or minimal (IP, URL path) | Deep semantic understanding of the input |
| **Optimality Criterion** | Minimize latency, maximize throughput, ensure fairness | Maximize task-specific quality, minimize cost, satisfy constraints |
| **State Dependency** | Server-side load metrics | Full conversational context, memory, budget, policy |
| **Failure Mode** | Retry on another equivalent server | Misrouting produces incorrect/suboptimal outputs; requires semantic recovery |
| **Granularity** | Per-request | Per-step within a multi-step agentic workflow |

**Mathematical Contrast:**

Traditional load balancing selects server $s_i$ to minimize expected response time:

$$
s^* = \arg\min_{s_i \in \mathcal{S}} \mathbb{E}[\text{ResponseTime}(s_i) \mid \text{Load}(s_i)]
$$

This is **content-agnostic** — the nature of the request does not influence the selection.

Agentic routing selects agent $a_i$ to maximize expected task quality:

$$
a^* = \arg\max_{a_i \in \mathcal{A}} \mathbb{E}[\text{Quality}(a_i, q) \mid q, C, S]
$$

This is **content-dependent** — the semantic content of $q$, the full context $C$, and the system state $S$ all influence the decision.

A critical corollary: traditional load balancers are **interchangeable** — any server can handle any request. In agentic routing, agents are **non-interchangeable** — routing a medical question to a code-generation agent produces a meaningfully wrong result, not merely a slower one.

---

### 1.1.4 Routing in the Context of LLM Orchestration

LLM orchestration frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, OpenAI Assistants API) implement routing as a core primitive. In this context, routing serves as the **control flow mechanism** of the orchestration layer.

**Orchestration Architecture with Routing:**

```
                    ┌─────────────┐
                    │   User      │
                    │   Query     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Router    │ ◄── Routing Decision Function R(q, C, S)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼────┐ ┌─────▼─────┐
       │  Agent A    │ │Agent B│ │  Agent C   │
       │  (Code Gen) │ │(RAG)  │ │  (Math)    │
       └──────┬──────┘ └──┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Aggregator │
                    │  / Output   │
                    └─────────────┘
```

**Routing as Orchestration Primitive:**

In LLM orchestration, routing appears at multiple levels:

1. **Inter-Agent Routing**: Selecting which agent handles the current task from a pool of specialized agents.

2. **Intra-Agent Routing**: Within a single agent, selecting which tool, retrieval source, or sub-model to invoke.

3. **Model-Tier Routing**: Selecting which LLM (e.g., GPT-4o vs. GPT-4o-mini vs. Claude Haiku) serves a given request based on complexity.

4. **Workflow Routing**: Determining which branch of a multi-step workflow to follow based on intermediate results.

**Formal Orchestration Graph:**

An orchestration graph $G = (V, E, R)$ consists of:
- $V$: vertices representing agents, tools, and processing nodes
- $E$: directed edges representing possible transitions
- $R$: routing functions at each decision node $v \in V_{\text{decision}}$ that select the outgoing edge $e \in \text{Out}(v)$

The execution trace $\tau$ through the orchestration graph is:

$$
\tau = (v_0, e_1, v_1, e_2, v_2, \ldots, v_T)
$$

where each transition $v_{t-1} \xrightarrow{e_t} v_t$ is determined by the routing function $R_{v_{t-1}}(q_t, C_t, S_t)$.

---

## 1.2 Taxonomy of Routing Strategies

### 1.2.1 Static Routing

Static routing strategies use **predetermined, fixed rules** that do not adapt based on learned parameters or runtime feedback. They are deterministic, interpretable, and fast, but limited in their ability to handle ambiguous or novel inputs.

#### Rule-Based Routing

Rule-based routing encodes expert knowledge as a set of conditional rules:

$$
R_{\text{rule}}(q) = \begin{cases} a_1 & \text{if } \phi_1(q) = \text{True} \\ a_2 & \text{if } \phi_2(q) = \text{True} \\ \vdots \\ a_{\text{default}} & \text{otherwise} \end{cases}
$$

where each $\phi_k: \mathcal{Q} \rightarrow \{\text{True}, \text{False}\}$ is a **predicate function** evaluated on the input query.

**Rule Evaluation Order Matters:**

Rules are evaluated in priority order. If $\phi_1$ and $\phi_2$ both match, the first rule wins. This creates an implicit priority hierarchy:

$$
a^* = a_k \quad \text{where} \quad k = \min\{j : \phi_j(q) = \text{True}\}
$$

**Advantages:**
- Fully interpretable and auditable
- Zero latency overhead (no model inference required)
- Easy to debug and modify
- Deterministic behavior

**Limitations:**
- Brittle: fails on paraphrased or novel inputs
- Requires exhaustive manual enumeration of rules
- Cannot generalize beyond explicitly coded patterns
- Maintenance burden grows linearly with system complexity

**Implementation Example:**

```python
class RuleBasedRouter:
    def __init__(self):
        self.rules = [
            (lambda q: any(kw in q.lower() for kw in ["code", "python", "function", "debug"]),
             "code_agent"),
            (lambda q: any(kw in q.lower() for kw in ["search", "find", "look up", "latest"]),
             "search_agent"),
            (lambda q: any(kw in q.lower() for kw in ["calculate", "math", "solve", "equation"]),
             "math_agent"),
        ]
        self.default_agent = "general_agent"
    
    def route(self, query: str) -> str:
        for predicate, agent in self.rules:
            if predicate(query):
                return agent
        return self.default_agent
```

#### Keyword/Regex Matching

A specialized form of rule-based routing where predicates are defined via keyword sets or regular expressions:

$$
\phi_k(q) = \begin{cases} \text{True} & \text{if } \exists \, w \in \mathcal{W}_k : w \in \text{tokens}(q) \\ \text{False} & \text{otherwise} \end{cases}
$$

where $\mathcal{W}_k$ is the keyword set associated with route $k$.

**Regex-based routing** extends this with pattern matching:

$$
\phi_k(q) = \text{regex\_match}(p_k, q)
$$

where $p_k$ is a compiled regular expression pattern.

**Example Patterns:**

```python
import re

ROUTE_PATTERNS = {
    "code_agent":   re.compile(r"\b(write|generate|debug|refactor)\b.*\b(code|function|class|script)\b", re.I),
    "sql_agent":    re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|JOIN|WHERE)\b", re.I),
    "math_agent":   re.compile(r"(\d+\s*[\+\-\*\/\^]\s*\d+|solve|integral|derivative|equation)", re.I),
    "search_agent": re.compile(r"\b(search|find|look\s*up|what\s+is|who\s+is|latest)\b", re.I),
}
```

**Limitations of Keyword/Regex:**
- "Can you help me write a poem?" matches `write` → misrouted to `code_agent`
- Semantic ambiguity is unresolvable without deeper understanding
- Multilingual support requires per-language pattern sets
- Pattern maintenance becomes intractable at scale

#### Intent Classification via Fixed Taxonomy

A more structured form of static routing where a **pre-trained classifier** maps queries to a fixed set of intent categories, each mapped to a route:

$$
\text{Intent}(q) = \arg\max_{i \in \{1, \ldots, K\}} P_\phi(y = i \mid q)
$$

$$
R(q) = \text{RouteMap}[\text{Intent}(q)]
$$

where $P_\phi$ is a classifier (e.g., fine-tuned BERT, logistic regression over TF-IDF features) and $\text{RouteMap}: \{1, \ldots, K\} \rightarrow \mathcal{A}$ is a fixed mapping from intents to agents.

**Architecture:**

```
Query → Tokenizer → Encoder → Classification Head → Intent Label → Route Map → Agent
```

**Training Data Format:**

| Query | Intent | Route |
|---|---|---|
| "Write a Python function to sort a list" | `code_generation` | `code_agent` |
| "What is the capital of France?" | `factual_qa` | `search_agent` |
| "Solve $\int_0^1 x^2 dx$" | `math` | `math_agent` |

**Advantages over keyword matching:**
- Handles paraphrasing and linguistic variation
- Generalizes to unseen phrasings within trained categories
- Probabilistic output enables confidence-based fallback

**Limitations:**
- Fixed taxonomy cannot accommodate new intents without retraining
- Requires labeled training data
- Classification errors propagate to routing errors
- Cannot handle multi-intent queries (e.g., "Write code to solve this math problem")

---

### 1.2.2 Dynamic Routing

Dynamic routing strategies use **learned models or runtime computation** to make routing decisions, enabling adaptation to novel inputs and changing system conditions.

#### LLM-as-Router: Using a Language Model to Select Downstream Paths

The most flexible dynamic routing strategy: use a language model itself to analyze the input and select the appropriate route.

**Mechanism:**

A router LLM receives the query along with descriptions of available agents and produces a routing decision:

$$
a^* = \text{LLM}_{\text{router}}\left(q, \{(\text{name}_i, \text{desc}_i)\}_{i=1}^{n}\right)
$$

**Prompt Template:**

```
You are a routing agent. Given a user query, select the most appropriate 
agent to handle it.

Available agents:
1. code_agent: Handles code generation, debugging, and refactoring tasks
2. search_agent: Handles factual queries requiring web search
3. math_agent: Handles mathematical computations and proofs
4. creative_agent: Handles creative writing, brainstorming, and ideation

User query: {query}

Respond with ONLY the agent name that should handle this query.
```

**Advantages:**
- Handles semantic nuance, ambiguity, and novel inputs
- No training data required (zero-shot routing)
- Can reason about multi-intent queries
- Can provide explanations for routing decisions

**Disadvantages:**
- Adds latency (LLM inference on the critical path)
- Adds cost (every query requires a router LLM call)
- Non-deterministic (same query may route differently)
- Prompt-sensitive (routing quality depends heavily on prompt engineering)

**Latency Mitigation:**

Use a small, fast model (e.g., GPT-4o-mini, Claude Haiku, Phi-3-mini) as the router while reserving larger models as downstream agents. The router model need not be the most capable—it only needs to classify, not generate.

#### Embedding-Based Semantic Routing

Routing based on **vector similarity** between the query embedding and pre-computed route embeddings. This is detailed further in Section 1.3.

**Core Idea:**

Each route $r_i$ is represented by a set of exemplar utterances. These are embedded into a shared vector space. At inference time, the query is embedded and compared against route representations:

$$
a^* = \arg\max_{r_i \in \mathcal{R}} \text{sim}(\mathbf{e}_q, \mathbf{e}_{r_i})
$$

where $\text{sim}$ is cosine similarity and $\mathbf{e}_q, \mathbf{e}_{r_i} \in \mathbb{R}^d$ are embeddings.

#### Confidence-Threshold Routing

A routing strategy that uses the **confidence of an initial model's prediction** to decide whether to escalate to a more capable (and expensive) model:

$$
R(q) = \begin{cases} a_{\text{small}} & \text{if } \text{Conf}(a_{\text{small}}, q) \geq \tau \\ a_{\text{large}} & \text{otherwise} \end{cases}
$$

where $\text{Conf}(a_{\text{small}}, q)$ is a confidence measure of the small model's output.

**Confidence Measures:**

1. **Token-level log-probability**: Average log-probability of generated tokens:

$$
\text{Conf}(q) = \frac{1}{T}\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, q)
$$

2. **Perplexity-based**: Low perplexity → high confidence:

$$
\text{PPL}(q) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, q)\right)
$$

3. **Self-consistency**: Generate $K$ responses; high agreement → high confidence:

$$
\text{Conf}_{\text{SC}}(q) = \frac{1}{\binom{K}{2}} \sum_{i < j} \text{sim}(y^{(i)}, y^{(j)})
$$

4. **Calibrated classifier**: Train a separate model to predict whether the small model's output is correct:

$$
\text{Conf}(q) = P_\psi(\text{correct} \mid q, y_{\text{small}})
$$

**Cascading Architecture:**

```
Query → Small Model → Confidence Check → [High] → Return Small Model Output
                                        → [Low]  → Large Model → Return Large Model Output
```

This pattern reduces average cost while maintaining quality, since most queries (often 70-80%) can be handled by the small model.

---

### 1.2.3 Hierarchical Routing

Hierarchical routing decomposes the routing decision into **multiple levels**, each operating at a different granularity.

#### Coarse-to-Fine Routing Cascades

The routing decision is decomposed into a sequence of increasingly specific classifications:

$$
\text{Level 1: } d_1 = R_1(q) \in \{\text{technical}, \text{creative}, \text{analytical}\}
$$

$$
\text{Level 2: } d_2 = R_2(q, d_1) \in \begin{cases} \{\text{code}, \text{devops}, \text{data}\} & \text{if } d_1 = \text{technical} \\ \{\text{writing}, \text{art}, \text{music}\} & \text{if } d_1 = \text{creative} \\ \ldots \end{cases}
$$

$$
\text{Level 3: } a^* = R_3(q, d_1, d_2) \in \mathcal{A}_{d_1, d_2}
$$

**Advantages:**
- Reduces the decision space at each level: instead of choosing among $n$ agents directly, each level chooses among $\sqrt[L]{n}$ options for $L$ levels
- Enables reuse: Level 1 classifier can be shared across deployments
- Supports graceful degradation: if Level 2 fails, Level 1 decision still narrows the candidate set

**Formal Complexity Analysis:**

- **Flat routing**: Decision complexity $O(n)$ where $n = |\mathcal{A}|$
- **Hierarchical routing** with $L$ levels and branching factor $b$: Decision complexity $O(L \cdot b)$ where $n = b^L$, so $L = \log_b n$ and total complexity is $O(b \log_b n)$

For $n = 64$ agents with $b = 4$: flat routing compares 64 options; hierarchical routing compares $4 + 4 + 4 = 12$ options across 3 levels.

#### Multi-Level Dispatch Trees

A hierarchical routing structure implemented as a **decision tree** where each internal node is a routing function and each leaf is an agent:

```
                        ┌─────────────┐
                        │  Root Router│
                        │  (Domain)   │
                        └──────┬──────┘
                   ┌───────────┼───────────┐
              ┌────▼────┐ ┌───▼────┐ ┌────▼────┐
              │Technical│ │Creative│ │Business │
              │ Router  │ │ Router │ │ Router  │
              └────┬────┘ └───┬────┘ └────┬────┘
            ┌──────┼──────┐   │       ┌───┼───┐
         ┌──▼─┐ ┌─▼──┐ ┌─▼─┐  │       ┌──▼┐ ┌▼──┐
         │Code│ │Data│ │Dev│ ...      │Fin│ │Mkt│
         │Gen │ │Sci │ │Ops│          │   │ │   │
         └────┘ └────┘ └───┘          └───┘ └───┘
```

**Implementation:**

```python
class DispatchTree:
    def __init__(self, classifier, children: dict):
        """
        classifier: function q -> category
        children: dict mapping category -> DispatchTree or Agent
        """
        self.classifier = classifier
        self.children = children
    
    def route(self, query: str, context: dict) -> str:
        category = self.classifier(query, context)
        child = self.children.get(category, self.children.get("default"))
        if isinstance(child, DispatchTree):
            return child.route(query, context)
        return child  # leaf agent
```

---

### 1.2.4 Adaptive Routing

Adaptive routing strategies **learn and improve** routing decisions over time based on observed outcomes.

#### Reinforcement Learning-Based Route Selection

Model the routing problem as a **Markov Decision Process (MDP)**:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

where:
- $\mathcal{S}$: state space (encoded query + context + system state)
- $\mathcal{A}$: action space (set of available agents/routes)
- $P(s' \mid s, a)$: transition probability (how the system state changes after routing to agent $a$)
- $R(s, a)$: reward signal (quality of the agent's output, latency, cost)
- $\gamma \in [0, 1)$: discount factor

The optimal routing policy $\pi^*$ maximizes expected cumulative reward:

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]
$$

**Reward Design for Routing:**

$$
R(s, a) = \alpha \cdot \text{Quality}(a, q) - \beta \cdot \text{Cost}(a) - \delta \cdot \text{Latency}(a) + \epsilon \cdot \text{UserSatisfaction}
$$

where $\alpha, \beta, \delta, \epsilon$ are tunable hyperparameters.

**Challenges:**
- Reward is delayed (quality is only observed after the agent completes processing)
- State space is high-dimensional (natural language queries)
- Action space may change dynamically (agents may become unavailable)
- Exploration is costly (routing to a suboptimal agent wastes resources and degrades user experience)

#### Contextual Bandits for Routing

A simplified RL formulation where routing decisions are **single-step** (no sequential dependency between routing decisions):

$$
a^* = \arg\max_{a \in \mathcal{A}} \hat{r}(a \mid x)
$$

where $x$ is the context vector (encoded query + system state) and $\hat{r}(a \mid x)$ is the estimated reward for selecting agent $a$ given context $x$.

**Formal Framework:**

At each round $t$:
1. Observe context $x_t$ (encoded query features)
2. Select action $a_t \in \mathcal{A}$ (choose an agent)
3. Observe reward $r_t$ (quality/cost of agent's output)
4. Update the reward estimator $\hat{r}$

**Algorithms:**

1. **LinUCB (Linear Upper Confidence Bound):**

For each agent $a$, maintain parameters $\mathbf{A}_a \in \mathbb{R}^{d \times d}$, $\mathbf{b}_a \in \mathbb{R}^d$:

$$
\hat{\theta}_a = \mathbf{A}_a^{-1}\mathbf{b}_a
$$

$$
\text{UCB}_a(x_t) = \hat{\theta}_a^\top x_t + \alpha \sqrt{x_t^\top \mathbf{A}_a^{-1} x_t}
$$

$$
a_t = \arg\max_{a \in \mathcal{A}} \text{UCB}_a(x_t)
$$

The first term is the **exploitation** component (estimated reward), and the second is the **exploration** bonus (uncertainty).

2. **Thompson Sampling:**

Maintain a posterior distribution over reward parameters for each agent. At each round, sample parameters from the posterior and select the agent with the highest sampled reward:

$$
\tilde{\theta}_a \sim P(\theta_a \mid \text{history})
$$

$$
a_t = \arg\max_{a \in \mathcal{A}} \tilde{\theta}_a^\top x_t
$$

**Regret Bound:**

The expected cumulative regret of LinUCB after $T$ rounds is:

$$
\text{Regret}(T) = O\left(d\sqrt{T \log T}\right)
$$

where $d$ is the context dimension. This is **sublinear**, meaning the average per-round regret converges to zero.

#### Online Learning and Exploration-Exploitation in Route Selection

The exploration-exploitation tradeoff is central to adaptive routing:

- **Exploitation**: Route to the agent with the highest estimated quality (based on historical data)
- **Exploration**: Occasionally route to less-used agents to gather information about their current performance

**$\epsilon$-Greedy Strategy:**

$$
a_t = \begin{cases} \arg\max_{a \in \mathcal{A}} \hat{r}(a \mid x_t) & \text{with probability } 1 - \epsilon \\ \text{Uniform}(\mathcal{A}) & \text{with probability } \epsilon \end{cases}
$$

With $\epsilon$ decaying over time: $\epsilon_t = \min\left(1, \frac{c \cdot |\mathcal{A}|}{d^2 \cdot t}\right)$ where $c$ and $d$ are constants.

**Non-Stationary Environments:**

Agent capabilities may change over time (model updates, infrastructure changes, concept drift). Adaptive routing must handle **non-stationarity** by:

1. **Sliding window**: Only use recent $W$ observations to estimate rewards
2. **Exponential discounting**: Weight recent observations more heavily:

$$
\hat{r}_t(a \mid x) = \frac{\sum_{s=1}^{t} \gamma^{t-s} r_s \cdot \mathbb{1}[a_s = a]}{\sum_{s=1}^{t} \gamma^{t-s} \cdot \mathbb{1}[a_s = a]}
$$

3. **Change detection**: Use statistical tests (e.g., CUSUM, Page-Hinkley) to detect performance shifts and reset estimators

---

## 1.3 Semantic Routing

Semantic routing uses **dense vector representations** of queries and routes to make routing decisions based on **meaning** rather than surface-level text patterns.

### 1.3.1 Embedding Space Construction for Route Matching

**Core Principle:**

Each route is defined by a set of **exemplar utterances** — representative queries that should be directed to that route. These exemplars are embedded into a dense vector space using a pre-trained embedding model.

**Route Definition:**

A route $r_i$ is defined by:

$$
r_i = (\text{name}_i, \text{desc}_i, \mathcal{U}_i)
$$

where $\mathcal{U}_i = \{u_1^{(i)}, u_2^{(i)}, \ldots, u_{m_i}^{(i)}\}$ is the set of exemplar utterances for route $i$.

**Embedding Computation:**

For each exemplar utterance $u_j^{(i)}$, compute its embedding:

$$
\mathbf{e}_j^{(i)} = \text{Embed}(u_j^{(i)}) \in \mathbb{R}^d
$$

where $\text{Embed}(\cdot)$ is a pre-trained sentence embedding model (e.g., OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2`, Cohere `embed-v3`).

**Route Centroid:**

The route is represented by the centroid of its exemplar embeddings:

$$
\mathbf{e}_{r_i} = \frac{1}{m_i}\sum_{j=1}^{m_i} \mathbf{e}_j^{(i)}
$$

**Alternative Route Representations:**

1. **Centroid** (as above): simple, fast, but loses distributional information
2. **All exemplars**: compare query against every exemplar; use max or top-$k$ average:

$$
\text{score}(q, r_i) = \max_{j \in \{1, \ldots, m_i\}} \text{sim}(\mathbf{e}_q, \mathbf{e}_j^{(i)})
$$

3. **Cluster-based**: Cluster exemplars within each route; represent route by cluster centroids to capture multi-modal intent distributions

**Embedding Model Selection Criteria:**

| Criterion | Consideration |
|---|---|
| **Semantic quality** | Must capture meaning, not just lexical overlap |
| **Dimensionality** | Higher $d$ → better discrimination but slower computation |
| **Domain alignment** | Models trained on domain-specific data may outperform general-purpose models |
| **Inference speed** | Router is on the critical path; embedding must be fast |
| **Multilingual support** | Required if queries arrive in multiple languages |

**Index Structure:**

For large numbers of routes and exemplars, a vector index (FAISS, Annoy, ScaNN, HNSW) enables sublinear-time nearest-neighbor lookup:

$$
\text{ANN}(\mathbf{e}_q, \mathcal{I}) \rightarrow \{(r_i, \text{score}_i)\}_{i=1}^{k}
$$

where $\mathcal{I}$ is the pre-built index over route embeddings.

---

### 1.3.2 Cosine Similarity Scoring

The standard similarity metric for semantic routing is **cosine similarity**:

$$
\text{sim}(q, r_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{r_i}}{\|\mathbf{e}_q\| \|\mathbf{e}_{r_i}\|}
$$

**Properties:**

- Range: $[-1, 1]$ (for normalized embeddings, effectively $[0, 1]$ for most modern embedding models)
- Scale-invariant: independent of vector magnitude
- Equivalent to the cosine of the angle $\theta$ between vectors: $\text{sim} = \cos(\theta)$

**Routing Decision:**

$$
a^* = \arg\max_{r_i \in \mathcal{R}} \text{sim}(q, r_i)
$$

**Why Cosine Similarity?**

Modern embedding models produce vectors on or near the unit hypersphere. For unit-normalized vectors ($\|\mathbf{e}\| = 1$):

$$
\text{sim}(q, r_i) = \mathbf{e}_q \cdot \mathbf{e}_{r_i} = \mathbf{e}_q^\top \mathbf{e}_{r_i}
$$

This is equivalent to the dot product, and:

$$
\|\mathbf{e}_q - \mathbf{e}_{r_i}\|^2 = 2(1 - \text{sim}(q, r_i))
$$

So maximizing cosine similarity is equivalent to minimizing Euclidean distance on the unit hypersphere.

**Alternative Metrics:**

1. **Dot Product**: $\text{sim}(q, r_i) = \mathbf{e}_q^\top \mathbf{e}_{r_i}$ — does not normalize, biased toward longer vectors

2. **Euclidean Distance**: $\text{dist}(q, r_i) = \|\mathbf{e}_q - \mathbf{e}_{r_i}\|_2$ — used with L2-normalized embeddings

3. **Mahalanobis Distance**: $\text{dist}(q, r_i) = \sqrt{(\mathbf{e}_q - \mathbf{e}_{r_i})^\top \Sigma^{-1} (\mathbf{e}_q - \mathbf{e}_{r_i})}$ — accounts for covariance structure of the embedding space

**Batch Scoring Efficiency:**

For $n$ routes, compute all scores simultaneously via matrix multiplication:

$$
\mathbf{s} = \mathbf{E}_{\mathcal{R}} \cdot \mathbf{e}_q
$$

where $\mathbf{E}_{\mathcal{R}} \in \mathbb{R}^{n \times d}$ is the matrix of route embeddings and $\mathbf{s} \in \mathbb{R}^n$ is the score vector. This is $O(n \cdot d)$ and highly parallelizable on GPU.

---

### 1.3.3 Thresholding and Fallback Mechanisms

Not every query matches an available route. A critical component of semantic routing is **knowing when no route is appropriate** and triggering a fallback.

**Threshold-Based Routing:**

$$
a^* = \begin{cases} \arg\max_{r_i} \text{sim}(q, r_i) & \text{if } \max_{r_i} \text{sim}(q, r_i) \geq \tau \\ a_{\text{fallback}} & \text{otherwise} \end{cases}
$$

where $\tau \in [0, 1]$ is the **routing confidence threshold**.

**Threshold Selection:**

The threshold $\tau$ controls the precision-recall tradeoff:

- **High $\tau$**: High precision (few misroutes) but low recall (many queries fallback to the default agent)
- **Low $\tau$**: High recall (most queries are routed) but low precision (more misroutes)

**Optimal Threshold Selection via Validation:**

Given a labeled validation set $\{(q_j, r_j^*)\}_{j=1}^{N}$:

$$
\tau^* = \arg\max_\tau \left[ \text{Precision}(\tau) \cdot \text{Recall}(\tau) \right]
$$

or equivalently, maximize the $F_1$ score as a function of $\tau$.

**Margin-Based Thresholding:**

Instead of thresholding the absolute score, threshold the **margin** between the best and second-best route:

$$
\text{margin}(q) = \text{sim}(q, r_{(1)}) - \text{sim}(q, r_{(2)})
$$

where $r_{(1)}, r_{(2)}$ are the top-2 routes by similarity.

$$
a^* = \begin{cases} r_{(1)} & \text{if } \text{margin}(q) \geq \tau_{\text{margin}} \\ a_{\text{fallback}} & \text{otherwise} \end{cases}
$$

A small margin indicates ambiguity — the query is equidistant between two routes and may require human review or a more sophisticated routing strategy.

**Fallback Strategies:**

| Strategy | Description |
|---|---|
| **Default Agent** | Route to a general-purpose agent capable of handling any query |
| **Human Escalation** | Forward to a human operator for manual routing |
| **LLM Re-routing** | Use an LLM-based router (Section 1.4) as a fallback for ambiguous queries |
| **Clarification** | Ask the user to clarify their intent before routing |
| **Multi-Agent Fan-out** | Send to multiple candidate agents and select the best response |

---

### 1.3.4 Few-Shot Route Calibration

Semantic routing requires **calibration** — selecting representative exemplar utterances for each route and tuning similarity thresholds.

**Exemplar Selection Strategy:**

For each route $r_i$, the exemplar set $\mathcal{U}_i$ should:

1. **Cover the semantic space** of queries that should be routed to $r_i$
2. **Include boundary cases** — queries that are near the decision boundary with adjacent routes
3. **Be diverse** — avoid redundant exemplars that cluster in a small region of embedding space
4. **Be concise** — enough exemplars for good coverage, but not so many that computation is expensive

**Optimal Number of Exemplars:**

Empirically, 5–20 exemplars per route provides a good balance. Formally, the route representation quality can be measured by the **average intra-route variance**:

$$
\text{Var}(r_i) = \frac{1}{m_i} \sum_{j=1}^{m_i} \|\mathbf{e}_j^{(i)} - \mathbf{e}_{r_i}\|^2
$$

Adding exemplars reduces this variance until saturation.

**Active Learning for Route Calibration:**

1. Start with a small set of exemplars per route
2. Route real queries and collect routing decisions
3. Identify **misrouted** and **low-confidence** queries
4. Have a human annotator label these queries with the correct route
5. Add them as new exemplars to the appropriate route
6. Repeat until routing accuracy stabilizes

**Calibration via Temperature Scaling:**

Apply a temperature parameter $T$ to sharpen or soften the similarity distribution:

$$
P(r_i \mid q) = \frac{\exp(\text{sim}(q, r_i) / T)}{\sum_j \exp(\text{sim}(q, r_j) / T)}
$$

- $T \rightarrow 0$: Hard routing (winner-take-all)
- $T = 1$: Standard softmax
- $T \rightarrow \infty$: Uniform distribution (no discrimination)

Optimal $T$ is selected via **Expected Calibration Error (ECE)** on a validation set.

---

### 1.3.5 Hybrid Routing: Combining Semantic + Rule-Based

Hybrid routing combines the interpretability and reliability of rule-based routing with the flexibility of semantic routing:

**Architecture:**

```
Query → Rule-Based Router → [Match Found] → Agent
                           → [No Match]   → Semantic Router → [Above Threshold] → Agent
                                                             → [Below Threshold] → Fallback
```

**Formal Definition:**

$$
R_{\text{hybrid}}(q) = \begin{cases}
R_{\text{rule}}(q) & \text{if } \exists \, k : \phi_k(q) = \text{True} \\
R_{\text{semantic}}(q) & \text{if } \max_{r_i} \text{sim}(q, r_i) \geq \tau \\
a_{\text{fallback}} & \text{otherwise}
\end{cases}
$$

**Priority Ordering:**

Rules are checked first because they encode **high-confidence, high-priority routing decisions** (e.g., safety-critical queries must always go to a moderation agent). Semantic routing handles the long tail of queries that don't match any explicit rule.

**Weighted Combination:**

Alternatively, combine rule-based and semantic scores:

$$
\text{score}(q, r_i) = w_{\text{rule}} \cdot \mathbb{1}[\phi_i(q)] + w_{\text{sem}} \cdot \text{sim}(q, r_i)
$$

$$
a^* = \arg\max_{r_i} \text{score}(q, r_i)
$$

where $w_{\text{rule}}$ and $w_{\text{sem}}$ are weights reflecting confidence in each signal.

**Implementation:**

```python
class HybridRouter:
    def __init__(self, rule_router, semantic_router, threshold=0.8):
        self.rule_router = rule_router
        self.semantic_router = semantic_router
        self.threshold = threshold
    
    def route(self, query: str) -> str:
        # Priority 1: Rule-based routing
        rule_result = self.rule_router.route(query)
        if rule_result is not None:
            return rule_result
        
        # Priority 2: Semantic routing with threshold
        route, score = self.semantic_router.route(query)
        if score >= self.threshold:
            return route
        
        # Priority 3: Fallback
        return "general_agent"
```

---

## 1.4 LLM-Based Routing

### 1.4.1 Function Calling as Routing Primitive

Modern LLMs (GPT-4, Claude, Gemini) support **function calling** (tool use), where the model selects and parameterizes a function from a provided schema. This mechanism serves as a natural routing primitive.

**Mechanism:**

The LLM receives a set of function definitions (tools) alongside the user query. Instead of generating a text response, it outputs a **structured function call** specifying which function to invoke and with what arguments.

**Function Definition as Route Definition:**

Each "function" corresponds to a route/agent:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "code_agent",
        "description": "Handles code generation, debugging, and programming tasks",
        "parameters": {
          "type": "object",
          "properties": {
            "language": {"type": "string", "description": "Programming language"},
            "task_type": {"type": "string", "enum": ["generate", "debug", "refactor", "explain"]}
          },
          "required": ["task_type"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "search_agent",
        "description": "Searches the web for factual, up-to-date information",
        "parameters": {
          "type": "object",
          "properties": {
            "search_query": {"type": "string", "description": "The search query"}
          },
          "required": ["search_query"]
        }
      }
    }
  ]
}
```

**Advantages of Function Calling as Routing:**

1. **Structured output**: The routing decision is a structured JSON object, not free-form text; no parsing required
2. **Parameter extraction**: The LLM simultaneously routes AND extracts relevant parameters
3. **Native support**: Major LLM providers support this natively with fine-tuned function-calling behavior
4. **Multi-tool selection**: Some models support `parallel_tool_calls`, enabling simultaneous routing to multiple agents

**Formal Mapping:**

$$
\text{LLM}(q, \mathcal{F}) \rightarrow (f_i, \text{args}_i)
$$

where $\mathcal{F} = \{f_1, \ldots, f_n\}$ is the set of function definitions and $(f_i, \text{args}_i)$ is the selected function with extracted arguments.

The routing decision is then $a^* = f_i$ with additional context from $\text{args}_i$.

---

### 1.4.2 Chain-of-Thought Routing Decisions

For complex routing decisions, a **Chain-of-Thought (CoT)** approach enables the router LLM to reason step-by-step before selecting a route.

**Motivation:**

Simple queries (e.g., "Write a Python sort function") are easy to route. Complex or ambiguous queries (e.g., "I need to analyze customer churn data, build a predictive model, and present findings to stakeholders") require decomposition and reasoning.

**CoT Routing Prompt:**

```
Analyze the following user query and determine the best agent to handle it.

Think step by step:
1. What is the primary intent of the query?
2. What capabilities are required (code generation, data analysis, search, etc.)?
3. Are there multiple sub-tasks that need different agents?
4. Which agent best matches the dominant requirement?

Available agents: [code_agent, data_agent, search_agent, writing_agent]

User query: {query}

Reasoning:
<think step by step>

Selected agent: <agent_name>
```

**Multi-Step Decomposition Routing:**

For multi-intent queries, CoT routing can produce a **routing plan** — an ordered sequence of agent invocations:

$$
\text{Plan}(q) = [(a_1, \text{subtask}_1), (a_2, \text{subtask}_2), \ldots, (a_k, \text{subtask}_k)]
$$

**Implementation:**

```python
ROUTING_COT_PROMPT = """
You are a routing agent. Analyze the query and produce a routing plan.

Available agents and their capabilities:
{agent_descriptions}

User query: {query}

Instructions:
1. Decompose the query into sub-tasks
2. For each sub-task, identify the best agent
3. Determine the execution order (some sub-tasks depend on others)
4. Output a JSON routing plan

Output format:
{{
  "reasoning": "...",
  "plan": [
    {{"step": 1, "agent": "...", "subtask": "...", "depends_on": []}},
    {{"step": 2, "agent": "...", "subtask": "...", "depends_on": [1]}}
  ]
}}
"""
```

**Tradeoff Analysis:**

| Aspect | Direct Routing | CoT Routing |
|---|---|---|
| Latency | Lower (single classification) | Higher (reasoning + classification) |
| Accuracy on simple queries | Comparable | Comparable |
| Accuracy on complex queries | Lower | Significantly higher |
| Token cost | Lower | Higher |
| Interpretability | Low (opaque decision) | High (reasoning trace) |

---

### 1.4.3 Structured Output for Route Selection (JSON Schema Enforcement)

Routing decisions must be **machine-parseable**. Structured output enforcement ensures the LLM produces valid, schema-conformant routing decisions.

**JSON Schema for Routing:**

```json
{
  "type": "object",
  "properties": {
    "selected_agent": {
      "type": "string",
      "enum": ["code_agent", "search_agent", "math_agent", "creative_agent"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "reasoning": {
      "type": "string"
    },
    "extracted_params": {
      "type": "object"
    }
  },
  "required": ["selected_agent", "confidence"]
}
```

**Enforcement Mechanisms:**

1. **Constrained Decoding**: Restrict the token sampling to only produce valid JSON tokens at each step. Implemented via **grammar-guided generation** (e.g., Outlines, Guidance, LMFE):

$$
P'(y_t \mid y_{<t}) = \begin{cases} \frac{P(y_t \mid y_{<t})}{\sum_{y' \in \mathcal{V}_{\text{valid}}} P(y' \mid y_{<t})} & \text{if } y_t \in \mathcal{V}_{\text{valid}} \\ 0 & \text{otherwise} \end{cases}
$$

where $\mathcal{V}_{\text{valid}}$ is the set of valid next tokens given the partial JSON and the schema.

2. **API-Level Enforcement**: OpenAI's `response_format: { type: "json_schema", ... }` and Anthropic's tool-use structured outputs.

3. **Post-Hoc Validation**: Generate free-form text, parse as JSON, validate against schema, retry on failure.

**Reliability Hierarchy:**

$$
\text{Constrained Decoding} > \text{API-Level Enforcement} > \text{Post-Hoc Validation}
$$

Constrained decoding guarantees valid output in a single pass. Post-hoc validation may require multiple retries, increasing latency.

---

### 1.4.4 Multi-Model Routing: Selecting Among Heterogeneous LLMs

A critical routing scenario: given multiple LLMs with different capability-cost-latency profiles, select the optimal model for each query.

#### Cost-Performance Tradeoff

$$
\min_{m_i} \; C(m_i) \quad \text{s.t.} \quad Q(m_i, q) \geq \tau
$$

where:
- $C(m_i)$: cost per query for model $m_i$ (measured in dollars per million tokens)
- $Q(m_i, q)$: expected quality of model $m_i$'s response to query $q$
- $\tau$: minimum acceptable quality threshold

**Lagrangian Relaxation:**

$$
\mathcal{L}(m_i, \lambda) = C(m_i) + \lambda(\tau - Q(m_i, q))
$$

$$
m^* = \arg\min_{m_i} \max_{\lambda \geq 0} \mathcal{L}(m_i, \lambda)
$$

**Model Tier Examples:**

| Model | Cost ($/1M tokens) | Quality (MMLU) | Latency (TTFT) |
|---|---|---|---|
| GPT-4o | $5.00 / $15.00 | 88.7% | ~300ms |
| GPT-4o-mini | $0.15 / $0.60 | 82.0% | ~150ms |
| Claude 3.5 Sonnet | $3.00 / $15.00 | 88.7% | ~250ms |
| Claude 3.5 Haiku | $0.80 / $4.00 | 75.2% | ~100ms |
| Llama 3.1 70B (self-hosted) | ~$0.50 | 82.0% | ~200ms |

**Optimal Strategy:**

Route 70-80% of queries (simple ones) to the cheapest model, achieving $5\text{-}10\times$ cost reduction with minimal quality loss.

#### Capability-Based Model Selection

Different models excel at different tasks. Capability-based routing selects the model with the highest capability for the identified task type:

$$
m^* = \arg\max_{m_i \in \mathcal{M}} \text{Capability}(m_i, \text{TaskType}(q))
$$

**Capability Matrix:**

| Task Type | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro | Llama 3.1 70B |
|---|---|---|---|---|
| Code Generation | 9/10 | 9/10 | 8/10 | 7/10 |
| Mathematical Reasoning | 8/10 | 8/10 | 9/10 | 6/10 |
| Creative Writing | 8/10 | 9/10 | 7/10 | 6/10 |
| Multilingual | 8/10 | 7/10 | 9/10 | 5/10 |
| Long Context | 7/10 | 8/10 | 10/10 | 6/10 |

**Implementation:**

```python
CAPABILITY_MATRIX = {
    "code":      {"gpt-4o": 0.9, "claude-sonnet": 0.9, "gemini-pro": 0.8},
    "math":      {"gpt-4o": 0.8, "claude-sonnet": 0.8, "gemini-pro": 0.9},
    "creative":  {"gpt-4o": 0.8, "claude-sonnet": 0.9, "gemini-pro": 0.7},
    "long_ctx":  {"gpt-4o": 0.7, "claude-sonnet": 0.8, "gemini-pro": 1.0},
}

def capability_route(query: str, task_type: str, cost_weight: float = 0.3):
    scores = {}
    for model, capability in CAPABILITY_MATRIX[task_type].items():
        cost = MODEL_COSTS[model]
        scores[model] = capability - cost_weight * cost
    return max(scores, key=scores.get)
```

#### Latency-Aware Routing

For real-time applications, routing must respect **latency constraints**:

$$
m^* = \arg\max_{m_i} Q(m_i, q) \quad \text{s.t.} \quad \text{Latency}(m_i) \leq L_{\max}
$$

**Dynamic Latency Estimation:**

Maintain a running estimate of each model's latency using an **exponential moving average**:

$$
\hat{L}_{m_i}^{(t)} = \alpha \cdot L_{m_i}^{(t)} + (1 - \alpha) \cdot \hat{L}_{m_i}^{(t-1)}
$$

where $L_{m_i}^{(t)}$ is the observed latency of the last call to model $m_i$ and $\alpha \in (0, 1)$ is the smoothing factor.

**Latency-Adjusted Scoring:**

$$
\text{score}(m_i, q) = Q(m_i, q) - \beta \cdot \max(0, \hat{L}_{m_i} - L_{\max})
$$

The penalty term $\max(0, \hat{L}_{m_i} - L_{\max})$ penalizes models that exceed the latency budget, making them less likely to be selected.

---

## 1.5 Routing Architecture Patterns

### 1.5.1 Single-Dispatcher Pattern

The simplest routing architecture: a single router component receives all incoming queries and dispatches each to exactly one agent.

```
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Agent A      Agent B      Agent C
```

**Characteristics:**

- **Single point of decision**: All routing logic is centralized
- **Simple to implement and debug**: One component to monitor
- **Bottleneck risk**: The router becomes a throughput bottleneck under high load
- **Single point of failure**: Router failure halts the entire system

**Formal Model:**

$$
\text{Output} = \text{Agent}_{R(q, C, S)}(q, C)
$$

One routing decision, one agent invocation, one output.

**When to Use:**
- Small number of agents ($n \leq 10$)
- Low query volume (router is not a bottleneck)
- Simple routing logic (classification is straightforward)

---

### 1.5.2 Router-Chain Pattern

Multiple routers are chained sequentially, with each router refining or augmenting the routing decision:

```
Query → Router₁ (Domain) → Router₂ (Task Type) → Router₃ (Model Selection) → Agent
```

**Formal Model:**

$$
d_1 = R_1(q, C, S)
$$
$$
d_2 = R_2(q, C, S, d_1)
$$
$$
a^* = R_3(q, C, S, d_1, d_2)
$$

Each router operates on a different aspect of the routing decision and passes its decision downstream as additional context.

**Example:**

1. **Router₁ (Intent Classification)**: Determines whether the query is about coding, writing, research, or general conversation
2. **Router₂ (Complexity Assessment)**: Determines whether the query is simple, moderate, or complex
3. **Router₃ (Model Selection)**: Based on intent + complexity, selects the appropriate model tier

**Advantages:**
- Separation of concerns: each router is specialized
- Modular: individual routers can be updated independently
- Composable: new routing dimensions can be added by inserting a new router in the chain

**Disadvantages:**
- Increased latency: sequential execution of multiple routers
- Error propagation: an error at Router₁ propagates to all downstream routers

---

### 1.5.3 Routing with State Machines

Routing decisions are governed by a **finite state machine (FSM)** that transitions between states based on routing outcomes and agent outputs:

**Formal Definition:**

$$
\text{FSM} = (\mathcal{Q}_{\text{states}}, \Sigma, \delta, q_0, F)
$$

where:
- $\mathcal{Q}_{\text{states}}$: finite set of states
- $\Sigma$: input alphabet (events: user input, agent output, system events)
- $\delta: \mathcal{Q}_{\text{states}} \times \Sigma \rightarrow \mathcal{Q}_{\text{states}}$: transition function
- $q_0$: initial state
- $F$: set of accepting/terminal states

**Each state maps to a routing action:**

$$
\text{Action}: \mathcal{Q}_{\text{states}} \rightarrow \mathcal{A} \cup \{\text{terminate}\}
$$

**Example State Machine for Customer Support:**

```
                 ┌──────────┐
                 │  START    │
                 └─────┬────┘
                       │ classify_intent
                 ┌─────▼─────┐
            ┌────┤  CLASSIFY  ├────┐
            │    └───────────┘     │
       billing                 technical
            │                      │
    ┌───────▼──────┐     ┌───────▼──────┐
    │BILLING_AGENT │     │ TECH_AGENT   │
    └───────┬──────┘     └───────┬──────┘
            │ resolved?          │ resolved?
       ┌────┼────┐           ┌───┼────┐
      yes   │   no          yes  │   no
       │    │    │           │   │    │
   ┌───▼┐  │  ┌─▼────┐ ┌──▼┐ │ ┌──▼─────┐
   │DONE│  │  │ESCAL.│ │DONE││ │ESCALATE│
   └────┘  │  └──────┘ └────┘  └────────┘
           │                   │
           └───────────────────┘
```

**Implementation:**

```python
from enum import Enum, auto

class State(Enum):
    START = auto()
    CLASSIFY = auto()
    BILLING_AGENT = auto()
    TECH_AGENT = auto()
    ESCALATE = auto()
    DONE = auto()

class RoutingStateMachine:
    def __init__(self):
        self.state = State.START
        self.transitions = {
            (State.START, "any"):       State.CLASSIFY,
            (State.CLASSIFY, "billing"):  State.BILLING_AGENT,
            (State.CLASSIFY, "technical"):State.TECH_AGENT,
            (State.BILLING_AGENT, "resolved"):   State.DONE,
            (State.BILLING_AGENT, "unresolved"): State.ESCALATE,
            (State.TECH_AGENT, "resolved"):      State.DONE,
            (State.TECH_AGENT, "unresolved"):    State.ESCALATE,
        }
    
    def transition(self, event: str) -> State:
        key = (self.state, event)
        if key in self.transitions:
            self.state = self.transitions[key]
        else:
            self.state = self.transitions.get((self.state, "any"), State.ESCALATE)
        return self.state
    
    def get_agent(self) -> str:
        STATE_AGENT_MAP = {
            State.CLASSIFY: "classifier_agent",
            State.BILLING_AGENT: "billing_agent",
            State.TECH_AGENT: "tech_agent",
            State.ESCALATE: "human_agent",
        }
        return STATE_AGENT_MAP.get(self.state)
```

**Advantages:**
- Explicit, auditable control flow
- Handles multi-turn, stateful conversations naturally
- Easy to visualize and reason about
- Supports guard conditions, timeouts, and error states

**Limitations:**
- Rigid: does not handle dynamic, unstructured workflows
- State explosion for complex systems with many possible transitions
- Does not learn or adapt from experience

---

### 1.5.4 Routing DAGs (Directed Acyclic Graphs)

For complex workflows with **parallel execution and dependencies**, routing is modeled as a **Directed Acyclic Graph (DAG)**.

**Formal Definition:**

$$
G = (V, E)
$$

where:
- $V = \{v_1, v_2, \ldots, v_n\}$: nodes representing agents/processing steps
- $E \subseteq V \times V$: directed edges representing data dependencies
- Acyclicity constraint: $\nexists$ cycle in $E$

**Execution Semantics:**

A node $v_i$ can execute when all its predecessors have completed:

$$
\text{Ready}(v_i) = \forall v_j \in \text{Parents}(v_i): \text{Complete}(v_j)
$$

**Topological Sort** determines a valid execution order. Nodes with no dependencies can execute **in parallel**.

**Example DAG:**

```
User Query
    │
    ├──────────────┐
    ▼              ▼
[Search Agent] [Code Agent]     ← Parallel execution
    │              │
    ▼              ▼
[Summarizer]   [Tester]         ← Parallel execution
    │              │
    └──────┬───────┘
           ▼
    [Synthesis Agent]            ← Waits for both branches
           │
           ▼
    [Final Output]
```

**Critical Path Analysis:**

The **minimum total latency** of the DAG is determined by the **critical path** — the longest path from source to sink:

$$
\text{Latency}(G) = \max_{\text{path } P \text{ from source to sink}} \sum_{v \in P} \text{Latency}(v)
$$

This is computed in $O(|V| + |E|)$ via topological sort.

**Implementation with asyncio:**

```python
import asyncio
from typing import Dict, List, Any

class DAGRouter:
    def __init__(self):
        self.nodes: Dict[str, callable] = {}
        self.edges: Dict[str, List[str]] = {}  # node -> list of dependencies
    
    def add_node(self, name: str, agent: callable, depends_on: List[str] = None):
        self.nodes[name] = agent
        self.edges[name] = depends_on or []
    
    async def execute(self, query: str) -> Dict[str, Any]:
        results = {}
        completed = set()
        pending = set(self.nodes.keys())
        
        async def run_node(name: str):
            # Wait for dependencies
            dep_results = {dep: results[dep] for dep in self.edges[name]}
            result = await self.nodes[name](query, dep_results)
            results[name] = result
            completed.add(name)
        
        while pending:
            ready = [n for n in pending 
                     if all(d in completed for d in self.edges[n])]
            if not ready:
                raise RuntimeError("Deadlock detected: circular dependency")
            pending -= set(ready)
            await asyncio.gather(*[run_node(n) for n in ready])
        
        return results
```

---

### 1.5.5 Event-Driven Routing

In event-driven architectures, routing is triggered by **events** rather than synchronous function calls:

**Architecture:**

```
Event Bus / Message Queue
    │
    ├── Event: "query.received" → Router subscribes → Publishes "route.code_agent"
    ├── Event: "route.code_agent" → Code Agent subscribes → Publishes "agent.response"
    ├── Event: "agent.response" → Aggregator subscribes → Publishes "response.ready"
    └── Event: "response.ready" → Output Handler subscribes → Returns to user
```

**Formal Model:**

An event-driven routing system is defined by:

$$
\text{EDS} = (\mathcal{E}, \mathcal{H}, \mathcal{B})
$$

where:
- $\mathcal{E}$: set of event types
- $\mathcal{H}: \mathcal{E} \rightarrow 2^{\mathcal{A}}$: event-to-handler mapping (which agents subscribe to which events)
- $\mathcal{B}$: the event bus/message broker

**Routing as Event Emission:**

The router consumes a `query.received` event and emits a `route.<agent>` event:

$$
\text{Router}: e_{\text{query}} \rightarrow e_{\text{route}(a^*)}
$$

**Advantages:**
- **Decoupled**: Router and agents are independent components communicating via events
- **Scalable**: Multiple instances of each agent can subscribe to the same event
- **Resilient**: Failed agent invocations can be retried or dead-letter queued
- **Observable**: All events are logged, enabling full trace reconstruction

**Technologies:**
- Apache Kafka for high-throughput event streaming
- RabbitMQ / Redis Streams for lightweight queuing
- AWS EventBridge / Google Pub/Sub for cloud-native deployments

---

## 1.6 Routing Evaluation and Optimization

### 1.6.1 Route Accuracy Metrics

**Route Accuracy** is the primary metric: the fraction of queries correctly routed to the intended agent.

$$
\text{RouteAccuracy} = \frac{|\{q_i : R(q_i) = r_i^*\}|}{N}
$$

where $r_i^*$ is the ground-truth correct route for query $q_i$.

**Per-Route Metrics:**

For each route $r_k$:

$$
\text{Precision}(r_k) = \frac{|\{q_i : R(q_i) = r_k \land r_i^* = r_k\}|}{|\{q_i : R(q_i) = r_k\}|}
$$

$$
\text{Recall}(r_k) = \frac{|\{q_i : R(q_i) = r_k \land r_i^* = r_k\}|}{|\{q_i : r_i^* = r_k\}|}
$$

$$
F_1(r_k) = \frac{2 \cdot \text{Precision}(r_k) \cdot \text{Recall}(r_k)}{\text{Precision}(r_k) + \text{Recall}(r_k)}
$$

**Confusion Matrix:**

A $K \times K$ confusion matrix $\mathbf{M}$ where $M_{ij}$ counts the number of queries with true route $i$ that were routed to route $j$.

**Weighted Route Accuracy:**

Not all misroutes are equally costly. A misroute from `code_agent` to `search_agent` is less severe than a misroute from `safety_agent` to `general_agent`. Define a **misroute cost matrix** $\mathbf{W}$ where $W_{ij}$ is the cost of routing a query from true route $i$ to route $j$:

$$
\text{WeightedCost} = \frac{1}{N}\sum_{i=1}^{N} W_{r_i^*, R(q_i)}
$$

---

### 1.6.2 Latency Overhead of Routing Layers

The routing layer adds latency to every request. This overhead must be measured and minimized.

**Total Request Latency Decomposition:**

$$
L_{\text{total}} = L_{\text{preprocessing}} + L_{\text{routing}} + L_{\text{agent}} + L_{\text{postprocessing}}
$$

**Routing Latency by Strategy:**

| Strategy | Typical Latency | Bottleneck |
|---|---|---|
| Rule-based | $< 1$ ms | None (in-memory) |
| Regex matching | $1\text{-}5$ ms | Pattern compilation |
| Intent classifier (BERT) | $5\text{-}20$ ms | Model inference |
| Embedding + similarity | $10\text{-}50$ ms | Embedding computation |
| LLM-as-router (small) | $100\text{-}500$ ms | LLM inference |
| LLM-as-router (large) | $500\text{-}2000$ ms | LLM inference |
| CoT routing | $1000\text{-}5000$ ms | Multi-step LLM reasoning |

**Routing Latency Budget Rule of Thumb:**

$$
L_{\text{routing}} \leq 0.1 \cdot L_{\text{agent}}
$$

The routing layer should consume no more than 10% of the total agent processing time. If the agent takes 2 seconds, the router should complete in under 200ms.

**Optimization Techniques:**

1. **Caching**: Cache routing decisions for repeated/similar queries:

$$
R_{\text{cached}}(q) = \begin{cases} \text{Cache}[h(q)] & \text{if } h(q) \in \text{Cache} \\ R(q) & \text{otherwise (compute and cache)} \end{cases}
$$

where $h(q)$ is a hash or approximate embedding of $q$.

2. **Batch Routing**: Accumulate queries and route them in a single batch inference

3. **Speculative Routing**: Begin executing the most likely agent while the router finalizes its decision; abort if misrouted

4. **Tiered Routing**: Try fast rules first; fall back to embedding-based routing; fall back to LLM routing only if needed (Section 1.3.5 hybrid approach)

---

### 1.6.3 Misrouting Detection and Recovery

**Misrouting** occurs when the router selects an incorrect agent. Detection and recovery mechanisms are essential for system reliability.

**Detection Strategies:**

1. **Agent Self-Reporting**: The agent detects that the query is outside its capability and reports inability:

```python
class Agent:
    async def handle(self, query: str) -> AgentResponse:
        if not self.can_handle(query):
            return AgentResponse(
                status="MISROUTED",
                message="This query is outside my capabilities",
                suggested_agent="math_agent"
            )
        # Process normally...
```

2. **Output Quality Monitoring**: A quality monitor evaluates the agent's output and flags low-quality responses:

$$
\text{Misrouted}(q, a, y) = \mathbb{1}[\text{QualityScore}(q, y) < \tau_{\text{quality}}]
$$

3. **Confidence Monitoring**: The agent's confidence in its own output is below a threshold:

$$
\text{Misrouted}(q, a) = \mathbb{1}[\text{AgentConfidence}(a, q) < \tau_{\text{conf}}]
$$

4. **Latency Anomaly**: The agent takes unexpectedly long, suggesting it is struggling with an out-of-domain query

**Recovery Strategies:**

1. **Re-routing**: Route to the next-best agent:

$$
a_{\text{fallback}} = \arg\max_{a_j \neq a_i} P(a_j \mid q, C, S)
$$

2. **Escalation**: Route to a more capable (and expensive) general-purpose agent

3. **Multi-Agent Consensus**: Send to multiple agents simultaneously and select the best output:

$$
y^* = \arg\max_{y_k} \text{QualityScore}(q, y_k)
$$

4. **Human-in-the-Loop**: Escalate to a human operator for queries that resist automated routing

**Recovery Latency Budget:**

$$
L_{\text{recovery}} = L_{\text{detection}} + L_{\text{reroute}} + L_{\text{agent}_2}
$$

This is typically $2\text{-}3\times$ the normal request latency. Minimizing detection latency is critical.

---

### 1.6.4 A/B Testing Routing Strategies

Systematic comparison of routing strategies requires **controlled experimentation**.

**A/B Test Design:**

1. **Hypothesis**: "Semantic routing (Strategy B) achieves higher route accuracy than rule-based routing (Strategy A)"
2. **Randomization**: Randomly assign incoming queries to Strategy A or Strategy B with probability $p$ and $1 - p$ (typically $p = 0.5$)
3. **Metrics**: Measure route accuracy, latency, cost, and downstream task quality for each group
4. **Duration**: Run until statistical significance is achieved

**Statistical Test:**

For route accuracy comparison, use a **two-proportion z-test**:

$$
z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}
$$

where $\hat{p}_A, \hat{p}_B$ are the observed route accuracies and $\hat{p} = \frac{n_A \hat{p}_A + n_B \hat{p}_B}{n_A + n_B}$ is the pooled proportion.

Reject the null hypothesis (no difference) if $|z| > z_{\alpha/2}$ (e.g., $z_{0.025} = 1.96$ for 95% confidence).

**Sample Size Calculation:**

For detecting a minimum effect size $\delta$ with power $1 - \beta$:

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\hat{p}(1 - \hat{p})}{\delta^2}
$$

**Multi-Armed Bandit Alternative:**

Instead of fixed A/B splits, use a **multi-armed bandit** approach that dynamically allocates traffic to better-performing strategies:

$$
P(\text{Strategy } k) = \frac{\exp(\hat{\mu}_k / T)}{\sum_j \exp(\hat{\mu}_j / T)}
$$

This reduces the cost of exploration by quickly shifting traffic away from inferior strategies (Thompson Sampling or UCB-based allocation).

---

### 1.6.5 Routing Calibration and Drift Detection

**Calibration** ensures that the router's confidence scores accurately reflect the true probability of correct routing:

$$
P(\text{correct route} \mid \text{confidence} = c) = c \quad \forall c \in [0, 1]
$$

**Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|
$$

where $B$ is the number of bins, $n_b$ is the number of queries in bin $b$, $\text{acc}(b)$ is the actual accuracy in bin $b$, and $\text{conf}(b)$ is the average confidence in bin $b$.

**Reliability Diagram:**

Plot $\text{acc}(b)$ vs. $\text{conf}(b)$ for each bin. A perfectly calibrated router produces points on the diagonal $y = x$.

**Calibration Techniques:**

1. **Platt Scaling**: Learn a logistic function mapping raw scores to calibrated probabilities:

$$
P_{\text{calibrated}}(r_i \mid q) = \sigma(w \cdot s_i + b)
$$

where $s_i$ is the raw routing score, and $w, b$ are learned on a held-out calibration set.

2. **Temperature Scaling**: Adjust the softmax temperature (Section 1.3.4)

3. **Isotonic Regression**: Non-parametric calibration via monotonic regression

**Drift Detection:**

Over time, the distribution of incoming queries may shift (**data drift**), or agent capabilities may change (**concept drift**). Routing decisions trained on historical data may become suboptimal.

**Detection Methods:**

1. **Population Stability Index (PSI):**

$$
\text{PSI} = \sum_{b=1}^{B} (p_b^{\text{new}} - p_b^{\text{ref}}) \cdot \ln\left(\frac{p_b^{\text{new}}}{p_b^{\text{ref}}}\right)
$$

where $p_b^{\text{new}}$ and $p_b^{\text{ref}}$ are the proportions of queries falling in bin $b$ for the new and reference distributions, respectively.

- $\text{PSI} < 0.1$: No significant drift
- $0.1 \leq \text{PSI} < 0.25$: Moderate drift, investigate
- $\text{PSI} \geq 0.25$: Significant drift, recalibrate

2. **Kolmogorov-Smirnov Test**: Compare the distribution of routing scores between a reference period and the current period:

$$
D = \sup_x |F_{\text{ref}}(x) - F_{\text{new}}(x)|
$$

3. **Routing Accuracy Monitoring**: Track route accuracy over time via windowed metrics:

$$
\text{Accuracy}_{[t-W, t]} = \frac{1}{W}\sum_{i=t-W}^{t} \mathbb{1}[R(q_i) = r_i^*]
$$

Alert if accuracy drops below a threshold.

**Automated Recalibration Pipeline:**

```
Monitor → Detect Drift → Collect New Labeled Data → Retrain/Recalibrate Router → Deploy → Monitor
```

---

## 1.7 Case Studies and Implementations

### 1.7.1 Semantic Router Libraries

**`semantic-router` (by Aurelio AI)**

An open-source library implementing embedding-based semantic routing with minimal overhead.

**Architecture:**

```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder

# Define routes with exemplar utterances
code_route = Route(
    name="code_agent",
    utterances=[
        "Write a Python function to sort a list",
        "Debug this JavaScript code",
        "Refactor my class to use dependency injection",
        "Generate a REST API endpoint in FastAPI",
        "How do I implement a binary search tree?",
    ]
)

search_route = Route(
    name="search_agent",
    utterances=[
        "What is the latest news about AI?",
        "Who won the Nobel Prize in Physics 2024?",
        "Find information about climate change",
        "What is the population of Tokyo?",
        "Look up the current stock price of NVIDIA",
    ]
)

math_route = Route(
    name="math_agent",
    utterances=[
        "Solve the integral of x^2 from 0 to 1",
        "What is the derivative of sin(x)*cos(x)?",
        "Calculate the eigenvalues of this matrix",
        "Prove that the sum of first n integers is n(n+1)/2",
        "Solve this system of linear equations",
    ]
)

# Initialize the route layer
encoder = OpenAIEncoder(name="text-embedding-3-small")
route_layer = RouteLayer(
    encoder=encoder,
    routes=[code_route, search_route, math_route]
)

# Route a query
result = route_layer("Write me a quicksort implementation in Rust")
# result.name → "code_agent"
```

**Internal Mechanism:**

1. On initialization, all exemplar utterances are embedded and stored
2. Route centroids (or individual exemplar vectors) are pre-computed
3. At query time:
   - Embed the query: $\mathbf{e}_q = \text{Embed}(q)$
   - Compute cosine similarity against all route representations
   - Return the route with the highest similarity above the threshold

**Performance Characteristics:**
- Embedding latency: ~20-50ms (API call) or ~5ms (local model)
- Similarity computation: $< 1$ms for $< 100$ routes
- Total routing latency: $\sim 25\text{-}55$ms

---

### 1.7.2 OpenAI Function Calling as Routing

OpenAI's function calling API provides a production-grade routing mechanism where the LLM itself acts as the router.

**Implementation Pattern:**

```python
import openai
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "route_to_code_agent",
            "description": "Route to the code generation agent for programming tasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language"
                    },
                    "task_description": {
                        "type": "string",
                        "description": "What code needs to be written"
                    }
                },
                "required": ["task_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_search_agent",
            "description": "Route to the search agent for factual queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query to execute"
                    }
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_general_agent",
            "description": "Route to general conversation agent for chat, advice, creative writing",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic of conversation"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

async def llm_route(query: str) -> dict:
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",  # Use cheap model for routing
        messages=[
            {"role": "system", "content": "You are a routing agent. Select the appropriate tool to handle the user's request."},
            {"role": "user", "content": query}
        ],
        tools=tools,
        tool_choice="required"  # Force tool selection
    )
    
    tool_call = response.choices[0].message.tool_calls[0]
    return {
        "agent": tool_call.function.name.replace("route_to_", ""),
        "params": json.loads(tool_call.function.arguments)
    }
```

**Key Design Decisions:**

1. **`tool_choice="required"`**: Forces the LLM to select a tool, preventing it from generating a text response instead of routing

2. **Use `gpt-4o-mini` for routing**: The routing model need not be the most capable; it only needs to classify, not generate high-quality content. Using a small model reduces routing latency from ~500ms to ~150ms and cost by ~30×.

3. **Parameter extraction**: The function calling mechanism simultaneously routes AND extracts structured parameters, eliminating a separate extraction step

4. **Schema enforcement**: The JSON schema in the tool definition constrains the output format, ensuring reliable parsing

**Comparison with Direct Prompting:**

| Aspect | Function Calling | Direct Prompting |
|---|---|---|
| Output format | Guaranteed JSON | May produce invalid format |
| Parameter extraction | Built-in | Requires separate parsing |
| Reliability | High (constrained output) | Medium (may refuse or hallucinate) |
| Multi-tool selection | Supported natively | Requires custom logic |
| Cost | Slightly higher (tool tokens count) | Slightly lower |

---

### 1.7.3 Multi-Agent Routing in Production Systems

**Production Architecture: Enterprise Customer Support System**

```
                         ┌──────────────┐
                         │ API Gateway  │
                         │ (Rate Limit, │
                         │  Auth)       │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Router      │
                         │  (Hybrid:    │
                         │   Rules +    │
                         │   Semantic + │
                         │   LLM)       │
                         └──────┬───────┘
                    ┌───────────┼───────────┐──────────┐
                    │           │           │          │
              ┌─────▼────┐ ┌───▼───┐ ┌─────▼────┐ ┌──▼───────┐
              │ Billing  │ │ Tech  │ │ Account  │ │ General  │
              │ Agent    │ │Support│ │ Mgmt     │ │ Chat     │
              │ (GPT-4o) │ │Agent  │ │ Agent    │ │(GPT-4o-  │
              │          │ │(Sonn.)│ │(Haiku)   │ │ mini)    │
              └─────┬────┘ └───┬───┘ └─────┬────┘ └──┬───────┘
                    │          │           │          │
                    │     ┌────▼────┐      │          │
                    │     │Tools:   │      │          │
                    │     │- DB     │      │          │
                    │     │- Jira   │      │          │
                    │     │- Docs   │      │          │
                    │     └─────────┘      │          │
                    └──────────┼───────────┘──────────┘
                               │
                        ┌──────▼───────┐
                        │  Response    │
                        │  Quality     │
                        │  Monitor     │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  Feedback    │
                        │  Loop        │
                        │  (Bandit)    │
                        └──────────────┘
```

**Routing Logic (Three-Tier Hybrid):**

```python
class ProductionRouter:
    def __init__(self):
        self.rule_router = RuleBasedRouter()     # Tier 1: ~0ms
        self.semantic_router = SemanticRouter()   # Tier 2: ~25ms
        self.llm_router = LLMRouter()            # Tier 3: ~150ms
        self.bandit = ContextualBandit()          # Adaptive learning
    
    async def route(self, query: str, context: dict) -> RoutingDecision:
        # Tier 1: Safety-critical rules (always checked first)
        if self.rule_router.is_safety_critical(query):
            return RoutingDecision(
                agent="safety_agent",
                method="rule",
                confidence=1.0
            )
        
        # Tier 2: Semantic routing
        semantic_result = self.semantic_router.route(query)
        if semantic_result.confidence >= 0.85:
            # High confidence: use semantic result, but apply bandit adjustment
            adjusted = self.bandit.adjust(query, semantic_result)
            return adjusted
        
        # Tier 3: LLM routing for ambiguous queries
        llm_result = await self.llm_router.route(query, context)
        return RoutingDecision(
            agent=llm_result.agent,
            method="llm",
            confidence=llm_result.confidence,
            reasoning=llm_result.reasoning
        )
    
    def record_outcome(self, query: str, decision: RoutingDecision, 
                       outcome: dict):
        """Feedback loop: update bandit model with observed outcome"""
        reward = self._compute_reward(outcome)
        self.bandit.update(query, decision.agent, reward)
```

**Production Monitoring Dashboard Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| Route Accuracy | $\geq 95\%$ | $< 90\%$ |
| Routing Latency (P50) | $< 30$ ms | $> 100$ ms |
| Routing Latency (P99) | $< 200$ ms | $> 500$ ms |
| Fallback Rate | $< 10\%$ | $> 20\%$ |
| Misroute Detection Rate | $\geq 99\%$ | $< 95\%$ |
| Route Distribution Entropy | Stable $\pm 5\%$ | Shift $> 15\%$ |
| Cost per Routed Query | $< \$0.001$ | $> \$0.005$ |

**Key Production Lessons:**

1. **Start simple, add complexity as needed.** Rule-based routing handles 60-70% of queries. Semantic routing handles another 20-25%. LLM routing is only needed for the remaining 5-15%.

2. **The router is a single point of failure.** Implement redundancy: if the primary router fails, fall back to a simpler (rule-based) router rather than failing entirely.

3. **Monitor route distribution, not just accuracy.** A sudden shift in route distribution (e.g., 80% of queries going to `billing_agent` instead of the usual 30%) indicates either a real-world event (billing issue) or a routing bug.

4. **Invest in misroute recovery.** Even with 95% routing accuracy, 5% of queries are misrouted. At 10,000 queries/day, that is 500 misrouted queries. Automated recovery (re-routing based on agent self-reporting) is essential.

5. **Use the cheapest viable model for routing.** Routing is a classification task, not a generation task. A $0.15/M-token model performs comparably to a $15/M-token model for routing decisions, yielding $100\times$ cost savings on the routing layer.

6. **Implement circuit breakers.** If an agent is consistently failing (e.g., external API is down), the router should automatically stop routing to that agent and redirect to a fallback:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_time=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_time = recovery_time
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED = normal, OPEN = blocking
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"
    
    def is_available(self) -> bool:
        if self.state == "CLOSED":
            return True
        if time.time() - self.last_failure_time > self.recovery_time:
            self.state = "HALF_OPEN"
            return True  # Allow one probe request
        return False
```

---

**Summary of Chapter 1:**

Routing is the **control-plane intelligence** of agentic AI systems. It is formally defined as a decision function $R: (q, C, S) \rightarrow a_i$ that maps queries, context, and system state to the optimal agent or processing path. The routing strategy landscape spans from simple rule-based matching (fast, brittle) through embedding-based semantic routing (flexible, moderate latency) to LLM-based routing (maximally flexible, highest latency). Production systems employ **hybrid architectures** that layer these strategies in a tiered cascade, with adaptive learning (contextual bandits) continuously optimizing routing decisions based on observed outcomes. Routing evaluation requires rigorous metrics (accuracy, latency, calibration), drift detection, and A/B testing to maintain system reliability over time.