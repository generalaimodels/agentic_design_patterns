

# Chapter 7: Memory Management

---

## 7.1 Definition and Formal Framework

### 7.1.1 What is Memory in Agentic Systems

Memory in agentic systems is the computational infrastructure that enables an agent to **store, organize, retrieve, update, and selectively forget information** across temporal boundaries—spanning individual reasoning steps, multi-turn conversations, and persistent cross-session interactions. It is the mechanism through which an agent transcends the ephemeral, stateless nature of individual LLM forward passes and achieves **temporal coherence**, **experiential learning**, and **contextual awareness**.

Formally, memory in an agentic system is a stateful module:

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{W}, \mathcal{R}, \mathcal{U}, \mathcal{D}, \mathcal{C} \rangle
$$

where:

- $\mathcal{S}$: The storage substrate—the physical or logical medium holding memory entries (vector databases, key-value stores, graphs, etc.).
- $\mathcal{W}: \text{Information} \times \mathcal{S} \rightarrow \mathcal{S}'$: The **write** (encoding) operation that commits new information to storage.
- $\mathcal{R}: \text{Query} \times \mathcal{S} \rightarrow \{(m_i, \alpha_i)\}$: The **read** (retrieval) operation that fetches relevant memories given a query, each with relevance score $\alpha_i$.
- $\mathcal{U}: \text{NewInfo} \times \mathcal{S} \rightarrow \mathcal{S}'$: The **update** operation that modifies existing memories.
- $\mathcal{D}: \text{Criterion} \times \mathcal{S} \rightarrow \mathcal{S}'$: The **delete** (forgetting) operation that removes memories based on specified criteria.
- $\mathcal{C}: \mathcal{S} \rightarrow \mathcal{S}'$: The **consolidation** operation that compresses, merges, and reorganizes memories for efficiency.

**Fundamental distinctions from traditional databases:**

| Dimension | Traditional Database | Agent Memory |
|---|---|---|
| **Query type** | Structured (SQL, key lookup) | Semantic (natural language, embedding similarity) |
| **Schema** | Fixed, predefined | Flexible, evolving |
| **Write trigger** | Explicit application logic | Agent autonomously decides what to memorize |
| **Retrieval relevance** | Exact match or predefined indices | Learned similarity in embedding space |
| **Temporal dynamics** | Static until modified | Decaying, consolidating, context-dependent |
| **Purpose** | Data persistence | Cognitive augmentation |

**Why memory is architecturally essential for agents:**

1. **Statefulness across interactions**: Without memory, every LLM call is independent—the agent cannot track progress on multi-step tasks, learn from past failures, or maintain user context across sessions.

2. **Knowledge accumulation**: Agents operating in dynamic environments encounter novel information that is not in the LLM's pre-training data. Memory enables runtime knowledge acquisition.

3. **Personalization**: To serve individual users effectively, agents must remember user preferences, history, and context—information that varies per user and cannot be baked into model weights.

4. **Error correction**: By remembering past mistakes and their corrections, agents can avoid repeating errors—a form of experiential learning without gradient updates.

5. **Coordination in multi-agent systems**: Shared memory enables agents to communicate asynchronously, share discoveries, and maintain collective state (cf. Chapter 6, blackboard architecture).

---

### 7.1.2 Memory as a Function

The core abstraction of agent memory is a **retrieval function** that, given a query and a temporal context, returns a ranked set of relevant memories:

$$
\mathcal{M}: (q, t) \rightarrow \{(k_i, v_i, \alpha_i)\}_{i=1}^{K}
$$

where:

- $q$: The **query**—a natural language string, an embedding vector, or a structured query object representing the agent's current information need.
- $t$: The **current time**—a temporal index that modulates retrieval behavior (e.g., recency weighting, decay filtering).
- $k_i$: The **key** of the $i$-th retrieved memory—an identifier or indexing representation used for matching against the query (typically an embedding vector).
- $v_i$: The **stored content** (value) of the $i$-th memory—the actual information payload (text, structured data, code snippets, observation records, etc.).
- $\alpha_i \in [0, 1]$: The **relevance score** of memory $i$ at time $t$—a composite measure incorporating semantic similarity, temporal recency, and importance.

**Formal decomposition of relevance:**

The relevance score $\alpha_i$ is computed as a weighted combination of multiple factors:

$$
\alpha_i = f\left(\text{sim}(q, k_i),\; \text{recency}(t, t_i),\; \text{importance}(m_i),\; \text{access\_count}(m_i)\right)
$$

A common instantiation:

$$
\alpha_i = w_s \cdot \text{sim}(\mathbf{e}_q, \mathbf{e}_{k_i}) + w_r \cdot \text{recency}(t, t_i) + w_p \cdot \text{importance}(m_i)
$$

subject to $w_s + w_r + w_p = 1$, where:

- $\text{sim}(\mathbf{e}_q, \mathbf{e}_{k_i})$: Cosine similarity between query embedding and memory key embedding:

$$
\text{sim}(\mathbf{e}_q, \mathbf{e}_{k_i}) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{k_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{k_i}\|}
$$

- $\text{recency}(t, t_i)$: Temporal decay function measuring how recently memory $m_i$ was created or last accessed:

$$
\text{recency}(t, t_i) = e^{-\lambda (t - t_i)}
$$

where $\lambda > 0$ is the decay rate parameter and $t_i$ is the timestamp of memory $m_i$.

- $\text{importance}(m_i) \in [0, 1]$: An intrinsic importance score assigned at write time, reflecting how critical the information is regardless of the current query.

**Memory entry schema:**

Each memory entry $m_i$ is a structured record:

$$
m_i = \langle \text{id}_i, k_i, v_i, \mathbf{e}_i, t_{\text{created}}, t_{\text{accessed}}, n_{\text{access}}, \text{importance}_i, \text{source}_i, \text{type}_i, \text{metadata}_i \rangle
$$

| Field | Type | Description |
|---|---|---|
| $\text{id}_i$ | UUID | Unique memory identifier |
| $k_i$ | String | Human-readable key or summary |
| $v_i$ | String/Object | Full content payload |
| $\mathbf{e}_i$ | $\mathbb{R}^d$ | Embedding vector for similarity search |
| $t_{\text{created}}$ | Timestamp | When the memory was first stored |
| $t_{\text{accessed}}$ | Timestamp | When the memory was last retrieved |
| $n_{\text{access}}$ | Integer | How many times the memory has been retrieved |
| $\text{importance}_i$ | $[0, 1]$ | Intrinsic importance score |
| $\text{source}_i$ | Enum | Origin: user input, tool output, agent reasoning, etc. |
| $\text{type}_i$ | Enum | Category: episodic, semantic, procedural |
| $\text{metadata}_i$ | Dict | Additional structured attributes |

---

### 7.1.3 Why LLMs Need External Memory (Context Window Limitations)

Large language models operate within a **fixed context window** $C_{\max}$ tokens. This architectural constraint creates fundamental limitations that external memory systems must overcome.

**Limitation 1: Finite Context Capacity**

The transformer's self-attention mechanism has computational complexity $O(n^2 d)$ where $n$ is the sequence length and $d$ is the embedding dimension. Even with recent advances extending context windows to 128K–1M tokens, the capacity remains finite:

$$
|\text{All information the agent has ever encountered}| \gg C_{\max}
$$

For a long-running agent that processes 10,000 tokens per interaction across 1,000 interactions, the total information volume is $10^7$ tokens—far exceeding any context window.

**Limitation 2: Attention Degradation over Long Contexts**

Empirical evidence (Liu et al., 2024, "Lost in the Middle") demonstrates that transformer attention does not uniformly weight all context positions. Information in the **middle** of long contexts is significantly less likely to be attended to:

$$
P(\text{recall} | \text{position } p, \text{context length } n) \approx \begin{cases}
\text{high} & \text{if } p \approx 0 \text{ or } p \approx n \\
\text{low} & \text{if } p \approx n/2
\end{cases}
$$

This U-shaped attention curve means that simply concatenating all information into the context is **lossy even when it fits**. External memory with targeted retrieval ensures that relevant information is placed in high-attention positions.

**Limitation 3: Cost Scaling**

LLM inference cost scales linearly (or super-linearly for attention computation) with context length:

$$
\text{Cost}(n) = c_{\text{input}} \cdot n + c_{\text{output}} \cdot m + c_{\text{attention}} \cdot n^2
$$

where $n$ is input length, $m$ is output length. For a context of 100K tokens at \$3/million input tokens:

$$
\text{Cost per call} = 100{,}000 \times 3 \times 10^{-6} = \$0.30
$$

Over 1,000 calls per day: \$300/day. External memory with selective retrieval (pulling only 2K relevant tokens instead of 100K) reduces this by $50\times$.

**Limitation 4: No Cross-Session Persistence**

LLM context is ephemeral—it exists only during a single API call. Between sessions, all context is lost unless explicitly saved and re-injected. External memory provides **persistence across sessions**, enabling:

- Multi-day task execution.
- User preference learning over weeks/months.
- Organizational knowledge accumulation.

**Limitation 5: No Selective Forgetting**

The context window treats all tokens equally (modulo attention patterns). There is no mechanism to explicitly "forget" irrelevant information or "highlight" critical information within the context. External memory enables:

- Importance-weighted storage and retrieval.
- Explicit deletion of outdated or incorrect information.
- Prioritized retrieval of the most relevant subset.

**The Memory Augmentation Principle:**

$$
\text{EffectiveContext}(t) = \underbrace{C_{\text{system}}}_{\text{instructions}} + \underbrace{\mathcal{R}(q_t, t)}_{\text{retrieved memories}} + \underbrace{C_{\text{recent}}}_{\text{current conversation}} + \underbrace{C_{\text{tools}}}_{\text{tool descriptions}}
$$

The agent's effective context at time $t$ is dynamically assembled from multiple sources, with retrieved memories filling the knowledge gap that the finite context window cannot span.

---

### 7.1.4 Cognitive Science Inspiration: Human Memory Models

Agent memory design draws heavily from cognitive science models of human memory. Understanding these models illuminates why certain architectural choices are effective.

**Atkinson-Shiffrin Multi-Store Model (1968):**

```
External        Sensory          Short-Term        Long-Term
Stimulus  ──→  Register   ──→   Memory (STM)  ──→  Memory (LTM)
                (< 1 sec)       (15-30 sec,       (unlimited
                                 7 ± 2 items)      capacity)
                   │                  │                  │
                   ▼                  ▼                  ▼
               Forgotten         Forgotten           Retrieval
               if not            if not               (recall,
               attended          rehearsed            recognition)
```

**Mapping to agent memory:**

| Human Memory | Agent Memory Analog | Implementation |
|---|---|---|
| Sensory register | Raw input buffer | Current observation, tool outputs |
| Short-term memory (STM) | Working memory / context window | Last $k$ conversation turns |
| Long-term memory (LTM) | Persistent vector store / database | Vector DB, knowledge graph |
| Rehearsal (STM → LTM) | Memory consolidation | Summarization, entity extraction |
| Encoding | Memory write | Embedding generation + storage |
| Retrieval | Memory read | Similarity search + ranking |
| Forgetting | Memory decay/deletion | Exponential decay, explicit deletion |

**Tulving's Memory Classification (1972):**

Endel Tulving distinguished three types of long-term memory:

**1. Episodic Memory**: Memory of specific events and experiences with temporal and contextual markers.

$$
m_{\text{episodic}} = \langle \text{event}, \text{time}, \text{place}, \text{context}, \text{emotional\_valence} \rangle
$$

Agent analog: Logs of specific interactions, tool call sequences, error incidents, and their outcomes. "Last Tuesday, when I tried to deploy the model, the Docker build failed because of a missing dependency."

**2. Semantic Memory**: General world knowledge and facts, divorced from specific episodes.

$$
m_{\text{semantic}} = \langle \text{concept}, \text{properties}, \text{relations} \rangle
$$

Agent analog: Extracted facts, user preferences, domain knowledge. "The user prefers Python over Java." "API rate limit for service X is 100 requests/minute."

**3. Procedural Memory**: Knowledge of how to perform tasks—skills and procedures.

$$
m_{\text{procedural}} = \langle \text{task\_pattern}, \text{action\_sequence}, \text{success\_conditions} \rangle
$$

Agent analog: Successful action sequences, debugging strategies, workflow templates. "When deploying to production: first run tests, then build Docker image, then push to registry, then update Kubernetes deployment."

**Ebbinghaus Forgetting Curve (1885):**

Hermann Ebbinghaus quantified the exponential decay of memory retention over time:

$$
R(t) = e^{-t/S}
$$

where $R(t)$ is the retention probability at time $t$ after learning, and $S$ is the memory **strength** (inversely related to the decay rate $\lambda = 1/S$). This curve informs the design of temporal decay functions in agent memory systems.

**Spaced Repetition Effect:** Memories that are accessed repeatedly at increasing intervals develop stronger retention:

$$
S_{n+1} = S_n \cdot (1 + \delta \cdot f(\Delta t_n))
$$

where $S_n$ is the strength after the $n$-th review, $\Delta t_n$ is the interval between reviews, and $f$ is an increasing function rewarding longer successful intervals. In agent memory, this translates to **strengthening memories that are repeatedly retrieved across diverse queries**—increasing their importance score with each access.

**Levels of Processing Theory (Craik & Lockhart, 1972):**

Deeper semantic processing at encoding time leads to stronger, more durable memories. In agent memory:

- **Shallow encoding**: Store the raw text of a conversation turn verbatim. Fast but low retrieval quality.
- **Deep encoding**: Extract entities, relationships, sentiment, and implications before storage. Slower but dramatically better retrieval:

$$
\text{RetrievalQuality} \propto \text{ProcessingDepth}(\text{encoding})
$$

This motivates the use of **LLM-powered encoding** that semantically processes information before storage, rather than simply storing raw text.

---

## 7.2 Memory Taxonomy

### 7.2.1 Short-Term / Working Memory

Short-term (working) memory holds information that is **immediately relevant** to the current task or conversation. It is characterized by limited capacity, rapid access, and transient lifetime.

#### Conversation Buffer

The simplest form of working memory: store the complete conversation history and include it in every LLM call.

$$
\mathcal{M}_{\text{buffer}} = [\text{msg}_1, \text{msg}_2, \ldots, \text{msg}_t]
$$

**Token cost at turn $t$:**

$$
|\mathcal{M}_{\text{buffer}}(t)| = \sum_{i=1}^{t} |\text{msg}_i| \approx t \cdot \bar{\ell}
$$

where $\bar{\ell}$ is the average token count per message.

**Failure condition:** The buffer exceeds the context window:

$$
|\mathcal{M}_{\text{buffer}}(t)| + |C_{\text{system}}| + |C_{\text{tools}}| > C_{\max}
$$

For $\bar{\ell} = 200$ tokens and $C_{\max} = 128{,}000$ tokens, this occurs after approximately $640$ turns—seemingly large, but multi-agent systems with verbose tool outputs can reach this limit rapidly.

**Implementation:**

```python
class ConversationBuffer:
    def __init__(self, max_tokens: int):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add(self, message: Message):
        self.messages.append(message)
    
    def get_context(self) -> List[Message]:
        return self.messages
    
    def token_count(self) -> int:
        return sum(count_tokens(m.content) for m in self.messages)
    
    def is_overflowing(self, reserved_tokens: int) -> bool:
        return self.token_count() + reserved_tokens > self.max_tokens
```

#### Sliding Window Memory: Retaining Last $k$ Turns

To bound memory size, retain only the most recent $k$ turns:

$$
\mathcal{M}_{\text{window}}(t) = [\text{msg}_{t-k+1}, \text{msg}_{t-k+2}, \ldots, \text{msg}_t]
$$

**Token cost:** Bounded at $O(k \cdot \bar{\ell})$, independent of total conversation length $t$.

**Information loss:** All information from turns $1$ through $t - k$ is irrecoverably lost. This is acceptable when:
- The task is predominantly local (recent context is sufficient).
- Older information has been consolidated into long-term memory.

**Choosing $k$:** The optimal window size balances information retention against context cost:

$$
k^* = \arg\max_k \left[\text{TaskPerformance}(k) - \lambda \cdot \text{Cost}(k)\right]
$$

Empirically, $k = 10$–$20$ turns provides a good balance for most conversational agents. For code generation agents with long outputs, $k = 3$–$5$ may be necessary.

**Implementation with overlap:**

```python
class SlidingWindowMemory:
    def __init__(self, window_size: int):
        self.messages = deque(maxlen=window_size)
    
    def add(self, message: Message):
        self.messages.append(message)
    
    def get_context(self) -> List[Message]:
        return list(self.messages)
```

#### Token-Bounded Memory

A more flexible approach than fixed turn count: bound the total token count rather than the number of turns:

$$
|\mathcal{M}_{\text{working}}| \leq C_{\max}
$$

This accommodates variable-length messages—a single long tool output might consume the equivalent of 10 short conversational turns.

**Eviction policy when capacity is exceeded:**

1. **FIFO (First-In, First-Out)**: Remove oldest messages first.
2. **Importance-weighted FIFO**: Remove the oldest message with the lowest importance score.
3. **LRU (Least Recently Used)**: Remove messages that have been least recently referenced in reasoning.
4. **Summarize-and-evict**: Summarize the oldest messages before removing them, preserving key information in compressed form.

**Token-bounded implementation:**

```python
class TokenBoundedMemory:
    def __init__(self, max_tokens: int, eviction="summarize"):
        self.messages = []
        self.max_tokens = max_tokens
        self.eviction = eviction
    
    def add(self, message: Message):
        self.messages.append(message)
        while self._total_tokens() > self.max_tokens:
            self._evict()
    
    def _evict(self):
        if self.eviction == "fifo":
            self.messages.pop(0)
        elif self.eviction == "summarize":
            # Summarize oldest N messages into one
            n = min(5, len(self.messages) // 2)
            oldest = self.messages[:n]
            summary = self.llm.summarize(oldest)
            self.messages = [Message("system", summary)] + self.messages[n:]
    
    def _total_tokens(self) -> int:
        return sum(count_tokens(m.content) for m in self.messages)
```

#### Summary Memory: Compressing History

Summary memory maintains a **running summary** of the conversation that is updated incrementally as new messages arrive:

$$
\text{Summary}_t = \text{LLM}\left(\text{Summary}_{t-1}, \text{msg}_t, \text{"Update the summary with this new information"}\right)
$$

**Token cost:** $O(S)$ where $S$ is the fixed summary length (typically 500–2000 tokens), regardless of conversation length $t$.

**Information fidelity:**

Define the information retention rate:

$$
\rho(t) = \frac{I(\text{Summary}_t; \text{History}_{1:t})}{H(\text{History}_{1:t})}
$$

where $I$ is mutual information and $H$ is entropy. Perfect retention ($\rho = 1$) is impossible for $S \ll t \cdot \bar{\ell}$ by the data processing inequality. The practical goal is to maximize $\rho$ subject to the size constraint.

**Progressive summarization strategy:**

Rather than maintaining a single monolithic summary, use a hierarchical approach:

```
Level 0: Current turn (full detail, ~200 tokens)
Level 1: Last 5 turns (full detail, ~1000 tokens)
Level 2: Last 20 turns (summarized, ~500 tokens)
Level 3: Full history (highly compressed summary, ~300 tokens)
```

**Implementation:**

```python
class SummaryMemory:
    def __init__(self, llm, max_summary_tokens=1000):
        self.llm = llm
        self.summary = ""
        self.recent_messages = deque(maxlen=5)
        self.max_summary_tokens = max_summary_tokens
    
    def add(self, message: Message):
        self.recent_messages.append(message)
        
        # Update summary when recent buffer is full
        if len(self.recent_messages) == self.recent_messages.maxlen:
            self._consolidate()
    
    def _consolidate(self):
        recent_text = "\n".join(
            f"{m.role}: {m.content}" for m in self.recent_messages
        )
        self.summary = self.llm.invoke(f"""
            Current summary of conversation:
            {self.summary}
            
            New messages to incorporate:
            {recent_text}
            
            Produce an updated summary that captures all important 
            information. Keep it under {self.max_summary_tokens} tokens.
            Prioritize: key decisions, facts learned, pending tasks, 
            and user preferences.
        """)
    
    def get_context(self) -> str:
        recent = "\n".join(
            f"{m.role}: {m.content}" for m in self.recent_messages
        )
        return f"Conversation Summary:\n{self.summary}\n\nRecent:\n{recent}"
```

---

### 7.2.2 Long-Term Memory

Long-term memory persists **beyond the current conversation** and grows over the agent's entire operational lifetime. It is the agent's accumulated knowledge base.

#### Persistent Storage Across Sessions

Long-term memory enables agents to:

- Remember user preferences across days, weeks, or months.
- Accumulate domain knowledge from every interaction.
- Learn from past successes and failures.
- Maintain project context across multiple work sessions.

**Persistence requirements:**

$$
\text{Lifetime}(\mathcal{M}_{\text{LTM}}) \gg \text{Lifetime}(\text{session})
$$

**Storage durability guarantee:**

$$
P(\text{data loss} | \text{hardware failure}) < \epsilon_{\text{durability}}
$$

where $\epsilon_{\text{durability}} = 10^{-11}$ (eleven nines durability) for production systems, achieved through replication and checkpointing.

#### Episodic Memory: Specific Interaction Histories

Episodic memory stores **specific events and interactions** with full contextual detail—the "what happened, when, where, and with whom."

**Schema:**

$$
m_{\text{episodic}} = \langle \text{event\_id}, \text{timestamp}, \text{participants}, \text{actions}, \text{outcome}, \text{context}, \text{lessons\_learned} \rangle
$$

**Examples:**

```json
{
  "event_id": "ep_2024_001",
  "timestamp": "2024-11-15T14:30:00Z",
  "type": "task_execution",
  "description": "Attempted to deploy ML model v2.3 to production",
  "actions_taken": [
    "Ran test suite → all passed",
    "Built Docker image → success",
    "Pushed to registry → success",
    "Deployed to k8s → FAILED: OOM on startup"
  ],
  "outcome": "failure",
  "root_cause": "Model required 16GB RAM, pod limit was 8GB",
  "resolution": "Increased pod memory limit to 20GB, redeployed successfully",
  "lessons_learned": "Always check model memory footprint against pod limits before deploying"
}
```

**Retrieval patterns for episodic memory:**

1. **Temporal retrieval**: "What happened yesterday during deployment?"

$$
\mathcal{R}_{\text{temporal}}(t_{\text{start}}, t_{\text{end}}) = \{m_i : t_{\text{start}} \leq t_i \leq t_{\text{end}}\}
$$

2. **Similarity retrieval**: "Have we encountered a similar error before?"

$$
\mathcal{R}_{\text{similar}}(q) = \text{Top-}k\left(\text{sim}(\mathbf{e}_q, \mathbf{e}_{m_i})\right)
$$

3. **Outcome-based retrieval**: "What strategies worked for deployment failures?"

$$
\mathcal{R}_{\text{outcome}}(q, \text{outcome}=\text{success}) = \{m_i : \text{sim}(q, m_i) > \tau \wedge m_i.\text{outcome} = \text{success}\}
$$

#### Semantic Memory: Extracted Facts and Knowledge

Semantic memory stores **distilled knowledge**—facts, relationships, rules, and generalizations extracted from experiences, divorced from the specific episodes that produced them.

**Schema:**

$$
m_{\text{semantic}} = \langle \text{subject}, \text{predicate}, \text{object}, \text{confidence}, \text{source\_episodes}, \text{last\_verified} \rangle
$$

**Examples:**

```
(User, prefers, Python) [confidence: 0.95, verified: 2024-11-10]
(ProjectX, uses, PostgreSQL) [confidence: 0.90, verified: 2024-11-14]
(API_ServiceY, rate_limit, 100/min) [confidence: 0.85, verified: 2024-11-01]
(ModelV2, requires_ram, 16GB) [confidence: 1.0, verified: 2024-11-15]
```

**Semantic memory construction from episodes:**

$$
\mathcal{M}_{\text{semantic}} = \text{Extract}(\mathcal{M}_{\text{episodic}})
$$

The extraction function uses the LLM to identify generalizable facts from specific episodes:

```python
def extract_semantic_memories(episodic_memory: EpisodicMemory, llm) -> List[SemanticMemory]:
    recent_episodes = episodic_memory.get_recent(n=20)
    
    extraction_prompt = f"""
    From these interaction records, extract general facts, 
    user preferences, system configurations, and learned rules.
    
    Episodes:
    {format_episodes(recent_episodes)}
    
    For each fact, provide:
    - Subject, Predicate, Object (triple format)
    - Confidence (0-1)
    - Source episode IDs
    
    Only extract facts that are likely to be useful in future interactions.
    Do NOT extract ephemeral details.
    """
    
    raw_facts = llm.invoke(extraction_prompt)
    return parse_semantic_memories(raw_facts)
```

#### Procedural Memory: Learned Procedures and Strategies

Procedural memory stores **action patterns and strategies** that the agent has learned through experience—the "how to" knowledge.

**Schema:**

$$
m_{\text{procedural}} = \langle \text{task\_pattern}, \text{preconditions}, \text{action\_sequence}, \text{success\_rate}, \text{avg\_steps}, \text{last\_used} \rangle
$$

**Examples:**

```yaml
- task_pattern: "Deploy ML model to Kubernetes"
  preconditions:
    - "Tests pass"
    - "Docker image builds"
    - "Sufficient cluster resources"
  action_sequence:
    1. "Run full test suite"
    2. "Check model memory requirements"
    3. "Verify cluster has sufficient resources"
    4. "Build and push Docker image"
    5. "Apply Kubernetes deployment manifest"
    6. "Monitor pod startup for 60 seconds"
    7. "Run smoke tests against deployment"
  success_rate: 0.92
  times_used: 13
  last_used: "2024-11-15"
  failure_modes:
    - "OOM: increase pod memory limit"
    - "Image pull error: check registry credentials"
```

**Procedural memory retrieval and application:**

When the agent encounters a task similar to a stored procedure:

$$
P(\text{use\_stored\_procedure} | \text{sim}(q, m_{\text{proc}}) > \tau, \text{success\_rate}(m_{\text{proc}}) > \sigma) = \text{high}
$$

The agent retrieves the procedure, adapts it to the current context, and executes it—dramatically reducing planning time and improving success rates compared to planning from scratch.

**Procedural memory vs. few-shot examples:**

| Feature | Procedural Memory | Few-Shot Examples |
|---|---|---|
| Source | Agent's own experience | Developer-curated |
| Adaptivity | Evolves with experience | Static |
| Context-specificity | Tailored to agent's environment | Generic |
| Success rate tracking | Yes | No |
| Failure mode documentation | Yes | Rarely |

---

### 7.2.3 Sensory / Perceptual Memory

Sensory memory is the transient buffer that holds **raw input data** before it is processed and selectively encoded into working or long-term memory.

#### Raw Input Buffering

In agentic systems, sensory memory corresponds to:

- The raw user message before parsing.
- Tool output before extraction.
- API responses before filtering.
- Web page content before summarization.

**Buffer characteristics:**

$$
\mathcal{M}_{\text{sensory}} = \{(x_i, t_i, \text{modality}_i) : t - \Delta t_{\text{buffer}} \leq t_i \leq t\}
$$

where $\Delta t_{\text{buffer}}$ is the buffer duration (typically one interaction cycle). Sensory memory is **not persisted**—it exists only long enough for the agent to decide what to commit to working or long-term memory.

**Gating mechanism (what to attend to):**

Not all sensory input is worth processing. An attention gate filters raw inputs:

$$
\text{Attended} = \{x_i \in \mathcal{M}_{\text{sensory}} : \text{Relevance}(x_i, \text{current\_task}) > \tau_{\text{attend}}\}
$$

This mirrors the human attentional filter that prevents sensory overload.

#### Multi-Modal Memory (Images, Audio Embeddings)

For multimodal agents, sensory memory must handle diverse data types:

**Image memory:**

$$
\mathbf{e}_{\text{image}} = \text{VisionEncoder}(\text{image}) \in \mathbb{R}^{d_v}
$$

Store both the embedding (for retrieval) and a compressed representation of the image (for reconstruction). Vision-language models (e.g., CLIP) produce embeddings that live in a shared text-image space, enabling cross-modal retrieval:

$$
\text{sim}(\mathbf{e}_{\text{text\_query}}, \mathbf{e}_{\text{image}}) = \frac{\mathbf{e}_{\text{text}} \cdot \mathbf{e}_{\text{image}}}{\|\mathbf{e}_{\text{text}}\| \cdot \|\mathbf{e}_{\text{image}}\|}
$$

**Audio memory:**

$$
\mathbf{e}_{\text{audio}} = \text{AudioEncoder}(\text{audio\_segment}) \in \mathbb{R}^{d_a}
$$

Audio can be stored as: raw waveform (expensive), mel-spectrogram features (moderate), transcription text (cheap, lossy), or learned embeddings (compact, semantic).

**Multi-modal memory entry:**

```python
@dataclass
class MultiModalMemory:
    text_content: Optional[str]
    text_embedding: Optional[np.ndarray]
    image_data: Optional[bytes]
    image_embedding: Optional[np.ndarray]
    audio_transcription: Optional[str]
    audio_embedding: Optional[np.ndarray]
    modality: str  # "text", "image", "audio", "multimodal"
    timestamp: float
    metadata: dict
```

---

## 7.3 Memory Storage Backends

### 7.3.1 Vector Databases (Pinecone, Weaviate, Chroma, Qdrant)

Vector databases are purpose-built for storing and retrieving high-dimensional embedding vectors via **approximate nearest neighbor (ANN)** search.

**Core operation:**

Given a query embedding $\mathbf{e}_q \in \mathbb{R}^d$ and a collection of stored embeddings $\{\mathbf{e}_1, \ldots, \mathbf{e}_N\}$, find:

$$
\mathcal{R} = \text{Top-}k \left\{i : \text{sim}(\mathbf{e}_q, \mathbf{e}_i)\right\}
$$

**Distance/similarity metrics:**

| Metric | Formula | Properties |
|---|---|---|
| Cosine similarity | $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | Direction-based, $\in [-1, 1]$ |
| L2 (Euclidean) distance | $\|\mathbf{a} - \mathbf{b}\|_2$ | Magnitude-sensitive |
| Dot product | $\mathbf{a} \cdot \mathbf{b}$ | Requires normalized vectors for meaningful comparison |

**ANN indexing algorithms:**

| Algorithm | Time Complexity | Space Complexity | Accuracy | Use Case |
|---|---|---|---|---|
| **HNSW** (Hierarchical Navigable Small World) | $O(\log N)$ | $O(N \cdot M)$ | 95-99% recall | General purpose, production default |
| **IVF** (Inverted File Index) | $O(\sqrt{N})$ | $O(N)$ | 90-95% recall | Large-scale, memory-constrained |
| **PQ** (Product Quantization) | $O(N)$ | $O(N/r)$ compression ratio $r$ | 85-95% recall | Very large datasets, compressed |
| **Flat/Brute Force** | $O(N)$ | $O(N)$ | 100% recall | Small datasets ($< 10K$) |

**HNSW algorithm detail:**

HNSW constructs a multi-layer graph where:
- Layer 0 contains all vectors.
- Higher layers contain progressively fewer vectors (logarithmic sampling).
- Each layer is a navigable small-world graph with edges connecting nearby vectors.

Search proceeds top-down:

$$
\text{Start at top layer} \rightarrow \text{greedy search to nearest in layer} \rightarrow \text{descend to next layer} \rightarrow \cdots \rightarrow \text{fine search in layer 0}
$$

The construction parameter $M$ (maximum edges per node) and $\text{ef}_{\text{construction}}$ (beam width during construction) control the accuracy-speed trade-off:

$$
\text{Recall} \uparrow \text{ as } M \uparrow, \text{ef}_{\text{search}} \uparrow
$$

$$
\text{Speed} \downarrow \text{ as } M \uparrow, \text{ef}_{\text{search}} \uparrow
$$

**Vector database comparison:**

| Feature | Pinecone | Weaviate | Chroma | Qdrant |
|---|---|---|---|---|
| Deployment | Managed cloud | Self-hosted / cloud | In-process / server | Self-hosted / cloud |
| Index type | Proprietary | HNSW | HNSW (via hnswlib) | HNSW |
| Metadata filtering | Yes (server-side) | Yes (GraphQL) | Yes (basic) | Yes (payload filters) |
| Max vectors | Billions | Millions-billions | Millions | Billions |
| Hybrid search | Vector + keyword | Vector + BM25 | Vector only | Vector + keyword |
| Multitenancy | Native | Native | Collection-based | Collection-based |

**Implementation example (Chroma):**

```python
import chromadb
from sentence_transformers import SentenceTransformer

class VectorMemoryStore:
    def __init__(self, collection_name: str, embedding_model: str):
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.encoder = SentenceTransformer(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store(self, memory_id: str, content: str, metadata: dict):
        embedding = self.encoder.encode(content).tolist()
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: dict = None) -> List[dict]:
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter
        )
        return [
            {"id": id, "content": doc, "metadata": meta, "distance": dist}
            for id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0]
            )
        ]
```

---

### 7.3.2 Key-Value Stores

Key-value stores provide **exact-match retrieval** using explicit keys. They complement vector databases for memories that are best accessed by known identifiers rather than semantic similarity.

**Use cases in agent memory:**

| Key Pattern | Value | Example |
|---|---|---|
| `user:{user_id}:preferences` | JSON preferences object | `{"language": "python", "verbose": true}` |
| `session:{session_id}:summary` | Session summary text | "Discussed deployment strategy for project X..." |
| `fact:{entity}:{attribute}` | Fact value | `fact:ServiceY:rate_limit → "100/min"` |
| `procedure:{task_type}` | Procedure template | YAML action sequence |

**Advantages over vector databases:**

1. **Exact retrieval**: No approximation—$O(1)$ lookup guaranteed.
2. **Structured access**: Keys can encode hierarchy and relationships.
3. **Atomic updates**: Simple read-modify-write operations.
4. **Low latency**: Sub-millisecond retrieval.

**Implementations:** Redis, DynamoDB, etcd, LevelDB.

```python
class KeyValueMemory:
    def __init__(self, redis_client):
        self.store = redis_client
    
    def set_fact(self, entity: str, attribute: str, value: str,
                 confidence: float, ttl: int = None):
        key = f"fact:{entity}:{attribute}"
        data = json.dumps({
            "value": value,
            "confidence": confidence,
            "timestamp": time.time()
        })
        self.store.set(key, data, ex=ttl)
    
    def get_fact(self, entity: str, attribute: str) -> Optional[dict]:
        key = f"fact:{entity}:{attribute}"
        data = self.store.get(key)
        return json.loads(data) if data else None
    
    def get_all_facts_about(self, entity: str) -> dict:
        pattern = f"fact:{entity}:*"
        keys = self.store.keys(pattern)
        return {
            k.decode().split(":")[-1]: json.loads(self.store.get(k))
            for k in keys
        }
```

---

### 7.3.3 Graph Databases for Relational Memory

Graph databases store memories as **nodes and edges**, naturally representing entities and their relationships. This is particularly powerful for semantic memory where facts are inherently relational.

**Graph memory model:**

$$
\mathcal{G} = (V, E, \phi_V, \phi_E)
$$

where:
- $V$: Set of entity nodes (persons, objects, concepts).
- $E \subseteq V \times V$: Set of relationship edges.
- $\phi_V: V \rightarrow \mathbb{R}^{d_v}$: Node attribute function (properties, embeddings).
- $\phi_E: E \rightarrow \mathbb{R}^{d_e}$: Edge attribute function (relationship type, strength, timestamp).

**Advantages for agent memory:**

1. **Relational queries**: "What does the user know about Python?" traverses from `User` node through `knows` edges to `Python` node and its neighbors.
2. **Path-based reasoning**: Multi-hop queries like "Who manages the team that owns the failing service?" require graph traversal.
3. **Schema evolution**: New entity types and relationship types can be added without schema migration.
4. **Contextual retrieval**: Retrieving an entity brings along its neighborhood—providing rich context.

**Knowledge graph construction from agent experience:**

```python
class GraphMemory:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def add_entity(self, entity_id: str, entity_type: str, 
                   properties: dict):
        query = f"""
        MERGE (e:{entity_type} {{id: $id}})
        SET e += $properties
        SET e.updated_at = timestamp()
        """
        self.driver.execute_query(query, id=entity_id, 
                                  properties=properties)
    
    def add_relationship(self, from_id: str, to_id: str,
                        rel_type: str, properties: dict = None):
        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $properties
        SET r.updated_at = timestamp()
        """
        self.driver.execute_query(
            query, from_id=from_id, to_id=to_id,
            properties=properties or {}
        )
    
    def query_context(self, entity_id: str, depth: int = 2) -> dict:
        query = """
        MATCH path = (e {id: $id})-[*1..$depth]-(related)
        RETURN path
        """
        result = self.driver.execute_query(
            query, id=entity_id, depth=depth
        )
        return self._format_subgraph(result)
    
    def extract_and_store(self, text: str, llm):
        """Use LLM to extract entities and relationships, then store."""
        extraction = llm.invoke(f"""
            Extract entities and relationships from this text:
            {text}
            
            Output as JSON:
            {{
              "entities": [{{"id": "...", "type": "...", "properties": {{}}}}],
              "relationships": [{{"from": "...", "to": "...", "type": "...", "properties": {{}}}}]
            }}
        """)
        
        parsed = json.loads(extraction)
        for entity in parsed["entities"]:
            self.add_entity(entity["id"], entity["type"], 
                          entity.get("properties", {}))
        for rel in parsed["relationships"]:
            self.add_relationship(rel["from"], rel["to"], 
                                rel["type"], rel.get("properties", {}))
```

---

### 7.3.4 Relational Databases for Structured Memory

Relational databases (PostgreSQL, SQLite) store memories in **structured tables** with strong consistency guarantees, ACID transactions, and powerful SQL querying.

**Use cases:**

| Table | Columns | Purpose |
|---|---|---|
| `interactions` | id, timestamp, user_id, query, response, tokens_used, success | Interaction logging |
| `user_preferences` | user_id, key, value, confidence, updated_at | Preference tracking |
| `task_history` | task_id, description, status, start_time, end_time, steps | Task execution audit |
| `tool_usage` | tool_name, call_count, avg_latency, error_rate, last_used | Tool performance metrics |

**Advantages:**

1. **Complex queries**: SQL enables arbitrary filtering, joining, aggregating:

```sql
SELECT tool_name, AVG(latency_ms), COUNT(*) 
FROM tool_usage 
WHERE timestamp > NOW() - INTERVAL '7 days' 
GROUP BY tool_name 
HAVING error_rate > 0.1
ORDER BY error_rate DESC;
```

2. **Transactional safety**: ACID guarantees prevent data corruption during concurrent access.
3. **Analytics**: Agent performance metrics, usage patterns, and error trends.

**Integration with vector search (pgvector):**

PostgreSQL with the pgvector extension enables **hybrid storage**: relational data + vector embeddings in a single database:

```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    memory_type VARCHAR(20),
    importance FLOAT,
    created_at TIMESTAMP,
    accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    metadata JSONB
);

-- Similarity search with metadata filtering
SELECT id, content, importance, 
       1 - (embedding <=> $query_embedding) AS similarity
FROM memories
WHERE memory_type = 'semantic'
  AND importance > 0.5
  AND created_at > NOW() - INTERVAL '30 days'
ORDER BY embedding <=> $query_embedding
LIMIT 10;
```

---

### 7.3.5 Hybrid Storage Architectures

Production agent memory systems typically combine multiple storage backends, each optimized for different access patterns:

**Recommended hybrid architecture:**

```
                    ┌─────────────────────────────┐
                    │      MEMORY MANAGER          │
                    │   (Unified Interface)        │
                    └──────┬──────┬───────┬────────┘
                           │      │       │
                    ┌──────┴──┐ ┌─┴───┐ ┌─┴────────┐
                    │ Vector  │ │ KV   │ │ Graph    │
                    │ DB      │ │Store │ │ DB       │
                    │(Qdrant) │ │(Redis)│ │(Neo4j)  │
                    └─────────┘ └──────┘ └──────────┘
                    Semantic     Exact     Relational
                    search      lookup    traversal
                           │      │       │
                    ┌──────┴──────┴───────┴────────┐
                    │     PostgreSQL (pgvector)      │
                    │  (Structured data + vectors    │
                    │   + transaction logs)           │
                    └────────────────────────────────┘
                    Persistent storage, analytics,
                    audit trail
```

**Router logic:**

```python
class HybridMemoryManager:
    def __init__(self, vector_store, kv_store, graph_store, sql_store):
        self.vector = vector_store
        self.kv = kv_store
        self.graph = graph_store
        self.sql = sql_store
    
    def retrieve(self, query: str, context: dict) -> List[Memory]:
        results = []
        
        # 1. Check KV store for exact matches (user prefs, known facts)
        if context.get("user_id"):
            prefs = self.kv.get_all_facts_about(
                f"user:{context['user_id']}"
            )
            results.extend(prefs)
        
        # 2. Vector similarity search for semantic retrieval
        semantic_results = self.vector.retrieve(query, top_k=10)
        results.extend(semantic_results)
        
        # 3. Graph traversal for relational context
        entities = extract_entities(query)
        for entity in entities:
            subgraph = self.graph.query_context(entity, depth=2)
            results.extend(subgraph)
        
        # 4. SQL query for structured data
        if needs_structured_data(query):
            sql_results = self.sql.query(
                build_sql_query(query, context)
            )
            results.extend(sql_results)
        
        # Deduplicate and rank
        results = deduplicate(results)
        results = rank_by_relevance(results, query)
        
        return results[:self.max_results]
```

---

## 7.4 Memory Operations

### 7.4.1 Memory Write (Encoding)

The write operation commits new information from the agent's experience into memory storage. This is not a trivial dump—it involves **deciding what to memorize**, **extracting structured information**, and **generating appropriate representations**.

#### What to Memorize: Importance Scoring

Not all information encountered by an agent is worth storing. The agent must evaluate the **importance** of each piece of information:

$$
\text{importance}(x) = P(\text{useful in future} | x, \text{context})
$$

**Importance scoring criteria:**

| Criterion | Description | Example |
|---|---|---|
| **Novelty** | Is this information new? | A new user preference not previously recorded |
| **Contradiction** | Does this contradict existing memory? | User changed their preferred language |
| **Task relevance** | Is this relevant to ongoing or future tasks? | A discovered API endpoint |
| **Emotional/priority signal** | Does the user emphasize this? | "This is really important: always use HTTPS" |
| **Generalizability** | Can this be applied to future situations? | A debugging strategy that worked |
| **Specificity** | Is this specific enough to be actionable? | An exact configuration value vs. "it was slow" |

**LLM-based importance scoring:**

```python
def score_importance(information: str, context: str, llm) -> float:
    prompt = f"""
    Rate the importance of storing this information for future 
    interactions on a scale of 0.0 to 1.0.
    
    Information: {information}
    Current context: {context}
    
    Consider:
    - Will this be useful in future interactions? (0.3 weight)
    - Is this a new fact not already known? (0.2 weight)
    - Is this a user preference or constraint? (0.2 weight)
    - Does this represent a learned strategy? (0.15 weight)
    - Is this specific and actionable? (0.15 weight)
    
    Output a single float between 0.0 and 1.0.
    """
    score = float(llm.invoke(prompt).strip())
    return max(0.0, min(1.0, score))
```

**Importance threshold:**

$$
\text{Store}(x) \iff \text{importance}(x) > \tau_{\text{store}}
$$

Setting $\tau_{\text{store}}$ involves a precision-recall trade-off:
- $\tau_{\text{store}} = 0.3$ (low): Stores most information. High recall, potential noise, larger storage.
- $\tau_{\text{store}} = 0.7$ (high): Only stores critical information. High precision, risk of missing useful information, smaller storage.

Empirically, $\tau_{\text{store}} \in [0.4, 0.6]$ balances well for most applications.

#### Information Extraction from Conversations

Raw conversation text is rarely the optimal storage format. Structured extraction produces more retrievable memories:

**Extraction pipeline:**

$$
\text{Raw text} \xrightarrow{\text{Entity extraction}} \text{Entities} \xrightarrow{\text{Relation extraction}} \text{Triples} \xrightarrow{\text{Fact normalization}} \text{Canonical facts}
$$

```python
def extract_memories_from_conversation(messages: List[Message], 
                                        llm) -> List[Memory]:
    conversation_text = format_messages(messages)
    
    extraction = llm.invoke(f"""
    Extract all memorable information from this conversation.
    
    Conversation:
    {conversation_text}
    
    For each piece of information, output:
    {{
      "type": "episodic|semantic|procedural",
      "content": "The actual information",
      "key_entities": ["entity1", "entity2"],
      "importance": 0.0-1.0,
      "category": "preference|fact|procedure|decision|lesson"
    }}
    
    Rules:
    - Extract user preferences explicitly stated or implied
    - Extract factual information about systems, projects, people
    - Extract successful procedures or strategies
    - Extract decisions and their rationale
    - Do NOT extract trivial conversational filler
    - Do NOT extract information that changes frequently (e.g., "it's Tuesday")
    """)
    
    return parse_memories(extraction)
```

#### Embedding Generation

For vector-based storage and retrieval, each memory must be encoded as a dense vector:

$$
\mathbf{e} = \text{Encoder}(m) \in \mathbb{R}^d
$$

**Embedding model selection criteria:**

| Model | Dimensions | MTEB Score | Speed | Use Case |
|---|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | 64.6 | API-bound | Production, high quality |
| `text-embedding-3-small` (OpenAI) | 1536 | 62.3 | API-bound | Production, cost-sensitive |
| `all-MiniLM-L6-v2` (SBERT) | 384 | 56.3 | Very fast | Local, real-time |
| `bge-large-en-v1.5` (BAAI) | 1024 | 64.2 | Moderate | Self-hosted, high quality |
| `nomic-embed-text-v1.5` | 768 | 62.2 | Fast | Local, good balance |
| `voyage-3` (Voyage AI) | 1024 | 67.0+ | API-bound | SoTA retrieval quality |

**Embedding quality impacts retrieval quality:**

$$
\text{Retrieval Recall@}k \propto \text{EmbeddingQuality}(\text{Encoder})
$$

Higher-quality embeddings produce better separation in vector space between semantically related and unrelated memories, directly improving retrieval precision.

**Chunking strategy for long documents:**

When storing long documents, they must be chunked before embedding:

$$
\text{Document} \rightarrow [\text{chunk}_1, \text{chunk}_2, \ldots, \text{chunk}_n]
$$

**Chunking methods:**

| Method | Description | Pros | Cons |
|---|---|---|---|
| Fixed-size | Split every $N$ tokens | Simple, predictable | May split mid-sentence |
| Sentence-based | Split at sentence boundaries | Preserves meaning units | Variable chunk size |
| Paragraph-based | Split at paragraph boundaries | Natural boundaries | Potentially large chunks |
| Recursive | Split by headers → paragraphs → sentences → tokens | Hierarchical, semantic | Complex implementation |
| Semantic | Split when embedding similarity between adjacent segments drops below threshold | Optimal semantic coherence | Computationally expensive |

**Optimal chunk size** depends on the retrieval task:

$$
\text{ChunkSize}^* = \arg\max_s \text{RetrievalF1}(s)
$$

Empirically, 256–512 tokens performs well for factual retrieval; 512–1024 tokens for procedural and contextual retrieval.

---

### 7.4.2 Memory Read (Retrieval)

Memory retrieval is the process of finding and returning the most relevant memories for a given query at a given time.

#### Similarity-Based Retrieval

The foundational retrieval mechanism uses embedding similarity:

$$
\mathcal{M}_{\text{retrieved}} = \text{Top-}k\left(\text{sim}(\mathbf{e}_q, \mathbf{e}_{m_i})\right)
$$

**Cosine similarity** is the default metric:

$$
\text{sim}_{\cos}(\mathbf{e}_q, \mathbf{e}_{m_i}) = \frac{\mathbf{e}_q^T \mathbf{e}_{m_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{m_i}\|} \in [-1, 1]
$$

**Maximum Inner Product Search (MIPS)** is an alternative when embeddings are not normalized:

$$
\text{sim}_{\text{MIPS}}(\mathbf{e}_q, \mathbf{e}_{m_i}) = \mathbf{e}_q^T \mathbf{e}_{m_i}
$$

**Limitations of pure similarity retrieval:**

1. **Recency blindness**: A highly similar but outdated memory may be retrieved over a less similar but current one.
2. **Popularity bias**: Frequently stored topics dominate, drowning out rare but important memories.
3. **Query sensitivity**: Minor rephrasing of the query can significantly change retrieval results due to embedding space geometry.

#### Recency-Weighted Retrieval

Incorporate temporal recency to prefer recent memories:

$$
\text{recency}(m_i, t) = e^{-\lambda(t - t_i)}
$$

where $t_i = \max(t_{\text{created}_i}, t_{\text{accessed}_i})$ (the more recent of creation and last access times).

**Decay rate $\lambda$ selection:**

| $\lambda$ | Half-life | Interpretation |
|---|---|---|
| 0.01 | ~69 time units | Very slow decay; long memory |
| 0.1 | ~6.9 time units | Moderate decay |
| 0.5 | ~1.4 time units | Rapid decay; strong recency bias |
| 1.0 | ~0.69 time units | Very rapid decay; almost only recent |

The half-life $t_{1/2} = \frac{\ln 2}{\lambda}$ gives the time after which a memory's recency score drops to 50%.

#### Importance-Weighted Retrieval

Incorporate the intrinsic importance score assigned during encoding:

$$
\text{importance}(m_i) \in [0, 1]
$$

High-importance memories (e.g., user safety constraints, critical system configurations) should be retrievable even when they are old and not highly similar to the current query.

#### Combined Scoring

The composite retrieval score combines all factors:

$$
s(m_i) = \alpha \cdot \text{sim}(q, m_i) + \beta \cdot \text{recency}(m_i) + \gamma \cdot \text{importance}(m_i)
$$

where $\alpha + \beta + \gamma = 1$ and the weights are hyperparameters tuned for the application.

**Generalized Park et al. (2023) scoring (from "Generative Agents"):**

$$
s(m_i) = \alpha \cdot \text{normalize}(\text{sim}(q, m_i)) + \beta \cdot \text{normalize}(\text{recency}(m_i, t)) + \gamma \cdot \text{normalize}(\text{importance}(m_i))
$$

where normalization maps each component to $[0, 1]$ using min-max scaling over the candidate set:

$$
\text{normalize}(x) = \frac{x - \min_j x_j}{\max_j x_j - \min_j x_j + \epsilon}
$$

**Context-dependent weight adjustment:**

The weights $\alpha, \beta, \gamma$ can be dynamically adjusted based on the query type:

$$
(\alpha, \beta, \gamma) = \begin{cases}
(0.7, 0.1, 0.2) & \text{if query is fact-seeking (favor similarity)} \\
(0.3, 0.5, 0.2) & \text{if query is "what did we just discuss?" (favor recency)} \\
(0.2, 0.1, 0.7) & \text{if query is safety-critical (favor importance)}
\end{cases}
$$

**Full retrieval implementation:**

```python
class MemoryRetriever:
    def __init__(self, vector_store, alpha=0.5, beta=0.3, gamma=0.2,
                 decay_rate=0.1):
        self.vector_store = vector_store
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_rate = decay_rate
    
    def retrieve(self, query: str, current_time: float,
                 top_k: int = 10, candidate_k: int = 50) -> List[Memory]:
        # Step 1: Get candidates via similarity search (broad net)
        candidates = self.vector_store.retrieve(query, top_k=candidate_k)
        
        # Step 2: Compute composite scores
        scored = []
        for mem in candidates:
            sim_score = 1 - mem["distance"]  # Convert distance to sim
            recency_score = math.exp(
                -self.decay_rate * (current_time - mem["metadata"]["timestamp"])
            )
            importance_score = mem["metadata"].get("importance", 0.5)
            
            composite = (self.alpha * self._normalize(sim_score, candidates, "sim") +
                        self.beta * self._normalize(recency_score, candidates, "rec") +
                        self.gamma * self._normalize(importance_score, candidates, "imp"))
            
            scored.append((mem, composite))
        
        # Step 3: Sort and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Update access metadata for retrieved memories
        for mem, score in scored[:top_k]:
            self._update_access(mem["id"])
        
        return [mem for mem, _ in scored[:top_k]]
    
    def _normalize(self, value, candidates, field):
        values = [self._get_field(c, field) for c in candidates]
        min_v, max_v = min(values), max(values)
        if max_v - min_v < 1e-8:
            return 0.5
        return (value - min_v) / (max_v - min_v)
```

---

### 7.4.3 Memory Update (Modification)

Memories must be updated when new information supersedes or refines existing stored knowledge.

#### Contradiction Resolution

When new information contradicts existing memory, the system must decide which to keep:

$$
\text{If } m_{\text{new}} \text{ contradicts } m_{\text{old}}: \quad \text{Resolve}(m_{\text{new}}, m_{\text{old}})
$$

**Resolution strategies:**

**1. Recency Wins (Last-Write-Wins):**

$$
\text{Result} = m_{\text{new}} \quad \text{(always prefer newer information)}
$$

Simple but may lose valid older information if the new information is incorrect.

**2. Confidence-Based:**

$$
\text{Result} = \begin{cases}
m_{\text{new}} & \text{if } \text{conf}(m_{\text{new}}) > \text{conf}(m_{\text{old}}) \\
m_{\text{old}} & \text{otherwise}
\end{cases}
$$

**3. Source-Authority-Based:**

$$
\text{Result} = \arg\max_{m \in \{m_{\text{new}}, m_{\text{old}}\}} \text{SourceAuthority}(m.\text{source})
$$

where $\text{SourceAuthority}(\text{user\_explicit}) > \text{SourceAuthority}(\text{agent\_inference}) > \text{SourceAuthority}(\text{tool\_output})$.

**4. LLM-Mediated Resolution:**

```python
def resolve_contradiction(old_memory, new_info, llm):
    resolution = llm.invoke(f"""
    Existing memory: {old_memory.content}
    (Stored: {old_memory.timestamp}, Confidence: {old_memory.confidence})
    
    New information: {new_info}
    
    These appear to contradict each other. Determine:
    1. Is this a genuine contradiction or compatible information?
    2. Which is more likely correct and why?
    3. Should the old memory be:
       a) Replaced entirely with the new information
       b) Updated to incorporate both (with nuance)
       c) Kept as-is (new information is unreliable)
    
    Output your decision and the resulting memory content.
    """)
    return parse_resolution(resolution)
```

#### Fact Updating

For factual memories, updates follow a **read-modify-write** pattern:

$$
v_{\text{updated}} = f_{\text{update}}(v_{\text{old}}, \Delta v)
$$

**Update types:**

| Update Type | Operation | Example |
|---|---|---|
| **Overwrite** | $v \leftarrow v_{\text{new}}$ | User changed email address |
| **Append** | $v \leftarrow v \cup v_{\text{new}}$ | User mentioned a new project |
| **Increment** | $v \leftarrow v + \Delta$ | Tool usage count increased |
| **Conditional** | $v \leftarrow v_{\text{new}}$ if condition | Update only if confidence improved |
| **Merge** | $v \leftarrow \text{LLM.merge}(v, v_{\text{new}})$ | Combine old and new descriptions |

**Version history for critical memories:**

For high-importance memories, maintain a version history to enable rollback:

```python
class VersionedMemory:
    def __init__(self, memory_id: str):
        self.id = memory_id
        self.versions = []  # [(timestamp, content, metadata)]
    
    def update(self, new_content: str, metadata: dict):
        self.versions.append((time.time(), new_content, metadata))
    
    @property
    def current(self):
        return self.versions[-1] if self.versions else None
    
    def rollback(self, n_versions: int = 1):
        if len(self.versions) > n_versions:
            self.versions = self.versions[:-n_versions]
```

---

### 7.4.4 Memory Delete (Forgetting)

Forgetting is essential for memory hygiene—without it, memory stores grow unboundedly, retrieval quality degrades, and outdated information pollutes the agent's context.

#### Decay Functions

Inspired by the Ebbinghaus forgetting curve, memories weaken over time unless reinforced:

$$
\text{strength}(m, t) = e^{-\lambda(t - t_0)}
$$

where $t_0 = \max(t_{\text{created}}, t_{\text{last\_accessed}})$.

**Generalized decay with reinforcement:**

Each time a memory is accessed, its strength is reinforced:

$$
\text{strength}(m, t) = \sum_{j=1}^{n_{\text{access}}} w_j \cdot e^{-\lambda(t - t_j)}
$$

where $t_j$ is the $j$-th access time and $w_j$ is the reinforcement weight. Under spaced repetition, memories accessed at well-spaced intervals develop greater aggregate strength.

**Power-law decay (alternative):**

Some empirical evidence suggests human forgetting follows a power law rather than exponential:

$$
\text{strength}(m, t) = (1 + \lambda(t - t_0))^{-\beta}
$$

Power-law decay is slower than exponential for large $(t - t_0)$, meaning old memories persist longer—arguably more realistic for important facts.

**Garbage collection trigger:**

$$
\text{Delete}(m_i) \iff \text{strength}(m_i, t) < \tau_{\text{forget}} \wedge \text{importance}(m_i) < \tau_{\text{protect}}
$$

Memories below the forget threshold **and** below the importance protection threshold are candidates for deletion. High-importance memories are never automatically deleted regardless of decay.

#### Explicit Deletion Triggers

Certain events should trigger immediate deletion:

1. **User request**: "Forget everything about my medical history."
2. **Contradiction resolution**: Old, proven-wrong facts should be deleted.
3. **Task completion**: Temporary task-specific memory can be cleaned up.
4. **Data expiration**: Memories with TTL (time-to-live) expire automatically.

#### Privacy-Driven Forgetting

Regulatory requirements (GDPR, CCPA) mandate the ability to delete user data:

$$
\text{RightToErasure}(\text{user}) \implies \forall m_i : m_i.\text{user\_id} = \text{user} \implies \text{Delete}(m_i)
$$

**Implementation requirements:**

1. **Complete deletion**: Remove from all storage backends (vector store, KV store, graph, SQL).
2. **Embedding deletion**: Remove embedding vectors—unlike text, embeddings may encode information that cannot be partially redacted.
3. **Derivative deletion**: Information derived from deleted memories must also be removed (e.g., summaries that incorporated the deleted information).
4. **Audit trail**: Maintain a record that deletion occurred (without storing the deleted content).

```python
class PrivacyAwareForgetting:
    def forget_user(self, user_id: str):
        # 1. Find all memories associated with user
        memories = self.sql.query(
            "SELECT id FROM memories WHERE user_id = %s", user_id
        )
        
        # 2. Find derived memories (summaries, extracted facts)
        derived = self.sql.query(
            "SELECT id FROM memories WHERE source_memory_id IN (%s)",
            [m.id for m in memories]
        )
        
        all_to_delete = memories + derived
        
        # 3. Delete from all backends
        for mem in all_to_delete:
            self.vector_store.delete(mem.id)
            self.kv_store.delete(f"memory:{mem.id}")
            self.graph_store.delete_node(mem.id)
            self.sql.execute(
                "DELETE FROM memories WHERE id = %s", mem.id
            )
        
        # 4. Record deletion event (without content)
        self.sql.execute(
            "INSERT INTO deletion_log (user_id, count, timestamp) "
            "VALUES (%s, %s, NOW())",
            user_id, len(all_to_delete)
        )
```

---

## 7.5 Memory Consolidation and Compression

### 7.5.1 Summarization-Based Compression

As memory stores grow, raw episodic memories consume increasing storage and degrade retrieval quality. Summarization compresses multiple memories into distilled representations.

**Consolidation function:**

$$
\mathcal{M}_{\text{consolidated}} = \text{Summarize}(\{m_1, m_2, \ldots, m_n\}) \quad \text{where } |\mathcal{M}_{\text{consolidated}}| \ll \sum_i |m_i|
$$

**Compression ratio:**

$$
r = \frac{|\text{original memories}|}{|\text{consolidated memory}|}
$$

Typical compression ratios: $r = 5$–$20\times$ for episodic-to-semantic consolidation.

**Strategies:**

**1. Periodic batch summarization:**

Every $N$ interactions, summarize the accumulated episodic memories:

```python
def consolidate_periodic(episodic_store, llm, batch_size=50):
    recent = episodic_store.get_unconsolidated(limit=batch_size)
    
    if len(recent) < batch_size:
        return  # Not enough to consolidate
    
    summary = llm.invoke(f"""
    Summarize these {len(recent)} interaction records into a 
    concise knowledge summary. Preserve:
    - Key decisions and their rationale
    - Learned facts and user preferences
    - Successful strategies and failure lessons
    - Important entities and relationships
    
    Records:
    {format_records(recent)}
    """)
    
    # Store consolidated summary
    semantic_store.add(summary, type="consolidated_summary",
                       source_ids=[r.id for r in recent])
    
    # Mark originals as consolidated (but don't delete immediately)
    for record in recent:
        record.consolidated = True
        episodic_store.update(record)
```

**2. Incremental summarization (running summary):**

Update a running summary with each new memory (cf. Summary Memory in 7.2.1):

$$
\text{Summary}_{t} = \text{LLM}(\text{Summary}_{t-1}, m_t)
$$

**3. Cluster-then-summarize:**

Group semantically similar memories, then summarize each cluster:

$$
\text{Clusters} = \text{KMeans}(\{\mathbf{e}_{m_i}\}, K)
$$

$$
\text{Summary}_k = \text{LLM}(\{m_i : \text{cluster}(m_i) = k\}) \quad \forall k \in [K]
$$

This produces $K$ focused summaries, each covering a distinct topic.

---

### 7.5.2 Entity Extraction and Knowledge Graph Construction

Extract structured entities and relationships from unstructured memories and store them in a knowledge graph—transforming episodic memories into semantic memory.

**Extraction pipeline:**

$$
\text{Episodic Memories} \xrightarrow{\text{NER + RE}} \text{Entity-Relation Triples} \xrightarrow{\text{Dedup + Normalize}} \text{Knowledge Graph}
$$

**Entity-Relation extraction prompt:**

```python
def extract_knowledge_graph(memories: List[str], llm) -> KnowledgeGraph:
    extraction = llm.invoke(f"""
    From these memory records, extract a knowledge graph.
    
    Records:
    {format_memories(memories)}
    
    Output format:
    ENTITIES:
    - (entity_id, entity_type, description)
    
    RELATIONSHIPS:
    - (source_entity, relationship_type, target_entity, confidence)
    
    Entity types: Person, Project, Tool, Concept, Organization, 
                  Preference, Configuration
    Relationship types: uses, prefers, manages, depends_on, 
                        is_part_of, knows, created, configured_as
    
    Rules:
    - Merge entities that refer to the same real-world object
    - Assign confidence 0.0-1.0 based on how strongly supported 
      the relationship is
    - Include timestamps where relevant
    """)
    
    return parse_knowledge_graph(extraction)
```

**Graph merging:**

When new triples overlap with existing graph nodes, merge operations must handle:

1. **Node deduplication**: Different surface forms for the same entity ("Python 3", "Python3", "Python 3.x" → single node).
2. **Relationship strengthening**: If the same relationship is extracted multiple times, increase its confidence:

$$
\text{conf}_{\text{new}} = 1 - (1 - \text{conf}_{\text{old}}) \cdot (1 - \text{conf}_{\text{evidence}})
$$

3. **Contradiction detection**: If a new relationship contradicts an existing one (e.g., "User prefers Python" vs. "User prefers Rust"), flag for resolution.

---

### 7.5.3 Hierarchical Memory: Detail Levels

Organize memories at multiple levels of abstraction, analogous to how humans remember events at varying granularity:

**Hierarchy:**

```
Level 3 (Abstract):   "The user is a data scientist who prefers Python"
                       ↑ (consolidation)
Level 2 (Summary):    "Over 15 interactions, the user consistently 
                       chose Python for data tasks, mentioned pandas 
                       and sklearn frequently"
                       ↑ (consolidation)
Level 1 (Detailed):   "2024-11-10: User asked to analyze CSV data 
                       using Python pandas"
                       "2024-11-11: User wrote sklearn pipeline"
                       "2024-11-12: User preferred Python over R 
                       for visualization"
                       ↑ (raw)
Level 0 (Raw):        Full conversation transcripts
```

**Retrieval at appropriate granularity:**

$$
\text{Level} = f(\text{query\_specificity}, \text{available\_context\_budget})
$$

- High-specificity query ("What pandas function did I use on Nov 10?") → Level 0/1.
- Medium-specificity query ("What tools does the user prefer for data analysis?") → Level 2.
- Low-specificity query ("Tell me about this user") → Level 3.

**Implementation:**

```python
class HierarchicalMemory:
    def __init__(self, levels=4):
        self.levels = [MemoryStore(level=i) for i in range(levels)]
    
    def consolidate_upward(self, from_level: int):
        """Consolidate memories from level i to level i+1."""
        if from_level >= len(self.levels) - 1:
            return
        
        memories = self.levels[from_level].get_unconsolidated()
        if len(memories) < self.consolidation_threshold:
            return
        
        summary = self.llm.summarize(
            memories, 
            target_level=from_level + 1
        )
        self.levels[from_level + 1].add(summary)
        
        for mem in memories:
            mem.consolidated = True
            self.levels[from_level].update(mem)
    
    def retrieve(self, query: str, budget_tokens: int) -> List[Memory]:
        results = []
        remaining_budget = budget_tokens
        
        # Start from highest level (cheapest) and drill down
        for level in reversed(range(len(self.levels))):
            level_results = self.levels[level].retrieve(
                query, 
                max_tokens=remaining_budget // (level + 1)
            )
            results.extend(level_results)
            remaining_budget -= sum(
                count_tokens(r.content) for r in level_results
            )
            
            if remaining_budget <= 0:
                break
        
        return results
```

---

### 7.5.4 Memory Merging and Deduplication

Over time, memory stores accumulate duplicates and near-duplicates that waste storage and pollute retrieval results.

**Deduplication strategies:**

**1. Exact deduplication**: Hash-based:

$$
\text{Duplicate}(m_i, m_j) \iff \text{hash}(m_i.\text{content}) = \text{hash}(m_j.\text{content})
$$

**2. Near-duplicate detection**: Embedding-based:

$$
\text{NearDuplicate}(m_i, m_j) \iff \text{sim}(\mathbf{e}_{m_i}, \mathbf{e}_{m_j}) > \tau_{\text{dedup}}
$$

where $\tau_{\text{dedup}} \in [0.90, 0.98]$ (high threshold to avoid false positives).

**3. Semantic merging**: When near-duplicates are found, merge rather than delete:

$$
m_{\text{merged}} = \text{LLM}(\text{"Merge these memories into one comprehensive entry: "} m_i, m_j)
$$

The merged memory inherits:
- The higher importance score: $\text{importance}_{\text{merged}} = \max(\text{importance}_i, \text{importance}_j)$.
- The combined access count: $n_{\text{merged}} = n_i + n_j$.
- The earlier creation time: $t_{\text{merged}} = \min(t_i, t_j)$.

**Deduplication algorithm:**

```python
def deduplicate_memory_store(store, threshold=0.95):
    all_memories = store.get_all()
    embeddings = np.array([m.embedding for m in all_memories])
    
    # Compute pairwise similarities efficiently
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find near-duplicate pairs
    duplicates = set()
    merge_groups = []
    
    for i in range(len(all_memories)):
        if i in duplicates:
            continue
        group = [i]
        for j in range(i + 1, len(all_memories)):
            if j not in duplicates and similarity_matrix[i][j] > threshold:
                group.append(j)
                duplicates.add(j)
        if len(group) > 1:
            merge_groups.append(group)
    
    # Merge each group
    for group in merge_groups:
        memories = [all_memories[idx] for idx in group]
        merged = merge_memories(memories, llm)
        store.add(merged)
        for mem in memories:
            store.delete(mem.id)
```

---

## 7.6 Context Window Management

### 7.6.1 Context Packing Strategies

The assembled context for each LLM call must be carefully packed to maximize information density within the token budget while maintaining comprehension.

**Context budget allocation:**

$$
C_{\max} = C_{\text{system}} + C_{\text{memory}} + C_{\text{tools}} + C_{\text{conversation}} + C_{\text{output\_reserve}}
$$

**Solving for memory budget:**

$$
C_{\text{memory}} = C_{\max} - C_{\text{system}} - C_{\text{tools}} - C_{\text{conversation}} - C_{\text{output\_reserve}}
$$

**Example allocation for 128K context window:**

| Component | Tokens | Percentage |
|---|---|---|
| System prompt | 2,000 | 1.6% |
| Retrieved memories | 8,000 | 6.3% |
| Tool descriptions | 3,000 | 2.3% |
| Current conversation | 110,000 | 85.9% |
| Output reserve | 5,000 | 3.9% |
| **Total** | **128,000** | **100%** |

**Packing optimization:**

Given a set of retrieved memories $\{m_1, \ldots, m_K\}$ with relevance scores $\{\alpha_1, \ldots, \alpha_K\}$ and token costs $\{c_1, \ldots, c_K\}$, and a memory token budget $B$, the packing problem is a variant of the **0-1 knapsack problem**:

$$
\max \sum_{i=1}^{K} \alpha_i \cdot x_i \quad \text{subject to} \quad \sum_{i=1}^{K} c_i \cdot x_i \leq B, \quad x_i \in \{0, 1\}
$$

This is NP-hard in general, but for typical $K$ (10–50 memories), dynamic programming or greedy approximation is efficient:

**Greedy algorithm (2-approximation):**

```python
def pack_memories(memories, budget):
    # Sort by relevance/token ratio (density)
    sorted_mems = sorted(
        memories,
        key=lambda m: m.relevance / max(m.token_count, 1),
        reverse=True
    )
    
    packed = []
    remaining = budget
    
    for mem in sorted_mems:
        if mem.token_count <= remaining:
            packed.append(mem)
            remaining -= mem.token_count
    
    return packed
```

---

### 7.6.2 Attention Sink and Positional Encoding Considerations

The placement of memories within the context affects how effectively the LLM attends to them.

**Attention sink phenomenon:**

Transformer models exhibit an **attention sink** where the first few tokens of the input receive disproportionately high attention, regardless of their semantic content. This is because the softmax operation needs to distribute attention somewhere, and the initial tokens serve as default attention targets:

$$
\text{attn}(i, j) = \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_l \exp(q_i^T k_l / \sqrt{d})}
$$

The first tokens $j \approx 0$ accumulate attention mass as a "sink."

**Implications for memory placement:**

1. **Beginning of context**: Receives high attention. Place system instructions and critical memory here.
2. **End of context (recent conversation)**: Also receives high attention due to recency effects in training. Place the current query and most relevant memories here.
3. **Middle of context**: Receives the least attention ("lost in the middle"). Avoid placing critical information here.

**Optimal memory ordering:**

$$
C = [\underbrace{C_{\text{system}}}_{\text{Position 0: high attention}}, \underbrace{C_{\text{memory\_high\_importance}}}_{\text{Early}}, \underbrace{C_{\text{memory\_moderate}}}_{\text{Middle (lower attention)}}, \underbrace{C_{\text{tools}}}_{\text{Before conversation}}, \underbrace{C_{\text{conversation}}}_{\text{End: high attention}}]
$$

**Positional encoding impact:**

For models using RoPE (Rotary Position Embeddings), the effective attention between positions $i$ and $j$ decays with relative distance:

$$
\text{RoPE attn}(i, j) \propto g(i - j)
$$

where $g$ is a decreasing function of distance. This means memories placed **closer to the query** receive stronger attention.

**Practical recommendation:** Place the most relevant retrieved memories **immediately before the user's query** to maximize attention:

```
[System prompt]
[Tool descriptions]
[Relevant memories]      ← Close to query
[Recent conversation]    ← Close to query
[User's current query]   ← Generates response
```

---

### 7.6.3 Dynamic Context Assembly

The context is **dynamically assembled** for each LLM call based on the current query, task state, and available information:

$$
C_{\text{assembled}} = [C_{\text{system}}, C_{\text{memory}}, C_{\text{tools}}, C_{\text{user}}]
$$

**Assembly pipeline:**

```python
class DynamicContextAssembler:
    def __init__(self, memory_manager, tool_registry, max_tokens):
        self.memory = memory_manager
        self.tools = tool_registry
        self.max_tokens = max_tokens
    
    def assemble(self, system_prompt: str, conversation: List[Message],
                 current_query: str) -> List[Message]:
        # Calculate fixed costs
        system_tokens = count_tokens(system_prompt)
        conversation_tokens = sum(
            count_tokens(m.content) for m in conversation
        )
        output_reserve = 4096  # Reserve for response
        
        # Available budget for memory and tools
        available = (self.max_tokens - system_tokens - 
                    conversation_tokens - output_reserve)
        
        # Allocate between memory and tools
        tool_budget = min(available * 0.3, 3000)
        memory_budget = available - tool_budget
        
        # Retrieve relevant memories
        memories = self.memory.retrieve(
            query=current_query,
            max_tokens=int(memory_budget)
        )
        
        # Select relevant tools
        tools = self.tools.select_relevant(
            query=current_query,
            max_tokens=int(tool_budget)
        )
        
        # Assemble context
        context = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=f"Relevant memories:\n{format_memories(memories)}"),
            SystemMessage(content=f"Available tools:\n{format_tools(tools)}"),
            *conversation,
            HumanMessage(content=current_query)
        ]
        
        # Verify total doesn't exceed limit
        total = sum(count_tokens(m.content) for m in context)
        assert total + output_reserve <= self.max_tokens, \
            f"Context overflow: {total} + {output_reserve} > {self.max_tokens}"
        
        return context
```

---

### 7.6.4 Context Priority Ordering

When the total desired context exceeds $C_{\max}$, a priority system determines what gets included and what is dropped:

**Priority levels:**

$$
\text{Priority}(C_i) \in \{P_{\text{critical}}, P_{\text{high}}, P_{\text{medium}}, P_{\text{low}}, P_{\text{optional}}\}
$$

| Priority | Content Type | Drop Policy |
|---|---|---|
| $P_{\text{critical}}$ | System instructions, safety constraints | Never drop |
| $P_{\text{high}}$ | Current query, most recent turn | Never drop |
| $P_{\text{high}}$ | Top-1 retrieved memory | Drop only if space impossible |
| $P_{\text{medium}}$ | Recent conversation (last 3 turns) | Summarize if needed |
| $P_{\text{medium}}$ | Top 2-5 retrieved memories | Drop by relevance |
| $P_{\text{low}}$ | Tool descriptions for less relevant tools | Drop least relevant |
| $P_{\text{low}}$ | Older conversation turns | Summarize aggressively |
| $P_{\text{optional}}$ | Additional context, examples | Drop first |

**Cascading truncation algorithm:**

```python
def truncate_context(context_blocks, max_tokens):
    """Cascading truncation from lowest to highest priority."""
    
    # Sort blocks by priority (lowest first)
    sorted_blocks = sorted(context_blocks, key=lambda b: b.priority)
    
    total = sum(b.token_count for b in sorted_blocks)
    
    if total <= max_tokens:
        return sorted_blocks  # Everything fits
    
    # Drop from lowest priority upward
    for block in sorted_blocks:
        if total <= max_tokens:
            break
        
        if block.priority == Priority.CRITICAL:
            continue  # Never drop critical blocks
        
        if block.droppable:
            total -= block.token_count
            block.included = False
        elif block.summarizable:
            old_tokens = block.token_count
            block.content = summarize(block.content, 
                                      target_tokens=old_tokens // 3)
            total -= (old_tokens - block.token_count)
    
    return [b for b in sorted_blocks if b.included]
```

---

## 7.7 Evaluation of Memory Systems

### 7.7.1 Retrieval Precision and Recall

Memory retrieval quality is measured using standard information retrieval metrics adapted for the agent memory context.

**Precision@$k$:**

$$
\text{Precision@}k = \frac{|\{\text{relevant memories}\} \cap \{\text{retrieved top-}k\}|}{k}
$$

The fraction of retrieved memories that are actually relevant to the query.

**Recall@$k$:**

$$
\text{Recall@}k = \frac{|\{\text{relevant memories}\} \cap \{\text{retrieved top-}k\}|}{|\{\text{relevant memories}\}|}
$$

The fraction of all relevant memories that were successfully retrieved.

**F1@$k$:**

$$
\text{F1@}k = 2 \cdot \frac{\text{Precision@}k \cdot \text{Recall@}k}{\text{Precision@}k + \text{Recall@}k}
$$

**Mean Reciprocal Rank (MRR):**

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

where $\text{rank}_i$ is the position of the first relevant memory in the retrieval results for query $i$.

**Normalized Discounted Cumulative Gain (nDCG@$k$):**

For graded relevance (memories can be partially relevant):

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
$$

$$
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

where $\text{IDCG@}k$ is the ideal DCG (achieved by ranking all memories by true relevance).

**Evaluation protocol:**

```python
def evaluate_memory_retrieval(memory_system, test_queries):
    metrics = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    
    for query, ground_truth_relevant_ids in test_queries:
        retrieved = memory_system.retrieve(query, top_k=10)
        retrieved_ids = [m.id for m in retrieved]
        
        # Precision@10
        relevant_retrieved = set(retrieved_ids) & set(ground_truth_relevant_ids)
        precision = len(relevant_retrieved) / len(retrieved_ids)
        recall = len(relevant_retrieved) / len(ground_truth_relevant_ids)
        
        # MRR
        rr = 0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in ground_truth_relevant_ids:
                rr = 1.0 / rank
                break
        
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["mrr"].append(rr)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

### 7.7.2 Memory Staleness Detection

Memory staleness occurs when stored information is **no longer accurate** due to changes in the real world, user preferences, or system configurations.

**Staleness score:**

$$
\text{staleness}(m_i, t) = 1 - e^{-\mu(t - t_{\text{verified}_i})}
$$

where $t_{\text{verified}_i}$ is the last time memory $m_i$ was verified as accurate, and $\mu$ is the staleness rate.

**Staleness risk factors:**

| Factor | Staleness Rate | Example |
|---|---|---|
| **Volatile facts** (prices, status) | High ($\mu > 1.0$) | "The API endpoint is currently healthy" |
| **Preferences** | Medium ($\mu \approx 0.1$) | "User prefers dark mode" |
| **Procedures** | Low-medium ($\mu \approx 0.05$) | "Deploy by running `make deploy`" |
| **Identity facts** | Very low ($\mu \approx 0.001$) | "User's name is Alice" |
| **Physical constants** | Zero ($\mu = 0$) | "$\pi \approx 3.14159$" |

**Proactive staleness management:**

```python
class StalenessDetector:
    def __init__(self, staleness_rates: Dict[str, float]):
        self.rates = staleness_rates
    
    def get_stale_memories(self, memory_store, threshold=0.7):
        all_memories = memory_store.get_all()
        stale = []
        
        for mem in all_memories:
            rate = self.rates.get(mem.type, 0.1)
            staleness = 1 - math.exp(
                -rate * (time.time() - mem.last_verified)
            )
            if staleness > threshold:
                stale.append((mem, staleness))
        
        return sorted(stale, key=lambda x: x[1], reverse=True)
    
    def schedule_verification(self, memory_store, llm):
        stale_memories = self.get_stale_memories(memory_store)
        
        for mem, staleness_score in stale_memories[:10]:
            verification = llm.invoke(f"""
                Memory: {mem.content}
                Stored: {mem.timestamp}
                Staleness risk: {staleness_score:.2f}
                
                Is this information likely still accurate?
                If not, what is the correct current information?
            """)
            
            if verification.still_accurate:
                mem.last_verified = time.time()
            else:
                mem.content = verification.corrected_content
                mem.last_verified = time.time()
            
            memory_store.update(mem)
```

---

### 7.7.3 Task Performance with vs. without Memory

The ultimate evaluation of a memory system is its **impact on task performance**. This is measured through ablation studies comparing agent performance with and without memory:

**Experimental design:**

$$
\Delta_{\text{memory}} = \text{Performance}(\text{Agent} + \mathcal{M}) - \text{Performance}(\text{Agent} - \mathcal{M})
$$

**Performance dimensions:**

| Metric | With Memory | Without Memory | Expected $\Delta$ |
|---|---|---|---|
| Task success rate | Higher | Lower | +10-30% |
| Steps to completion | Fewer | More | -20-40% |
| User satisfaction | Higher | Lower | +15-25% |
| Repeated errors | Fewer | More | -50-80% |
| Personalization quality | High | None | +100% |
| Token consumption | Lower (cached) | Higher (re-derive) | -30-50% |

**Controlled ablation protocol:**

```python
def memory_ablation_study(agent, tasks, memory_configs):
    """
    memory_configs: [
        "no_memory",
        "short_term_only",
        "long_term_semantic",
        "long_term_episodic",
        "full_memory"
    ]
    """
    results = {}
    
    for config in memory_configs:
        agent.set_memory_config(config)
        config_results = []
        
        for task in tasks:
            agent.reset()
            outcome = agent.execute(task)
            config_results.append({
                "success": outcome.success,
                "steps": outcome.steps,
                "tokens_used": outcome.tokens,
                "time_seconds": outcome.duration,
                "quality_score": evaluate_quality(outcome, task)
            })
        
        results[config] = aggregate_results(config_results)
    
    # Statistical significance testing
    for config in memory_configs[1:]:
        p_value = ttest_ind(
            [r["quality_score"] for r in results["full_memory"]],
            [r["quality_score"] for r in results[config]]
        ).pvalue
        print(f"full_memory vs {config}: p={p_value:.4f}")
    
    return results
```

---

### 7.7.4 Scalability Under Growing Memory Size

As the memory store grows over time, system performance must degrade gracefully.

**Scalability dimensions:**

**1. Retrieval latency vs. memory size:**

$$
L_{\text{retrieval}}(N) = \begin{cases}
O(\log N) & \text{HNSW (typical)} \\
O(\sqrt{N}) & \text{IVF} \\
O(N) & \text{Brute force}
\end{cases}
$$

**Target:** Retrieval latency should remain under 100ms for $N \leq 10^7$ memories.

**2. Retrieval quality vs. memory size:**

As $N$ grows, more irrelevant memories compete with relevant ones, potentially degrading precision:

$$
\text{Precision@}k(N) \approx \frac{R}{R + f(N)}
$$

where $R$ is the number of truly relevant memories (approximately constant for a given query) and $f(N)$ is the number of false positives that score above the relevance threshold. For well-designed embeddings, $f(N)$ grows sub-linearly with $N$.

**3. Storage cost vs. memory size:**

$$
\text{StorageCost}(N) = N \cdot (d \cdot \text{sizeof}(\text{float}) + \bar{c}_{\text{metadata}})
$$

For $d = 1536$ dimensions, float32: each vector costs $1536 \times 4 = 6{,}144$ bytes $\approx 6$ KB. For 10 million memories: $\sim 60$ GB for vectors alone.

**Scalability testing protocol:**

```python
def scalability_test(memory_system, test_queries, 
                     memory_sizes=[1e3, 1e4, 1e5, 1e6, 1e7]):
    results = {}
    
    for target_size in memory_sizes:
        # Populate memory to target size
        memory_system.populate(target_size)
        
        latencies = []
        precisions = []
        
        for query, relevant_ids in test_queries:
            start = time.perf_counter()
            retrieved = memory_system.retrieve(query, top_k=10)
            latency = time.perf_counter() - start
            
            precision = compute_precision(retrieved, relevant_ids)
            
            latencies.append(latency)
            precisions.append(precision)
        
        results[target_size] = {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "mean_precision": np.mean(precisions),
            "storage_mb": memory_system.storage_size_mb()
        }
    
    return results
```

**Expected results table:**

| Memory Size $N$ | Mean Latency (ms) | P99 Latency (ms) | Precision@10 | Storage (GB) |
|---|---|---|---|---|
| $10^3$ | 2 | 5 | 0.92 | 0.006 |
| $10^4$ | 5 | 12 | 0.90 | 0.06 |
| $10^5$ | 12 | 30 | 0.87 | 0.6 |
| $10^6$ | 25 | 65 | 0.83 | 6.0 |
| $10^7$ | 55 | 150 | 0.78 | 60.0 |

**Mitigation strategies for large-scale memory:**

1. **Tiered storage**: Hot memories (recent, high-importance) in fast in-memory index; cold memories in disk-based index.
2. **Index partitioning**: Partition memory by namespace (user, project, domain) and search only relevant partitions.
3. **Periodic pruning**: Remove low-importance, high-staleness memories.
4. **Hierarchical consolidation**: Aggressively consolidate older memories into higher-level summaries.
5. **Pre-filtering**: Use metadata filters before vector search to reduce the candidate set:

$$
N_{\text{effective}} = N \cdot P(\text{passes metadata filter}) \ll N
$$

---

**Chapter Summary — Key Takeaways:**

1. Agent memory is a structured computational module $\mathcal{M}: (q, t) \rightarrow \{(k_i, v_i, \alpha_i)\}$ that extends LLMs beyond their finite context windows, enabling temporal coherence, experiential learning, and personalization.

2. Memory taxonomy mirrors cognitive science: working memory (bounded, transient), long-term memory (episodic, semantic, procedural), and sensory memory (raw buffering). Each type serves distinct functions with different storage characteristics.

3. Storage backends must be selected based on access patterns: vector databases for semantic retrieval, key-value stores for exact lookups, graph databases for relational queries, and relational databases for structured analytics. Production systems use hybrid architectures.

4. Memory operations (write, read, update, delete) each involve non-trivial design decisions: importance scoring for writes, composite relevance ranking for reads, contradiction resolution for updates, and decay-based or privacy-driven forgetting for deletes.

5. Consolidation through summarization, entity extraction, hierarchical abstraction, and deduplication is essential for maintaining memory quality and retrieval efficiency as stores grow.

6. Context window management—including budget allocation, attention-aware placement, dynamic assembly, and priority-based truncation—is the critical interface between memory systems and LLM inference.

7. Evaluation requires multi-dimensional assessment: retrieval quality (precision, recall, nDCG), staleness detection, task performance ablation, and scalability under growing memory size. The ultimate metric is downstream task success improvement attributable to the memory system.