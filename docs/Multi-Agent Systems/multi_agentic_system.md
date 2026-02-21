

# Chapter 6: Multi-Agent Systems

---

## 6.1 Definition and Formal Framework

### 6.1.1 What are Multi-Agent Systems (MAS) in Agentic AI

A Multi-Agent System (MAS) is a computational architecture comprising multiple autonomous agents that interact, communicate, coordinate, and potentially compete within a shared or distributed environment to collectively solve problems that exceed the capabilities, knowledge, or reliability of any single agent operating in isolation. In the context of agentic AI powered by large language models, MAS represents the transition from **monolithic intelligence** to **distributed cognitive architectures** where specialized LLM-backed agents collaborate through structured protocols.

**The fundamental principle** underpinning MAS design is the **cognitive division of labor**: complex tasks are decomposed not merely into subtasks (as in single-agent planning) but into **cognitive roles**—each instantiated as a distinct agent with specialized instructions, tools, memory, and behavioral constraints.

Formally, a multi-agent system differs from a single-agent system along five orthogonal axes:

| Axis | Single-Agent | Multi-Agent |
|---|---|---|
| **Decision-making** | Centralized in one entity | Distributed across $n$ entities |
| **Knowledge** | Unified context window | Distributed, potentially asymmetric |
| **Action space** | One agent's tool set | Union (or partition) of tool sets |
| **State observability** | Agent observes all it can | Each agent may observe different subsets |
| **Communication** | Internal (within context) | External (inter-agent messages) |

**Why multi-agent over single-agent?** The justification is not merely engineering convenience but stems from fundamental computational and cognitive arguments:

1. **Context window limitations**: A single LLM has a finite context window $C_{\max}$. For tasks requiring information volume $I > C_{\max}$, distributing information across agents allows the collective system to operate over effectively unbounded context:

$$
I_{\text{effective}} = \sum_{i=1}^{n} C_{\max}^{(i)} \gg C_{\max}
$$

2. **Specialization advantage**: An agent prompted with a narrow specialist persona (e.g., "You are an expert security auditor") consistently outperforms a generalist agent on domain-specific subtasks. This follows from the **conditional capability** of LLMs:

$$
P(\text{correct} | \text{specialist prompt}, \text{domain task}) > P(\text{correct} | \text{generalist prompt}, \text{domain task})
$$

3. **Error reduction via redundancy**: Multiple agents independently solving the same problem and aggregating results reduce error rates. If each agent has error probability $\epsilon < 0.5$ and agents are independent, a majority vote among $n$ agents yields error probability:

$$
P(\text{majority wrong}) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} \epsilon^k (1-\epsilon)^{n-k} \xrightarrow{n \to \infty} 0
$$

4. **Adversarial robustness**: Dedicated critic or red-team agents can identify flaws that a single self-critiquing agent overlooks due to **self-consistency bias**—the tendency for a single LLM to rate its own outputs favorably.

5. **Parallelism**: Independent subtasks can be executed concurrently by different agents, reducing wall-clock time from $O(n)$ (sequential) to $O(1)$ (fully parallel) for $n$ independent subtasks.

---

### 6.1.2 Formal MAS Definition

A Multi-Agent System is formally defined as the tuple:

$$
\mathcal{M} = \langle \{A_1, \ldots, A_n\}, \mathcal{E}, \mathcal{P}, \mathcal{C} \rangle
$$

**Component-by-component specification:**

---

**Component 1: Agent Set $\{A_1, \ldots, A_n\}$**

Each agent $A_i$ is itself a tuple:

$$
A_i = \langle \text{id}_i, \theta_i, \mathcal{T}_i, \mathcal{O}_i, \pi_i, M_i, R_i \rangle
$$

where:

- $\text{id}_i$: Unique identifier for the agent.
- $\theta_i$: The agent's underlying LLM parameters (all agents may share the same base LLM, differentiated only by prompts, or use distinct models).
- $\mathcal{T}_i \subseteq \mathcal{T}_{\text{global}}$: The tool set accessible to agent $i$. Tool partitioning enforces specialization:

$$
\mathcal{T}_i \cap \mathcal{T}_j = \emptyset \quad \text{(strict specialization, no tool overlap)}
$$

$$
\mathcal{T}_i \cap \mathcal{T}_j \neq \emptyset \quad \text{(shared capabilities with overlap)}
$$

- $\mathcal{O}_i \subseteq \mathcal{O}_{\text{global}}$: The observation space accessible to agent $i$. In partially observable multi-agent systems:

$$
\mathcal{O}_i \neq \mathcal{O}_j \quad \text{(asymmetric information)}
$$

- $\pi_i$: The agent's policy, which maps its local observation and message history to actions:

$$
\pi_i: \mathcal{O}_i \times \mathcal{H}_i^{\text{msg}} \rightarrow \mathcal{A}_i \cup \mathcal{C}_{\text{out}}
$$

where $\mathcal{A}_i$ is the action space and $\mathcal{C}_{\text{out}}$ is the set of outgoing messages.

- $M_i$: The agent's memory (working memory, episodic memory, shared memory references).
- $R_i$: The agent's role specification—a natural language description that constrains behavior (the system prompt).

**Agent heterogeneity:** In LLM-based MAS, heterogeneity is achieved through three mechanisms:

$$
\text{Heterogeneity}(A_i, A_j) = \underbrace{d(\theta_i, \theta_j)}_{\text{model difference}} + \underbrace{d(R_i, R_j)}_{\text{role difference}} + \underbrace{d(\mathcal{T}_i, \mathcal{T}_j)}_{\text{tool difference}}
$$

Even when $\theta_i = \theta_j$ (same base model), distinct roles $R_i \neq R_j$ and tools $\mathcal{T}_i \neq \mathcal{T}_j$ create functionally different agents.

---

**Component 2: Shared Environment $\mathcal{E}$**

The environment $\mathcal{E}$ is the external world with which agents interact:

$$
\mathcal{E} = \langle S, T_{\text{env}}, O_{\text{env}}, R_{\text{env}} \rangle
$$

- $S$: Environment state space.
- $T_{\text{env}}: S \times \mathcal{A}_1 \times \cdots \times \mathcal{A}_n \rightarrow \Delta(S)$: Joint transition function depending on **all** agents' actions (agents' actions may interact):

$$
s_{t+1} \sim T_{\text{env}}(s_t, a_t^{(1)}, a_t^{(2)}, \ldots, a_t^{(n)})
$$

- $O_{\text{env}}: S \times \{1, \ldots, n\} \rightarrow \mathcal{O}_i$: Observation function that may provide different observations to different agents.
- $R_{\text{env}}: S \times \mathcal{A}^n \rightarrow \mathbb{R}^n$: Reward function that assigns potentially different rewards to different agents.

In many LLM-agent scenarios, the environment includes: file systems, databases, web browsers, APIs, code execution sandboxes, or other software systems.

---

**Component 3: Protocol $\mathcal{P}$**

The protocol defines the **rules of interaction**—when agents can act, in what order, and under what constraints:

$$
\mathcal{P} = \langle \Sigma, \delta, \sigma_0, F, \text{Turn} \rangle
$$

- $\Sigma$: Set of protocol states (e.g., "planning phase," "execution phase," "review phase").
- $\delta: \Sigma \times \mathcal{E}_{\text{event}} \rightarrow \Sigma$: Protocol transition function.
- $\sigma_0 \in \Sigma$: Initial protocol state.
- $F \subseteq \Sigma$: Terminal protocol states (task complete, failure, timeout).
- $\text{Turn}: \Sigma \rightarrow 2^{\{A_1, \ldots, A_n\}}$: Specifies which agents may act in each protocol state.

**Protocol types:**

| Protocol Type | Turn Function | Example |
|---|---|---|
| **Round-robin** | $\text{Turn}(\sigma_t) = \{A_{(t \mod n) + 1}\}$ | Sequential debate |
| **Free-form** | $\text{Turn}(\sigma_t) = \{A_1, \ldots, A_n\}$ | Asynchronous collaboration |
| **Token-based** | $\text{Turn}(\sigma_t) = \{A_i : \text{hasToken}(A_i)\}$ | Controlled access |
| **Priority-based** | $\text{Turn}(\sigma_t) = \{\arg\max_i \text{Priority}(A_i, \sigma_t)\}$ | Hierarchical systems |

---

**Component 4: Communication Channel $\mathcal{C}$**

The communication channel defines how agents exchange information:

$$
\mathcal{C} = \langle \mathcal{L}, \mathcal{G}_{\text{topology}}, \text{Capacity}, \text{Reliability} \rangle
$$

- $\mathcal{L}$: Communication language (natural language messages, structured JSON, function calls, shared memory writes).
- $\mathcal{G}_{\text{topology}} = (V, E)$: Communication graph where $V = \{A_1, \ldots, A_n\}$ and edge $(A_i, A_j) \in E$ means $A_i$ can send messages to $A_j$.
- $\text{Capacity}$: Maximum message length (in tokens) per communication round.
- $\text{Reliability}$: Probability that a sent message is received correctly (typically 1.0 in software systems, but message loss/corruption can model system failures).

**Communication topology types:**

```
Fully Connected:        Star (Hub-and-Spoke):     Ring:
 A₁ ─── A₂              A₁   A₂   A₃           A₁ → A₂
 │  ╲╱  │                 ╲   │   ╱              ↑     ↓
 │  ╱╲  │                  ╲  │  ╱               A₄ ← A₃
 A₃ ─── A₄                  Hub

Hierarchical:           Bipartite:
    Manager              Planners ──── Executors
   ╱   │   ╲             P₁ ──── E₁
  W₁   W₂   W₃          P₂ ──── E₂
 ╱ ╲                     P₃ ──── E₃
W₁₁ W₁₂
```

**Formal message passing:**

A message $m$ from agent $A_i$ to agent $A_j$ at time $t$ is:

$$
m = \langle \text{sender}: A_i, \; \text{receiver}: A_j, \; \text{type}: \tau, \; \text{content}: c, \; \text{timestamp}: t \rangle
$$

where $\tau \in \{\text{inform}, \text{request}, \text{propose}, \text{accept}, \text{reject}, \text{delegate}, \text{report}\}$ follows speech act theory.

---

### 6.1.3 Single-Agent vs. Multi-Agent: When and Why

The decision between single-agent and multi-agent architectures is a fundamental design choice with deep implications. This decision should be based on rigorous analysis of task characteristics, not architectural preference.

**Decision Framework:**

Define the task complexity vector:

$$
\mathbf{v}_{\text{task}} = \langle D, H, K, P, R, F \rangle
$$

where:
- $D$: Diversity of required expertise domains
- $H$: Task horizon (number of steps)
- $K$: Knowledge volume required ($K / C_{\max}$ ratio)
- $P$: Parallelizability (fraction of independent subtasks)
- $R$: Reliability requirement (target success rate)
- $F$: Failure tolerance (cost of incorrect outputs)

**Decision criteria:**

$$
\text{Use MAS if:} \quad w_D \cdot D + w_H \cdot H + w_K \cdot \frac{K}{C_{\max}} + w_P \cdot P + w_R \cdot R + w_F \cdot F > \tau_{\text{MAS}}
$$

| Factor | Favors Single-Agent | Favors Multi-Agent |
|---|---|---|
| **Expertise diversity** $D$ | Task needs one domain | Task spans $\geq 3$ distinct domains |
| **Horizon** $H$ | Short ($< 10$ steps) | Long ($> 20$ steps with diverse step types) |
| **Knowledge volume** $K$ | Fits in one context window | Exceeds single context window |
| **Parallelizability** $P$ | Purely sequential dependencies | $> 40\%$ independent subtasks |
| **Reliability** $R$ | Moderate ($< 90\%$) acceptable | High ($> 99\%$) required |
| **Failure cost** $F$ | Low (easy to retry) | High (irreversible or expensive) |

**Quantitative analysis of the accuracy argument:**

If a single agent has per-step accuracy $p$ and the task has $H$ dependent steps, overall success probability is:

$$
P_{\text{single}}(\text{success}) = p^H
$$

For $p = 0.95$ and $H = 20$: $P_{\text{single}} = 0.95^{20} \approx 0.358$.

With multi-agent verification (a second agent checks each step, detecting errors with probability $q$):

$$
P_{\text{MAS}}(\text{success}) = (p + (1-p) \cdot q)^H
$$

For $q = 0.8$: $P_{\text{MAS}} = (0.95 + 0.05 \cdot 0.8)^{20} = 0.99^{20} \approx 0.818$.

This demonstrates the **multiplicative benefit** of multi-agent verification for long-horizon tasks.

**When single-agent is strictly superior:**

1. **Simple tasks**: When $H \leq 3$ and $D = 1$, the overhead of inter-agent communication exceeds the benefit.
2. **Tight latency budgets**: When response time $< 2$ seconds, the communication overhead of MAS is prohibitive.
3. **Strong coherence requirements**: When the output must be a single unified document with perfectly consistent style, voice, and narrative flow.

$$
\text{Cost}_{\text{MAS}} = \text{Cost}_{\text{compute}} + \text{Cost}_{\text{communication}} + \text{Cost}_{\text{coordination}}
$$

MAS is justified only when:

$$
\text{Benefit}_{\text{MAS}} = \text{Quality}_{\text{MAS}} - \text{Quality}_{\text{single}} > \text{Cost}_{\text{MAS}} - \text{Cost}_{\text{single}}
$$

---

### 6.1.4 Emergent Behavior in Multi-Agent Systems

**Emergence** occurs when the collective behavior of a multi-agent system exhibits properties that were not explicitly programmed into any individual agent. Formally:

$$
\text{Emergent Property } \phi: \quad \phi(\mathcal{M}) = \text{True} \quad \text{but} \quad \forall i: \phi(A_i) = \text{False}
$$

The emergent property $\phi$ exists at the system level but not at the individual agent level.

**Categories of emergence in LLM-based MAS:**

**1. Constructive Emergence (Positive)**

Multiple agents produce solutions that none could individually:

- **Creative synthesis**: A "scientist" agent proposes a hypothesis, a "statistician" agent identifies methodological flaws, a "domain expert" agent adds nuance—the resulting research plan is more rigorous than any agent's solo output.
- **Error correction chains**: Agent $A$ makes an error, agent $B$ identifies and corrects it, agent $A$ confirms the correction—achieving accuracy neither achieves alone.

**2. Emergent Communication Protocols**

When agents are given broad communication freedom, they may develop **implicit conventions**:

- Standardized formatting for requests vs. reports.
- Development of shorthand references to previously established concepts.
- Implicit priority signaling through message urgency markers.

**3. Social Loafing (Negative Emergence)**

In some MAS configurations, agents exhibit **diffusion of responsibility**—each agent produces lower-quality outputs assuming other agents will catch errors:

$$
\text{Quality}(A_i | \text{solo}) > \text{Quality}(A_i | \text{MAS with } n \text{ agents})
$$

This mirrors the psychological phenomenon in human teams and can be mitigated through explicit accountability mechanisms (each agent's contribution is individually evaluated).

**4. Emergent Specialization**

Even when agents start with identical capabilities, interaction dynamics can drive functional specialization. Agent $A_1$ may, through the conversation trajectory, naturally assume a leadership role while $A_2$ becomes the implementer—even without explicit role assignment.

**5. Cascade Failures (Negative Emergence)**

An error by one agent propagates through the system, amplified by each subsequent agent:

$$
\text{Error}_{t+1} = f(\text{Error}_t) \quad \text{where } f \text{ is expansive: } \|f(x)\| > \|x\| \text{ for } \|x\| > 0
$$

This necessitates **error containment** mechanisms: checkpoints, independent verification, and circuit breakers.

**Formal analysis via information-theoretic emergence:**

The degree of emergence can be quantified using **mutual information** between agents' individual behaviors and the system's collective behavior:

$$
\text{Emergence}(\mathcal{M}) = I(Y_{\text{system}}; A_1, \ldots, A_n) - \sum_{i=1}^{n} I(Y_{\text{system}}; A_i)
$$

where $Y_{\text{system}}$ is the system-level output. Positive values indicate synergistic interactions (super-additive information contribution); negative values indicate redundancy.

---

## 6.2 Multi-Agent Architectures

### 6.2.1 Centralized Orchestration (Hub-and-Spoke)

In centralized orchestration, a single **orchestrator agent** (the hub) receives all tasks, decomposes them, dispatches subtasks to **specialist agents** (the spokes), collects their results, and synthesizes the final output.

#### Single Orchestrator Dispatching to Specialist Agents

**Architecture:**

```
                    ┌──────────────────┐
                    │   ORCHESTRATOR   │
                    │   (Hub Agent)    │
                    │                  │
                    │ • Task analysis  │
                    │ • Decomposition  │
                    │ • Dispatch       │
                    │ • Aggregation    │
                    │ • Quality ctrl   │
                    └────┬───┬───┬─────┘
                    ╱    │       │    ╲
                   ╱     │       │     ╲
         ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
         │Agent S₁│ │Agent S₂│ │Agent S₃│ │Agent S₄│
         │(Coder) │ │(Writer)│ │(Tester)│ │(Review)│
         └────────┘ └────────┘ └────────┘ └────────┘
```

**Formal specification:**

The orchestrator implements a function:

$$
\text{Orchestrator}: (T, s_0) \rightarrow \langle (A_{k_1}, T_1), (A_{k_2}, T_2), \ldots, (A_{k_m}, T_m) \rangle
$$

mapping the task $T$ and initial state $s_0$ to a sequence of (agent, subtask) assignments.

**Orchestrator algorithm:**

```python
class CentralizedOrchestrator:
    def __init__(self, llm, specialist_agents: Dict[str, Agent]):
        self.llm = llm
        self.specialists = specialist_agents
        self.state = SharedState()
    
    def execute(self, task: str) -> str:
        # Phase 1: Analyze and decompose
        plan = self.llm.invoke(f"""
            Analyze this task and decompose it into subtasks.
            Available specialists: {list(self.specialists.keys())}
            Task: {task}
            
            For each subtask, specify:
            - Subtask description
            - Assigned specialist
            - Dependencies (which subtasks must complete first)
            - Expected output format
        """)
        subtasks = parse_plan(plan)
        
        # Phase 2: Topological execution
        results = {}
        for batch in topological_sort(subtasks):
            # Execute independent subtasks in parallel
            batch_results = parallel_execute([
                self.dispatch(st, results) for st in batch
            ])
            results.update(batch_results)
        
        # Phase 3: Synthesize final result
        return self.synthesize(task, results)
    
    def dispatch(self, subtask, prior_results):
        agent = self.specialists[subtask.assigned_agent]
        context = self.build_context(subtask, prior_results)
        return agent.execute(subtask.description, context)
    
    def synthesize(self, original_task, all_results):
        return self.llm.invoke(f"""
            Original task: {original_task}
            All subtask results: {all_results}
            Synthesize a final, coherent response.
        """)
```

#### Central State Management

The orchestrator maintains a **shared state object** accessible to all specialist agents:

$$
\mathcal{S}_{\text{shared}} = \{(k_1, v_1), (k_2, v_2), \ldots\}
$$

**State management operations:**

| Operation | Semantics | Concurrency Control |
|---|---|---|
| `read(key)` | Retrieve current value | No lock required |
| `write(key, value)` | Update value | Write lock required |
| `append(key, value)` | Add to list-valued entry | Append lock required |
| `cas(key, expected, new)` | Compare-and-swap | Atomic operation |

**Advantages of centralized orchestration:**

1. **Simplicity**: Single point of control makes the system easy to reason about and debug.
2. **Global optimization**: The orchestrator has full visibility into all subtasks and can optimize assignment.
3. **Deterministic execution order**: The orchestrator controls sequencing explicitly.
4. **Easy monitoring**: All communication passes through the hub, enabling comprehensive logging.

**Disadvantages:**

1. **Single point of failure**: If the orchestrator fails, the entire system halts.
2. **Bottleneck**: The orchestrator becomes a throughput bottleneck as $n$ grows:

$$
\text{Throughput} \leq \frac{C_{\text{orchestrator}}}{\text{AvgDispatchTime}}
$$

3. **Limited scalability**: Orchestrator context window must accommodate information about all $n$ agents and all subtask results.
4. **No agent autonomy**: Specialist agents cannot adapt or redirect without orchestrator intervention.

---

### 6.2.2 Decentralized / Peer-to-Peer

In decentralized architectures, there is **no central controller**. Agents communicate directly with each other and make autonomous decisions about task execution.

#### Agent-to-Agent Direct Communication

**Architecture:**

```
    A₁ ←───→ A₂
    ↕  ╲   ╱  ↕
    ↕    ╳    ↕
    ↕  ╱   ╲  ↕
    A₃ ←───→ A₄
```

Each agent $A_i$ maintains:
- A local task queue $Q_i$.
- A local state estimate $\hat{s}_i$.
- A neighbor set $\mathcal{N}(i) = \{j : (A_i, A_j) \in E\}$.

**Communication protocol:**

At each round $t$, each agent:
1. Processes incoming messages from neighbors.
2. Updates local state based on messages and own observations.
3. Decides whether to execute a task, request help, or offer assistance.
4. Sends outgoing messages to relevant neighbors.

**Formal agent loop:**

$$
\text{For each agent } A_i \text{ at time } t:
$$

$$
\hat{s}_i^{(t)} = \text{Update}(\hat{s}_i^{(t-1)}, o_i^{(t)}, \{m_j^{(t)} : j \in \mathcal{N}(i)\})
$$

$$
(a_i^{(t)}, \{m_i^{(t)} \rightarrow j : j \in \mathcal{N}(i)\}) = \pi_i(\hat{s}_i^{(t)}, Q_i)
$$

#### Consensus Protocols

When decentralized agents must agree on a shared decision (e.g., which agent handles a task, what the current state is), they require **consensus protocols**.

**Average Consensus:** Agents iteratively average their estimates with neighbors:

$$
x_i^{(t+1)} = w_{ii} x_i^{(t)} + \sum_{j \in \mathcal{N}(i)} w_{ij} x_j^{(t)}
$$

where $w_{ij}$ are mixing weights satisfying $\sum_j w_{ij} = 1$, $w_{ij} \geq 0$, and $w_{ij} = 0$ if $j \notin \mathcal{N}(i) \cup \{i\}$.

Under connectivity conditions, this converges to the global average:

$$
\lim_{t \to \infty} x_i^{(t)} = \frac{1}{n} \sum_{j=1}^{n} x_j^{(0)} \quad \forall i
$$

**Convergence rate** depends on the spectral gap of the mixing matrix $W$:

$$
\|x^{(t)} - \bar{x}\mathbf{1}\|_2 \leq \lambda_2(W)^t \|x^{(0)} - \bar{x}\mathbf{1}\|_2
$$

where $\lambda_2(W)$ is the second-largest eigenvalue of $W$. Smaller $\lambda_2$ means faster convergence.

**In LLM MAS context**, consensus can be implemented through structured debate rounds where agents exchange opinions and converge on a decision:

```python
def consensus_round(agents, topic, max_rounds=5):
    opinions = {a.id: a.initial_opinion(topic) for a in agents}
    
    for round_num in range(max_rounds):
        new_opinions = {}
        for agent in agents:
            neighbor_opinions = {
                n.id: opinions[n.id] 
                for n in agent.neighbors
            }
            new_opinions[agent.id] = agent.update_opinion(
                own_opinion=opinions[agent.id],
                neighbor_opinions=neighbor_opinions,
                round_num=round_num
            )
        
        # Check convergence
        if all_agree(new_opinions):
            return new_opinions  # Consensus reached
        opinions = new_opinions
    
    # Fallback: majority vote
    return majority_vote(opinions)
```

**Advantages of decentralized MAS:**

1. **No single point of failure**: System remains operational even if some agents fail.
2. **Scalability**: Adding agents does not require modifying a central controller.
3. **Privacy**: Agents can operate on local data without sharing everything centrally.

**Disadvantages:**

1. **Convergence time**: Reaching consensus takes multiple communication rounds.
2. **Coordination difficulty**: Without global state, agents may duplicate work or create conflicts.
3. **Debugging complexity**: No single log captures the full system trajectory.

---

### 6.2.3 Hierarchical Multi-Agent

Hierarchical architectures organize agents into **layers of authority and responsibility**, mirroring organizational hierarchies in human enterprises.

#### Manager-Worker Hierarchies

**Architecture:**

```
            ┌──────────────┐
            │  CEO Agent   │  (Level 0: Strategic)
            └──────┬───────┘
         ┌─────────┼─────────┐
    ┌────┴────┐ ┌──┴───┐ ┌───┴────┐
    │Manager₁ │ │Mgr₂  │ │Mgr₃   │  (Level 1: Tactical)
    │(Backend)│ │(Front)│ │(DevOps)│
    └────┬────┘ └──┬───┘ └───┬────┘
    ┌────┼────┐    │     ┌───┼────┐
   ┌┴┐  ┌┴┐ ┌┴┐  ┌┴┐   ┌┴┐ ┌┴┐ ┌┴┐
   │W│  │W│ │W│  │W│   │W│ │W│ │W│  (Level 2: Operational)
   └─┘  └─┘ └─┘  └─┘   └─┘ └─┘ └─┘
```

**Formal hierarchy:** A hierarchy is a tree $\mathcal{H} = (V, E, \text{level})$ where:
- $V$ is the set of agents.
- $E$ is the set of authority edges (parent → child).
- $\text{level}: V \rightarrow \{0, 1, \ldots, L\}$ assigns each agent to a hierarchy level.

**Properties:**

1. **Span of control**: Each manager supervises at most $k$ direct reports:

$$
|\text{children}(A_i)| \leq k \quad \forall A_i \text{ where } \text{level}(A_i) < L
$$

2. **Authority gradient**: Higher-level agents can override lower-level decisions:

$$
\text{Authority}(A_i) > \text{Authority}(A_j) \iff \text{level}(A_i) < \text{level}(A_j)
$$

3. **Abstraction gradient**: Higher levels deal with abstract strategy; lower levels with concrete execution:

$$
\text{Abstraction}(A_i) \propto \frac{1}{\text{level}(A_i) + 1}
$$

#### Recursive Delegation

A key pattern in hierarchical MAS is **recursive delegation**: each manager decomposes its assigned task and delegates subtasks to its workers, who may further delegate:

$$
\text{Delegate}(A_{\text{manager}}, T) = \begin{cases}
\text{Execute}(T) & \text{if } \text{level}(A_{\text{manager}}) = L \\
\{(A_j, T_j) : A_j \in \text{children}(A_{\text{manager}})\} & \text{otherwise}
\end{cases}
$$

where $T = \bigcup_j T_j$ and the subtasks are generated by the manager's LLM:

$$
\{T_1, \ldots, T_k\} = \text{LLM}_{\text{manager}}(T, \text{capabilities of children})
$$

**Manager agent responsibilities:**

```python
class ManagerAgent:
    def __init__(self, llm, workers: List[Agent], level: int):
        self.llm = llm
        self.workers = workers
        self.level = level
    
    def handle_task(self, task: str) -> str:
        # 1. Decompose task into subtasks
        decomposition = self.llm.invoke(f"""
            You are a Level-{self.level} manager.
            Your workers and their capabilities:
            {[(w.id, w.role) for w in self.workers]}
            
            Task: {task}
            
            Decompose into subtasks and assign to workers.
            Specify execution order and dependencies.
        """)
        
        assignments = parse_assignments(decomposition)
        
        # 2. Dispatch and monitor
        results = {}
        for batch in topological_sort(assignments):
            for assignment in batch:
                worker = self.get_worker(assignment.worker_id)
                result = worker.handle_task(assignment.subtask)
                
                # 3. Quality check
                quality = self.evaluate_result(assignment, result)
                if quality < self.quality_threshold:
                    result = self.request_revision(worker, assignment, result)
                
                results[assignment.id] = result
        
        # 4. Synthesize and report to superior
        return self.synthesize(task, results)
```

**Advantages of hierarchical MAS:**

1. **Scalability**: The tree structure scales to large numbers of agents with $O(\log n)$ communication depth.
2. **Clear accountability**: Each agent has a well-defined superior and subordinates.
3. **Cognitive load management**: Each agent deals with only $k$ direct reports, keeping context manageable.
4. **Natural task decomposition**: The hierarchy enforces progressive refinement from abstract to concrete.

**Disadvantages:**

1. **Communication latency**: Information must traverse the tree—$O(L)$ hops for leaf-to-leaf communication.
2. **Bottleneck at intermediate nodes**: Manager agents can become overloaded.
3. **Rigidity**: Difficult to adapt to tasks that don't decompose hierarchically.

---

### 6.2.4 Blackboard Architecture

The blackboard architecture uses a **shared knowledge repository** (the blackboard) that all agents can read from and write to. Agents are **knowledge sources** that opportunistically contribute when they can make progress on some aspect of the problem.

#### Shared Knowledge Space

**Architecture:**

```
┌─────────────────────────────────────────┐
│              BLACKBOARD                 │
│                                         │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Problem   │ │ Partial   │ │ Final   │ │
│  │ Statement │ │ Solutions │ │ Results │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│  ┌──────────┐ ┌──────────┐             │
│  │ Hypotheses│ │ Evidence │             │
│  └──────────┘ └──────────┘             │
└────┬──────┬──────┬──────┬──────┬───────┘
     │      │      │      │      │
   ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
   │KS₁│  │KS₂│  │KS₃│  │KS₄│  │KS₅│
   │   │  │   │  │   │  │   │  │   │
   └───┘  └───┘  └───┘  └───┘  └───┘
  Knowledge Sources (Specialist Agents)
```

**Formal specification:**

$$
\mathcal{B} = \langle \mathcal{K}, \{KS_1, \ldots, KS_n\}, \text{Control} \rangle
$$

- $\mathcal{K}$: The knowledge store—a structured repository with levels of abstraction:

$$
\mathcal{K} = \mathcal{K}_{\text{raw}} \cup \mathcal{K}_{\text{partial}} \cup \mathcal{K}_{\text{hypothesis}} \cup \mathcal{K}_{\text{solution}}
$$

- $KS_i$: Knowledge source $i$—an agent that monitors the blackboard and contributes when it can make progress.
- $\text{Control}$: The control mechanism that selects which knowledge source acts next.

#### Opportunistic Problem Solving

Each knowledge source $KS_i$ has:
- A **trigger condition** $\psi_i(\mathcal{K})$: a predicate over the blackboard state that determines when $KS_i$ can contribute.
- An **action function** $f_i(\mathcal{K}) \rightarrow \Delta\mathcal{K}$: the knowledge update it makes.

**Control loop:**

$$
\text{while } \mathcal{K} \not\models G:
$$

$$
\quad \text{Eligible} = \{KS_i : \psi_i(\mathcal{K}) = \text{True}\}
$$

$$
\quad KS^* = \text{Control.select}(\text{Eligible})
$$

$$
\quad \mathcal{K} \leftarrow \mathcal{K} \cup f_{KS^*}(\mathcal{K})
$$

**Implementation in LLM MAS:**

```python
class BlackboardSystem:
    def __init__(self, knowledge_sources: List[KnowledgeSource]):
        self.blackboard = {}
        self.sources = knowledge_sources
        self.history = []
    
    def solve(self, problem: str, max_iterations=20):
        self.blackboard["problem"] = problem
        self.blackboard["partial_solutions"] = []
        self.blackboard["evidence"] = []
        
        for iteration in range(max_iterations):
            # Find eligible knowledge sources
            eligible = [
                ks for ks in self.sources
                if ks.can_contribute(self.blackboard)
            ]
            
            if not eligible:
                break  # No agent can make progress
            
            # Select highest-priority contributor
            selected = max(eligible, key=lambda ks: 
                          ks.priority(self.blackboard))
            
            # Agent reads blackboard, contributes
            contribution = selected.contribute(self.blackboard)
            
            # Update blackboard
            self.blackboard = merge(self.blackboard, contribution)
            self.history.append((selected.id, contribution))
            
            # Check if solution is complete
            if self.is_solved():
                return self.blackboard["solution"]
        
        return self.synthesize_best_effort()
```

**Advantages:**

1. **Flexibility**: Agents contribute opportunistically—no rigid execution order.
2. **Modularity**: New knowledge sources can be added without modifying existing ones.
3. **Incremental refinement**: The solution evolves as agents progressively enrich the blackboard.

**Disadvantages:**

1. **Control complexity**: The control mechanism must intelligently schedule agents.
2. **Contention**: Multiple agents writing to the blackboard simultaneously can create conflicts.
3. **Convergence uncertainty**: No guarantee the system will converge to a solution.

---

### 6.2.5 Market-Based / Auction Architectures

Market-based architectures use **economic mechanisms** to allocate tasks to agents. Agents bid on tasks based on their capabilities and current load, and tasks are awarded to the highest bidder (or most suitable agent).

#### Task Allocation via Bidding

**Auction mechanism:**

1. **Task announcement**: A task $T$ is announced with its requirements $\text{Req}(T)$.
2. **Bid generation**: Each agent $A_i$ submits a bid $b_i(T)$ representing its estimated quality and cost:

$$
b_i(T) = \langle \text{quality}_i, \text{cost}_i, \text{time}_i, \text{confidence}_i \rangle
$$

3. **Winner selection**: The auctioneer selects the winning agent:

$$
A^* = \arg\max_i \text{Score}(b_i(T))
$$

where the scoring function may be:

$$
\text{Score}(b_i) = w_q \cdot \text{quality}_i - w_c \cdot \text{cost}_i - w_t \cdot \text{time}_i + w_f \cdot \text{confidence}_i
$$

4. **Execution and payment**: The winning agent executes the task and receives a reward.

#### Contract Net Protocol

The **Contract Net Protocol** (Smith, 1980) is the foundational protocol for market-based MAS:

```
Manager (Task Announcer)          Contractors (Bidding Agents)
        │                              │  │  │
        │──── Announce Task ──────────→│  │  │
        │                              │  │  │
        │←──── Bid (Agent 1) ─────────│  │  │
        │←──── Bid (Agent 2) ────────────│  │
        │←──── No Bid (Agent 3) ────────────│
        │                              │  │  │
        │──── Award (Agent 1) ────────→│  │  │
        │──── Reject (Agent 2) ──────────→│  │
        │                              │  │  │
        │←──── Result ─────────────────│  │  │
        │                              │  │  │
```

**LLM-agent implementation:**

```python
class ContractNetProtocol:
    def __init__(self, manager: Agent, contractors: List[Agent]):
        self.manager = manager
        self.contractors = contractors
    
    def allocate_task(self, task: str) -> Tuple[Agent, str]:
        # Step 1: Task announcement
        announcement = self.manager.announce(task)
        
        # Step 2: Collect bids
        bids = []
        for agent in self.contractors:
            bid = agent.generate_bid(announcement)
            # Bid contains: estimated quality, time, 
            #   capability match, current load
            if bid.willing:
                bids.append((agent, bid))
        
        if not bids:
            raise NoContractorsAvailable(task)
        
        # Step 3: Evaluate bids
        scored_bids = [
            (agent, bid, self.manager.evaluate_bid(bid, task))
            for agent, bid in bids
        ]
        
        # Step 4: Award contract
        winner_agent, winner_bid, _ = max(
            scored_bids, key=lambda x: x[2]
        )
        
        # Step 5: Execute
        result = winner_agent.execute(task)
        
        # Step 6: Verify result
        if self.manager.verify(result, task):
            return winner_agent, result
        else:
            # Re-auction to remaining agents
            remaining = [a for a, _ in bids if a != winner_agent]
            return self.reallocate(task, remaining)
```

**Advantages:**

1. **Load balancing**: Tasks naturally flow to less-loaded agents.
2. **Capability matching**: Agents bid only on tasks they can handle.
3. **Decoupled scaling**: New agents join by simply participating in auctions.

**Disadvantages:**

1. **Communication overhead**: Each task requires an auction round.
2. **Strategic behavior**: Agents may strategically misrepresent bids.
3. **Suboptimality for tightly coupled tasks**: Tasks with strong interdependencies don't decompose well into independent auctions.

---

## 6.3 Agent Roles and Specialization

### 6.3.1 Role Definition and Assignment

A **role** $R_i$ is a specification that constrains an agent's behavior, expertise, communication style, and objectives:

$$
R_i = \langle \text{Name}, \text{Description}, \text{Capabilities}, \text{Constraints}, \text{Objectives}, \text{Style} \rangle
$$

**Role specification components:**

| Component | Description | Example |
|---|---|---|
| **Name** | Role identifier | "Senior Code Reviewer" |
| **Description** | Natural language persona | "You are a meticulous code reviewer with 15 years of experience..." |
| **Capabilities** | What the agent can do | ["read_code", "write_review", "run_linter"] |
| **Constraints** | What the agent must not do | "Never approve code with security vulnerabilities" |
| **Objectives** | What the agent aims to achieve | "Ensure code quality, catch bugs, improve readability" |
| **Style** | Communication pattern | "Provide constructive feedback with specific line references" |

**Formal role assignment problem:**

Given a set of roles $\{R_1, \ldots, R_m\}$ and a set of agents $\{A_1, \ldots, A_n\}$, find an assignment $\sigma: \{R_1, \ldots, R_m\} \rightarrow \{A_1, \ldots, A_n\}$ that maximizes overall task performance:

$$
\sigma^* = \arg\max_\sigma \sum_{j=1}^{m} \text{Fitness}(A_{\sigma(j)}, R_j)
$$

where $\text{Fitness}(A_i, R_j)$ measures how well agent $A_i$ performs role $R_j$. When all agents share the same base LLM, fitness depends primarily on how well the role's system prompt activates relevant capabilities:

$$
\text{Fitness}(A_i, R_j) = P_\theta(\text{high quality output} | \text{SystemPrompt}(R_j), \text{model}(\theta_i))
$$

---

### 6.3.2 Persona-Based Agent Design

Persona-based design creates agents with rich, detailed identities that go beyond simple role descriptions. The hypothesis is that detailed personas activate more relevant knowledge and reasoning patterns in the LLM.

**Persona specification template:**

```
Name: Dr. Sarah Chen
Role: Machine Learning Research Scientist
Background: PhD in Statistical Learning Theory from MIT (2018).
  Published 23 papers in NeurIPS, ICML, and ICLR. Expert in 
  optimization theory and neural network generalization.
  Previously worked at DeepMind on reinforcement learning.
Personality: Rigorous, skeptical of unjustified claims, values 
  mathematical precision. Will always ask for formal proofs or 
  empirical evidence before accepting conclusions.
Communication Style: Formal academic tone. Uses mathematical 
  notation extensively. Cites relevant literature.
Typical Phrases: "What's the convergence guarantee?", 
  "Have you controlled for confounders?", "Let me formalize this..."
Limitations: Acknowledges when a question is outside her expertise.
  Defers to domain experts for domain-specific knowledge.
```

**Why personas work — conditional generation hypothesis:**

$$
P(\text{expert output} | \text{detailed persona}) > P(\text{expert output} | \text{simple role})
$$

The detailed persona provides a richer conditioning signal, steering the LLM's generation distribution toward the subset of its training data most relevant to the expert domain.

**Empirical evidence:** Studies (Li et al., 2023; Park et al., 2023) demonstrate that LLMs with detailed personas:
- Produce more consistent outputs across interactions.
- Better maintain specialized knowledge domains.
- Exhibit more realistic and diverse reasoning patterns.
- Reduce the tendency to default to generic responses.

---

### 6.3.3 Dynamic Role Allocation

In dynamic role allocation, agent roles are not fixed at design time but **adapt during task execution** based on task requirements, agent performance, and evolving context.

**Trigger conditions for role reallocation:**

1. **Performance degradation**: If agent $A_i$ in role $R_j$ consistently underperforms:

$$
\text{Reallocate if } \quad \text{AvgQuality}(A_i, R_j, \text{recent\_k}) < \tau_{\text{min}}
$$

2. **Task phase transition**: Different roles are needed in different phases of the task:

$$
R_i^{(t)} = f(\text{phase}(t))
$$

For example, during a software project: requirements analysis phase needs "Product Manager" and "User Researcher" roles; implementation phase needs "Developer" and "Architect" roles; testing phase needs "QA Engineer" and "Security Auditor" roles.

3. **Agent availability**: If an agent fails or becomes overloaded, its role must be reassigned:

$$
\text{If } \text{Load}(A_i) > \tau_{\text{load}}: \quad R_j \leftarrow \text{Reassign}(R_j, A_i \rightarrow A_k)
$$

**Dynamic role allocation algorithm:**

```python
class DynamicRoleAllocator:
    def __init__(self, agents, roles, llm):
        self.agents = agents
        self.roles = roles
        self.llm = llm
        self.assignment = {}
        self.performance_history = defaultdict(list)
    
    def allocate(self, task_context):
        # LLM determines which roles are needed
        needed_roles = self.llm.invoke(f"""
            Given the current task context: {task_context}
            Available roles: {self.roles}
            Which roles are needed? Prioritize by importance.
        """)
        
        # Assign agents to roles using capability matching
        for role in needed_roles:
            best_agent = max(
                self.available_agents(),
                key=lambda a: self.capability_score(a, role)
            )
            self.assignment[role] = best_agent
    
    def monitor_and_reallocate(self):
        for role, agent in self.assignment.items():
            recent_quality = self.performance_history[(agent.id, role.id)][-5:]
            if mean(recent_quality) < self.quality_threshold:
                # Find a better agent for this role
                alternatives = [a for a in self.agents if a != agent]
                best_alt = max(alternatives, 
                             key=lambda a: self.capability_score(a, role))
                if self.capability_score(best_alt, role) > \
                   self.capability_score(agent, role):
                    self.reassign(role, agent, best_alt)
```

---

### 6.3.4 Specialist vs. Generalist Agents

**Specialist agents** are optimized for a narrow domain:
- Detailed system prompts focused on one area.
- Domain-specific tools only.
- Deep few-shot examples from their domain.

**Generalist agents** handle a broad range of tasks:
- General-purpose system prompts.
- Access to all tools.
- Diverse few-shot examples.

**Trade-off analysis:**

Define the quality function for agent $A$ with specialization level $\sigma \in [0, 1]$ (0 = pure generalist, 1 = pure specialist) on a task with domain match $d \in [0, 1]$:

$$
Q(\sigma, d) = \underbrace{Q_{\text{base}} + \alpha \cdot \sigma \cdot d}_{\text{specialist advantage when domain matches}} - \underbrace{\beta \cdot \sigma \cdot (1 - d)}_{\text{specialist penalty when domain mismatches}}
$$

For perfect domain match ($d = 1$): $Q(1, 1) = Q_{\text{base}} + \alpha$ (specialist wins).

For no domain match ($d = 0$): $Q(1, 0) = Q_{\text{base}} - \beta$ (specialist loses).

For generalist: $Q(0, d) = Q_{\text{base}}$ (constant regardless of domain).

**Optimal system design** uses specialists for predictable, well-defined subtasks and generalists as fallback or for tasks that cross domain boundaries.

$$
A_i = \begin{cases}
\text{Specialist}(D_j) & \text{if subtask domain is known and } D_j \text{ matches} \\
\text{Generalist} & \text{if subtask domain is ambiguous or cross-domain}
\end{cases}
$$

---

### 6.3.5 Meta-Agents: Agents That Manage Other Agents

A **meta-agent** operates at a higher level of abstraction, managing the composition, configuration, and lifecycle of other agents:

$$
A_{\text{meta}}: (\text{Task}, \text{AgentPool}) \rightarrow (\text{TeamComposition}, \text{ExecutionPlan}, \text{MonitoringPolicy})
$$

**Meta-agent responsibilities:**

1. **Team assembly**: Decide how many agents are needed and what roles they should play.
2. **Protocol selection**: Choose the communication and coordination protocol.
3. **Performance monitoring**: Track agent-level and system-level metrics.
4. **Intervention**: Replace underperforming agents, adjust roles, modify protocols.
5. **Post-mortem analysis**: Analyze task execution to improve future team compositions.

**Meta-agent decision flow:**

```python
class MetaAgent:
    def __init__(self, llm, agent_registry):
        self.llm = llm
        self.registry = agent_registry  # Available agent templates
    
    def assemble_team(self, task: str) -> Team:
        # Analyze task requirements
        analysis = self.llm.invoke(f"""
            Analyze this task and determine the optimal team:
            Task: {task}
            
            Determine:
            1. Number of agents needed
            2. Required roles and specializations
            3. Communication topology (centralized/decentralized/hierarchical)
            4. Coordination protocol (round-robin/free-form/phase-based)
            5. Quality assurance mechanism
        """)
        
        team_spec = parse_team_specification(analysis)
        
        # Instantiate agents
        team = Team()
        for role_spec in team_spec.roles:
            agent = self.registry.create_agent(
                model=role_spec.preferred_model,
                system_prompt=role_spec.persona,
                tools=role_spec.tools,
                role=role_spec.role_name
            )
            team.add(agent, role=role_spec.role_name)
        
        team.set_topology(team_spec.topology)
        team.set_protocol(team_spec.protocol)
        
        return team
    
    def monitor_execution(self, team: Team, task: str):
        while not team.is_complete():
            metrics = team.get_metrics()
            
            if metrics.stalled:
                self.intervene_stall(team)
            if metrics.quality_drop:
                self.intervene_quality(team)
            if metrics.budget_exceeded:
                self.intervene_budget(team)
            
            time.sleep(self.monitoring_interval)
```

---

## 6.4 Coordination Mechanisms

### 6.4.1 Turn-Taking Protocols

Turn-taking protocols define the **temporal ordering** of agent actions, preventing communication collisions and ensuring structured interaction.

**Protocol types:**

**1. Static Round-Robin:**

$$
\text{Speaker}(t) = A_{(t \mod n) + 1}
$$

Each agent speaks in fixed order. Simple but inflexible—irrelevant agents still take turns.

**2. Dynamic Turn-Taking (Moderated):**

A moderator agent decides who speaks next based on context:

$$
\text{Speaker}(t) = \text{Moderator}\left(\text{history}_{<t}, \{A_1, \ldots, A_n\}\right)
$$

**3. Raise-Hand Protocol:**

Agents signal when they have something to contribute. The moderator selects from signaling agents:

$$
\text{Candidates}(t) = \{A_i : A_i.\text{wants\_to\_speak}(t) = \text{True}\}
$$

$$
\text{Speaker}(t) = \text{Moderator.select}(\text{Candidates}(t))
$$

**4. Priority-Based:**

Agents are assigned dynamic priorities. The highest-priority agent with pending output speaks:

$$
\text{Speaker}(t) = \arg\max_{A_i \in \text{Candidates}(t)} \text{Priority}(A_i, t)
$$

Priority can depend on role importance, urgency of contribution, or recency of last turn.

**Implementation:**

```python
class TurnManager:
    def __init__(self, agents, protocol="moderated", moderator=None):
        self.agents = agents
        self.protocol = protocol
        self.moderator = moderator
        self.turn_count = 0
        self.history = []
    
    def next_speaker(self) -> Agent:
        if self.protocol == "round_robin":
            speaker = self.agents[self.turn_count % len(self.agents)]
        
        elif self.protocol == "moderated":
            speaker_id = self.moderator.select_next_speaker(
                self.history, self.agents
            )
            speaker = self.get_agent(speaker_id)
        
        elif self.protocol == "raise_hand":
            candidates = [
                a for a in self.agents 
                if a.has_contribution(self.history)
            ]
            if not candidates:
                return None  # No one has anything to add
            speaker = self.moderator.select_from(candidates, self.history)
        
        self.turn_count += 1
        return speaker
```

---

### 6.4.2 Shared State and Scratchpads

A **shared scratchpad** is a mutable workspace accessible to all agents, enabling persistent information sharing beyond ephemeral messages.

**Scratchpad structure:**

$$
\mathcal{S}_{\text{pad}} = \{(\text{key}_i, \text{value}_i, \text{author}_i, \text{timestamp}_i, \text{version}_i)\}
$$

**Operations:**

| Operation | Description | Concurrency Semantics |
|---|---|---|
| `write(key, value)` | Set a value | Last-writer-wins or versioned |
| `read(key)` | Get current value | Always allowed |
| `append(key, value)` | Add to a list | Atomic append |
| `lock(key)` | Exclusive access | Blocking until released |
| `subscribe(key, callback)` | React to changes | Event-driven |

**Conflict resolution strategies:**

When multiple agents write to the same key:

1. **Last-writer-wins (LWW)**: Simple but may lose information.
2. **Version vectors**: Track causality to detect conflicts:

$$
\text{Conflict}(v_1, v_2) \iff v_1 \not\leq v_2 \wedge v_2 \not\leq v_1
$$

3. **CRDT (Conflict-free Replicated Data Types)**: Use data structures that are mathematically guaranteed to converge:

$$
x_1 \sqcup x_2 = x_2 \sqcup x_1 \quad \text{(commutativity)}
$$

$$
x \sqcup x = x \quad \text{(idempotency)}
$$

4. **LLM-mediated merging**: When conflicts arise, an LLM merges the conflicting values:

$$
v_{\text{merged}} = \text{LLM}(\text{"Merge these conflicting updates: "} v_1, v_2)
$$

**Practical example — collaborative document editing:**

```python
class SharedScratchpad:
    def __init__(self):
        self.data = {}
        self.versions = {}
        self.locks = {}
        self.subscribers = defaultdict(list)
    
    def write(self, key, value, author):
        if key in self.locks and self.locks[key] != author:
            raise LockConflict(f"{key} is locked by {self.locks[key]}")
        
        self.versions[key] = self.versions.get(key, 0) + 1
        self.data[key] = {
            "value": value,
            "author": author,
            "version": self.versions[key],
            "timestamp": time.time()
        }
        
        # Notify subscribers
        for callback in self.subscribers[key]:
            callback(key, value, author)
    
    def read(self, key):
        return self.data.get(key, {}).get("value")
    
    def get_section(self, section_name):
        """Get all keys under a section for organized workspace."""
        return {k: v for k, v in self.data.items() 
                if k.startswith(section_name)}
```

---

### 6.4.3 Task Queues and Work Stealing

**Task queues** provide a decoupled mechanism for distributing work: producers add tasks; consumers pull and execute tasks.

**Architecture:**

```
         Task Producers                    Task Consumers
  ┌───┐  ┌───┐  ┌───┐              ┌───┐  ┌───┐  ┌───┐
  │ P₁│  │ P₂│  │ P₃│              │ W₁│  │ W₂│  │ W₃│
  └─┬─┘  └─┬─┘  └─┬─┘              └─┬─┘  └─┬─┘  └─┬─┘
    │      │      │                   │      │      │
    ▼      ▼      ▼                   ▲      ▲      ▲
  ┌────────────────────────────────────────────────────┐
  │              TASK QUEUE                             │
  │  [T₅] [T₄] [T₃] [T₂] [T₁] →→→                   │
  │  Priority Queue / FIFO / Weighted Fair Queue        │
  └────────────────────────────────────────────────────┘
```

**Work stealing** extends basic task queues: when a worker's local queue is empty, it **steals** tasks from busy workers' queues:

$$
\text{If } |Q_i| = 0: \quad \text{steal from } A_j \text{ where } j = \arg\max_k |Q_k|
$$

**Work stealing algorithm:**

```python
class WorkStealingScheduler:
    def __init__(self, workers: List[Agent]):
        self.workers = workers
        self.local_queues = {w.id: deque() for w in workers}
        self.global_queue = deque()
    
    def submit(self, task):
        # Route to least-loaded worker
        target = min(self.workers, 
                    key=lambda w: len(self.local_queues[w.id]))
        self.local_queues[target.id].append(task)
    
    def worker_loop(self, worker):
        while True:
            # Try local queue first
            task = self.try_dequeue(worker.id)
            
            if task is None:
                # Try global queue
                task = self.try_global_dequeue()
            
            if task is None:
                # Steal from busiest worker
                task = self.steal(worker.id)
            
            if task is None:
                time.sleep(0.1)  # No work available
                continue
            
            result = worker.execute(task)
            self.report_result(task.id, result)
    
    def steal(self, thief_id):
        victims = sorted(
            self.workers,
            key=lambda w: len(self.local_queues[w.id]),
            reverse=True
        )
        for victim in victims:
            if victim.id != thief_id and self.local_queues[victim.id]:
                # Steal from the back of victim's queue
                return self.local_queues[victim.id].pop()
        return None
```

---

### 6.4.4 Voting and Consensus Mechanisms

When multiple agents must agree on a decision, **voting mechanisms** aggregate preferences:

**1. Majority Voting:**

$$
y^* = \arg\max_y \sum_{i=1}^{n} \mathbb{1}[\text{vote}(A_i) = y]
$$

Simple and effective when agents are independent. Requires $n \geq 3$ (odd) to avoid ties.

**2. Weighted Voting:**

$$
y^* = \arg\max_y \sum_{i=1}^{n} w_i \cdot \mathbb{1}[\text{vote}(A_i) = y]
$$

where $w_i$ reflects agent $A_i$'s expertise or past accuracy on similar decisions:

$$
w_i = \frac{\text{AccuracyHistory}(A_i)}{\sum_j \text{AccuracyHistory}(A_j)}
$$

**3. Approval Voting:**

Each agent approves or disapproves each option. The option with the most approvals wins:

$$
y^* = \arg\max_y \sum_{i=1}^{n} \text{approve}(A_i, y) \in \{0, 1\}
$$

**4. Borda Count:**

Each agent ranks options. Points are assigned based on rank position. The option with the most total points wins:

$$
y^* = \arg\max_y \sum_{i=1}^{n} (m - \text{rank}_i(y))
$$

where $m$ is the total number of options.

**5. Condorcet Method:**

An option wins if it would defeat every other option in pairwise majority comparisons:

$$
y^* \text{ is Condorcet winner if } \forall y' \neq y^*: |\{i : y^* \succ_i y'\}| > \frac{n}{2}
$$

**Arrow's Impossibility Theorem** establishes that no ranked voting system can simultaneously satisfy unrestricted domain, Pareto efficiency, independence of irrelevant alternatives, and non-dictatorship. This is a fundamental limitation that motivates the use of domain-specific voting mechanisms.

**LLM-specific voting implementation:**

```python
def multi_agent_vote(agents, question, method="weighted_majority"):
    # Collect votes with confidence
    votes = []
    for agent in agents:
        response = agent.decide(question)
        votes.append({
            "agent": agent.id,
            "answer": response.answer,
            "confidence": response.confidence,
            "reasoning": response.reasoning
        })
    
    if method == "weighted_majority":
        # Weight by confidence
        answer_scores = defaultdict(float)
        for vote in votes:
            answer_scores[vote["answer"]] += vote["confidence"]
        return max(answer_scores, key=answer_scores.get)
    
    elif method == "debate_then_vote":
        # Agents debate, then re-vote
        debate_transcript = conduct_debate(agents, question, votes)
        revised_votes = [
            agent.revise_vote(question, debate_transcript)
            for agent in agents
        ]
        return majority_vote(revised_votes)
```

---

### 6.4.5 Leader Election in Decentralized MAS

In decentralized MAS without a pre-designated orchestrator, agents must **elect a leader** dynamically to coordinate activity.

**Bully Algorithm (adapted for LLM agents):**

1. Any agent that detects the need for coordination initiates an election.
2. It sends an "election" message to all agents with higher capability scores.
3. If no higher-capability agent responds, the initiator becomes the leader.
4. If a higher-capability agent responds, it takes over the election process.

**Formal specification:**

$$
\text{Leader} = \arg\max_{A_i \in \text{Active}} \text{CapabilityScore}(A_i)
$$

where:

$$
\text{CapabilityScore}(A_i) = \alpha \cdot \text{ModelQuality}(A_i) + \beta \cdot \text{DomainExpertise}(A_i, T) + \gamma \cdot \text{AvailableCapacity}(A_i)
$$

**Raft-inspired consensus for LLM MAS:**

Adapt the Raft consensus algorithm for agent coordination:

1. **Leader election**: Agents use randomized timeouts. The first to timeout becomes a candidate and requests votes.
2. **Log replication**: The leader receives all task submissions and replicates the execution log to followers.
3. **Safety**: The leader handles all decision-making; followers mirror the leader's decisions.

```python
class RaftAgent:
    def __init__(self, agent_id, peers):
        self.id = agent_id
        self.peers = peers
        self.state = "follower"  # follower, candidate, leader
        self.current_term = 0
        self.voted_for = None
        self.election_timeout = random.uniform(150, 300)  # ms
    
    def start_election(self):
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.id
        votes = 1  # Vote for self
        
        for peer in self.peers:
            response = peer.request_vote(self.current_term, self.id)
            if response.vote_granted:
                votes += 1
        
        if votes > len(self.peers) // 2:
            self.state = "leader"
            self.begin_coordination()
        else:
            self.state = "follower"
```

---

## 6.5 Cooperative Multi-Agent Patterns

### 6.5.1 Debate and Discussion

#### Multi-Agent Debate for Improved Reasoning

Multi-agent debate leverages the principle that **adversarial scrutiny improves reasoning quality**. Multiple agents independently solve a problem, then engage in structured argumentation where they critique each other's solutions and iteratively refine toward a consensus answer.

**Formal framework:**

Given a question $x$, $n$ agents each produce an initial answer:

$$
y_i^{(0)} = \text{LLM}(x, R_i) \quad \text{for } i = 1, \ldots, n
$$

In each debate round $r = 1, 2, \ldots, R_{\max}$:

$$
y_i^{(r)} = \text{LLM}\left(x, R_i, \text{DebateHistory}^{(<r)}, \{y_j^{(r-1)} : j \neq i\}\right)
$$

Each agent updates its answer after seeing all other agents' previous answers and the accumulated debate transcript.

**Convergence criterion:**

$$
\text{Converged} \iff \forall i, j: y_i^{(r)} = y_j^{(r)} \quad \text{or} \quad r = R_{\max}
$$

#### Structured Argumentation

The debate output is resolved through a resolution function:

$$
y^* = \text{Resolve}(\{y_i\}_{i=1}^{n}, \text{debate\_transcript})
$$

**Resolution strategies:**

1. **Majority after debate**: After $R$ rounds, take the majority answer:

$$
y^* = \arg\max_y \sum_{i=1}^{n} \mathbb{1}[y_i^{(R)} = y]
$$

2. **Judge resolution**: A separate judge agent evaluates the debate:

$$
y^* = \text{Judge}(\text{debate\_transcript}, x)
$$

3. **Confidence-weighted**: After debate, each agent provides a confidence score:

$$
y^* = \arg\max_y \sum_{i=1}^{n} c_i^{(R)} \cdot \mathbb{1}[y_i^{(R)} = y]
$$

**Mathematical justification (Condorcet Jury Theorem extension):**

If each agent has independent probability $p > 0.5$ of being correct after debate, and debate does not reduce this probability, then:

$$
P(\text{majority correct}) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k} \xrightarrow{n \to \infty} 1
$$

However, debate introduces **correlation** between agents' answers, which can either help (if the correct answer gains converts through strong arguments) or hurt (if a confident-but-wrong agent sways others).

**Effective independence after debate:**

$$
P(\text{majority correct}) \geq 1 - \exp\left(-\frac{2(p_{\text{eff}} - 0.5)^2 n}{1 + (n-1)\rho}\right)
$$

where $p_{\text{eff}}$ is the effective per-agent accuracy post-debate and $\rho$ is the correlation between agents' final answers. Low $\rho$ (independent agents) and high $p_{\text{eff}}$ (debate improves accuracy) maximize benefit.

**Implementation:**

```python
class MultiAgentDebate:
    def __init__(self, agents: List[Agent], judge: Agent = None,
                 max_rounds: int = 3):
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
    
    def debate(self, question: str) -> str:
        # Round 0: Independent answers
        answers = {}
        for agent in self.agents:
            answers[agent.id] = agent.answer(question)
        
        transcript = []
        
        # Debate rounds
        for round_num in range(1, self.max_rounds + 1):
            new_answers = {}
            for agent in self.agents:
                others = {
                    aid: ans for aid, ans in answers.items() 
                    if aid != agent.id
                }
                
                response = agent.debate_round(
                    question=question,
                    own_previous_answer=answers[agent.id],
                    others_answers=others,
                    round_number=round_num,
                    transcript=transcript
                )
                new_answers[agent.id] = response.answer
                transcript.append({
                    "round": round_num,
                    "agent": agent.id,
                    "argument": response.argument,
                    "answer": response.answer,
                    "confidence": response.confidence
                })
            
            answers = new_answers
            
            # Check convergence
            if len(set(answers.values())) == 1:
                return list(answers.values())[0]
        
        # Resolution
        if self.judge:
            return self.judge.resolve(question, transcript)
        else:
            return majority_vote(answers)
```

---

### 6.5.2 Collaborative Writing/Coding

Multiple agents collaborate on producing a single artifact (document, codebase, report) with each agent responsible for different aspects.

**Collaboration patterns:**

**1. Sectional Division:**

Each agent writes a different section:

$$
\text{Document} = \text{Merge}(\text{Section}_1^{A_1}, \text{Section}_2^{A_2}, \ldots, \text{Section}_k^{A_k})
$$

**2. Role-Based Pipeline:**

```
Architect → designs structure
  ↓
Developer → writes code
  ↓
Reviewer → reviews and critiques
  ↓
Developer → revises based on review
  ↓
Tester → writes and runs tests
  ↓
Developer → fixes bugs found by tester
```

**3. Pair Programming (Dual-Agent):**

Two agents alternate—one writes, the other reviews in real-time:

$$
\text{for } t = 1, 2, \ldots:
$$

$$
\quad \text{code}_t = A_{\text{writer}}(\text{task}, \text{code}_{<t}, \text{feedback}_{t-1})
$$

$$
\quad \text{feedback}_t = A_{\text{reviewer}}(\text{task}, \text{code}_t)
$$

**Collaborative coding implementation (MetaGPT-style):**

```python
class CollaborativeCodingTeam:
    def __init__(self):
        self.product_manager = Agent(role="Product Manager")
        self.architect = Agent(role="Software Architect")
        self.developer = Agent(role="Senior Developer")
        self.reviewer = Agent(role="Code Reviewer")
        self.tester = Agent(role="QA Engineer")
        self.shared_repo = SharedRepository()
    
    def develop(self, requirements: str) -> CodeBase:
        # Phase 1: Product specification
        spec = self.product_manager.create_spec(requirements)
        
        # Phase 2: Architecture design
        design = self.architect.design(spec)
        self.shared_repo.write("design.md", design)
        
        # Phase 3: Implementation
        for module in design.modules:
            code = self.developer.implement(module, design)
            self.shared_repo.write(module.filepath, code)
            
            # Phase 4: Review
            review = self.reviewer.review(code, design, spec)
            if review.has_issues:
                revised_code = self.developer.revise(code, review)
                self.shared_repo.write(module.filepath, revised_code)
        
        # Phase 5: Testing
        tests = self.tester.write_tests(self.shared_repo, spec)
        test_results = self.tester.run_tests(tests, self.shared_repo)
        
        if test_results.failures:
            # Phase 6: Bug fixing
            fixes = self.developer.fix_bugs(test_results, self.shared_repo)
            self.shared_repo.apply(fixes)
        
        return self.shared_repo.get_codebase()
```

---

### 6.5.3 Division of Labor

Formal division of labor partitions the task into non-overlapping subtasks assigned to agents based on capability:

$$
T = T_1 \uplus T_2 \uplus \cdots \uplus T_n \quad \text{(disjoint partition)}
$$

$$
\text{Assignment}: T_i \rightarrow A_{\sigma(i)} \quad \text{where } \sigma \text{ is a permutation}
$$

**Optimal assignment (Hungarian algorithm analogy):**

Given a cost matrix $C$ where $C_{ij}$ is the cost (inverse quality) of agent $A_j$ performing subtask $T_i$:

$$
\sigma^* = \arg\min_\sigma \sum_{i=1}^{n} C_{i,\sigma(i)}
$$

This is the **linear assignment problem**, solvable in $O(n^3)$ time.

**In LLM MAS**, the cost matrix is estimated by having each agent self-assess its capability for each subtask:

$$
C_{ij} = 1 - P(\text{success} | A_j \text{ performs } T_i) \approx 1 - \text{LLM}_{A_j}(\text{"Can you do } T_i \text{? Rate 0-1"})
$$

---

### 6.5.4 Ensemble Aggregation Across Agents

When multiple agents produce outputs for the same task, ensemble aggregation combines their outputs for improved quality.

**Aggregation methods:**

**1. Best-of-N Selection:**

$$
y^* = \arg\max_{y_i} \text{Score}(y_i) \quad \text{where } y_i = A_i(x)
$$

The scoring function can be an LLM judge, a reward model, or a task-specific metric.

**2. Mixture Aggregation:**

For numerical or probabilistic outputs:

$$
y^* = \sum_{i=1}^{n} w_i \cdot y_i \quad \text{where } \sum_i w_i = 1
$$

**3. LLM-Based Synthesis:**

A synthesis agent reads all outputs and produces a unified result:

$$
y^* = A_{\text{synth}}(x, y_1, y_2, \ldots, y_n)
$$

**4. Self-Consistency (Majority Voting):**

$$
y^* = \arg\max_y \sum_{i=1}^{n} \mathbb{1}[y_i = y]
$$

**Theoretical grounding — Bias-Variance decomposition for ensembles:**

For regression-like aggregation:

$$
\text{MSE}_{\text{ensemble}} = \underbrace{\text{Bias}^2}_{\text{same as individual}} + \underbrace{\frac{1}{n}\text{Variance}}_{\text{reduced by } 1/n}
$$

This shows the $1/n$ variance reduction benefit of ensembling, contingent on agent diversity (low correlation).

For classification via majority vote, the ensemble error rate:

$$
\epsilon_{\text{ensemble}} \leq \exp\left(-\frac{n}{2}(1 - 2\epsilon)^2\right)
$$

where $\epsilon$ is the individual error rate (assuming independence), demonstrating exponential improvement.

---

## 6.6 Competitive and Adversarial Multi-Agent

### 6.6.1 Red Team / Blue Team Architectures

Red team/blue team architectures use **adversarial cooperation**: one team (red) tries to find flaws while the other team (blue) tries to produce robust outputs.

**Architecture:**

```
        ┌────────────────┐      ┌────────────────┐
        │   BLUE TEAM    │      │   RED TEAM     │
        │                │      │                │
        │ • Generator    │      │ • Attacker     │
        │ • Defender     │      │ • Critic       │
        │ • Patcher      │      │ • Adversary    │
        └───────┬────────┘      └───────┬────────┘
                │                       │
                │    ┌──────────┐       │
                └───→│ ARTIFACT │←──────┘
                     │ (output) │
                     └──────────┘
                          │
                     ┌────┴────┐
                     │  JUDGE  │
                     └─────────┘
```

**Interaction protocol:**

1. Blue team produces an output (code, text, plan).
2. Red team attacks it (finds bugs, logical flaws, adversarial inputs, security vulnerabilities).
3. Blue team patches based on red team findings.
4. Iterate until red team cannot find further issues or iteration budget is exhausted.

**Convergence guarantee:**

The quality improves monotonically:

$$
Q(\text{output}^{(r+1)}) \geq Q(\text{output}^{(r)}) \quad \forall r
$$

because each red team attack either finds a flaw (which blue team fixes, improving quality) or fails to find a flaw (quality unchanged).

The convergence bound depends on the red team's **coverage** of the flaw space $\mathcal{F}$:

$$
P(\text{residual flaw after } R \text{ rounds}) \leq (1 - p_{\text{detect}})^R
$$

where $p_{\text{detect}}$ is the per-round probability that the red team detects any remaining flaw.

**Implementation:**

```python
class RedBlueTeam:
    def __init__(self, blue_agents, red_agents, judge, max_rounds=5):
        self.blue = blue_agents
        self.red = red_agents
        self.judge = judge
        self.max_rounds = max_rounds
    
    def adversarial_refinement(self, task: str) -> str:
        # Blue team generates initial output
        output = self.blue.generate(task)
        
        for round_num in range(self.max_rounds):
            # Red team attacks
            attacks = self.red.attack(output, task)
            
            if not attacks.found_issues:
                break  # Red team could not find flaws
            
            # Blue team defends/patches
            output = self.blue.defend(output, attacks, task)
            
            # Judge evaluates
            quality = self.judge.evaluate(output, task)
            if quality >= self.quality_threshold:
                break
        
        return output
```

---

### 6.6.2 Adversarial Robustness Testing

Dedicated adversary agents systematically probe an agent system for weaknesses:

**Testing categories:**

| Category | Adversary Strategy | Goal |
|---|---|---|
| **Prompt injection** | Inject malicious instructions | Test instruction following robustness |
| **Edge case generation** | Generate unusual inputs | Test boundary behavior |
| **Consistency attacks** | Ask the same question differently | Test answer consistency |
| **Resource exhaustion** | Trigger infinite loops, excessive API calls | Test resource management |
| **Goal hijacking** | Subtly redirect the agent's goal | Test goal adherence |

**Formal adversarial testing framework:**

$$
\text{Robustness}(A) = \min_{\delta \in \mathcal{B}_\epsilon} \text{Quality}(A(x + \delta))
$$

where $\mathcal{B}_\epsilon$ is the perturbation ball of radius $\epsilon$ around the nominal input $x$. The adversary searches for the perturbation $\delta$ that maximally degrades quality.

---

### 6.6.3 Game-Theoretic Interactions

When agents have potentially conflicting objectives, game theory provides the formal framework for analyzing and designing their interactions.

#### Nash Equilibria in Multi-Agent Settings

A **Nash Equilibrium** is a strategy profile where no agent can unilaterally improve its utility by changing its strategy:

$$
\forall i, \; u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i
$$

where:
- $s_i^*$: Agent $i$'s equilibrium strategy.
- $s_{-i}^*$: All other agents' equilibrium strategies.
- $u_i$: Agent $i$'s utility function.
- $S_i$: Agent $i$'s strategy space.

**Nash's Existence Theorem:** Every finite game (finite players, finite strategies) has at least one Nash equilibrium in mixed strategies.

**Relevant game-theoretic concepts for LLM MAS:**

**1. Cooperative Games:** Agents form coalitions to maximize joint utility:

$$
v(C) = \max_{\text{joint strategy}} \sum_{i \in C} u_i \quad \text{for coalition } C \subseteq \{1, \ldots, n\}
$$

The **Shapley value** provides a fair allocation of the joint payoff:

$$
\phi_i(v) = \sum_{C \subseteq N \setminus \{i\}} \frac{|C|!(n - |C| - 1)!}{n!} \left[v(C \cup \{i\}) - v(C)\right]
$$

This quantifies each agent's marginal contribution, averaged over all possible orderings of coalition formation.

**2. Mechanism Design (Inverse Game Theory):** Design the rules of interaction so that self-interested agents' equilibrium behavior achieves a desired system-level outcome:

$$
\text{Design } (\mathcal{P}, \mathcal{C}) \text{ such that } \text{NE}(\mathcal{M}) \in \text{DesiredOutcomes}
$$

**3. Principal-Agent Problems:** The orchestrator (principal) delegates to specialist agents (agents) who may have misaligned incentives. The challenge is designing contracts (prompts, reward structures) that align agent behavior with the principal's objectives:

$$
\max_{\text{contract}} u_{\text{principal}} \quad \text{subject to} \quad \text{agent's best response given contract}
$$

**In LLM MAS**, game-theoretic analysis is relevant when:
- Multiple agents compete for limited resources (tokens, tool access, compute).
- Agent outputs are used in adversarial settings (security, negotiations).
- The system must incentivize truthful reporting (agents should not strategically misrepresent their confidence or capabilities).

---

## 6.7 Scalability and Practical Considerations

### 6.7.1 Communication Overhead: $O(n^2)$ in Fully Connected Topologies

In a fully connected communication topology, every agent can communicate with every other agent. The number of directed communication channels is:

$$
|E| = n(n-1) \in O(n^2)
$$

If each communication round involves every pair exchanging messages, the total message volume per round is:

$$
M_{\text{total}} = n(n-1) \cdot \bar{m}
$$

where $\bar{m}$ is the average message size in tokens.

**Cost analysis:**

If each message requires an LLM inference (the receiving agent processes the message), the per-round compute cost is:

$$
C_{\text{round}} = n \cdot \text{Cost}\left(\text{LLM inference with } (n-1) \cdot \bar{m} \text{ input tokens}\right)
$$

Since each agent must read messages from $n-1$ others, the input context grows linearly with $n$, and the cost grows super-linearly:

$$
C_{\text{round}} \in O(n^2 \cdot \bar{m}) \quad \text{(for linear-cost models)}
$$

$$
C_{\text{round}} \in O(n^2 \cdot \bar{m}^2) \quad \text{(for quadratic-attention models)}
$$

**Mitigation strategies:**

| Strategy | Communication Complexity | Trade-off |
|---|---|---|
| **Fully connected** | $O(n^2)$ | Maximum information but maximum cost |
| **Star topology** | $O(n)$ | Hub bottleneck, single point of failure |
| **Ring topology** | $O(n)$ | Slow propagation: $O(n)$ rounds for global info |
| **Hierarchical** | $O(n \log n)$ | Good balance; depth = $O(\log n)$ |
| **Sparse random** | $O(n \cdot k)$, $k \ll n$ | Good scalability, some info loss |
| **Broadcast + filter** | $O(n)$ | Each agent filters relevant messages |

**Practical recommendation:** For $n \leq 5$ agents, fully connected is feasible. For $5 < n \leq 20$, use hierarchical or star topology. For $n > 20$, use sparse topologies with explicit routing protocols.

---

### 6.7.2 Token Budget Explosion in Multi-Agent Conversations

In multi-agent conversations, the token consumption grows alarmingly because each agent's context includes the entire conversation history:

**Token growth analysis:**

Let $\ell$ denote the average tokens per agent turn. After $R$ rounds with $n$ agents:

$$
\text{Total tokens consumed} = \sum_{r=1}^{R} \sum_{i=1}^{n} \underbrace{(r \cdot n \cdot \ell)}_{\text{context at round } r} = n \cdot \ell \cdot \sum_{r=1}^{R} r \cdot n = n^2 \cdot \ell \cdot \frac{R(R+1)}{2}
$$

$$
\text{Total tokens} \in O(n^2 R^2 \ell)
$$

For $n = 5$ agents, $R = 10$ rounds, $\ell = 500$ tokens per turn:

$$
\text{Total} = 25 \cdot 500 \cdot 55 = 687{,}500 \text{ tokens}
$$

At \$3/million input tokens, this costs approximately \$2.06 for a single task. For 1000 tasks/day, costs reach \$2,060/day.

**Mitigation strategies:**

1. **Conversation summarization**: Periodically compress conversation history:

$$
\text{Context}_t = \text{Summarize}(\text{History}_{<t-k}) \oplus \text{History}_{t-k:t}
$$

This reduces context from $O(R \cdot n \cdot \ell)$ to $O(S + k \cdot n \cdot \ell)$ where $S$ is the summary length and $k$ is the recent window size.

2. **Selective communication**: Agents only receive messages relevant to them:

$$
\text{Context}(A_i, t) = \text{filter}(\text{History}_{<t}, \text{relevance}(A_i))
$$

3. **Structured message formats**: Use concise structured messages instead of verbose natural language:

```json
{"from": "Reviewer", "type": "issue", "severity": "high",
 "file": "auth.py", "line": 42, "description": "SQL injection risk"}
```

vs. verbose natural language paragraphs.

4. **Hierarchical summarization**: In hierarchical architectures, managers summarize their team's work before reporting upward:

$$
\text{Report}_{\text{manager}} = \text{Summarize}(\{\text{Report}_{w_i} : w_i \in \text{workers}\})
$$

---

### 6.7.3 Latency Amplification

Multi-agent systems amplify latency through **sequential dependencies** and **communication overhead**.

**Latency model:**

For a pipeline of $k$ sequential agent interactions, each with latency $L_i$:

$$
L_{\text{total}} = \sum_{i=1}^{k} L_i + \sum_{j=1}^{k-1} L_{\text{comm}}^{(j)}
$$

where $L_{\text{comm}}^{(j)}$ is the communication latency between step $j$ and step $j+1$.

For agents in parallel with a synchronization barrier:

$$
L_{\text{parallel}} = \max_{i \in [n]} L_i + L_{\text{sync}}
$$

**Amdahl's Law for MAS:**

If fraction $f$ of the task is inherently sequential and $(1-f)$ is parallelizable across $n$ agents:

$$
\text{Speedup}(n) = \frac{1}{f + \frac{1-f}{n}} \leq \frac{1}{f}
$$

For $f = 0.3$ (30% sequential): maximum speedup = $1/0.3 \approx 3.33\times$ regardless of agent count.

**Mitigation strategies:**

1. **Speculative execution**: Start downstream agents before upstream completes, using predicted outputs.
2. **Async communication**: Agents proceed without waiting for all responses.
3. **Pipeline parallelism**: Overlap computation stages across different tasks.
4. **Caching**: Cache common agent responses for reuse.

---

### 6.7.4 Debugging and Tracing Multi-Agent Interactions

Debugging MAS is qualitatively harder than single-agent debugging because:

1. **Non-determinism**: Multiple agents interacting creates exponentially many possible execution paths.
2. **Distributed state**: No single agent holds the complete system state.
3. **Emergent behaviors**: System-level bugs may not be attributable to any single agent.
4. **Blame attribution**: When the output is wrong, which agent caused the error?

**Tracing infrastructure requirements:**

```python
class MultiAgentTracer:
    def __init__(self):
        self.traces = []
        self.span_stack = []
    
    def start_span(self, agent_id, action_type, metadata):
        span = {
            "span_id": uuid4(),
            "agent_id": agent_id,
            "action_type": action_type,
            "start_time": time.time(),
            "parent_span": self.span_stack[-1]["span_id"] 
                          if self.span_stack else None,
            "metadata": metadata,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "in_progress"
        }
        self.span_stack.append(span)
        self.traces.append(span)
        return span
    
    def end_span(self, span, result, status="success"):
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        span["result_summary"] = summarize(result)
        span["status"] = status
        self.span_stack.pop()
    
    def get_agent_timeline(self, agent_id):
        return [t for t in self.traces if t["agent_id"] == agent_id]
    
    def get_critical_path(self):
        """Find the longest sequential dependency chain."""
        # Build dependency DAG and find longest path
        ...
    
    def blame_analysis(self, final_error):
        """Trace backwards to find the agent that introduced the error."""
        ...
```

**Observability dimensions:**

| Dimension | What to Log | Why |
|---|---|---|
| **Message flow** | All inter-agent messages with timestamps | Reconstruct communication sequence |
| **State snapshots** | Shared state at each modification | Track state evolution |
| **Decision rationale** | Each agent's reasoning trace | Understand why actions were taken |
| **Token usage** | Input/output tokens per agent per turn | Cost attribution and optimization |
| **Latency breakdown** | Time per agent per operation | Identify bottlenecks |
| **Error propagation** | Which agent's error affected which downstream agents | Root cause analysis |

---

### 6.7.5 Failure Propagation and Isolation

In MAS, a failure in one agent can cascade through the system:

**Failure propagation model:**

Define the **failure influence graph** $G_F = (V, E_F)$ where edge $(A_i, A_j) \in E_F$ means a failure in $A_i$ can cause $A_j$ to fail. The probability of cascade failure:

$$
P(\text{system failure}) = 1 - \prod_{i=1}^{n} (1 - p_i \cdot \text{influence}(A_i))
$$

where $p_i$ is agent $i$'s individual failure probability and $\text{influence}(A_i)$ is the fraction of the system affected by $A_i$'s failure.

**Isolation mechanisms:**

1. **Circuit breakers**: If an agent fails $k$ times consecutively, stop sending it tasks:

$$
\text{CircuitBreaker}(A_i) = \begin{cases}
\text{CLOSED} & \text{if recent failures} < k \\
\text{OPEN} & \text{if recent failures} \geq k \\
\text{HALF-OPEN} & \text{after timeout, try one request}
\end{cases}
$$

2. **Bulkhead pattern**: Isolate agent pools so that failures in one pool don't affect others:

```
Pool A (coding tasks):  [A₁, A₂, A₃]  ← isolated
Pool B (review tasks):  [A₄, A₅]      ← isolated
Pool C (testing tasks): [A₆, A₇]      ← isolated
```

3. **Fallback agents**: Designate backup agents for critical roles:

$$
A_{\text{active}} = \begin{cases}
A_{\text{primary}} & \text{if } A_{\text{primary}} \text{ is healthy} \\
A_{\text{backup}} & \text{otherwise}
\end{cases}
$$

4. **Graceful degradation**: If non-critical agents fail, the system continues with reduced capability rather than total failure:

$$
Q_{\text{degraded}} = Q_{\text{full}} \cdot \frac{|\text{healthy agents}|}{n}
$$

5. **Timeout-based isolation**: Kill agent processes that exceed time budgets:

$$
\text{Kill}(A_i) \text{ if } \text{ElapsedTime}(A_i) > T_{\max}(R_i)
$$

---

## 6.8 Frameworks and Implementations

### 6.8.1 AutoGen

**AutoGen** (Microsoft Research) is a framework for building multi-agent conversation systems where agents are defined as **conversable entities** that interact through natural language messages.

**Core concepts:**

| Concept | Description |
|---|---|
| **ConversableAgent** | Base class; every agent can send/receive messages |
| **AssistantAgent** | LLM-backed agent that generates responses |
| **UserProxyAgent** | Represents human; can execute code, solicit human input |
| **GroupChat** | Multi-agent conversation with turn management |
| **GroupChatManager** | Orchestrates speaker selection in group chats |

**Architecture:**

$$
\text{AutoGen MAS} = \text{GroupChatManager}(\{A_1, \ldots, A_n\}, \text{SpeakerSelectionPolicy})
$$

**Speaker selection policies:**
- `auto`: LLM-based speaker selection.
- `round_robin`: Sequential turns.
- `random`: Random agent speaks next.
- `manual`: Human selects the next speaker.

**Example — Software Development Team:**

```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "..."}]

# Define agents
product_manager = autogen.AssistantAgent(
    name="ProductManager",
    system_message="""You are a product manager. Analyze requirements, 
    create user stories, and define acceptance criteria. Do NOT write code.""",
    llm_config={"config_list": config_list}
)

architect = autogen.AssistantAgent(
    name="Architect",
    system_message="""You are a software architect. Design system 
    architecture, define APIs, and choose technology stack. 
    Provide architecture diagrams in text format.""",
    llm_config={"config_list": config_list}
)

developer = autogen.AssistantAgent(
    name="Developer",
    system_message="""You are a senior developer. Write clean, 
    well-documented code. Follow the architect's design. 
    Include error handling and type hints.""",
    llm_config={"config_list": config_list}
)

reviewer = autogen.AssistantAgent(
    name="Reviewer",
    system_message="""You are a code reviewer. Review code for bugs, 
    security issues, performance problems, and style violations. 
    Be specific about line numbers and suggestions.""",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "workspace"},
    is_termination_msg=lambda x: "TASK COMPLETE" in x.get("content", "")
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, product_manager, architect, developer, reviewer],
    messages=[],
    max_round=30,
    speaker_selection_method="auto"
)
manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    llm_config={"config_list": config_list}
)

# Initiate
user_proxy.initiate_chat(
    manager, 
    message="Build a REST API for a todo list application with user auth."
)
```

**Strengths:**
- Flexible conversation patterns (two-agent, group chat, nested chats).
- Built-in code execution with sandboxing.
- Human-in-the-loop support.
- Nested chats enable hierarchical MAS.

**Limitations:**
- Speaker selection in large groups can be unreliable.
- No built-in persistent state management beyond conversation history.
- Debugging complex group chats is difficult.

---

### 6.8.2 CrewAI

**CrewAI** focuses on **role-based agent collaboration** with emphasis on defined roles, goals, and backstories. It models multi-agent systems as "crews" of agents working on "tasks."

**Core abstractions:**

| Concept | Description |
|---|---|
| **Agent** | Autonomous entity with role, goal, backstory, tools |
| **Task** | Unit of work with description, expected output, assigned agent |
| **Crew** | Collection of agents and tasks with a process definition |
| **Process** | Execution strategy: sequential, hierarchical |

**Key differentiators:**
- **Strong role enforcement**: Agents stay in character through detailed role specifications.
- **Task delegation**: Agents can delegate subtasks to other agents.
- **Memory**: Short-term, long-term, and entity memory built in.

**Example:**

```python
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="""You are a veteran researcher at a leading think tank. 
    Known for your ability to find the most relevant information 
    and present it in a clear, concise manner.""",
    tools=[search_tool, arxiv_tool],
    verbose=True,
    allow_delegation=True
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned content strategist known for 
    transforming complex technical topics into engaging narratives. 
    You have a deep understanding of what makes content shareable.""",
    tools=[search_tool],
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest AI agent frameworks released in 2024.",
    expected_output="A detailed report with key findings, comparisons, "
                    "and market analysis.",
    agent=researcher
)

writing_task = Task(
    description="Write a blog post based on the research findings.",
    expected_output="A 1500-word engaging blog post with sections, "
                    "examples, and a conclusion.",
    agent=writer,
    context=[research_task]  # Depends on research output
)

# Assemble crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

**Process types:**

| Process | Execution Flow | Use Case |
|---|---|---|
| `sequential` | Tasks execute in order; output flows to next | Pipeline workflows |
| `hierarchical` | Manager agent delegates and coordinates | Complex projects |

---

### 6.8.3 LangGraph Multi-Agent

**LangGraph** (LangChain) models multi-agent systems as **stateful graphs** where nodes are agent functions and edges are conditional transitions.

**Core concepts:**

| Concept | Description |
|---|---|
| **StateGraph** | Directed graph with typed state |
| **Node** | Agent or function that transforms state |
| **Edge** | Transition between nodes (conditional or unconditional) |
| **State** | Typed dictionary passed between nodes |
| **Checkpointing** | Automatic state persistence for recovery |

**Key architectural advantage:** LangGraph provides **explicit control flow** over agent interactions, unlike the implicit conversation-based flow of AutoGen.

**Multi-agent supervisor pattern:**

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

# Define specialist agents
research_agent = create_react_agent(
    model, tools=[search_tool, arxiv_tool],
    state_modifier="You are a research specialist."
)

coding_agent = create_react_agent(
    model, tools=[python_repl, file_tools],
    state_modifier="You are a coding specialist."
)

# Define supervisor
def supervisor(state: MessagesState):
    """Route to appropriate specialist or finish."""
    response = model.invoke([
        SystemMessage(content="""You are a supervisor managing:
        - 'researcher': for information gathering
        - 'coder': for writing/running code
        Choose which agent to delegate to, or 'FINISH' if done."""),
        *state["messages"]
    ])
    
    if "FINISH" in response.content:
        return {"next": "FINISH"}
    elif "researcher" in response.content.lower():
        return {"next": "researcher"}
    elif "coder" in response.content.lower():
        return {"next": "coder"}

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", research_agent)
workflow.add_node("coder", coding_agent)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next"),
    {"researcher": "researcher", "coder": "coder", "FINISH": END}
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")

app = workflow.compile(checkpointer=MemorySaver())
```

**Strengths:**
- Explicit, inspectable control flow.
- Built-in checkpointing and state persistence.
- Supports cycles (iterative refinement loops).
- Human-in-the-loop at any node.
- Streaming support.

**Limitations:**
- More boilerplate than AutoGen/CrewAI for simple use cases.
- Requires explicit graph design (less "autonomous" agent behavior).

---

### 6.8.4 MetaGPT

**MetaGPT** (Hong et al., 2023) models multi-agent collaboration after **real-world software development processes**, with agents playing roles from a Standardized Operating Procedure (SOP).

**Core innovation:** MetaGPT enforces structured communication through **standardized documents** rather than free-form chat. Each agent produces well-defined artifacts (PRDs, design documents, code, test reports) that serve as the communication medium.

**Agent roles and their artifacts:**

| Role | Artifact Produced | Artifact Consumed |
|---|---|---|
| Product Manager | Product Requirements Document (PRD) | User requirements |
| Architect | System design, API specifications | PRD |
| Project Manager | Task breakdown, sprint plan | System design |
| Engineer | Code implementation | Task breakdown, API specs |
| QA Engineer | Test cases, test reports | Code, PRD |

**Key design principles:**

1. **Structured communication**: Documents replace chat messages, reducing ambiguity:

$$
\text{Communication}_{\text{MetaGPT}} \in \{\text{PRD}, \text{DesignDoc}, \text{Code}, \text{TestReport}\}
$$

vs.

$$
\text{Communication}_{\text{AutoGen}} \in \{\text{free-form natural language}\}
$$

2. **Executable feedback**: Code is actually executed and test results are fed back, providing grounded feedback.

3. **Role-constrained output**: Each agent is constrained to produce only its designated artifact type, preventing role confusion.

**Architecture diagram:**

```
Requirements ──→ [Product Manager] ──PRD──→ [Architect]
                                              │
                                         Design Doc
                                              │
                                              ▼
                                      [Project Manager]
                                              │
                                         Task List
                                              │
                                              ▼
                                        [Engineer(s)]
                                              │
                                           Code
                                              │
                                              ▼
                                       [QA Engineer]
                                              │
                                        Test Report
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                 Tests Pass         Tests Fail
                                    │                   │
                                    ▼                   ▼
                                 RELEASE          [Engineer] fixes
                                                        │
                                                        └──→ [QA] retests
```

---

### 6.8.5 CAMEL

**CAMEL** (Communicative Agents for "Mind" Exploration of Large Language Model Society; Li et al., 2023) explores **role-playing** between LLM agents through structured dialogue.

**Core concept:** Two agents are assigned complementary roles (e.g., "AI researcher" and "Python programmer") and engage in a task-oriented conversation where one acts as the **task specifier** (user role) and the other as the **task executor** (assistant role).

**CAMEL protocol:**

1. **Role assignment**: The system assigns one agent the "user" role and another the "assistant" role:

$$
(A_{\text{user}}, R_{\text{user}}) \quad \text{and} \quad (A_{\text{asst}}, R_{\text{asst}})
$$

2. **Task specification**: A task specifier agent refines the initial vague idea into a concrete task:

$$
T_{\text{specific}} = A_{\text{specifier}}(T_{\text{vague}}, R_{\text{user}}, R_{\text{asst}})
$$

3. **Role-playing conversation**: Agents converse within their assigned roles to collaboratively complete the task:

$$
\text{for } t = 1, 2, \ldots: \quad m_t^{\text{user}} = A_{\text{user}}(T, m_{<t}) \quad \Rightarrow \quad m_t^{\text{asst}} = A_{\text{asst}}(T, m_{\leq t})
$$

4. **Inception prompting**: Specialized prompts prevent agents from deviating from their roles, a technique called "inception prompting":

```
Never forget you are a {assistant_role} and I am a {user_role}.
Never flip roles! Never instruct me!
We must complete the task collaboratively.
The task is: {task}
You must respond with a specific solution step.
```

**Research contributions:**

1. **Role-playing dataset generation**: CAMEL generates large-scale conversational datasets by enumerating role pairs and tasks.
2. **Behavioral analysis**: Studies emergent behaviors (flattening, deception, hallucination) in multi-agent role-play.
3. **Scalable exploration**: Combinatorial explosion of role pairs enables broad exploration of LLM social behavior.

**Framework comparison matrix:**

| Feature | AutoGen | CrewAI | LangGraph | MetaGPT | CAMEL |
|---|---|---|---|---|---|
| **Architecture** | Conversation-based | Role-based crew | Graph-based | SOP-based | Role-playing |
| **Control flow** | Implicit (chat) | Sequential/Hierarchical | Explicit graph | Pipeline | Turn-taking |
| **Communication** | Free-form messages | Task outputs | State mutations | Structured docs | Role-play dialogue |
| **Code execution** | Built-in | Via tools | Via tools | Built-in + testing | Via tools |
| **Human-in-loop** | Yes (UserProxy) | Limited | Yes (interrupt) | No | No |
| **State management** | Conversation history | Implicit | Explicit typed state | Artifacts | Conversation |
| **Checkpointing** | No | No | Yes (built-in) | No | No |
| **Best for** | Flexible multi-agent chat | Structured team workflows | Complex control flows | Software development | Research & data generation |
| **Scalability** | Moderate | Moderate | High | Moderate | Low-moderate |
| **Debugging** | Difficult | Moderate | Best (graph visualization) | Moderate (artifact inspection) | Moderate |

---

**Chapter Summary — Key Takeaways:**

1. Multi-agent systems extend single-agent capabilities through cognitive division of labor, exploiting specialization, redundancy, and parallelism. The formal framework $\mathcal{M} = \langle \{A_1, \ldots, A_n\}, \mathcal{E}, \mathcal{P}, \mathcal{C} \rangle$ captures agents, environment, protocol, and communication channel.

2. Architectural choices (centralized, decentralized, hierarchical, blackboard, market-based) fundamentally determine scalability, fault tolerance, and coordination efficiency. No single architecture dominates—the optimal choice depends on task structure.

3. Agent roles and specialization are the primary mechanism for creating functional diversity from identical base LLMs. Persona-based design, dynamic role allocation, and meta-agents provide increasing sophistication.

4. Coordination mechanisms (turn-taking, shared state, task queues, voting, leader election) are essential infrastructure. The choice of mechanism determines communication patterns and efficiency.

5. Cooperative patterns (debate, collaborative coding, division of labor, ensemble aggregation) improve output quality through structured interaction. Adversarial patterns (red/blue team, game-theoretic analysis) improve robustness through intentional opposition.

6. Scalability is bounded by $O(n^2)$ communication complexity, token budget explosion ($O(n^2 R^2)$), and latency amplification. Practical MAS must employ topology optimization, message compression, and failure isolation.

7. Production frameworks (AutoGen, CrewAI, LangGraph, MetaGPT, CAMEL) offer different trade-offs between flexibility, structure, debuggability, and ease of use. LangGraph provides the strongest engineering foundations; MetaGPT provides the most structured software development workflow; AutoGen provides the most flexible conversational patterns.