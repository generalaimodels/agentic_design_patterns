

# Chapter 19: Prioritization

---

## 19.1 Definition and Formal Framework

### 19.1.1 What is Prioritization in Agentic Workflows

Prioritization is the systematic process by which an autonomous agent—or an orchestration layer governing multiple agents—**determines the relative ordering, urgency, and resource allocation** across a set of pending tasks, sub-goals, tool invocations, and inter-agent communications. Unlike simple FIFO (first-in, first-out) processing, prioritization introduces a **discriminative ordering function** that assigns differential processing precedence based on task attributes, environmental context, system constraints, and strategic objectives.

**Formal Definition.** Let $\mathcal{T} = \{T_1, T_2, \ldots, T_n\}$ denote a finite task set pending execution at time $t$. Prioritization is a mapping:

$$\pi: \mathcal{T} \rightarrow \mathbb{R}$$

that assigns a scalar priority value $\pi(T_i)$ to each task $T_i$, inducing a total preorder $\preceq_\pi$ on $\mathcal{T}$ such that:

$$T_i \preceq_\pi T_j \iff \pi(T_i) \leq \pi(T_j)$$

The agent's execution engine selects the next task as:

$$T^* = \arg\max_{T_i \in \mathcal{T}_{\text{ready}}} \pi(T_i)$$

where $\mathcal{T}_{\text{ready}} \subseteq \mathcal{T}$ is the subset of tasks whose dependency constraints are fully satisfied.

**Why Prioritization is Critical in Agentic Systems.** Agentic workflows differ from traditional software pipelines in several fundamental ways that make prioritization non-trivial:

| Property | Traditional Pipeline | Agentic Workflow |
|---|---|---|
| Task arrival | Deterministic, batch | Stochastic, streaming |
| Task dependencies | Static DAG | Dynamic, discovered at runtime |
| Resource cost | Predictable | Variable (LLM tokens, API calls) |
| Task value | Uniform | Heterogeneous, context-dependent |
| Failure modes | Binary | Partial, recoverable, cascading |
| Environmental context | Fixed | Continuously shifting |

**Key Roles of Prioritization:**

1. **Resource efficiency**: LLM inference, tool calls, and memory retrievals carry non-trivial computational and monetary costs. Prioritization ensures that the highest-value tasks consume resources first, maximizing utility per unit cost.

2. **Latency management**: User-facing tasks demand lower latency than background maintenance operations. Without prioritization, critical responses may be delayed behind lower-value computations.

3. **Dependency-aware execution**: Tasks form directed acyclic graphs (DAGs) where certain outputs feed into downstream computations. Prioritization must respect these structural constraints while optimizing the execution order within feasible orderings.

4. **Goal alignment**: An agent pursuing multiple sub-goals simultaneously must allocate attention proportional to each sub-goal's contribution to the overarching objective, requiring continuous priority reassessment.

5. **Graceful degradation**: Under resource contention or rate limiting, the agent must shed load intelligently—deprioritizing or deferring low-value tasks while preserving critical functionality.

**Prioritization in the Agent Loop.** Within a canonical Observe-Orient-Decide-Act (OODA) agent loop, prioritization operates at the **Decide** phase:

```
while not terminated:
    observations = perceive(environment)          # Observe
    context = update_world_model(observations)     # Orient
    task_queue = generate_candidate_tasks(context)  # Orient
    ordered_tasks = prioritize(task_queue, context) # Decide  ← PRIORITIZATION
    result = execute(ordered_tasks.top())           # Act
    update_state(result)
```

Prioritization is therefore not a one-time computation but a **recurrent, state-dependent decision** that is re-evaluated at every iteration of the agent loop as context evolves.

---

### 19.1.2 Priority Function: $p(T_i) = f(\text{urgency}(T_i), \text{importance}(T_i), \text{cost}(T_i), \text{dependency}(T_i))$

The priority function $p: \mathcal{T} \times \mathcal{C} \rightarrow \mathbb{R}$ maps each task $T_i$ and the current context $\mathcal{C}$ to a scalar priority score. This function must integrate multiple heterogeneous signals into a coherent ranking. We formalize each component axis and their composition.

---

**Axis 1: Urgency — $u(T_i, t)$**

Urgency captures the time-sensitivity of a task. It is a function of elapsed time, deadlines, and temporal decay of task value.

$$u(T_i, t) = \begin{cases} \frac{1}{t_{\text{deadline}}(T_i) - t} & \text{if } t < t_{\text{deadline}}(T_i) \\ +\infty & \text{if } t \geq t_{\text{deadline}}(T_i) \end{cases}$$

For tasks without hard deadlines, a **soft urgency** model uses exponential decay of task freshness:

$$u_{\text{soft}}(T_i, t) = e^{\lambda (t - t_{\text{arrival}}(T_i))}$$

where $\lambda > 0$ is the urgency growth rate. Higher $\lambda$ values cause tasks to become urgent more rapidly as they age, penalizing long queue residence times.

**Generalized urgency** combines both deadline-driven and aging-driven components:

$$u(T_i, t) = \alpha \cdot \frac{1}{\max(\epsilon, \; t_{\text{deadline}}(T_i) - t)} + (1 - \alpha) \cdot e^{\lambda(t - t_{\text{arrival}}(T_i))}$$

where $\alpha \in [0, 1]$ balances deadline pressure against age-based urgency, and $\epsilon > 0$ prevents division by zero.

---

**Axis 2: Importance — $\iota(T_i)$**

Importance measures the intrinsic value or strategic significance of completing a task, independent of time pressure. This is typically derived from:

- **Goal alignment score**: How directly the task contributes to the agent's terminal objective.
- **User-specified priority tier**: Explicit labels (critical, high, medium, low) mapped to numerical values.
- **Expected downstream impact**: Number and value of dependent tasks that are blocked until $T_i$ completes.

Formally, if $\mathcal{G}$ is the agent's goal hierarchy with goal weights $w_g$:

$$\iota(T_i) = \sum_{g \in \mathcal{G}} w_g \cdot \text{relevance}(T_i, g)$$

where $\text{relevance}(T_i, g) \in [0, 1]$ quantifies the degree to which task $T_i$ advances goal $g$.

For tasks with downstream dependents, importance is augmented by **dependency fan-out**:

$$\iota_{\text{aug}}(T_i) = \iota(T_i) + \gamma \sum_{T_j \in \text{dependents}(T_i)} \iota(T_j)$$

where $\gamma \in (0, 1)$ is a discount factor preventing unbounded recursion in deep dependency chains.

---

**Axis 3: Cost — $\kappa(T_i)$**

Cost represents the resources required to execute a task: LLM tokens, API call latency, monetary expense, memory consumption, or compute cycles.

$$\kappa(T_i) = \sum_{r \in \mathcal{R}} \beta_r \cdot c_r(T_i)$$

where $\mathcal{R}$ is the set of resource types, $c_r(T_i)$ is the estimated consumption of resource $r$ by task $T_i$, and $\beta_r$ is the per-unit cost weight for resource $r$.

Cost enters the priority function as an **efficiency modifier**. Two canonical formulations exist:

**Value-density formulation** (analogous to knapsack value density):

$$p_{\text{density}}(T_i) = \frac{\iota(T_i) \cdot u(T_i, t)}{\kappa(T_i)}$$

**Cost-penalized formulation**:

$$p_{\text{penalized}}(T_i) = \iota(T_i) \cdot u(T_i, t) - \mu \cdot \kappa(T_i)$$

where $\mu > 0$ is the cost sensitivity coefficient.

---

**Axis 4: Dependency — $\delta(T_i)$**

Dependency captures the structural position of a task within the task DAG. A task that blocks many downstream operations should receive elevated priority to maximize overall throughput.

Let $G = (\mathcal{T}, E)$ be the task dependency DAG where $(T_i, T_j) \in E$ indicates $T_j$ depends on $T_i$. The dependency factor is:

$$\delta(T_i) = |\text{descendants}(T_i, G)| + \eta \cdot \text{critical\_path\_membership}(T_i, G)$$

where $\text{descendants}(T_i, G)$ is the transitive closure of tasks reachable from $T_i$, and $\text{critical\_path\_membership}(T_i, G) \in \{0, 1\}$ indicates whether $T_i$ lies on the longest path in the DAG.

Additionally, a task is **ready** only if all its predecessors have completed:

$$\text{ready}(T_i) = \bigwedge_{T_k \in \text{parents}(T_i)} \text{completed}(T_k)$$

---

**Composite Priority Function.** The general priority function aggregates all axes:

$$p(T_i, t) = w_u \cdot \hat{u}(T_i, t) + w_\iota \cdot \hat{\iota}(T_i) + w_\delta \cdot \hat{\delta}(T_i) - w_\kappa \cdot \hat{\kappa}(T_i)$$

where $\hat{(\cdot)}$ denotes normalized values (e.g., min-max or z-score normalization across the current task set) and $\{w_u, w_\iota, w_\delta, w_\kappa\}$ are learnable or hand-tuned weights satisfying $\sum w = 1$.

Alternatively, a **multiplicative formulation** prevents any single zero-valued axis from being overridden:

$$p(T_i, t) = u(T_i, t)^{w_u} \cdot \iota(T_i)^{w_\iota} \cdot \delta(T_i)^{w_\delta} \cdot \kappa(T_i)^{-w_\kappa}$$

Taking logarithms converts this to the additive form, confirming mathematical equivalence under log-transformation.

---

**Priority Function Properties (Desiderata):**

| Property | Formal Requirement | Rationale |
|---|---|---|
| Monotonicity in urgency | $u(T_i, t_1) \leq u(T_i, t_2)$ for $t_1 < t_2$ | Priority grows as deadlines approach |
| Transitivity | $p(T_i) \geq p(T_j)$ and $p(T_j) \geq p(T_k) \Rightarrow p(T_i) \geq p(T_k)$ | Consistent total ordering |
| Sensitivity | $\frac{\partial p}{\partial u} > 0$, $\frac{\partial p}{\partial \iota} > 0$, $\frac{\partial p}{\partial \kappa} < 0$ | Priority increases with urgency/importance, decreases with cost |
| Boundedness | $p(T_i) \in [p_{\min}, p_{\max}]$ for non-deadline tasks | Prevents numerical instability |
| Context-adaptivity | $p(T_i, \mathcal{C}_1) \neq p(T_i, \mathcal{C}_2)$ when $\mathcal{C}_1 \neq \mathcal{C}_2$ | Responds to environmental changes |

---

### 19.1.3 Prioritization vs. Scheduling vs. Routing

These three concepts are often conflated in agentic AI literature. They occupy distinct but interdependent layers of the task management hierarchy.

**Formal Definitions:**

**Prioritization** determines *what matters most*:
$$\pi: \mathcal{T} \rightarrow \mathbb{R} \quad \text{(ordering function)}$$

**Scheduling** determines *when to execute*:
$$\sigma: \mathcal{T} \rightarrow \mathbb{R}_{\geq 0} \quad \text{(time assignment function)}$$

**Routing** determines *where/who executes*:
$$\rho: \mathcal{T} \rightarrow \mathcal{A} \quad \text{(agent/resource assignment function)}$$

where $\mathcal{A}$ is the set of available agents or execution endpoints.

---

**Dimensional Comparison:**

| Dimension | Prioritization | Scheduling | Routing |
|---|---|---|---|
| Core question | What should be done first? | When should it be done? | Who/where should do it? |
| Input | Task attributes, context | Priority ordering, resource calendar | Task requirements, agent capabilities |
| Output | Ranked task list | Time-slot assignments | Agent-task mapping |
| Constraints | Dependency structure, deadlines | Resource availability, concurrency limits | Capacity, specialization, locality |
| Optimization target | Maximize aggregate value | Minimize makespan or tardiness | Minimize mismatch, balance load |
| Temporal scope | Relative ordering (atemporal) | Absolute temporal placement | Spatial/logical placement |
| Recurrence | Every decision cycle | Periodically or event-driven | Per-task or per-batch |

---

**Interaction Pattern:**

The three functions compose in a pipeline:

$$\text{Execution Plan} = \rho \circ \sigma \circ \pi(\mathcal{T})$$

1. **Prioritization** produces a ranked list: $[T_{(1)}, T_{(2)}, \ldots, T_{(n)}]$ where $\pi(T_{(1)}) \geq \pi(T_{(2)}) \geq \ldots$
2. **Scheduling** assigns temporal slots respecting the priority ordering and resource constraints: $\sigma(T_{(i)}) = t_i^{\text{start}}$
3. **Routing** maps each scheduled task to an executor: $\rho(T_{(i)}) = a_k \in \mathcal{A}$

**Critical distinction**: Prioritization is a *necessary precondition* for intelligent scheduling and routing. Without priority information, schedulers default to FIFO or round-robin policies that are provably suboptimal under heterogeneous task valuations.

**Example — Agentic Customer Support System:**

- A user submits a **billing dispute** (high importance), a **feature request** (low importance), and a **service outage report** (critical urgency).
- **Prioritization** ranks: outage > billing > feature request.
- **Scheduling** determines: outage is processed immediately, billing in 2 minutes, feature request queued for batch processing.
- **Routing** assigns: outage → specialized incident-response agent, billing → finance-domain agent, feature request → product feedback collector agent.

---

## 19.2 Priority Assignment Strategies

### 19.2.1 Static Priority Assignment

Static priority assignment pre-defines fixed priority levels that remain constant throughout the task's lifecycle. This is the simplest and most interpretable approach, suitable for systems with well-understood task taxonomies.

---

**Fixed Priority Levels (Critical, High, Medium, Low)**

A discrete priority space $\mathcal{P} = \{p_{\text{critical}}, p_{\text{high}}, p_{\text{medium}}, p_{\text{low}}\}$ with numerical mappings:

$$\text{priority\_map}: \mathcal{P} \rightarrow \mathbb{R} = \{p_{\text{critical}} \mapsto 4, \; p_{\text{high}} \mapsto 3, \; p_{\text{medium}} \mapsto 2, \; p_{\text{low}} \mapsto 1\}$$

**Formal Properties:**

- **Cardinality**: $|\mathcal{P}|$ is typically small (3–7 levels). The **Miller bound** ($7 \pm 2$) from cognitive science applies: too many levels reduce discriminability.
- **Deterministic**: $p(T_i) = \text{priority\_map}(\text{label}(T_i))$ — no runtime computation required.
- **Static**: $\frac{\partial p(T_i)}{\partial t} = 0$ — the priority does not change over time.

**Implementation:**

```python
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any
import heapq

class Priority(IntEnum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: str = field(compare=False)
    payload: Any = field(compare=False)

class StaticPriorityQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0  # Tie-breaking by arrival order
    
    def enqueue(self, task_id: str, priority: Priority, payload: Any):
        # Negate priority for max-heap behavior with heapq (min-heap)
        entry = (-priority.value, self._counter, PrioritizedTask(priority, task_id, payload))
        heapq.heappush(self._queue, entry)
        self._counter += 1
    
    def dequeue(self) -> PrioritizedTask:
        if not self._queue:
            raise IndexError("Queue is empty")
        _, _, task = heapq.heappop(self._queue)
        return task
    
    def peek(self) -> PrioritizedTask:
        if not self._queue:
            raise IndexError("Queue is empty")
        return self._queue[0][2]
    
    def __len__(self):
        return len(self._queue)
```

**Advantages:**
1. Zero computational overhead at assignment time.
2. Fully interpretable and auditable.
3. Compatible with any downstream scheduler.

**Limitations:**
1. Cannot adapt to changing conditions (approaching deadlines, resource scarcity).
2. Leads to **priority inversion** when a low-priority task holds a resource needed by a high-priority task.
3. Suffers from **priority clustering** when most tasks receive the same label, degenerating to FIFO within that level.

---

**User-Defined Priority Tags**

Users or upstream systems annotate tasks with explicit priority metadata:

$$T_i = (\text{payload}_i, \; \text{metadata}_i, \; \text{tag}_i)$$

where $\text{tag}_i \in \mathcal{P}$ is a user-supplied priority label.

**Tag Validation and Normalization:**

User-supplied priorities may be unreliable (e.g., every task marked "critical"). A normalization step enforces distributional constraints:

$$\text{tag}'_i = \begin{cases} \text{tag}_i & \text{if } \frac{|\{T_j : \text{tag}_j = \text{tag}_i\}|}{|\mathcal{T}|} \leq \theta_{\text{tag}_i} \\ \text{downgrade}(\text{tag}_i) & \text{otherwise} \end{cases}$$

where $\theta_{\text{tag}_i}$ is the maximum allowed proportion for priority level $\text{tag}_i$ (e.g., at most 10% of tasks may be "critical").

**Hierarchical Tag Resolution:**

When tasks originate from multiple users with different authority levels:

$$p_{\text{effective}}(T_i) = \text{priority\_map}(\text{tag}_i) \cdot \text{authority}(\text{user}(T_i))$$

where $\text{authority}: \mathcal{U} \rightarrow [0, 1]$ maps users to their priority-setting authority weight.

---

### 19.2.2 Dynamic Priority Computation

Dynamic priority computation assigns and continuously updates priorities based on runtime state, elapsed time, resource availability, and evolving context.

---

**Deadline-Based Prioritization**

$$p(T_i, t) \propto \frac{1}{t_{\text{deadline}}(T_i) - t}$$

This is the **Earliest Deadline First (EDF)** policy, which is optimal for single-processor, preemptive scheduling of periodic tasks (Liu & Layland, 1973).

**Full Formulation with Feasibility Check:**

$$p_{\text{EDF}}(T_i, t) = \begin{cases} \frac{1}{t_{\text{deadline}}(T_i) - t} & \text{if } t < t_{\text{deadline}}(T_i) \\ p_{\text{max}} + \text{penalty} \cdot (t - t_{\text{deadline}}(T_i)) & \text{if } t \geq t_{\text{deadline}}(T_i) \end{cases}$$

The penalty term for overdue tasks ensures they are processed immediately (if still valuable) or explicitly abandoned.

**Slack-Based Variant (Least Slack Time First):**

$$\text{slack}(T_i, t) = (t_{\text{deadline}}(T_i) - t) - \text{remaining\_execution\_time}(T_i)$$

$$p_{\text{LST}}(T_i, t) = \frac{1}{\max(\epsilon, \; \text{slack}(T_i, t))}$$

This accounts for the estimated remaining execution time, providing a more nuanced urgency signal than raw deadline proximity. Tasks with negative slack are already infeasible under sequential execution.

**Schedulability Condition (EDF):**

For a task set to be feasible under EDF on a single processor:

$$\sum_{i=1}^{n} \frac{e_i}{d_i} \leq 1$$

where $e_i$ is execution time and $d_i$ is the relative deadline for task $T_i$.

---

**Value-Based Prioritization**

Not all tasks have deadlines. Value-based prioritization assigns priority proportional to the **expected value** of task completion:

$$p_{\text{value}}(T_i) = \mathbb{E}[V(T_i)] = \sum_{o \in \text{outcomes}(T_i)} P(o) \cdot V(o)$$

where $V(o)$ is the value of outcome $o$ and $P(o)$ is its probability.

**Value with Temporal Discounting:**

$$p_{\text{value}}(T_i, t) = V(T_i) \cdot e^{-\rho (t_{\text{completion}}(T_i) - t)}$$

where $\rho > 0$ is the temporal discount rate reflecting the diminishing value of delayed task completion. This model is grounded in **hyperbolic discounting** from behavioral economics.

**Expected Value of Information (EVOI):**

For tasks that produce information (e.g., research queries, data retrievals), the priority can be set by the expected reduction in decision uncertainty:

$$p_{\text{EVOI}}(T_i) = H(\text{Decision} \mid \text{current info}) - \mathbb{E}_{o \sim T_i}[H(\text{Decision} \mid \text{current info} \cup o)]$$

where $H(\cdot)$ is Shannon entropy. Tasks that maximally reduce decision entropy receive highest priority.

---

**Cost-Weighted Prioritization**

Incorporating execution cost produces a **value-density** formulation analogous to the fractional knapsack problem:

$$p_{\text{cost-weighted}}(T_i) = \frac{V(T_i)}{\kappa(T_i)}$$

This prioritizes tasks with the highest value-to-cost ratio, maximizing total value delivered per unit of resource expenditure.

**Budget-Constrained Formulation:**

Given a remaining budget $B$ (in tokens, API calls, or compute-seconds):

$$\max_{\mathcal{S} \subseteq \mathcal{T}} \sum_{T_i \in \mathcal{S}} V(T_i) \quad \text{s.t.} \quad \sum_{T_i \in \mathcal{S}} \kappa(T_i) \leq B$$

This is the 0-1 knapsack problem (NP-hard), but the greedy approximation using cost-weighted priority achieves a 2-approximation for the 0-1 case and is optimal for the fractional relaxation.

**Dynamic Cost Estimation:**

LLM-based tasks have variable costs depending on prompt complexity, required reasoning depth, and output length. Cost estimation uses:

$$\hat{\kappa}(T_i) = \hat{\kappa}_{\text{input}}(T_i) + \hat{\kappa}_{\text{output}}(T_i) + \hat{\kappa}_{\text{tools}}(T_i)$$

where:
- $\hat{\kappa}_{\text{input}}(T_i) = \text{token\_count}(\text{prompt}(T_i)) \cdot c_{\text{input}}$
- $\hat{\kappa}_{\text{output}}(T_i) = \hat{L}_{\text{output}}(T_i) \cdot c_{\text{output}}$ (estimated output length × cost per token)
- $\hat{\kappa}_{\text{tools}}(T_i) = \sum_{k} P(\text{tool}_k \text{ invoked}) \cdot c_{\text{tool}_k}$ (expected tool invocation costs)

---

### 19.2.3 LLM-Based Priority Assessment

When task attributes are unstructured (natural language descriptions, ambiguous requirements), traditional scoring functions are insufficient. LLMs serve as **learned priority estimators** that can parse semantic content and assign priorities through in-context reasoning.

---

**Context-Aware Priority Scoring**

An LLM receives the task description, current agent state, and priority criteria, then outputs a numerical priority score:

$$p_{\text{LLM}}(T_i) = \text{LLM}(\text{prompt}_{\text{priority}}(T_i, \mathcal{C}, \mathcal{T}))$$

**Prompt Template:**

```
You are a priority assessment module for an autonomous agent system.

Current context:
- Active goals: {goals}
- Resource budget remaining: {budget}
- Time elapsed: {elapsed}
- Pending tasks: {task_summaries}

Task to evaluate:
- Description: {task_description}
- Requester: {requester}
- Dependencies: {dependencies}
- Estimated cost: {cost_estimate}

Evaluate this task on the following axes (each 0-10):
1. Urgency: How time-sensitive is this task?
2. Importance: How critical is this task to the overall objective?
3. Blocking factor: How many other tasks depend on this?
4. Cost efficiency: How favorable is the value-to-cost ratio?

Output a JSON object:
{
    "urgency": <float>,
    "importance": <float>,
    "blocking_factor": <float>,
    "cost_efficiency": <float>,
    "composite_priority": <float>,
    "reasoning": "<brief justification>"
}
```

**Calibration of LLM Scores:**

Raw LLM outputs may be poorly calibrated (e.g., clustering around 7/10). Post-processing calibration uses:

$$p_{\text{calibrated}}(T_i) = \Phi^{-1}\left(\text{rank}(p_{\text{LLM}}(T_i)) / (|\mathcal{T}| + 1)\right)$$

where $\Phi^{-1}$ is the inverse standard normal CDF, transforming ranked outputs into a Gaussian distribution. This ensures uniform separation between priority levels.

**Alternatively**, temperature-scaled softmax normalization:

$$p_{\text{normalized}}(T_i) = \frac{e^{p_{\text{LLM}}(T_i)/\tau}}{\sum_{j} e^{p_{\text{LLM}}(T_j)/\tau}}$$

where $\tau$ controls the sharpness of the priority distribution. Low $\tau$ produces winner-take-all priority; high $\tau$ produces more uniform priorities.

---

**Multi-Criteria Priority Ranking**

Instead of scoring individual tasks, the LLM directly produces a **ranking** over the entire task set, leveraging its ability to perform comparative reasoning:

$$\text{LLM}(\mathcal{T}, \mathcal{C}) \rightarrow \sigma: \{1, \ldots, n\} \rightarrow \mathcal{T}$$

where $\sigma$ is a permutation representing the priority ordering.

**Pairwise Comparison Approach:**

For improved consistency, use pairwise comparisons and aggregate:

$$\text{prefer}(T_i, T_j) = \text{LLM}(\text{"Which task has higher priority: } T_i \text{ or } T_j \text{?"})$$

Aggregate pairwise preferences using the **Bradley-Terry model**:

$$P(T_i \succ T_j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}}$$

where scores $\{s_i\}$ are estimated via maximum likelihood over the pairwise comparison results. This yields a consistent global ranking even when the LLM's pointwise scores are noisy.

**Complexity Consideration:** Pairwise comparison requires $O(n^2)$ LLM calls for $n$ tasks. For large task sets, use **Swiss-system tournament** sampling ($O(n \log n)$ comparisons) or **active sorting** algorithms that query the LLM only for uncertain pairs.

**Implementation with Consistency Verification:**

```python
import numpy as np
from itertools import combinations

class LLMPriorityRanker:
    def __init__(self, llm_client, consistency_threshold=0.8):
        self.llm = llm_client
        self.consistency_threshold = consistency_threshold
    
    def rank_tasks(self, tasks: list, context: dict) -> list:
        n = len(tasks)
        if n <= 1:
            return tasks
        
        # Pairwise comparison matrix
        pref_matrix = np.zeros((n, n))
        
        for i, j in combinations(range(n), 2):
            preference = self._compare(tasks[i], tasks[j], context)
            pref_matrix[i][j] = preference
            pref_matrix[j][i] = 1 - preference
        
        # Check transitivity consistency
        consistency = self._check_consistency(pref_matrix)
        
        if consistency < self.consistency_threshold:
            # Re-query inconsistent pairs
            pref_matrix = self._resolve_inconsistencies(pref_matrix, tasks, context)
        
        # Bradley-Terry score estimation
        scores = self._bradley_terry_mle(pref_matrix)
        
        # Return tasks sorted by descending score
        ranked_indices = np.argsort(-scores)
        return [tasks[i] for i in ranked_indices]
    
    def _compare(self, task_a, task_b, context):
        prompt = f"""Compare the priority of these two tasks given the current context.
        Context: {context}
        Task A: {task_a}
        Task B: {task_b}
        
        Which task should be executed first? 
        Respond with a probability (0.0 = definitely B first, 1.0 = definitely A first)."""
        
        response = self.llm.generate(prompt)
        return float(response)
    
    def _check_consistency(self, pref_matrix):
        n = pref_matrix.shape[0]
        violations = 0
        total_triples = 0
        for i, j, k in combinations(range(n), 3):
            total_triples += 1
            # Check transitivity: if i > j and j > k, then i > k
            if (pref_matrix[i][j] > 0.5 and pref_matrix[j][k] > 0.5 
                and pref_matrix[i][k] <= 0.5):
                violations += 1
        return 1.0 - (violations / max(1, total_triples))
    
    def _bradley_terry_mle(self, pref_matrix, max_iter=100, tol=1e-6):
        n = pref_matrix.shape[0]
        scores = np.ones(n)
        
        for _ in range(max_iter):
            old_scores = scores.copy()
            for i in range(n):
                numerator = np.sum(pref_matrix[i])
                denominator = np.sum(
                    (pref_matrix[i, j] + pref_matrix[j, i]) / 
                    (scores[i] + scores[j])
                    for j in range(n) if j != i
                )
                scores[i] = numerator / max(denominator, 1e-10)
            
            scores /= np.sum(scores)  # Normalize
            if np.max(np.abs(scores - old_scores)) < tol:
                break
        
        return scores
```

---

## 19.3 Task Scheduling and Queue Management

### 19.3.1 Priority Queues for Agent Task Management

A **priority queue** is the fundamental data structure for maintaining tasks in priority order with efficient insertion and extraction.

**Abstract Interface:**

$$\text{PriorityQueue} = \langle \text{insert}(T_i, p_i), \; \text{extract\_max}(), \; \text{peek}(), \; \text{update}(T_i, p_i'), \; \text{delete}(T_i) \rangle$$

**Implementation Options and Complexity:**

| Data Structure | Insert | Extract-Max | Update | Peek | Delete |
|---|---|---|---|---|---|
| Sorted Array | $O(n)$ | $O(1)$ | $O(n)$ | $O(1)$ | $O(n)$ |
| Binary Heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(1)$ | $O(\log n)$ |
| Fibonacci Heap | $O(1)^*$ | $O(\log n)^*$ | $O(1)^*$ | $O(1)$ | $O(\log n)^*$ |
| Skip List | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(1)$ | $O(\log n)$ |

$^*$Amortized complexity.

For agentic systems, the **update** operation is critical because dynamic re-prioritization requires modifying priorities of already-queued tasks. Binary heaps with an auxiliary hash map (for $O(1)$ task lookup) provide the best practical tradeoff.

**Multi-Level Priority Queue (MLPQ):**

For discrete priority levels, a more efficient structure uses one FIFO queue per priority level:

$$\text{MLPQ} = \{Q_{\text{critical}}, Q_{\text{high}}, Q_{\text{medium}}, Q_{\text{low}}\}$$

Extraction always selects from the highest non-empty queue:

$$\text{extract\_max}() = Q_k.\text{dequeue}() \quad \text{where } k = \max\{l : Q_l \neq \emptyset\}$$

This achieves $O(1)$ insertion and $O(L)$ extraction where $L = |\mathcal{P}|$ is the number of priority levels (constant for fixed level counts).

**Implementation with Dynamic Priorities:**

```python
import heapq
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

@dataclass
class AgentTask:
    task_id: str
    description: str
    created_at: float
    deadline: Optional[float] = None
    base_importance: float = 1.0
    estimated_cost: float = 1.0
    dependencies: list = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed

class DynamicPriorityQueue:
    """Priority queue with dynamic re-prioritization support."""
    
    def __init__(self, priority_fn: Callable[[AgentTask, float], float]):
        self._tasks: dict[str, AgentTask] = {}
        self._priority_fn = priority_fn
        self._removed = set()  # Lazy deletion markers
        self._heap = []
        self._counter = 0
    
    def add_task(self, task: AgentTask):
        self._tasks[task.task_id] = task
        priority = self._priority_fn(task, time.time())
        entry = (-priority, self._counter, task.task_id)
        heapq.heappush(self._heap, entry)
        self._counter += 1
    
    def reprioritize_all(self):
        """Rebuild heap with updated priorities (called periodically)."""
        self._heap = []
        self._counter = 0
        self._removed.clear()
        t = time.time()
        for task_id, task in self._tasks.items():
            if task.status == "pending":
                priority = self._priority_fn(task, t)
                entry = (-priority, self._counter, task_id)
                heapq.heappush(self._heap, entry)
                self._counter += 1
    
    def pop_highest(self) -> Optional[AgentTask]:
        while self._heap:
            neg_priority, _, task_id = heapq.heappop(self._heap)
            if task_id not in self._removed and task_id in self._tasks:
                if self._tasks[task_id].status == "pending":
                    task = self._tasks[task_id]
                    task.status = "running"
                    return task
        return None
    
    def remove_task(self, task_id: str):
        self._removed.add(task_id)
        if task_id in self._tasks:
            del self._tasks[task_id]
    
    def get_ready_tasks(self, completed_ids: set) -> list:
        """Return tasks whose dependencies are all completed."""
        ready = []
        for task in self._tasks.values():
            if task.status == "pending":
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        return ready
    
    def __len__(self):
        return sum(1 for t in self._tasks.values() if t.status == "pending")


# Example priority function
def composite_priority(task: AgentTask, current_time: float) -> float:
    # Urgency component
    if task.deadline is not None:
        time_remaining = max(0.001, task.deadline - current_time)
        urgency = 1.0 / time_remaining
    else:
        age = current_time - task.created_at
        urgency = 1.0 + 0.1 * age  # Mild aging factor
    
    # Value density
    value_density = task.base_importance / max(0.01, task.estimated_cost)
    
    # Composite
    w_urgency, w_value = 0.6, 0.4
    return w_urgency * urgency + w_value * value_density
```

---

### 19.3.2 Preemptive vs. Non-Preemptive Scheduling

**Non-Preemptive Scheduling:**

Once a task begins execution, it runs to completion regardless of newly arriving higher-priority tasks:

$$\text{If } T_i \text{ is executing at time } t \text{ and } T_j \text{ arrives with } p(T_j) > p(T_i): \quad T_j \text{ waits.}$$

**Properties:**
- Simpler implementation: no checkpointing or state-saving required.
- Predictable execution: each task completes without interruption.
- **Risk**: A long-running low-priority task can delay a critical task indefinitely (**priority inversion**).

**Preemptive Scheduling:**

A running task is interrupted when a higher-priority task arrives:

$$\text{If } T_i \text{ is executing at time } t \text{ and } T_j \text{ arrives with } p(T_j) > p(T_i): \quad T_i \text{ is suspended, } T_j \text{ starts.}$$

**Properties:**
- Lower worst-case latency for high-priority tasks.
- Requires state checkpointing for suspended tasks.
- **Risk**: Excessive context-switching overhead, especially for LLM-based tasks where re-prompting may be necessary.

**Preemption in LLM-Based Agents — Specific Challenges:**

| Challenge | Description | Mitigation |
|---|---|---|
| Stateful inference | LLM generation is autoregressive; pausing mid-generation loses KV-cache | Checkpoint KV-cache to memory |
| Tool call atomicity | External API calls cannot be partially rolled back | Define preemption boundaries between tool calls |
| Cost sunk fallacy | Tokens already consumed are irrecoverable | Preempt only if $V(T_j) - V(T_i) > \text{sunk\_cost}(T_i)$ |
| Context fragmentation | Resuming a suspended task may require re-establishing context | Store conversation state as resumable checkpoints |

**Formal Preemption Decision Rule:**

$$\text{preempt}(T_i, T_j) = \begin{cases} \text{true} & \text{if } p(T_j) - p(T_i) > \theta_{\text{preempt}} + \kappa_{\text{switch}} \\ \text{false} & \text{otherwise} \end{cases}$$

where $\theta_{\text{preempt}}$ is the minimum priority differential threshold and $\kappa_{\text{switch}}$ is the estimated context-switching cost (in priority-equivalent units).

**Hybrid Policy — Time-Sliced Preemption:**

Assign each priority level a maximum continuous execution quantum $q_l$:

$$q_l = q_{\text{base}} \cdot 2^{l-1}$$

Higher-priority tasks receive larger time quanta. After a quantum expires, the scheduler re-evaluates priorities and may preempt. This bounds worst-case latency while limiting preemption frequency.

---

### 19.3.3 Fair Scheduling Across Multiple Users/Tasks

When multiple users or goal threads compete for a shared agent, **fairness** ensures no single user or task category monopolizes resources.

**Formal Fairness Metrics:**

**Max-Min Fairness:**

An allocation is **max-min fair** if no user's allocation can be increased without decreasing the allocation of a user with equal or smaller allocation:

$$\text{maximize } \min_{u \in \mathcal{U}} \text{allocation}(u)$$

**Proportional Fairness:**

An allocation $\mathbf{x}$ is proportionally fair if for any alternative allocation $\mathbf{x'}$:

$$\sum_{u \in \mathcal{U}} \frac{x'_u - x_u}{x_u} \leq 0$$

Equivalently, proportional fairness maximizes $\sum_u \log(x_u)$.

**Weighted Fair Queuing (WFQ):**

Each user $u$ is assigned a weight $w_u$ representing their share of agent resources. The **virtual finish time** for a task $T_i$ from user $u$ is:

$$\text{VFT}(T_i) = \max(t_{\text{current}}, \text{VFT}_{\text{prev}}(u)) + \frac{\kappa(T_i)}{w_u}$$

Tasks are scheduled in order of increasing virtual finish time. This ensures that users with higher weights receive proportionally more service while still serving all users.

**Deficit Round Robin (DRR):**

Each user maintains a **deficit counter** $D_u$, initialized to 0. Each round:

1. $D_u \leftarrow D_u + w_u \cdot Q$ where $Q$ is the service quantum
2. While $D_u \geq \kappa(T_{u,\text{next}})$: execute $T_{u,\text{next}}$, $D_u \leftarrow D_u - \kappa(T_{u,\text{next}})$
3. If user's queue is empty: $D_u \leftarrow 0$

DRR achieves $O(1)$ per-packet scheduling complexity with provable long-term fairness guarantees.

**Implementation for Multi-Tenant Agent System:**

```python
from collections import defaultdict, deque

class FairPriorityScheduler:
    """Weighted Fair Queuing with priority support."""
    
    def __init__(self):
        self.user_queues: dict[str, deque] = defaultdict(deque)
        self.user_weights: dict[str, float] = {}
        self.virtual_time: float = 0.0
        self.user_vft: dict[str, float] = defaultdict(float)  # Virtual finish times
    
    def register_user(self, user_id: str, weight: float = 1.0):
        self.user_weights[user_id] = weight
    
    def submit_task(self, user_id: str, task: AgentTask):
        if user_id not in self.user_weights:
            self.register_user(user_id)
        
        # Compute virtual finish time
        start_vt = max(self.virtual_time, self.user_vft[user_id])
        finish_vt = start_vt + task.estimated_cost / self.user_weights[user_id]
        self.user_vft[user_id] = finish_vt
        
        self.user_queues[user_id].append((finish_vt, task))
    
    def next_task(self) -> Optional[AgentTask]:
        """Select task with smallest virtual finish time."""
        best_vft = float('inf')
        best_user = None
        
        for user_id, queue in self.user_queues.items():
            if queue:
                vft, _ = queue[0]
                if vft < best_vft:
                    best_vft = vft
                    best_user = user_id
        
        if best_user is None:
            return None
        
        vft, task = self.user_queues[best_user].popleft()
        self.virtual_time = max(self.virtual_time, vft)
        return task
```

---

### 19.3.4 Starvation Prevention

**Starvation** occurs when a low-priority task is indefinitely delayed because higher-priority tasks continually arrive and are served first.

**Formal Definition:**

Task $T_i$ suffers starvation if:

$$\lim_{t \to \infty} (t - t_{\text{arrival}}(T_i)) = \infty \quad \text{while } T_i.\text{status} = \text{pending}$$

i.e., the task waits indefinitely despite being in the queue.

---

**Strategy 1: Priority Aging**

Gradually increase the priority of waiting tasks:

$$p(T_i, t) = p_{\text{base}}(T_i) + \alpha \cdot (t - t_{\text{arrival}}(T_i))$$

where $\alpha > 0$ is the aging rate. After sufficient time $t^*$, any low-priority task will surpass any high-priority task:

$$t^* = t_{\text{arrival}}(T_i) + \frac{p_{\text{high}} - p_{\text{low}}}{\alpha}$$

**Bounded Aging** prevents over-promotion:

$$p(T_i, t) = \min\left(p_{\text{max}}, \; p_{\text{base}}(T_i) + \alpha \cdot (t - t_{\text{arrival}}(T_i))\right)$$

**Exponential Aging** for more aggressive starvation prevention:

$$p(T_i, t) = p_{\text{base}}(T_i) + \alpha \cdot \left(e^{\beta(t - t_{\text{arrival}}(T_i))} - 1\right)$$

where $\beta > 0$ controls the acceleration of priority growth.

---

**Strategy 2: Maximum Wait Time Guarantee**

Define a maximum allowable wait time $W_{\max}$ per priority level:

$$\text{If } (t - t_{\text{arrival}}(T_i)) > W_{\max}(p_{\text{base}}(T_i)): \quad p(T_i) \leftarrow p_{\text{critical}}$$

This provides a **hard guarantee** on worst-case latency for every priority tier.

Typical configuration:
- $W_{\max}(\text{low}) = 300\text{s}$
- $W_{\max}(\text{medium}) = 60\text{s}$
- $W_{\max}(\text{high}) = 10\text{s}$
- $W_{\max}(\text{critical}) = 0\text{s}$ (immediate)

---

**Strategy 3: Reserved Capacity**

Allocate a fixed fraction of agent capacity to each priority level:

$$\text{capacity}(l) = C_l \cdot C_{\text{total}}, \quad \sum_{l} C_l = 1$$

For example: 50% for critical, 25% for high, 15% for medium, 10% for low.

This guarantees that even under maximum load from higher-priority tasks, lower-priority tasks still receive a minimum service rate:

$$\text{throughput}_{\min}(l) \geq C_l \cdot \text{throughput}_{\text{total}}$$

---

**Strategy 4: Lottery Scheduling**

Each task receives lottery tickets proportional to its priority. The next task to execute is selected by drawing a random ticket:

$$P(\text{select } T_i) = \frac{p(T_i)}{\sum_{j} p(T_j)}$$

This provides probabilistic fairness: even a task with priority 1 has a non-zero probability of selection when competing against priority-100 tasks. In expectation, service rates are proportional to priorities, but no task receives zero service.

**Advantage over deterministic schemes:** Naturally avoids starvation without explicit aging or reserved capacity.

**Formal guarantee:**

$$\mathbb{E}[\text{wait time}(T_i)] = \frac{\sum_{j} p(T_j)}{p(T_i)} \cdot \bar{e}$$

where $\bar{e}$ is the mean task execution time. Wait time is bounded as long as $p(T_i) > 0$.

---

## 19.4 Multi-Objective Prioritization

### 19.4.1 Pareto-Optimal Task Ordering

When tasks must be evaluated on multiple incommensurable objectives (e.g., user satisfaction, cost minimization, latency), no single scalar priority function can capture all tradeoffs without imposing subjective weights. **Pareto optimality** provides a weight-free framework.

**Formal Definition:**

Let each task $T_i$ be characterized by an objective vector:

$$\mathbf{f}(T_i) = (f_1(T_i), f_2(T_i), \ldots, f_m(T_i)) \in \mathbb{R}^m$$

Task $T_i$ **Pareto-dominates** $T_j$ (written $T_i \succ_P T_j$) if and only if:

$$\forall k \in \{1, \ldots, m\}: f_k(T_i) \geq f_k(T_j) \quad \text{and} \quad \exists k: f_k(T_i) > f_k(T_j)$$

The **Pareto front** $\mathcal{F}^* \subseteq \mathcal{T}$ consists of all non-dominated tasks:

$$\mathcal{F}^* = \{T_i \in \mathcal{T} : \nexists T_j \in \mathcal{T} \text{ s.t. } T_j \succ_P T_i\}$$

**Application to Task Ordering:**

Given three objectives—urgency $u$, importance $\iota$, and cost-efficiency $\kappa^{-1}$—the Pareto front identifies tasks that are unambiguously high-priority (dominating others on all axes).

**Algorithm — Non-Dominated Sorting (from NSGA-II):**

1. Identify the first Pareto front $\mathcal{F}_1$ (non-dominated tasks).
2. Remove $\mathcal{F}_1$ from $\mathcal{T}$ and identify the next front $\mathcal{F}_2$.
3. Repeat until all tasks are assigned to fronts: $\mathcal{F}_1, \mathcal{F}_2, \ldots, \mathcal{F}_k$.
4. Process tasks in front order: all tasks in $\mathcal{F}_1$ before any in $\mathcal{F}_2$.

**Within-Front Ordering — Crowding Distance:**

For tasks within the same Pareto front (incomparable), use **crowding distance** to prioritize tasks in sparser regions of objective space, promoting diversity:

$$\text{CD}(T_i) = \sum_{k=1}^{m} \frac{f_k(T_{i+1}) - f_k(T_{i-1})}{f_k^{\max} - f_k^{\min}}$$

where tasks are sorted by each objective and boundary tasks receive infinite crowding distance.

```python
import numpy as np

def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Fast non-dominated sorting.
    objectives: (n_tasks, n_objectives) array where higher is better.
    Returns list of fronts, each front is a list of task indices.
    """
    n = objectives.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]
    
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
        
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)
    
    return [f for f in fronts if f]  # Remove empty final front


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if objective vector a Pareto-dominates b."""
    return np.all(a >= b) and np.any(a > b)


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Compute crowding distance for tasks in a Pareto front."""
    n = len(front)
    m = objectives.shape[1]
    distances = np.zeros(n)
    
    for k in range(m):
        sorted_indices = np.argsort(objectives[front, k])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        obj_range = (objectives[front[sorted_indices[-1]], k] - 
                     objectives[front[sorted_indices[0]], k])
        if obj_range == 0:
            continue
        
        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                objectives[front[sorted_indices[i + 1]], k] - 
                objectives[front[sorted_indices[i - 1]], k]
            ) / obj_range
    
    return distances
```

---

### 19.4.2 Weighted Multi-Criteria Decision Making

$$\text{score}(T_i) = \sum_{j=1}^{m} w_j \cdot c_j(T_i)$$

This is the **Simple Additive Weighting (SAW)** method, also known as the **Weighted Sum Model (WSM)**.

**Formal Setup:**

- $m$ criteria: $c_1, c_2, \ldots, c_m$ (each $c_j: \mathcal{T} \rightarrow \mathbb{R}$)
- Weight vector: $\mathbf{w} = (w_1, w_2, \ldots, w_m)$ with $w_j \geq 0$ and $\sum_j w_j = 1$

**Critical Requirement — Normalization:**

Criteria may have different scales and units (urgency in $[0, \infty)$, cost in dollars, importance in $[0, 10]$). Before aggregation, normalize to a common scale:

**Min-Max Normalization:**

$$\hat{c}_j(T_i) = \frac{c_j(T_i) - \min_{T \in \mathcal{T}} c_j(T)}{\max_{T \in \mathcal{T}} c_j(T) - \min_{T \in \mathcal{T}} c_j(T)}$$

**Z-Score Normalization:**

$$\hat{c}_j(T_i) = \frac{c_j(T_i) - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of criterion $j$ across the task set.

**Normalized Score:**

$$\text{score}(T_i) = \sum_{j=1}^{m} w_j \cdot \hat{c}_j(T_i)$$

---

**Advanced MCDM Methods:**

**TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution):**

1. Construct the weighted normalized decision matrix: $v_{ij} = w_j \cdot \hat{c}_j(T_i)$
2. Determine the ideal solution $\mathbf{v}^+$ and anti-ideal $\mathbf{v}^-$:

$$v_j^+ = \max_i v_{ij}, \quad v_j^- = \min_i v_{ij}$$

3. Compute distances to ideal and anti-ideal:

$$D_i^+ = \sqrt{\sum_j (v_{ij} - v_j^+)^2}, \quad D_i^- = \sqrt{\sum_j (v_{ij} - v_j^-)^2}$$

4. Compute the closeness coefficient:

$$\text{CC}(T_i) = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1]$$

5. Rank by decreasing $\text{CC}$.

**Advantage over SAW**: TOPSIS considers the geometric structure of the objective space, not just linear aggregation. It handles criteria with different preference directions (some to maximize, some to minimize) naturally.

**AHP (Analytic Hierarchy Process) for Weight Elicitation:**

When weights are uncertain, AHP elicits them through pairwise comparisons between criteria:

1. Construct pairwise comparison matrix $A$ where $a_{jk}$ indicates the relative importance of criterion $j$ over $k$ (Saaty scale: 1–9).
2. Compute the principal eigenvector of $A$: $A\mathbf{w} = \lambda_{\max} \mathbf{w}$
3. Normalize: $w_j \leftarrow w_j / \sum_k w_k$
4. Check consistency ratio: $\text{CR} = \frac{\lambda_{\max} - m}{(m-1) \cdot \text{RI}(m)}$ where $\text{RI}(m)$ is the random index. Accept if $\text{CR} < 0.1$.

---

### 19.4.3 Constraint-Based Priority Resolution

When tasks are subject to hard constraints (resource limits, ordering dependencies, mutual exclusion), priority assignment becomes a **constraint satisfaction problem (CSP)** or **constraint optimization problem (COP)**.

**Formal Formulation:**

$$\max_{\sigma} \sum_{i=1}^{n} p(T_{\sigma(i)}) \cdot \text{discount}(i)$$

subject to:

$$\forall (T_i, T_j) \in E_{\text{dep}}: \sigma^{-1}(T_i) < \sigma^{-1}(T_j) \quad \text{(dependency ordering)}$$

$$\forall t: \sum_{T_i \text{ active at } t} r_k(T_i) \leq R_k \quad \text{(resource capacity)}$$

$$\forall (T_i, T_j) \in \text{mutex}: \neg(\text{concurrent}(T_i, T_j)) \quad \text{(mutual exclusion)}$$

where $\sigma$ is a permutation (execution order), $\text{discount}(i) = \gamma^{i-1}$ rewards earlier execution of high-priority tasks, $r_k(T_i)$ is the consumption of resource $k$ by task $T_i$, and $R_k$ is total availability.

**Solution Approaches:**

1. **Topological sort with priority**: Perform topological ordering on the dependency DAG, breaking ties by priority. This is optimal when all tasks have unit execution time and the only constraints are dependencies.

2. **Integer Linear Programming (ILP)**: For exact solutions with complex constraints:

$$\text{Variables: } x_{it} \in \{0, 1\} \quad \text{(task } i \text{ starts at time } t\text{)}$$

$$\max \sum_{i,t} p(T_i) \cdot \gamma^t \cdot x_{it}$$

$$\text{s.t.} \quad \sum_t x_{it} = 1 \quad \forall i \quad \text{(each task scheduled exactly once)}$$

$$\sum_{i: t \in [\text{start}_i, \text{end}_i)} r_k(T_i) \leq R_k \quad \forall k, t$$

$$t_j \geq t_i + e_i \quad \forall (T_i, T_j) \in E_{\text{dep}}$$

3. **Constraint propagation + backtracking**: For moderate-sized task sets, domain-specific constraint propagation (arc consistency, forward checking) with priority-guided variable/value ordering.

**Example — Constraint Resolution in Practice:**

```python
from typing import Optional

class ConstraintAwarePriorityResolver:
    def __init__(self, tasks: list[AgentTask], 
                 dependencies: dict[str, list[str]],
                 resource_limits: dict[str, float]):
        self.tasks = {t.task_id: t for t in tasks}
        self.dependencies = dependencies  # task_id -> list of prerequisite task_ids
        self.resource_limits = resource_limits
        self.completed = set()
    
    def get_executable_ordered(self) -> list[AgentTask]:
        """Return dependency-respecting, priority-ordered list of ready tasks."""
        # Find tasks with all dependencies satisfied
        ready = []
        for task_id, task in self.tasks.items():
            if task.status != "pending":
                continue
            prereqs = self.dependencies.get(task_id, [])
            if all(p in self.completed for p in prereqs):
                ready.append(task)
        
        # Sort by priority (descending)
        ready.sort(key=lambda t: self._compute_priority(t), reverse=True)
        
        # Filter by resource feasibility
        feasible = []
        current_usage = {r: 0.0 for r in self.resource_limits}
        
        for task in ready:
            task_resources = self._estimate_resources(task)
            can_execute = all(
                current_usage[r] + task_resources.get(r, 0) <= limit
                for r, limit in self.resource_limits.items()
            )
            if can_execute:
                for r in current_usage:
                    current_usage[r] += task_resources.get(r, 0)
                feasible.append(task)
        
        return feasible
    
    def _compute_priority(self, task: AgentTask) -> float:
        # Number of tasks transitively blocked
        blocked_count = self._count_descendants(task.task_id)
        return task.base_importance * (1 + 0.5 * blocked_count)
    
    def _count_descendants(self, task_id: str, visited=None) -> int:
        if visited is None:
            visited = set()
        count = 0
        for tid, prereqs in self.dependencies.items():
            if task_id in prereqs and tid not in visited:
                visited.add(tid)
                count += 1 + self._count_descendants(tid, visited)
        return count
    
    def _estimate_resources(self, task: AgentTask) -> dict:
        return {"tokens": task.estimated_cost, "concurrent_slots": 1}
```

---

### 19.4.4 Dynamic Re-Prioritization on Context Change

In agentic systems, the context is non-stationary. Events that trigger re-prioritization include:

1. **New task arrival**: Alters relative priorities across the queue.
2. **Task completion/failure**: Unblocks dependents, may obsolete related tasks.
3. **Deadline approach**: Urgency of time-sensitive tasks increases.
4. **Resource state change**: Budget depletion, rate limit activation, new resource availability.
5. **Goal revision**: User provides new instructions or changes objectives.
6. **Environmental signal**: External data (market change, system alert) alters task value.

**Formal Framework — Event-Triggered Re-Prioritization:**

Define a context state $\mathcal{C}_t$ at time $t$ and a change detection function:

$$\Delta(\mathcal{C}_t, \mathcal{C}_{t-1}) = \|\mathbf{c}_t - \mathbf{c}_{t-1}\|$$

Re-prioritization is triggered when:

$$\Delta(\mathcal{C}_t, \mathcal{C}_{t-1}) > \theta_{\text{reprioritize}}$$

or at fixed intervals $\Delta t_{\text{reprioritize}}$.

**Incremental vs. Full Re-Prioritization:**

| Approach | Complexity | When to Use |
|---|---|---|
| Full recomputation | $O(n \log n)$ (re-sort) | Small task sets, major context changes |
| Incremental update | $O(k \log n)$ for $k$ affected tasks | Large task sets, minor context changes |
| Lazy evaluation | $O(1)$ per dequeue (recompute on extraction) | Rapidly changing priorities |

**Incremental Update Algorithm:**

```
On event e at time t:
    affected_tasks = identify_affected(e, task_queue)
    for T_i in affected_tasks:
        p_old = priority(T_i)
        p_new = compute_priority(T_i, context_t)
        if |p_new - p_old| > epsilon:
            update_priority(task_queue, T_i, p_new)
```

**Adaptive Re-Prioritization Frequency:**

The re-prioritization interval itself can be dynamic:

$$\Delta t_{\text{reprioritize}} = \frac{\Delta t_{\text{base}}}{\text{volatility}(\mathcal{C})}$$

where $\text{volatility}(\mathcal{C})$ measures the rate of context change. In stable environments, re-prioritize infrequently; in volatile environments, re-prioritize aggressively.

**Stability Constraint — Bounded Rank Change:**

To prevent oscillatory behavior ("priority thrashing"), impose a maximum rank change per re-prioritization cycle:

$$|\text{rank}_t(T_i) - \text{rank}_{t-1}(T_i)| \leq \Delta_{\text{max}}$$

This acts as a momentum term, smoothing priority trajectories. Formally, apply exponential moving average to priorities:

$$p_{\text{smooth}}(T_i, t) = \beta \cdot p(T_i, t) + (1 - \beta) \cdot p_{\text{smooth}}(T_i, t-1)$$

where $\beta \in (0, 1)$ controls responsiveness vs. stability.

---

## 19.5 Priority in Multi-Agent Systems

### 19.5.1 Inter-Agent Priority Negotiation

When multiple autonomous agents operate in a shared environment, they must negotiate task priorities to avoid conflicts, redundant work, and resource waste. Unlike centralized prioritization, multi-agent priority negotiation is **distributed, asynchronous, and potentially adversarial**.

**System Model:**

- Agent set: $\mathcal{A} = \{A_1, A_2, \ldots, A_N\}$
- Shared task pool: $\mathcal{T}$ (or per-agent task queues with cross-dependencies)
- Communication: message-passing (synchronous or asynchronous)
- Each agent has a local priority function: $p^{(k)}: \mathcal{T} \rightarrow \mathbb{R}$ for agent $A_k$

**Challenge**: Different agents may assign different priorities to the same task based on their local goals, capabilities, and information.

---

**Protocol 1: Contract Net Protocol (CNP)**

A classic negotiation protocol adapted for priority-aware task allocation:

1. **Announcement**: Manager agent broadcasts task $T_i$ with minimum required priority.
2. **Bidding**: Contractor agents assess their capacity and submit bids: $\text{bid}(A_k, T_i) = (p^{(k)}(T_i), \; \kappa^{(k)}(T_i), \; \text{ETA}^{(k)})$
3. **Awarding**: Manager selects the contractor with the best bid according to:

$$A^* = \arg\max_{A_k} \frac{p^{(k)}(T_i)}{\kappa^{(k)}(T_i)} \cdot \frac{1}{\text{ETA}^{(k)}}$$

4. **Confirmation**: Selected agent confirms acceptance and commits to the task.

**Priority-Aware Extension**: Agents include their current queue load in their bids, allowing the manager to estimate actual completion time:

$$\text{ETA}^{(k)} = \sum_{T_j \in \text{queue}(A_k): p^{(k)}(T_j) > p^{(k)}(T_i)} e(T_j) + e(T_i)$$

---

**Protocol 2: Consensus-Based Priority Aggregation**

When no single manager exists, agents must reach consensus on a global priority ordering.

**Voting-Based Aggregation:**

Each agent submits a priority ranking $\sigma^{(k)}$ over shared tasks. The aggregated ranking uses **Borda count**:

$$\text{Borda}(T_i) = \sum_{k=1}^{N} (n - \sigma^{(k)}(T_i))$$

where $\sigma^{(k)}(T_i)$ is the rank of $T_i$ in agent $A_k$'s ordering.

**Weighted Consensus:**

When agents have different expertise or authority levels:

$$p_{\text{consensus}}(T_i) = \sum_{k=1}^{N} \alpha_k \cdot p^{(k)}(T_i)$$

where $\alpha_k$ is agent $A_k$'s authority weight with $\sum_k \alpha_k = 1$.

**Convergence via Iterative Refinement:**

Agents exchange priority assessments and iteratively update their local priorities:

$$p^{(k)}_{t+1}(T_i) = (1 - \lambda) \cdot p^{(k)}_t(T_i) + \lambda \cdot \frac{1}{|\mathcal{N}(k)|} \sum_{j \in \mathcal{N}(k)} p^{(j)}_t(T_i)$$

where $\mathcal{N}(k)$ is the set of agent $A_k$'s communication neighbors and $\lambda \in (0, 1)$ is the convergence rate.

**Theorem (Convergence):** Under a connected communication graph and $\lambda \in (0, 1)$, this iterative scheme converges to the weighted average of initial priorities:

$$\lim_{t \to \infty} p^{(k)}_t(T_i) = \frac{1}{N} \sum_{j=1}^{N} p^{(j)}_0(T_i) \quad \forall k$$

This is a direct application of the **DeGroot consensus model** from social learning theory.

---

**Protocol 3: Auction-Based Priority Market**

Each agent has a priority budget $B_k$ and bids on tasks they wish to execute or have executed:

$$\text{bid}(A_k, T_i) = b_{ki} \leq B_k - \sum_{j \neq i} b_{kj}$$

Tasks are allocated to the highest bidder, and the priority order reflects market prices:

$$p_{\text{market}}(T_i) = \max_k b_{ki}$$

**Properties:**
- **Incentive compatible**: Agents bid truthfully in second-price (Vickrey) auctions.
- **Efficient**: Allocates tasks to agents that value them most.
- **Scalable**: Requires only $O(N)$ communication per task.

---

**Multi-Agent Priority Negotiation System Implementation:**

```python
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class AgentProfile:
    agent_id: str
    capabilities: set[str]      # Task types this agent can handle
    authority_weight: float     # Priority-setting authority
    current_load: float         # Current utilization [0, 1]
    priority_fn: Callable       # Local priority assessment function

class MultiAgentPriorityNegotiator:
    def __init__(self, agents: list[AgentProfile]):
        self.agents = {a.agent_id: a for a in agents}
        self.consensus_history = []
    
    def negotiate_priorities(self, tasks: list[AgentTask], 
                              method: str = "weighted_consensus") -> dict[str, float]:
        """
        Negotiate global priority ordering across all agents.
        Returns: dict mapping task_id -> consensus priority
        """
        if method == "weighted_consensus":
            return self._weighted_consensus(tasks)
        elif method == "contract_net":
            return self._contract_net(tasks)
        elif method == "auction":
            return self._vickrey_auction(tasks)
        else:
            raise ValueError(f"Unknown negotiation method: {method}")
    
    def _weighted_consensus(self, tasks: list[AgentTask]) -> dict[str, float]:
        consensus = {}
        total_weight = sum(a.authority_weight for a in self.agents.values())
        
        for task in tasks:
            weighted_sum = 0.0
            for agent in self.agents.values():
                local_priority = agent.priority_fn(task)
                weighted_sum += agent.authority_weight * local_priority
            
            consensus[task.task_id] = weighted_sum / total_weight
        
        return consensus
    
    def _contract_net(self, tasks: list[AgentTask]) -> dict[str, tuple[str, float]]:
        """Returns task_id -> (assigned_agent_id, priority)"""
        assignments = {}
        
        for task in tasks:
            best_agent = None
            best_score = -float('inf')
            
            for agent in self.agents.values():
                # Check capability
                if not self._can_handle(agent, task):
                    continue
                
                local_priority = agent.priority_fn(task)
                # Score considers priority, load, and estimated cost
                score = local_priority * (1 - agent.current_load)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent.agent_id
            
            if best_agent:
                assignments[task.task_id] = (best_agent, best_score)
        
        return assignments
    
    def _vickrey_auction(self, tasks: list[AgentTask]) -> dict[str, tuple[str, float]]:
        """Second-price auction: winner pays second-highest bid."""
        results = {}
        
        for task in tasks:
            bids = []
            for agent in self.agents.values():
                if self._can_handle(agent, task):
                    bid_value = agent.priority_fn(task) * (1 - agent.current_load)
                    bids.append((agent.agent_id, bid_value))
            
            if len(bids) >= 2:
                bids.sort(key=lambda x: x[1], reverse=True)
                winner = bids[0][0]
                price = bids[1][1]  # Second-price
                results[task.task_id] = (winner, price)
            elif len(bids) == 1:
                results[task.task_id] = (bids[0][0], bids[0][1])
        
        return results
    
    def _can_handle(self, agent: AgentProfile, task: AgentTask) -> bool:
        # Check if agent has required capabilities
        # Simplified: check if any capability keyword matches task description
        return agent.current_load < 0.95  # Not overloaded
```

---

### 19.5.2 Resource Contention Resolution

When multiple agents compete for shared resources (LLM API quotas, database connections, tool access, memory bandwidth), **resource contention** arises. Priority-aware contention resolution determines which agent's request is served when demand exceeds capacity.

**Formal Model:**

- Shared resource set: $\mathcal{R} = \{R_1, R_2, \ldots, R_q\}$
- Resource capacity: $\text{cap}(R_j)$ (maximum concurrent users or throughput)
- Request: $(A_k, R_j, T_i, p(T_i))$ — agent $A_k$ requests resource $R_j$ for task $T_i$ with priority $p(T_i)$
- Contention: $\sum_{k: \text{requesting}(A_k, R_j)} \text{demand}(A_k, R_j) > \text{cap}(R_j)$

---

**Resolution Strategy 1: Priority-Based Admission**

Serve requests in priority order until capacity is exhausted:

$$\text{admitted} = \text{top-k requests by } p(T_i) \text{ s.t. total demand} \leq \text{cap}(R_j)$$

**Resolution Strategy 2: Proportional Sharing**

Allocate resource capacity proportional to request priorities:

$$\text{alloc}(A_k, R_j) = \frac{p(T_{k,i})}{\sum_{k'} p(T_{k',i'})} \cdot \text{cap}(R_j)$$

This prevents starvation but may result in all agents receiving insufficient resources.

**Resolution Strategy 3: Priority-Weighted Queuing**

When resources are temporarily unavailable, requests enter a **priority queue** associated with each resource:

$$\text{waitqueue}(R_j) = \text{PriorityQueue ordered by } p(T_i)$$

When the resource becomes available, the highest-priority waiting request is served.

**Deadlock Prevention in Multi-Resource Contention:**

If tasks require multiple resources simultaneously, circular waits can cause **deadlock**. Standard prevention strategies:

1. **Resource ordering**: Impose a total order on resources; agents must acquire resources in order.

$$R_1 \prec R_2 \prec \ldots \prec R_q \quad \Rightarrow \quad \text{acquire in ascending order only}$$

2. **Timeout-based release**: If a resource is not acquired within $\tau_{\text{timeout}}$, release all held resources and retry.

3. **Priority-based preemption**: A higher-priority task can preempt a lower-priority task's resource holding (requires rollback capability).

---

**Formal Contention Resolution with Priority:**

The contention resolution problem can be modeled as a **concurrent resource allocation** optimization:

$$\max \sum_{k, j} p(T_{k}) \cdot x_{kj}$$

$$\text{s.t.} \quad \sum_k x_{kj} \leq \text{cap}(R_j) \quad \forall j$$

$$x_{kj} \in \{0, 1\} \quad \forall k, j$$

$$x_{kj} = 1 \text{ only if } A_k \text{ needs } R_j$$

This is a variant of the **weighted bipartite matching** problem, solvable in polynomial time via the Hungarian algorithm or network flow methods.

---

### 19.5.3 Priority Inheritance and Delegation

**Priority Inversion Problem:**

Priority inversion occurs when a high-priority task $T_H$ is blocked by a low-priority task $T_L$ that holds a needed resource, while a medium-priority task $T_M$ preempts $T_L$, effectively causing $T_H$ to wait for $T_M$ despite $p(T_H) > p(T_M) > p(T_L)$.

**Formal Description:**

$$T_H \xrightarrow{\text{blocked by}} T_L \xrightarrow{\text{preempted by}} T_M \quad \Rightarrow \quad T_H \text{ waits for } T_M$$

This violates the priority ordering invariant: $T_H$ should never wait for $T_M$ when $p(T_H) > p(T_M)$.

**Historical Note:** Priority inversion caused the Mars Pathfinder mission anomaly (1997), where a high-priority bus management task was blocked by a low-priority meteorological task, leading to system resets.

---

**Solution 1: Priority Inheritance Protocol (PIP)**

When a low-priority task $T_L$ blocks a high-priority task $T_H$, $T_L$ temporarily inherits $T_H$'s priority:

$$p_{\text{effective}}(T_L) = \max(p(T_L), \max_{T_H \in \text{blocked\_by}(T_L)} p(T_H))$$

**Properties:**
- $T_L$ can no longer be preempted by medium-priority tasks $T_M$ with $p(T_M) < p(T_H)$.
- Inheritance is transitive: if $T_L$ is blocked by $T_{LL}$, then $T_{LL}$ also inherits $p(T_H)$.
- Priority reverts to $p(T_L)$ when the blocking resource is released.

**Formal Algorithm:**

```
On task T_H blocked by resource held by T_L:
    if p(T_H) > p_effective(T_L):
        p_effective(T_L) ← p(T_H)
        if T_L is blocked by T_LL:
            propagate_inheritance(T_LL, p(T_H))  # Transitive

On T_L releasing resource:
    p_effective(T_L) ← max(p(T_L), max over remaining blockers)
```

**Limitation — Chained Blocking:**

Under PIP, the maximum blocking time for a task $T_H$ with $n$ lower-priority tasks each holding one resource is:

$$B_H \leq \sum_{i=1}^{n} e(T_{L_i}, \text{critical section}_i)$$

This can be $O(n)$ in the number of lower-priority tasks, which may be unacceptable for real-time systems.

---

**Solution 2: Priority Ceiling Protocol (PCP)**

Each resource $R_j$ is assigned a **priority ceiling** equal to the highest priority of any task that may use it:

$$\text{ceil}(R_j) = \max_{T_i \in \text{users}(R_j)} p(T_i)$$

A task $T_i$ may acquire resource $R_j$ only if:

$$p(T_i) > \max_{R_k \in \text{held\_by\_others}} \text{ceil}(R_k)$$

**Properties:**
- **Deadlock-free**: No circular waits are possible.
- **Bounded blocking**: Each task is blocked by at most one lower-priority critical section.

$$B_H \leq \max_{i} e(T_{L_i}, \text{critical section}_i)$$

- More restrictive than PIP (may unnecessarily delay tasks).

---

**Solution 3: Priority Delegation in Multi-Agent Systems**

In agentic contexts, priority delegation extends inheritance across agent boundaries. When agent $A_1$ with high-priority task $T_H$ depends on agent $A_2$ completing sub-task $T_S$:

$$p_{\text{delegated}}(T_S, A_2) = \max(p^{(2)}(T_S), p^{(1)}(T_H))$$

**Delegation Protocol:**

1. Agent $A_1$ sends delegation request: $\text{delegate}(T_H, T_S, p(T_H))$
2. Agent $A_2$ receives and elevates $T_S$'s priority in its local queue.
3. Agent $A_2$ acknowledges: $\text{ack}(T_S, \text{ETA})$
4. On completion: Agent $A_2$ sends result and reverts $T_S$'s priority.

**Implementation:**

```python
class PriorityInheritanceManager:
    """Manages priority inheritance across tasks and agents."""
    
    def __init__(self):
        self.base_priorities: dict[str, float] = {}
        self.effective_priorities: dict[str, float] = {}
        self.resource_holders: dict[str, str] = {}  # resource -> task_id
        self.blocked_on: dict[str, str] = {}  # task -> resource
        self.delegation_chain: dict[str, list[str]] = {}  # task -> chain of inheriting tasks
    
    def set_base_priority(self, task_id: str, priority: float):
        self.base_priorities[task_id] = priority
        if task_id not in self.effective_priorities:
            self.effective_priorities[task_id] = priority
    
    def acquire_resource(self, task_id: str, resource_id: str) -> bool:
        """Attempt to acquire a resource. Returns False if blocked."""
        if resource_id not in self.resource_holders:
            self.resource_holders[resource_id] = task_id
            return True
        
        holder = self.resource_holders[resource_id]
        if holder == task_id:
            return True  # Already held
        
        # Block and trigger inheritance
        self.blocked_on[task_id] = resource_id
        self._inherit_priority(task_id, holder)
        return False
    
    def release_resource(self, task_id: str, resource_id: str):
        """Release resource and revert inherited priorities."""
        if self.resource_holders.get(resource_id) == task_id:
            del self.resource_holders[resource_id]
            self._revert_priority(task_id)
            
            # Unblock highest-priority waiter
            waiters = [
                tid for tid, rid in self.blocked_on.items() 
                if rid == resource_id
            ]
            if waiters:
                highest = max(waiters, 
                            key=lambda t: self.effective_priorities.get(t, 0))
                del self.blocked_on[highest]
                self.resource_holders[resource_id] = highest
    
    def _inherit_priority(self, blocked_task: str, holder_task: str):
        """Propagate priority from blocked task to resource holder."""
        blocked_priority = self.effective_priorities.get(blocked_task, 0)
        holder_priority = self.effective_priorities.get(holder_task, 0)
        
        if blocked_priority > holder_priority:
            self.effective_priorities[holder_task] = blocked_priority
            
            # Track inheritance chain
            if holder_task not in self.delegation_chain:
                self.delegation_chain[holder_task] = []
            self.delegation_chain[holder_task].append(blocked_task)
            
            # Transitive: if holder is also blocked, propagate further
            if holder_task in self.blocked_on:
                next_resource = self.blocked_on[holder_task]
                if next_resource in self.resource_holders:
                    next_holder = self.resource_holders[next_resource]
                    self._inherit_priority(holder_task, next_holder)
    
    def _revert_priority(self, task_id: str):
        """Revert to base priority or max of remaining inherited priorities."""
        base = self.base_priorities.get(task_id, 0)
        
        # Check if still blocking anyone
        still_blocking = [
            self.effective_priorities.get(tid, 0)
            for tid, rid in self.blocked_on.items()
            if self.resource_holders.get(rid) == task_id
        ]
        
        self.effective_priorities[task_id] = max(
            [base] + still_blocking
        )
        
        # Clear delegation chain
        if task_id in self.delegation_chain:
            del self.delegation_chain[task_id]
    
    def get_effective_priority(self, task_id: str) -> float:
        return self.effective_priorities.get(
            task_id, self.base_priorities.get(task_id, 0)
        )
```

---

**Summary — Priority Inheritance/Delegation Decision Matrix:**

| Mechanism | Deadlock-Free | Max Blocking | Complexity | Multi-Agent |
|---|---|---|---|---|
| No protection | No | Unbounded | $O(1)$ | N/A |
| Priority Inheritance (PIP) | No | $O(n)$ critical sections | $O(n)$ propagation | Via message-passing |
| Priority Ceiling (PCP) | Yes | 1 critical section | $O(1)$ per acquisition | Requires global knowledge |
| Immediate Ceiling (ICPP) | Yes | 1 critical section | $O(1)$ per acquisition | Requires global knowledge |
| Priority Delegation | No (needs timeout) | Application-dependent | $O(\text{agents})$ | Native |

---

**End-to-End Prioritization Architecture for Agentic Systems:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIORITIZATION ENGINE                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  Task Intake  │──▶│  Priority    │──▶│  Priority Queue   │   │
│  │  (new tasks,  │   │  Computation │   │  (Binary Heap +   │   │
│  │   events)     │   │  (Static /   │   │   Index Map)      │   │
│  └──────────────┘   │  Dynamic /   │   └────────┬──────────┘   │
│                      │  LLM-based) │            │               │
│  ┌──────────────┐   └──────────────┘   ┌────────▼──────────┐   │
│  │  Context      │                      │  Constraint       │   │
│  │  Monitor      │─── trigger ─────────▶│  Resolver         │   │
│  │  (deadlines,  │   re-prioritize      │  (dependencies,   │   │
│  │   resources)  │                      │   resources,      │   │
│  └──────────────┘                      │   mutex)          │   │
│                                         └────────┬──────────┘   │
│  ┌──────────────┐                      ┌────────▼──────────┐   │
│  │  Fairness     │◀─────────────────────│  Scheduler        │   │
│  │  Controller   │   feedback           │  (preemptive /    │   │
│  │  (aging,      │─────────────────────▶│   non-preemptive) │   │
│  │   quotas)     │                      └────────┬──────────┘   │
│  └──────────────┘                      ┌────────▼──────────┐   │
│                                         │  Multi-Agent      │   │
│  ┌──────────────┐                      │  Negotiator       │   │
│  │  Priority     │◀─────────────────────│  (CNP / Auction / │   │
│  │  Inheritance  │   resource conflict  │   Consensus)      │   │
│  │  Manager      │─────────────────────▶│                   │   │
│  └──────────────┘                      └────────┬──────────┘   │
│                                                  │              │
│                                         ┌────────▼──────────┐   │
│                                         │  Execution Engine │   │
│                                         │  (Agent Workers)  │   │
│                                         └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

This architecture ensures that every task entering the system is evaluated across all priority dimensions, constrained by dependencies and resources, scheduled fairly across users, negotiated across agents, and protected from priority inversion—forming a complete, production-grade prioritization system for agentic AI workflows.