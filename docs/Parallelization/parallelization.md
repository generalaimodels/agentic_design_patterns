

# Chapter 2: Parallelization

---

## 2.1 Definition and Formal Framework

### 2.1.1 What is Parallelization in Agentic Workflows

Parallelization in agentic workflows is the **systematic decomposition and simultaneous execution** of sub-tasks across multiple agents, models, or processing units to reduce end-to-end latency, increase throughput, and improve output quality through redundancy. It is the computational strategy that transforms a sequential chain of LLM calls into a concurrent execution graph where independent operations proceed simultaneously.

In a naive agentic pipeline, tasks execute sequentially:

$$
L_{\text{sequential}} = \sum_{i=1}^{k} L(T_i)
$$

where $L(T_i)$ is the latency of task $T_i$. For $k = 5$ LLM calls each taking 2 seconds, total latency is 10 seconds.

With parallelization of independent tasks:

$$
L_{\text{parallel}} = \max_{i \in \text{parallel group}} L(T_i) + \sum_{j \in \text{sequential}} L(T_j)
$$

If 4 of those 5 calls are independent, total latency drops to $\max(2, 2, 2, 2) + 2 = 4$ seconds — a $2.5\times$ speedup.

**Why Parallelization is Non-Trivial in Agentic Systems:**

Unlike traditional parallel computing (matrix multiplication, data-parallel training), agentic parallelization faces unique challenges:

| Challenge | Description |
|---|---|
| **Non-uniform task latency** | LLM calls have highly variable latency (200ms–30s) depending on output length, model load, and complexity |
| **Semantic dependencies** | Task dependencies are determined by meaning, not data flow; an LLM must analyze the query to determine which sub-tasks are independent |
| **Stochastic outputs** | Each parallel branch produces a non-deterministic result; aggregation must handle semantic variability |
| **Cost amplification** | Parallelizing $k$ branches multiplies token cost by up to $k\times$ |
| **Shared mutable state** | Multiple agents may need to read/write shared memory, tools, or external systems |
| **Error propagation** | A failure in one branch may invalidate the entire execution or require re-planning |

**Taxonomy of Parallelization Purposes:**

1. **Latency Reduction**: Execute independent sub-tasks simultaneously to reduce wall-clock time
2. **Quality Improvement**: Generate multiple candidate solutions in parallel, then select or merge the best (ensemble/voting)
3. **Throughput Scaling**: Process multiple independent queries simultaneously (data parallelism)
4. **Robustness**: Redundant execution across multiple agents/models provides fault tolerance
5. **Exploration**: Parallel exploration of multiple reasoning paths or solution strategies

---

### 2.1.2 Task Decomposition for Parallelism

Task decomposition is the process of breaking a composite task $T$ into sub-tasks:

$$
T \rightarrow \{T_1, T_2, \ldots, T_k\}
$$

The **dependency structure** between sub-tasks defines the execution graph. Formally, the dependency relation is:

$$
T_i \prec T_j \iff T_j \text{ requires the output of } T_i
$$

Two sub-tasks $T_i$ and $T_j$ are **independent** (parallelizable) if and only if:

$$
T_i \not\prec T_j \land T_j \not\prec T_i
$$

This is precisely the condition under which $T_i$ and $T_j$ can execute concurrently without violating correctness.

**Dependency Types:**

1. **Data Dependency (Flow Dependency)**: $T_j$ uses the output of $T_i$ as input:

$$
\text{input}(T_j) \cap \text{output}(T_i) \neq \emptyset
$$

2. **Control Dependency**: $T_j$'s execution depends on a decision made during $T_i$:

$$
\text{execute}(T_j) = g(\text{output}(T_i))
$$

where $g$ is a predicate function determining whether $T_j$ should run.

3. **Resource Dependency**: $T_i$ and $T_j$ both require exclusive access to the same resource (API rate limit, shared memory):

$$
\text{resources}(T_i) \cap \text{resources}(T_j) \neq \emptyset
$$

4. **No Dependency**: $T_i$ and $T_j$ are fully independent — different inputs, different outputs, different resources.

**Decomposition Strategies:**

**Strategy 1: Static Decomposition (Design-Time)**

The developer explicitly defines the decomposition based on domain knowledge:

```python
# Static decomposition of a research report task
subtasks = {
    "literature_review": {"agent": "search_agent", "depends_on": []},
    "data_analysis":     {"agent": "code_agent",   "depends_on": []},
    "methodology":       {"agent": "writing_agent", "depends_on": []},
    "results":           {"agent": "writing_agent", "depends_on": ["data_analysis"]},
    "conclusion":        {"agent": "writing_agent", "depends_on": ["literature_review", "results"]},
}
```

**Strategy 2: Dynamic Decomposition (Runtime, LLM-Planned)**

An LLM analyzes the task and produces a decomposition plan at runtime:

```python
DECOMPOSITION_PROMPT = """
Analyze the following task and decompose it into sub-tasks.
For each sub-task, specify:
1. A unique ID
2. A description
3. Which sub-tasks it depends on (by ID)
4. Which agent should handle it

Task: {task}

Output a JSON decomposition plan.
"""
```

**Strategy 3: Hybrid Decomposition**

Use static templates for known task types and fall back to dynamic decomposition for novel tasks.

**Maximum Parallelism Degree:**

Given a dependency structure, the **maximum parallelism degree** $\pi$ is the size of the largest antichain in the partial order defined by $\prec$:

$$
\pi = \max\{|S| : S \subseteq \{T_1, \ldots, T_k\}, \forall T_i, T_j \in S: T_i \not\prec T_j \land T_j \not\prec T_i\}
$$

By Dilworth's theorem, this equals the minimum number of chains (sequential paths) needed to cover all tasks. Alternatively, $\pi$ equals the width of the DAG.

---

### 2.1.3 Speedup and Efficiency: Amdahl's Law Applied to Agentic Pipelines

**Amdahl's Law** provides the theoretical upper bound on speedup from parallelization:

$$
S(n) = \frac{1}{(1 - p) + \frac{p}{n}}
$$

where:
- $S(n)$: speedup factor with $n$ parallel workers
- $p$: fraction of the workload that is parallelizable ($0 \leq p \leq 1$)
- $(1 - p)$: fraction that is inherently sequential
- $n$: number of parallel workers/branches

**Key Implications:**

1. **Maximum speedup** as $n \rightarrow \infty$:

$$
\lim_{n \rightarrow \infty} S(n) = \frac{1}{1 - p}
$$

If only 80% of the pipeline is parallelizable ($p = 0.8$), maximum speedup is $\frac{1}{0.2} = 5\times$, regardless of how many parallel workers are deployed.

2. **Diminishing returns**: Adding more parallel branches yields decreasing marginal speedup. For $p = 0.9$:

| $n$ | $S(n)$ | Marginal Gain |
|---|---|---|
| 1 | 1.0× | — |
| 2 | 1.82× | +0.82× |
| 4 | 3.08× | +1.26× |
| 8 | 4.71× | +1.63× |
| 16 | 6.40× | +1.69× |
| 32 | 7.80× | +1.40× |
| ∞ | 10.0× | — |

**Amdahl's Law for Agentic Pipelines (Extended):**

Standard Amdahl's Law assumes uniform parallel task duration. In agentic systems, parallel tasks have **non-uniform latency**. The extended formulation is:

$$
L_{\text{total}} = L_{\text{seq}} + \max_{i \in \{1, \ldots, k\}} L(T_i^{\text{parallel}}) + L_{\text{overhead}}
$$

where:
- $L_{\text{seq}}$: total latency of inherently sequential components (routing, aggregation, state updates)
- $\max_{i} L(T_i^{\text{parallel}})$: latency determined by the slowest parallel branch
- $L_{\text{overhead}}$: orchestration overhead (task dispatch, synchronization, network)

The effective speedup is:

$$
S_{\text{eff}} = \frac{L_{\text{sequential\_total}}}{L_{\text{seq}} + \max_i L(T_i^{\text{parallel}}) + L_{\text{overhead}}}
$$

**Gustafson's Law (Alternative Perspective):**

Gustafson's Law provides a more optimistic view by assuming the problem size scales with available parallelism:

$$
S(n) = n - (1 - p)(n - 1) = 1 + (n - 1)p
$$

This is more applicable when parallelization enables processing **more data** (e.g., analyzing more documents in the same time budget).

**Efficiency:**

Parallel efficiency $E(n)$ measures how effectively $n$ workers are utilized:

$$
E(n) = \frac{S(n)}{n} = \frac{1}{n(1 - p) + p}
$$

An efficiency of 1.0 means perfect utilization; lower values indicate wasted resources. In agentic pipelines, efficiency is typically 0.3–0.7 due to non-uniform task latency and orchestration overhead.

---

### 2.1.4 Distinction: Parallelization vs. Concurrency vs. Distribution

These three concepts are frequently conflated but are technically distinct:

**Parallelism:**

$$
\text{Parallelism} = \text{Simultaneous execution of multiple tasks on multiple processing units}
$$

Multiple tasks execute at **literally the same instant** on separate hardware (GPUs, CPU cores, machines). True parallelism requires physical hardware parallelism.

**Concurrency:**

$$
\text{Concurrency} = \text{Managing multiple tasks whose execution may overlap in time}
$$

Tasks may **interleave** on a single processing unit, creating the **illusion** of simultaneity. In Python's `asyncio`, coroutines are concurrent but not parallel — they share a single thread, yielding control during I/O waits.

For agentic systems, concurrency is often sufficient because the bottleneck is **I/O-bound** (waiting for LLM API responses), not CPU-bound:

```
Thread 1: [Send LLM Call A] → [Wait...........] → [Receive A]
                                ↕ (yield control)
Thread 1: [Send LLM Call B] → [Wait.......] → [Receive B]
```

Both calls are in-flight simultaneously, even though only one thread manages them.

**Distribution:**

$$
\text{Distribution} = \text{Execution across multiple networked machines}
$$

Tasks execute on different physical machines, communicating via network protocols. Distribution introduces network latency, partial failure modes, and consistency challenges not present in local parallelism.

**Comparison Table:**

| Property | Parallelism | Concurrency | Distribution |
|---|---|---|---|
| **Hardware** | Multiple cores/GPUs/machines | Single or multiple cores | Multiple machines |
| **Execution model** | Truly simultaneous | Interleaved (cooperative or preemptive) | Truly simultaneous across network |
| **Primary benefit** | Reduce latency via simultaneous compute | Efficient I/O utilization | Scale beyond single machine |
| **Overhead** | Thread/process creation | Context switching, event loop | Network latency, serialization |
| **Failure mode** | Shared memory corruption | Deadlock, race conditions | Partial failures, network partitions |
| **Agentic use case** | Parallel LLM calls on multiple GPUs | Async/await for concurrent API calls | Multi-region agent deployment |

**In Practice for Agentic Systems:**

Most agentic parallelization is implemented as **concurrency** (async I/O), not true parallelism, because:
1. The bottleneck is network I/O (LLM API calls), not CPU computation
2. Python's GIL prevents true CPU parallelism in a single process
3. Async/await is simpler to reason about than threads or multiprocessing
4. LLM API endpoints handle parallelism on their side

```python
import asyncio

async def parallel_agent_calls(query: str, agents: list):
    """Concurrent (not parallel) execution of agent calls"""
    tasks = [agent.process(query) for agent in agents]
    results = await asyncio.gather(*tasks)  # All calls in-flight simultaneously
    return results
```

---

## 2.2 Parallelization Patterns

### 2.2.1 Sectioning (Fan-Out / Fan-In)

**Fan-Out / Fan-In** is the fundamental parallelization pattern: decompose a task into independent sub-tasks (fan-out), execute them concurrently, and aggregate results (fan-in).

```
                    ┌─────────────┐
                    │   Input     │
                    │   Query     │
                    └──────┬──────┘
                           │  Fan-Out
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          ┌───────┐    ┌───────┐    ┌───────┐
          │ Sub-  │    │ Sub-  │    │ Sub-  │
          │Task 1 │    │Task 2 │    │Task 3 │
          └───┬───┘    └───┬───┘    └───┬───┘
              │            │            │
              └────────────┼────────────┘
                           │  Fan-In
                    ┌──────▼──────┐
                    │ Aggregator  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Output    │
                    └─────────────┘
```

#### Independent Sub-Task Execution

Each sub-task $T_i$ operates on either:

1. **The same input** (different perspectives): Each agent processes the full query but focuses on a different aspect.

Example: "Analyze this company for investment" →
   - $T_1$: Financial analysis agent
   - $T_2$: Market position analysis agent
   - $T_3$: Risk assessment agent
   - $T_4$: ESG compliance agent

2. **Different partitions of the input** (data parallelism): A large document is split into chunks, each processed independently.

Example: "Summarize this 100-page document" →
   - $T_1$: Summarize pages 1–25
   - $T_2$: Summarize pages 26–50
   - $T_3$: Summarize pages 51–75
   - $T_4$: Summarize pages 76–100

**Formal Correctness Condition:**

Fan-out is correct only when sub-tasks are **truly independent**:

$$
\forall i \neq j: \text{output}(T_i) \text{ does not depend on } \text{output}(T_j)
$$

Violating this condition leads to **inconsistent** or **incomplete** results.

#### Result Aggregation Strategies

The aggregation function $A$ combines sub-task outputs into a final result:

$$
y = A(f_1(x), f_2(x), \ldots, f_k(x))
$$

**Strategy 1: Concatenation**

$$
A_{\text{concat}}(y_1, \ldots, y_k) = y_1 \oplus y_2 \oplus \ldots \oplus y_k
$$

where $\oplus$ denotes string/document concatenation. Appropriate when sub-tasks produce complementary (non-overlapping) content.

**Strategy 2: Merging (LLM-Based Synthesis)**

Use an LLM to synthesize sub-task outputs into a coherent whole:

$$
A_{\text{merge}}(y_1, \ldots, y_k) = \text{LLM}\left(\text{"Synthesize these results: "} \| y_1 \| \ldots \| y_k\right)
$$

This handles redundancy, resolves contradictions, and produces coherent output, but adds an additional LLM call.

**Strategy 3: Voting (for classification/selection tasks)**

$$
A_{\text{vote}}(y_1, \ldots, y_k) = \arg\max_c \sum_{i=1}^{k} \mathbb{1}[y_i = c]
$$

Detailed in Section 2.2.2.

**Strategy 4: Best-of-N Selection**

Generate $k$ candidates and select the best via a scoring function:

$$
A_{\text{best}}(y_1, \ldots, y_k) = \arg\max_{y_i} \text{Score}(y_i)
$$

where $\text{Score}$ may be a reward model, perplexity-based metric, or LLM-as-judge.

**Strategy 5: Structured Union**

When sub-tasks produce structured outputs (JSON, key-value pairs), merge by field:

```python
def structured_union(results: list[dict]) -> dict:
    merged = {}
    for result in results:
        for key, value in result.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[key] = resolve_conflict(merged[key], value)
    return merged
```

**Implementation:**

```python
import asyncio
from typing import List, Callable, Any

class FanOutFanIn:
    def __init__(self, agents: List[Callable], aggregator: Callable):
        self.agents = agents
        self.aggregator = aggregator
    
    async def execute(self, query: str) -> Any:
        # Fan-Out: dispatch to all agents concurrently
        tasks = [agent(query) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter failures
        successful = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        if not successful:
            raise RuntimeError(f"All {len(self.agents)} agents failed: {failures}")
        
        # Fan-In: aggregate results
        output = await self.aggregator(successful)
        return output
```

---

### 2.2.2 Voting / Ensembling

Voting produces robust decisions by aggregating opinions from multiple agents or multiple runs of the same agent.

#### Majority Voting

Each of $k$ agents independently produces a discrete output. The final output is the **mode** (most frequent value):

$$
\hat{y} = \arg\max_c \sum_{i=1}^{k} \mathbb{1}[f_i(x) = c]
$$

**Correctness via Condorcet's Jury Theorem:**

If each agent independently produces the correct answer with probability $p > 0.5$, then the probability that the majority vote is correct approaches 1 as $k \rightarrow \infty$:

$$
P(\text{majority correct}) = \sum_{j=\lceil k/2 \rceil}^{k} \binom{k}{j} p^j (1-p)^{k-j} \xrightarrow{k \rightarrow \infty} 1
$$

For $k = 5$ agents each with $p = 0.7$:

$$
P(\text{majority correct}) = \sum_{j=3}^{5} \binom{5}{j} (0.7)^j (0.3)^{5-j} = 0.837
$$

Each individual agent is 70% accurate, but the ensemble achieves 83.7% accuracy.

**Critical assumption**: Agents must be **independent** (not correlated errors). If all agents use the same model and training data, their errors are highly correlated, and voting provides minimal benefit. Diversification strategies include:

- Using different LLMs (GPT-4o, Claude Sonnet, Gemini Pro)
- Using different prompts or system instructions
- Using different temperatures
- Using different retrieval contexts (RAG with different corpora)

#### Weighted Voting

When agents have different reliability levels, weight their votes accordingly:

$$
\hat{y} = \arg\max_c \sum_{i=1}^{k} w_i \cdot \mathbb{1}[f_i(x) = c]
$$

where $w_i \geq 0$ and $\sum_i w_i = 1$.

**Weight Estimation:**

1. **Performance-based**: Weight proportional to historical accuracy:

$$
w_i = \frac{\text{accuracy}_i}{\sum_j \text{accuracy}_j}
$$

2. **Confidence-based**: Weight proportional to the agent's self-reported confidence:

$$
w_i = \text{Confidence}(f_i(x))
$$

3. **Learned weights**: Optimize weights on a validation set:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{(x, y) \in \mathcal{D}_{\text{val}}} \mathcal{L}\left(y, \arg\max_c \sum_i w_i \cdot \mathbb{1}[f_i(x) = c]\right)
$$

#### Self-Consistency Decoding

A powerful instantiation of voting for chain-of-thought reasoning, introduced by Wang et al. (2022):

**Procedure:**

1. Sample $k$ reasoning paths from the same LLM using temperature $T > 0$:

$$
(r^{(i)}, a^{(i)}) \sim P_{\text{LLM}}(\cdot \mid q; T) \quad \text{for } i = 1, \ldots, k
$$

where $r^{(i)}$ is the reasoning chain and $a^{(i)}$ is the final answer.

2. Take a majority vote over the **final answers** (ignoring the reasoning chains):

$$
\hat{y} = \arg\max_c \sum_{i=1}^{k} \mathbb{1}[a^{(i)} = c]
$$

**Why it works:**

- Correct reasoning paths tend to converge on the same answer through diverse routes
- Incorrect reasoning paths tend to produce diverse (scattered) wrong answers
- Majority vote amplifies the signal of correct reasoning

**Mathematical Justification:**

Let $p_{\text{correct}}$ be the probability that a single sample produces the correct answer $c^*$. The aggregated probability of the majority selecting $c^*$ is:

$$
P(\hat{y} = c^*) = \sum_{j=\lceil k/2 \rceil}^{k} \binom{k}{j} p_{\text{correct}}^j (1 - p_{\text{correct}})^{k-j}
$$

For $p_{\text{correct}} > 0.5$, this exceeds $p_{\text{correct}}$ for all $k \geq 3$.

**Optimal Number of Samples:**

The marginal quality gain from additional samples diminishes. Empirically, $k = 5$ to $k = 20$ provides the best cost-quality tradeoff. Formally, the expected accuracy improvement from adding sample $k+1$ is:

$$
\Delta(k) = P(\text{correct} \mid k+1) - P(\text{correct} \mid k)
$$

This approaches zero rapidly as $k$ grows.

**Cost Consideration:**

Self-consistency multiplies cost by $k$:

$$
C_{\text{SC}} = k \cdot C_{\text{single}}
$$

But since all $k$ calls execute in parallel, latency increases only by the overhead of the aggregation step, not by a factor of $k$.

**Implementation:**

```python
import asyncio
from collections import Counter

async def self_consistency(llm, query: str, k: int = 5, temperature: float = 0.7):
    """Self-consistency decoding with parallel sampling"""
    
    async def sample_once():
        response = await llm.generate(
            query, 
            temperature=temperature,
            max_tokens=1024
        )
        # Extract final answer from reasoning chain
        answer = extract_final_answer(response)
        return {"reasoning": response, "answer": answer}
    
    # Parallel sampling
    samples = await asyncio.gather(*[sample_once() for _ in range(k)])
    
    # Majority vote on final answers
    answers = [s["answer"] for s in samples]
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]
    
    # Return the best answer with a supporting reasoning chain
    confidence = vote_counts[best_answer] / k
    supporting = next(s for s in samples if s["answer"] == best_answer)
    
    return {
        "answer": best_answer,
        "confidence": confidence,
        "vote_distribution": dict(vote_counts),
        "reasoning": supporting["reasoning"]
    }
```

---

### 2.2.3 Pipeline Parallelism

Pipeline parallelism exploits **stage-level overlap** in multi-step chains. While one input is being processed by Stage 2, the next input can simultaneously be processed by Stage 1.

#### Stage-Level Parallelism in Multi-Step Chains

Consider a 3-stage pipeline: $S_1 \rightarrow S_2 \rightarrow S_3$

**Sequential execution** for 3 inputs:

```
Time →  |  t=0  |  t=1  |  t=2  |  t=3  |  t=4  |  t=5  |  t=6  |  t=7  |  t=8  |
S1:     | Inp1  |       |       | Inp2  |       |       | Inp3  |       |       |
S2:     |       | Inp1  |       |       | Inp2  |       |       | Inp3  |       |
S3:     |       |       | Inp1  |       |       | Inp2  |       |       | Inp3  |
```

Total time for $N$ inputs: $N \times k$ time units (where $k$ is the number of stages).

**Pipelined execution:**

```
Time →  |  t=0  |  t=1  |  t=2  |  t=3  |  t=4  |
S1:     | Inp1  | Inp2  | Inp3  |       |       |
S2:     |       | Inp1  | Inp2  | Inp3  |       |
S3:     |       |       | Inp1  | Inp2  | Inp3  |
```

Total time for $N$ inputs: $k + (N - 1)$ time units.

**Pipeline Speedup:**

$$
S_{\text{pipeline}} = \frac{N \times k}{k + (N - 1)} \xrightarrow{N \rightarrow \infty} k
$$

For large $N$, pipeline parallelism achieves a speedup factor equal to the number of stages $k$.

**Throughput:**

$$
\text{Throughput} = \frac{N}{k + (N-1)} \xrightarrow{N \rightarrow \infty} \frac{1}{1} = 1 \text{ input per time unit}
$$

compared to $\frac{1}{k}$ inputs per time unit without pipelining.

#### Micro-Batching Across Pipeline Stages

When pipeline stages have non-uniform latency, the slowest stage becomes the **bottleneck**:

$$
\text{Throughput}_{\text{pipeline}} = \frac{1}{\max_i L(S_i)}
$$

**Micro-batching** mitigates this by accumulating multiple inputs at each stage and processing them as a batch:

```python
class PipelineStage:
    def __init__(self, agent, batch_size=4, max_wait_ms=100):
        self.agent = agent
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.buffer = []
        self.next_stage = None
    
    async def submit(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            await self._flush()
    
    async def _flush(self):
        batch = self.buffer
        self.buffer = []
        # Process batch concurrently
        results = await asyncio.gather(*[self.agent(item) for item in batch])
        if self.next_stage:
            for result in results:
                await self.next_stage.submit(result)
```

**Optimal Batch Size Selection:**

The batch size $B$ balances latency and throughput:

$$
B^* = \arg\min_B \left[\frac{L_{\text{batch}}(B)}{B} + \lambda \cdot L_{\text{wait}}(B)\right]
$$

where $L_{\text{batch}}(B)$ is the latency to process a batch of size $B$ and $L_{\text{wait}}(B)$ is the average waiting time to fill a batch.

---

### 2.2.4 Data Parallelism

Data parallelism processes **multiple independent inputs** simultaneously, distributing the workload across agents or workers.

#### Batch Processing Across Multiple Inputs

Given $N$ independent queries $\{q_1, q_2, \ldots, q_N\}$ and $W$ workers, assign $\lceil N/W \rceil$ queries to each worker:

$$
L_{\text{data\_parallel}} = \lceil N/W \rceil \cdot L_{\text{per\_query}}
$$

compared to $N \cdot L_{\text{per\_query}}$ for sequential processing.

**Implementation:**

```python
import asyncio
from typing import List

async def data_parallel_process(
    queries: List[str], 
    agent, 
    max_concurrency: int = 10
) -> List[dict]:
    """Process multiple queries in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_one(query: str) -> dict:
        async with semaphore:
            return await agent.process(query)
    
    tasks = [process_one(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

#### Shard-Based Document Processing

For large documents that exceed a single LLM's context window or benefit from parallel analysis:

**Sharding Strategy:**

Given document $D$ of $N$ tokens and shard size $s$:

$$
D = D_1 \cup D_2 \cup \ldots \cup D_{\lceil N/s \rceil}
$$

with optional **overlap** $o$ between adjacent shards to preserve context continuity:

$$
D_i = D[(i-1)(s-o) : (i-1)(s-o) + s]
$$

**Overlap ensures** that information at shard boundaries is not lost. Typical overlap is 10-20% of shard size.

**Map Phase** (parallel):

$$
r_i = f(D_i) \quad \forall i \in \{1, \ldots, \lceil N/s \rceil\}
$$

**Reduce Phase** (aggregation):

$$
R = A(r_1, r_2, \ldots, r_{\lceil N/s \rceil})
$$

**Example: Parallel Document Summarization**

```python
async def parallel_summarize(document: str, chunk_size: int = 4000, 
                              overlap: int = 500, llm=None):
    # Shard the document
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        chunks.append(document[start:end])
        start += chunk_size - overlap
    
    # Map: summarize each chunk in parallel
    summaries = await asyncio.gather(*[
        llm.generate(f"Summarize this section:\n\n{chunk}")
        for chunk in chunks
    ])
    
    # Reduce: synthesize chunk summaries into a final summary
    combined = "\n\n".join([f"Section {i+1}: {s}" for i, s in enumerate(summaries)])
    final_summary = await llm.generate(
        f"Synthesize these section summaries into a coherent overall summary:\n\n{combined}"
    )
    
    return final_summary
```

---

### 2.2.5 Speculative Execution

Speculative execution **pre-computes results for likely future branches** before the branch decision is made, discarding unused results if the prediction was wrong.

#### Pre-Computation of Likely Branches

Given a decision point with $m$ possible outcomes $\{o_1, o_2, \ldots, o_m\}$:

1. **Predict** the most likely outcomes based on historical data or heuristics
2. **Speculatively execute** downstream tasks for the top-$k$ predicted outcomes in parallel
3. **When the actual decision is made**, use the pre-computed result if it matches; otherwise, compute from scratch

**Formal Model:**

Let $P(o_j \mid x)$ be the predicted probability of outcome $o_j$ given context $x$. Speculatively execute branches for outcomes with $P(o_j \mid x) \geq \tau_{\text{spec}}$:

$$
\text{Spec}(x) = \{o_j : P(o_j \mid x) \geq \tau_{\text{spec}}\}
$$

**Expected Latency:**

$$
L_{\text{spec}} = P(\text{hit}) \cdot L_{\text{parallel}} + P(\text{miss}) \cdot (L_{\text{parallel}} + L_{\text{recompute}})
$$

where $P(\text{hit}) = P(o_{\text{actual}} \in \text{Spec}(x))$.

If speculation accuracy is high ($P(\text{hit}) > 0.8$), speculative execution provides significant latency savings:

$$
L_{\text{spec}} < L_{\text{sequential}} \iff P(\text{hit}) > \frac{L_{\text{parallel}} + L_{\text{recompute}} - L_{\text{sequential}}}{L_{\text{recompute}}}
$$

**Example: Speculative Tool Call**

```python
async def speculative_execution(query: str, router, agents: dict):
    # Phase 1: Start routing AND speculatively execute likely agents
    route_task = asyncio.create_task(router.route(query))
    
    # Speculate: pre-execute top-2 most likely agents
    top_predictions = router.predict_top_k(query, k=2)
    spec_tasks = {
        agent_name: asyncio.create_task(agents[agent_name].process(query))
        for agent_name in top_predictions
    }
    
    # Phase 2: Get actual routing decision
    actual_route = await route_task
    
    # Phase 3: Use speculative result if available
    if actual_route in spec_tasks:
        result = await spec_tasks[actual_route]
        # Cancel unused speculative tasks
        for name, task in spec_tasks.items():
            if name != actual_route:
                task.cancel()
        return result
    else:
        # Speculation miss: cancel all speculative tasks and execute correctly
        for task in spec_tasks.values():
            task.cancel()
        return await agents[actual_route].process(query)
```

#### Rollback on Misprediction

When speculative execution produces an incorrect result (the actual branch differs from the speculated branch):

1. **Discard** the speculative result
2. **Cancel** any in-flight speculative computations
3. **Execute** the correct branch from scratch

**Cost of Misprediction:**

$$
C_{\text{mispredict}} = C_{\text{wasted\_speculation}} + C_{\text{correct\_execution}}
$$

The wasted cost is the token/compute cost of the discarded speculative branches. This cost is acceptable only if:

$$
P(\text{hit}) \cdot \text{Latency\_Savings} > P(\text{miss}) \cdot C_{\text{wasted}}
$$

**Speculative Decoding (LLM-Specific Application):**

In LLM inference, speculative decoding uses a small **draft model** to generate $k$ tokens speculatively, then the large **target model** verifies them in a single forward pass:

$$
\text{Draft tokens: } y_1, y_2, \ldots, y_k \sim P_{\text{draft}}(\cdot \mid x)
$$

$$
\text{Verify: } P_{\text{target}}(y_1, \ldots, y_k \mid x)
$$

If the target model agrees with $j$ tokens before rejecting, $j+1$ tokens are generated in the time of approximately one target model forward pass. This achieves $2\text{-}3\times$ speedup in token generation latency.

---

## 2.3 Dependency Analysis and Execution Graphs

### 2.3.1 Task Dependency Graph Construction

A task dependency graph $G = (V, E)$ formalizes the execution structure of a parallelized workflow:

- $V = \{v_1, v_2, \ldots, v_n\}$: set of tasks (vertices)
- $E \subseteq V \times V$: set of directed edges where $(v_i, v_j) \in E$ means $v_j$ depends on $v_i$ (i.e., $v_i$ must complete before $v_j$ can start)

**Properties:**

1. **Acyclicity**: The graph must be a DAG (Directed Acyclic Graph). Cycles indicate circular dependencies and make execution impossible.

2. **Source vertices**: Tasks with no incoming edges ($\text{in-degree}(v) = 0$) can start immediately.

3. **Sink vertices**: Tasks with no outgoing edges ($\text{out-degree}(v) = 0$) produce final outputs.

4. **Parallelizable sets**: Tasks at the same "level" (same longest distance from source) can execute in parallel.

**Construction Methods:**

**Method 1: Explicit Declaration**

```python
class TaskGraph:
    def __init__(self):
        self.tasks = {}       # name -> callable
        self.edges = {}       # name -> list of dependency names
    
    def add_task(self, name: str, fn: callable, depends_on: list = None):
        self.tasks[name] = fn
        self.edges[name] = depends_on or []
    
    def build(self):
        """Validate DAG and compute execution order"""
        # Check for cycles
        if self._has_cycle():
            raise ValueError("Circular dependency detected")
        # Compute topological order
        self.order = self._topological_sort()
        # Compute parallelism levels
        self.levels = self._compute_levels()
        return self

# Example usage
graph = TaskGraph()
graph.add_task("search", search_agent)
graph.add_task("analyze", analyze_agent)
graph.add_task("code", code_agent, depends_on=["analyze"])
graph.add_task("visualize", viz_agent, depends_on=["analyze"])
graph.add_task("report", report_agent, depends_on=["search", "code", "visualize"])
graph.build()
```

**Method 2: LLM-Generated Dependency Graph**

```python
DEPENDENCY_ANALYSIS_PROMPT = """
Given these sub-tasks, determine the dependency relationships.
A task depends on another if it needs that task's output.

Sub-tasks:
{tasks}

For each task, list its dependencies (tasks that must complete first).
Output JSON: {{"task_name": ["dependency1", "dependency2"], ...}}
"""
```

**Method 3: Automated Dependency Detection**

Analyze the input/output signatures of each task to infer dependencies:

$$
(v_i, v_j) \in E \iff \text{output\_type}(v_i) \in \text{input\_types}(v_j)
$$

---

### 2.3.2 Topological Sorting for Execution Ordering

A **topological sort** of a DAG produces a linear ordering of vertices such that for every edge $(v_i, v_j) \in E$, $v_i$ appears before $v_j$.

**Kahn's Algorithm (BFS-based):**

```python
from collections import deque

def topological_sort(tasks: dict, edges: dict) -> list:
    """
    tasks: {name: task_fn}
    edges: {name: [dependency_names]}
    Returns: list of task names in valid execution order
    """
    # Compute in-degrees
    in_degree = {name: 0 for name in tasks}
    reverse_edges = {name: [] for name in tasks}  # dependents
    for name, deps in edges.items():
        in_degree[name] = len(deps)
        for dep in deps:
            reverse_edges[dep].append(name)
    
    # Start with tasks that have no dependencies
    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    order = []
    
    while queue:
        # All tasks in the queue can execute in parallel
        current_level = list(queue)
        queue.clear()
        order.append(current_level)  # Group by parallelism level
        
        for task in current_level:
            for dependent in reverse_edges[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
    
    # Check for cycles
    if sum(len(level) for level in order) != len(tasks):
        raise ValueError("Cycle detected: not all tasks were processed")
    
    return order  # List of lists: each inner list is a parallelizable group
```

**Output for the Example Above:**

```
Level 0: [search, analyze]       ← execute in parallel
Level 1: [code, visualize]       ← execute in parallel (after analyze)
Level 2: [report]                ← execute after all dependencies
```

**Time Complexity:** $O(|V| + |E|)$

---

### 2.3.3 Critical Path Analysis

The **critical path** is the longest path through the DAG, measured by the sum of task durations. It determines the **minimum possible end-to-end latency** regardless of available parallelism.

$$
T_{\text{critical}} = \max_{\text{paths } P \text{ from source to sink}} \sum_{v \in P} t(v)
$$

where $t(v)$ is the execution time (latency) of task $v$.

**Computation via Dynamic Programming:**

For each vertex $v$, compute the **earliest start time** $\text{EST}(v)$ and **earliest finish time** $\text{EFT}(v)$:

$$
\text{EST}(v) = \max_{u \in \text{predecessors}(v)} \text{EFT}(u)
$$

$$
\text{EFT}(v) = \text{EST}(v) + t(v)
$$

For source vertices: $\text{EST}(v) = 0$.

The critical path length is:

$$
T_{\text{critical}} = \max_{v \in \text{sinks}} \text{EFT}(v)
$$

**Slack (Float) Analysis:**

The **slack** of a task is the amount of time it can be delayed without increasing the overall completion time:

$$
\text{Slack}(v) = \text{LST}(v) - \text{EST}(v)
$$

where $\text{LST}(v)$ is the latest start time (computed via backward pass):

$$
\text{LST}(v) = \min_{u \in \text{successors}(v)} \text{LST}(u) - t(v)
$$

Tasks with **zero slack** are on the critical path. Optimizing the overall latency requires reducing the duration of critical-path tasks.

**Implementation:**

```python
def critical_path_analysis(tasks: dict, edges: dict, durations: dict):
    """
    tasks: {name: ...}
    edges: {name: [dependency_names]}
    durations: {name: estimated_seconds}
    """
    # Forward pass: compute EST and EFT
    est = {}
    eft = {}
    order = topological_sort(tasks, edges)
    
    for level in order:
        for task in level:
            deps = edges[task]
            est[task] = max([eft[d] for d in deps], default=0)
            eft[task] = est[task] + durations[task]
    
    # Total completion time
    makespan = max(eft.values())
    
    # Backward pass: compute LST and LFT
    lst = {}
    lft = {}
    reverse_deps = {t: [] for t in tasks}
    for t, deps in edges.items():
        for d in deps:
            reverse_deps[d].append(t)
    
    for level in reversed(order):
        for task in level:
            succs = reverse_deps[task]
            lft[task] = min([lst[s] for s in succs], default=makespan)
            lst[task] = lft[task] - durations[task]
    
    # Slack and critical path
    slack = {t: lst[t] - est[t] for t in tasks}
    critical = [t for t, s in slack.items() if abs(s) < 1e-9]
    
    return {
        "makespan": makespan,
        "est": est, "eft": eft,
        "lst": lst, "lft": lft,
        "slack": slack,
        "critical_path": critical
    }
```

**Practical Implications for Agentic Systems:**

1. **Optimize the critical path**: If `code_agent` takes 5s and is on the critical path, switching to a faster model or optimizing the prompt saves 5s of total latency — regardless of how many parallel branches exist.

2. **Non-critical tasks can tolerate slowness**: If `search_agent` has 3s of slack, it can use a cheaper/slower model without impacting overall latency.

3. **Dynamic re-estimation**: Task durations are stochastic in agentic systems. Re-compute the critical path dynamically as actual durations become known to adapt the execution strategy.

---

### 2.3.4 Dynamic DAG Construction at Runtime

In many agentic workflows, the full dependency graph is not known at design time. An LLM planner constructs the DAG dynamically based on the specific query.

**Architecture:**

```
Query → Planner LLM → DAG Specification → DAG Executor → Results
```

**Planner Output Schema:**

```json
{
  "tasks": [
    {"id": "t1", "agent": "search_agent", "input": "...", "depends_on": []},
    {"id": "t2", "agent": "code_agent", "input": "...", "depends_on": []},
    {"id": "t3", "agent": "analysis_agent", "input": "...", "depends_on": ["t1", "t2"]},
    {"id": "t4", "agent": "report_agent", "input": "...", "depends_on": ["t3"]}
  ]
}
```

**Dynamic DAG Executor:**

```python
class DynamicDAGExecutor:
    def __init__(self, agents: dict):
        self.agents = agents
    
    async def execute(self, dag_spec: dict) -> dict:
        results = {}
        completed = set()
        tasks = {t["id"]: t for t in dag_spec["tasks"]}
        
        while len(completed) < len(tasks):
            # Find ready tasks (all dependencies met)
            ready = [
                tid for tid, task in tasks.items()
                if tid not in completed
                and all(d in completed for d in task["depends_on"])
            ]
            
            if not ready:
                raise RuntimeError("Deadlock: no ready tasks but not all complete")
            
            # Execute ready tasks in parallel
            async def run_task(tid):
                task = tasks[tid]
                dep_results = {d: results[d] for d in task["depends_on"]}
                agent = self.agents[task["agent"]]
                result = await agent.process(task["input"], dep_results)
                return tid, result
            
            batch_results = await asyncio.gather(*[run_task(tid) for tid in ready])
            
            for tid, result in batch_results:
                results[tid] = result
                completed.add(tid)
        
        return results
```

**Adaptive DAG Modification:**

The DAG can be modified during execution based on intermediate results:

1. **Branch Addition**: If an intermediate result reveals the need for additional processing, add new tasks to the DAG
2. **Branch Pruning**: If an intermediate result makes certain downstream tasks unnecessary, remove them
3. **Re-routing**: If an agent fails, replace the task with an alternative agent

$$
G_{t+1} = \text{Modify}(G_t, \text{IntermediateResults}_t)
$$

---

### 2.3.5 Cycle Detection and Resolution in Agent Workflows

Cycles in the task graph indicate **circular dependencies** that make execution impossible in a standard DAG framework.

**Cycle Detection:**

Use **DFS-based cycle detection**:

```python
def has_cycle(tasks: dict, edges: dict) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t: WHITE for t in tasks}
    
    def dfs(v):
        color[v] = GRAY
        for neighbor in get_dependents(v, edges):
            if color[neighbor] == GRAY:
                return True  # Back edge → cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[v] = BLACK
        return False
    
    return any(dfs(v) for v in tasks if color[v] == WHITE)
```

**Time Complexity:** $O(|V| + |E|)$

**Resolution Strategies:**

1. **Iterative Convergence (Acceptable Cycles)**: Some agentic workflows intentionally use cycles for iterative refinement:

$$
x_{t+1} = f(x_t) \quad \text{until} \quad \|x_{t+1} - x_t\| < \epsilon \quad \text{or} \quad t > T_{\max}
$$

These are not DAG cycles but rather **loops with termination conditions**. They should be modeled as **iteration nodes** in the DAG, not as true cycles:

```python
class IterativeNode:
    def __init__(self, agent, max_iterations=5, convergence_threshold=0.95):
        self.agent = agent
        self.max_iter = max_iterations
        self.threshold = convergence_threshold
    
    async def execute(self, input_data):
        result = input_data
        for i in range(self.max_iter):
            new_result = await self.agent.process(result)
            if self.converged(result, new_result):
                return new_result
            result = new_result
        return result  # Max iterations reached
```

2. **Cycle Breaking**: Identify the edge that introduces the cycle and remove it, replacing with a sequential dependency or an initial estimate.

3. **Unrolling**: Replace a cycle of depth $d$ with $d$ copies of the cycle nodes arranged sequentially.

---

## 2.4 Synchronization and Coordination

### 2.4.1 Barrier Synchronization

A **barrier** is a synchronization primitive that blocks all tasks until every task in a group has reached the barrier point.

**Formal Definition:**

A barrier $B$ over a set of tasks $\{T_1, \ldots, T_k\}$ ensures:

$$
\forall i: T_i \text{ does not proceed past } B \text{ until } \forall j: T_j \text{ has reached } B
$$

**Use Case in Agentic Systems:**

After a fan-out phase, all parallel branches must complete before the aggregation (fan-in) step begins. The barrier ensures the aggregator does not start with incomplete data.

**Implementation with `asyncio`:**

```python
import asyncio

class Barrier:
    def __init__(self, n: int):
        self.n = n
        self.count = 0
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()
    
    async def wait(self):
        async with self.lock:
            self.count += 1
            if self.count >= self.n:
                self.event.set()
        await self.event.wait()

# Usage
barrier = Barrier(n=3)

async def parallel_task(agent, query, barrier):
    result = await agent.process(query)
    await barrier.wait()  # Wait for all tasks
    return result
```

In practice, `asyncio.gather()` provides implicit barrier semantics — it returns only when all gathered coroutines have completed.

**Partial Barrier (Quorum-Based):**

A quorum barrier unblocks when $m$ out of $k$ tasks have completed ($m \leq k$):

$$
\text{Unblock when } |\{i : T_i \text{ complete}\}| \geq m
$$

This is useful when some agents may be slow or fail, and we want to proceed with partial results:

```python
async def quorum_gather(tasks, quorum: int, timeout: float = 30.0):
    """Wait for at least `quorum` tasks to complete, with timeout"""
    results = []
    pending = set(asyncio.ensure_future(t) for t in tasks)
    
    while len(results) < quorum and pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                results.append(task.result())
            except Exception:
                pass  # Skip failed tasks
    
    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    
    if len(results) < quorum:
        raise RuntimeError(f"Only {len(results)}/{quorum} tasks completed")
    
    return results
```

---

### 2.4.2 Futures and Promises in Agent Orchestration

A **Future** (or **Promise**) is a placeholder for a value that will be available at some point in the future. It decouples task submission from result retrieval.

**Formal Definition:**

A future $F$ over type $T$ has three states:

$$
\text{State}(F) \in \{\text{PENDING}, \text{RESOLVED}(v : T), \text{REJECTED}(e : \text{Error})\}
$$

**Operations:**

| Operation | Semantics |
|---|---|
| `create()` | Create a new future in PENDING state |
| `resolve(v)` | Transition from PENDING to RESOLVED with value $v$ |
| `reject(e)` | Transition from PENDING to REJECTED with error $e$ |
| `await` | Block until the future is RESOLVED or REJECTED, then return value or raise error |
| `then(f)` | Chain: when resolved, apply function $f$ to the result, returning a new future |
| `is_done()` | Non-blocking check: returns True if RESOLVED or REJECTED |

**In Python asyncio:**

```python
import asyncio

async def orchestrate_with_futures(query: str, agents: dict):
    # Create futures for each agent
    futures = {
        name: asyncio.ensure_future(agent.process(query))
        for name, agent in agents.items()
    }
    
    # Can do other work while agents are running...
    
    # Retrieve results when needed
    search_result = await futures["search"]
    code_result = await futures["code"]
    
    # Use results in downstream task
    final = await synthesis_agent.process(
        search_result=search_result,
        code_result=code_result
    )
    return final
```

**Chaining Futures (Composition):**

Futures can be composed to express complex dependency patterns:

$$
F_3 = F_1.\text{then}(f_A).\text{then}(f_B)
$$

is equivalent to $F_3 = f_B(f_A(\text{await } F_1))$.

**Combining Futures:**

| Combinator | Semantics |
|---|---|
| `gather(F_1, \ldots, F_k)` | Wait for ALL futures; return list of results |
| `wait(F_1, \ldots, F_k, return_when=FIRST_COMPLETED)` | Return as soon as ANY future completes |
| `race(F_1, \ldots, F_k)` | Return the result of the FIRST future to complete; cancel others |
| `all_settled(F_1, \ldots, F_k)` | Wait for ALL futures; return results and errors |

---

### 2.4.3 Callback Patterns

Callbacks are functions invoked upon completion of an asynchronous task:

**Direct Callback:**

```python
async def agent_with_callback(agent, query, on_complete):
    result = await agent.process(query)
    await on_complete(result)
    return result
```

**Event-Driven Callback System:**

```python
class EventEmitter:
    def __init__(self):
        self._handlers = {}
    
    def on(self, event: str, handler: callable):
        self._handlers.setdefault(event, []).append(handler)
    
    async def emit(self, event: str, data: any):
        for handler in self._handlers.get(event, []):
            await handler(data)

# Usage
emitter = EventEmitter()
emitter.on("task_complete", lambda r: log_result(r))
emitter.on("task_complete", lambda r: update_dashboard(r))
emitter.on("task_failed", lambda e: alert_ops(e))
```

**Callback Hell and Mitigation:**

Deeply nested callbacks are difficult to reason about. Modern Python mitigates this via `async/await` syntax, which provides sequential-looking code for asynchronous operations.

**Anti-pattern (callback hell):**

```python
agent_a.process(query, callback=lambda result_a:
    agent_b.process(result_a, callback=lambda result_b:
        agent_c.process(result_b, callback=lambda result_c:
            synthesizer.combine(result_a, result_b, result_c))))
```

**Preferred (async/await):**

```python
result_a = await agent_a.process(query)
result_b = await agent_b.process(result_a)
result_c = await agent_c.process(result_b)
final = await synthesizer.combine(result_a, result_b, result_c)
```

---

### 2.4.4 Map-Reduce for Agentic Tasks

Map-Reduce is a two-phase parallelization pattern directly applicable to agentic workflows.

**Phase 1 — Map (Parallel):**

Apply a function $f$ independently to each element of an input collection:

$$
\text{Map}: \{x_1, \ldots, x_n\} \xrightarrow{f} \{f(x_1), \ldots, f(x_n)\}
$$

All $n$ applications of $f$ are independent and execute in parallel.

**Phase 2 — Reduce (Aggregation):**

Combine all mapped results into a single output using an associative aggregation function $\oplus$:

$$
\text{Reduce}: \{y_1, \ldots, y_n\} \xrightarrow{\oplus} y_1 \oplus y_2 \oplus \ldots \oplus y_n
$$

**Total Latency:**

$$
L_{\text{MapReduce}} = \max_i L_{\text{map}}(x_i) + L_{\text{reduce}}
$$

**Hierarchical Reduce:**

For large $n$, the reduce phase itself can be parallelized via a **tree reduction**:

```
Level 0 (Map):     y1  y2  y3  y4  y5  y6  y7  y8
Level 1 (Reduce):  y12     y34     y56     y78
Level 2 (Reduce):  y1234           y5678
Level 3 (Reduce):  y_final
```

Tree reduction has $O(\log n)$ depth with $O(n)$ total work.

**Implementation:**

```python
async def map_reduce(
    items: list,
    map_fn: callable,
    reduce_fn: callable,
    max_concurrency: int = 10
):
    """
    Generic map-reduce for agentic tasks
    
    map_fn: async (item) -> result
    reduce_fn: async (result1, result2) -> combined_result
    """
    # Map phase: parallel execution with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_map(item):
        async with semaphore:
            return await map_fn(item)
    
    mapped = await asyncio.gather(*[bounded_map(item) for item in items])
    
    # Reduce phase: tree reduction for parallelism
    while len(mapped) > 1:
        pairs = []
        for i in range(0, len(mapped) - 1, 2):
            pairs.append((mapped[i], mapped[i + 1]))
        if len(mapped) % 2 == 1:
            pairs.append((mapped[-1], None))
        
        async def reduce_pair(pair):
            a, b = pair
            if b is None:
                return a
            return await reduce_fn(a, b)
        
        mapped = await asyncio.gather(*[reduce_pair(p) for p in pairs])
    
    return mapped[0]

# Example: Summarize a large set of documents
async def summarize_all(documents: list, llm):
    async def summarize_one(doc):
        return await llm.generate(f"Summarize:\n{doc}")
    
    async def merge_summaries(s1, s2):
        return await llm.generate(f"Merge these summaries:\n1: {s1}\n2: {s2}")
    
    return await map_reduce(documents, summarize_one, merge_summaries)
```

---

### 2.4.5 Handling Partial Failures and Stragglers

In parallel agentic execution, individual branches may fail or take unexpectedly long.

**Failure Modes:**

| Mode | Cause | Consequence |
|---|---|---|
| **Hard failure** | Agent crashes, API error, timeout | No result from that branch |
| **Soft failure** | Agent produces low-quality or irrelevant output | Result exists but is unreliable |
| **Straggler** | Agent takes much longer than peers | Delays overall completion (bottleneck) |

**Strategy 1: Retry with Exponential Backoff**

```python
async def retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
```

**Strategy 2: Timeout with Fallback**

```python
async def with_timeout(fn, timeout_seconds=30, fallback=None):
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if fallback:
            return await fallback()
        return None
```

**Strategy 3: Hedged Requests**

Send the same request to multiple agents/endpoints simultaneously and use the first response:

$$
y = \text{first}(f_1(x), f_2(x), \ldots, f_m(x))
$$

This reduces tail latency at the cost of increased resource usage.

```python
async def hedged_request(agents: list, query: str):
    """Send to all agents, return first successful result"""
    tasks = [asyncio.create_task(agent.process(query)) for agent in agents]
    
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    
    # Return the first successful result
    for task in done:
        if not task.exception():
            return task.result()
    
    raise RuntimeError("All hedged requests failed")
```

**Strategy 4: Graceful Degradation**

Proceed with partial results when some branches fail:

```python
async def graceful_parallel(agents: dict, query: str, 
                             required: set, optional: set):
    """
    Execute all agents in parallel.
    Fail if any required agent fails.
    Proceed without optional agents that fail.
    """
    all_agents = {**{k: agents[k] for k in required},
                  **{k: agents[k] for k in optional}}
    
    tasks = {
        name: asyncio.create_task(agent.process(query))
        for name, agent in all_agents.items()
    }
    
    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await asyncio.wait_for(task, timeout=30)
        except (asyncio.TimeoutError, Exception) as e:
            if name in required:
                raise RuntimeError(f"Required agent '{name}' failed: {e}")
            else:
                results[name] = None  # Optional agent failed gracefully
    
    return results
```

**Straggler Mitigation:**

The **straggler problem** is that the overall latency is determined by the slowest branch:

$$
L_{\text{total}} = \max_i L(T_i)
$$

If one task takes 10× longer than the average, it wastes all the latency savings from parallelization.

**Mitigation techniques:**

1. **Timeout + fallback**: Set aggressive timeouts; if exceeded, use a fallback agent or default result
2. **Speculative redundancy**: Launch duplicate requests to different endpoints; use the first to complete
3. **Work stealing**: If one agent finishes early, reassign it to help with a slow task (applicable for decomposable tasks)
4. **Adaptive timeout**: Set timeout proportional to expected duration:

$$
\tau_i = \hat{\mu}_i + 3\hat{\sigma}_i
$$

where $\hat{\mu}_i, \hat{\sigma}_i$ are the estimated mean and standard deviation of agent $i$'s latency.

---

## 2.5 Concurrency Control

### 2.5.1 Rate Limiting and Throttling

LLM API providers impose rate limits. Exceeding them causes request failures (HTTP 429). Agentic systems must manage concurrent requests within these constraints.

**Rate Limit Types:**

| Limit Type | Unit | Example |
|---|---|---|
| **Requests per minute (RPM)** | API calls/min | 500 RPM for GPT-4o |
| **Tokens per minute (TPM)** | Tokens/min | 30,000 TPM for GPT-4o |
| **Requests per day (RPD)** | API calls/day | 10,000 RPD |
| **Concurrent requests** | Simultaneous connections | 25 concurrent |

**Token Bucket Algorithm:**

The standard rate limiting algorithm:

```python
import asyncio
import time

class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int):
        """
        rate: tokens added per second
        capacity: maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        while True:
            async with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            # Wait for tokens to accumulate
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
    
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

# Usage: 500 RPM → ~8.33 requests/second
limiter = TokenBucketRateLimiter(rate=8.33, capacity=50)

async def rate_limited_call(llm, query):
    await limiter.acquire()
    return await llm.generate(query)
```

**Multi-Tier Rate Limiting:**

When multiple agents share the same API endpoint, coordinate rate limits across agents:

```python
class SharedRateLimiter:
    """Singleton rate limiter shared across all agents using the same API"""
    _instances = {}
    
    @classmethod
    def get(cls, api_name: str, rate: float, capacity: int):
        if api_name not in cls._instances:
            cls._instances[api_name] = TokenBucketRateLimiter(rate, capacity)
        return cls._instances[api_name]
```

---

### 2.5.2 Semaphore-Based Concurrency Control

A **semaphore** limits the number of concurrently executing tasks to $N$:

$$
\text{At any time: } |\{T_i : T_i \text{ is executing}\}| \leq N
$$

**Implementation:**

```python
import asyncio

class ConcurrencyController:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
    
    async def execute(self, fn, *args, **kwargs):
        async with self.semaphore:
            self.active_count += 1
            try:
                return await fn(*args, **kwargs)
            finally:
                self.active_count -= 1

# Usage
controller = ConcurrencyController(max_concurrent=5)

async def process_batch(queries, agent):
    tasks = [controller.execute(agent.process, q) for q in queries]
    return await asyncio.gather(*tasks)
```

**Weighted Semaphore:**

Different tasks consume different amounts of resources. A weighted semaphore tracks resource consumption:

```python
class WeightedSemaphore:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0
        self.condition = asyncio.Condition()
    
    async def acquire(self, weight: int):
        async with self.condition:
            while self.used + weight > self.capacity:
                await self.condition.wait()
            self.used += weight
    
    async def release(self, weight: int):
        async with self.condition:
            self.used -= weight
            self.condition.notify_all()

# Usage: limit total concurrent tokens to 100k
token_semaphore = WeightedSemaphore(capacity=100_000)

async def controlled_llm_call(llm, query, estimated_tokens):
    await token_semaphore.acquire(estimated_tokens)
    try:
        return await llm.generate(query)
    finally:
        await token_semaphore.release(estimated_tokens)
```

---

### 2.5.3 Deadlock Prevention in Multi-Agent Parallelism

**Deadlock** occurs when two or more tasks are each waiting for a resource held by another, forming a circular wait.

**Deadlock Conditions (Coffman Conditions):**

All four must hold simultaneously for deadlock to occur:

1. **Mutual exclusion**: At least one resource is non-shareable
2. **Hold and wait**: A task holding one resource waits for another
3. **No preemption**: Resources cannot be forcibly taken from a task
4. **Circular wait**: There exists a cycle of tasks, each waiting for a resource held by the next

**Deadlock Scenarios in Agentic Systems:**

1. **Shared tool access**: Agent A holds a database lock and waits for Agent B's API result; Agent B waits for the database lock held by Agent A.

2. **Circular data dependencies**: Agent A sends a query to Agent B and waits for the response; Agent B sends a query to Agent A and waits for the response.

3. **Resource exhaustion**: All semaphore slots are consumed by tasks that each spawn sub-tasks requiring semaphore slots.

**Prevention Strategies:**

**Strategy 1: Resource Ordering**

Impose a total order on resources; tasks must acquire resources in increasing order:

$$
\text{If } r_i < r_j \text{, then any task must acquire } r_i \text{ before } r_j
$$

This breaks the circular wait condition.

**Strategy 2: Timeout-Based Detection**

```python
async def safe_acquire(lock, timeout=10.0):
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logging.warning("Potential deadlock detected: lock acquisition timed out")
        return False
```

**Strategy 3: Avoid Hold-and-Wait**

Require tasks to acquire all needed resources atomically before executing:

```python
async def acquire_all(resources: list, timeout=10.0):
    """Atomically acquire all resources or none"""
    acquired = []
    try:
        for resource in sorted(resources, key=lambda r: r.id):
            success = await safe_acquire(resource, timeout)
            if not success:
                # Release all acquired resources (rollback)
                for r in reversed(acquired):
                    r.release()
                raise DeadlockRisk("Could not acquire all resources")
            acquired.append(resource)
        return acquired
    except Exception:
        for r in reversed(acquired):
            r.release()
        raise
```

**Strategy 4: Non-Blocking Design**

Design agents to never hold resources while waiting for external results. Use message passing instead of shared-state synchronization.

---

### 2.5.4 Idempotency Requirements

In parallel systems with retries and potential duplicate executions, **idempotency** ensures that executing an operation multiple times produces the same result as executing it once:

$$
f(f(x)) = f(x) \quad \text{(idempotent)}
$$

**Why Idempotency Matters:**

1. **Retries**: When an LLM call fails and is retried, the side effects should not double
2. **Duplicate requests**: Network issues may cause the same request to be sent twice
3. **Speculative execution**: Speculatively executed tasks may be re-executed after rollback

**Idempotency Strategies:**

| Strategy | Mechanism |
|---|---|
| **Idempotency keys** | Assign a unique ID to each request; reject duplicates |
| **Read-only operations** | Queries are naturally idempotent |
| **Deterministic writes** | Use `PUT` semantics (set to value) instead of `POST` (append) |
| **Deduplication layer** | Track processed request IDs and skip duplicates |

**Implementation:**

```python
class IdempotentExecutor:
    def __init__(self):
        self.completed = {}  # idempotency_key -> result
    
    async def execute(self, key: str, fn: callable, *args):
        if key in self.completed:
            return self.completed[key]  # Return cached result
        
        result = await fn(*args)
        self.completed[key] = result
        return result

# Usage
executor = IdempotentExecutor()

async def process_with_retry(query, agent, retries=3):
    key = hashlib.sha256(query.encode()).hexdigest()
    for attempt in range(retries):
        try:
            return await executor.execute(key, agent.process, query)
        except TransientError:
            continue
    raise MaxRetriesExceeded()
```

---

## 2.6 Cost and Latency Optimization

### 2.6.1 Token Budget Allocation Across Parallel Branches

Parallelization multiplies token consumption. A **token budget** constrains the total tokens consumed:

$$
\sum_{i=1}^{k} \text{tokens}(T_i) \leq B_{\text{total}}
$$

**Budget Allocation Strategies:**

**Strategy 1: Uniform Allocation**

$$
B_i = \frac{B_{\text{total}}}{k} \quad \forall i
$$

Simple but suboptimal — some tasks need more tokens than others.

**Strategy 2: Priority-Based Allocation**

Assign budgets proportional to task importance:

$$
B_i = \frac{w_i}{\sum_j w_j} \cdot B_{\text{total}}
$$

where $w_i$ is the priority weight of task $T_i$.

**Strategy 3: Adaptive Allocation**

Start with a minimum budget for each task. If a task needs more tokens, reallocate from tasks that finished under budget:

```python
class TokenBudgetManager:
    def __init__(self, total_budget: int, num_tasks: int):
        self.remaining = total_budget
        self.per_task_base = total_budget // num_tasks
        self.lock = asyncio.Lock()
    
    async def allocate(self, task_id: str, requested: int) -> int:
        async with self.lock:
            allowed = min(requested, self.remaining)
            self.remaining -= allowed
            return allowed
    
    async def return_unused(self, task_id: str, unused: int):
        async with self.lock:
            self.remaining += unused
```

**Enforcement via `max_tokens`:**

```python
async def budget_constrained_call(llm, query, max_tokens):
    return await llm.generate(
        query,
        max_tokens=max_tokens  # Hard limit on output tokens
    )
```

---

### 2.6.2 Early Termination and Short-Circuiting

**Early termination** stops parallel execution as soon as a sufficient result is obtained, saving cost on remaining branches.

**Patterns:**

**Pattern 1: First-Success Termination**

For tasks where any single correct answer suffices (e.g., finding a working solution):

$$
y = \text{first}_{i}(f_i(x) \text{ s.t. } \text{valid}(f_i(x)))
$$

```python
async def first_valid_result(agents, query, validator):
    """Return the first valid result; cancel remaining agents"""
    tasks = {
        asyncio.create_task(agent.process(query)): agent
        for agent in agents
    }
    
    while tasks:
        done, _ = await asyncio.wait(
            tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            result = task.result()
            if validator(result):
                # Cancel all remaining tasks
                for remaining in tasks:
                    if remaining != task:
                        remaining.cancel()
                return result
            del tasks[task]
    
    raise RuntimeError("No valid result from any agent")
```

**Pattern 2: Threshold-Based Termination**

For quality-scored tasks, terminate when a sufficiently good result is found:

$$
\text{Terminate if } \exists i: \text{Score}(f_i(x)) \geq \tau_{\text{quality}}
$$

**Pattern 3: Diminishing Returns Termination**

For voting/ensemble tasks, terminate when additional votes are unlikely to change the outcome:

$$
\text{Terminate if } P(\text{majority change} \mid \text{remaining votes}) < \epsilon
$$

If after 7 out of 10 votes, one answer has 6 votes, the remaining 3 votes cannot change the majority. Terminate early and save the cost of 3 LLM calls.

```python
def can_terminate_early(votes: dict, total: int, completed: int) -> bool:
    """Check if majority is already decided"""
    remaining = total - completed
    if not votes:
        return False
    leader_count = max(votes.values())
    runner_up = sorted(votes.values(), reverse=True)[1] if len(votes) > 1 else 0
    # Leader cannot be overtaken even if all remaining go to runner-up
    return leader_count > runner_up + remaining
```

---

### 2.6.3 Adaptive Parallelism: Dynamic Branch Count Selection

Not all queries benefit equally from parallelization. **Adaptive parallelism** dynamically adjusts the number of parallel branches based on query characteristics.

**Decision Function:**

$$
k^* = g(q, B_{\text{budget}}, L_{\text{target}})
$$

where $k^*$ is the optimal number of parallel branches, $B_{\text{budget}}$ is the remaining token budget, and $L_{\text{target}}$ is the target latency.

**Complexity-Based Branching:**

$$
k^* = \begin{cases}
1 & \text{if } \text{Complexity}(q) = \text{LOW} \\
3 & \text{if } \text{Complexity}(q) = \text{MEDIUM} \\
5 & \text{if } \text{Complexity}(q) = \text{HIGH} \\
\end{cases}
$$

**Budget-Constrained Branching:**

$$
k^* = \min\left(k_{\max}, \left\lfloor \frac{B_{\text{remaining}}}{B_{\text{per\_branch}}} \right\rfloor\right)
$$

**Confidence-Based Branching:**

Start with $k = 1$. If confidence is low, add more branches:

```python
async def adaptive_parallel(query, agent, confidence_threshold=0.85, max_k=5):
    results = []
    for k in range(1, max_k + 1):
        result = await agent.process(query, temperature=0.7)
        results.append(result)
        
        # Check if we have enough confidence
        answers = [extract_answer(r) for r in results]
        vote_counts = Counter(answers)
        top_count = vote_counts.most_common(1)[0][1]
        confidence = top_count / len(results)
        
        if confidence >= confidence_threshold:
            return vote_counts.most_common(1)[0][0], confidence
    
    # Max branches reached
    return vote_counts.most_common(1)[0][0], confidence
```

---

### 2.6.4 Cost Function

The total cost of a parallelized agentic workflow is:

$$
C_{\text{total}} = \sum_{i=1}^{k} c(T_i) + c_{\text{orchestration}}
$$

where:
- $c(T_i)$: cost of executing task $T_i$, measured in dollars:

$$
c(T_i) = \text{InputTokens}(T_i) \cdot p_{\text{input}} + \text{OutputTokens}(T_i) \cdot p_{\text{output}}
$$

where $p_{\text{input}}, p_{\text{output}}$ are per-token prices for the model used.

- $c_{\text{orchestration}}$: overhead cost of the orchestration layer (router inference, embedding computation, state management).

**Cost Amplification Factor:**

$$
\text{CAF} = \frac{C_{\text{parallel}}}{C_{\text{sequential}}} = \frac{\sum_{i=1}^{k} c(T_i) + c_{\text{orch}}}{c(T_{\text{single}})}
$$

For a 5-branch parallel system where each branch costs the same as the sequential version, $\text{CAF} = 5 + \frac{c_{\text{orch}}}{c(T_{\text{single}})}$.

**Cost-Latency Pareto Frontier:**

The optimal parallelization strategy lies on the **Pareto frontier** of cost vs. latency:

$$
\text{Pareto}(k) = \{(C(k), L(k)) : \nexists k' \text{ s.t. } C(k') \leq C(k) \land L(k') \leq L(k)\}
$$

Increasing $k$ reduces latency but increases cost. The optimal $k$ depends on the relative value of time vs. money:

$$
k^* = \arg\min_k \left[C(k) + \lambda \cdot L(k)\right]
$$

where $\lambda$ converts latency (seconds) to cost (dollars).

**Example:**

| $k$ (branches) | Latency | Cost | $\lambda = 0.01$ (cheap time) | $\lambda = 1.0$ (expensive time) |
|---|---|---|---|---|
| 1 | 10s | $0.01 | $0.11 | $10.01 |
| 3 | 4s | $0.03 | $0.07 | $4.03 |
| 5 | 3s | $0.05 | $0.08 | $3.05 |
| 10 | 2.5s | $0.10 | $0.125 | $2.60 |

For cheap time ($\lambda = 0.01$): $k^* = 3$ (minimize cost with some speedup)

For expensive time ($\lambda = 1.0$): $k^* = 10$ (minimize latency, cost is secondary)

---

## 2.7 Implementation Considerations

### 2.7.1 Async/Await Patterns for LLM Calls

The fundamental implementation pattern for agentic parallelization in Python is `asyncio`-based concurrency.

**Basic Pattern:**

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def parallel_llm_calls(prompts: list[str], model: str = "gpt-4o-mini"):
    """Execute multiple LLM calls concurrently"""
    
    async def single_call(prompt: str):
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    results = await asyncio.gather(*[single_call(p) for p in prompts])
    return results
```

**Structured Concurrency with Task Groups (Python 3.11+):**

```python
async def structured_parallel(prompts: list[str]):
    """Task group ensures cleanup on failure"""
    results = {}
    
    async with asyncio.TaskGroup() as tg:
        for i, prompt in enumerate(prompts):
            async def process(idx=i, p=prompt):
                results[idx] = await single_call(p)
            tg.create_task(process())
    
    return [results[i] for i in range(len(prompts))]
```

`TaskGroup` provides **structured concurrency**: if any task raises an exception, all other tasks in the group are cancelled, preventing resource leaks.

**Advanced Pattern: Streaming with Parallel Aggregation**

```python
async def parallel_stream_aggregate(prompts: list[str], aggregator):
    """Stream responses from parallel LLM calls and aggregate incrementally"""
    
    async def stream_call(prompt):
        chunks = []
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)
    
    results = await asyncio.gather(*[stream_call(p) for p in prompts])
    return await aggregator(results)
```

**Error Handling Pattern:**

```python
async def resilient_parallel(tasks: list[callable], 
                              min_required: int,
                              timeout: float = 30.0):
    """
    Execute tasks in parallel.
    Require at least min_required successes.
    Individual timeouts per task.
    """
    async def safe_execute(task):
        try:
            return await asyncio.wait_for(task(), timeout=timeout)
        except asyncio.TimeoutError:
            return ParallelError("timeout")
        except Exception as e:
            return ParallelError(str(e))
    
    results = await asyncio.gather(*[safe_execute(t) for t in tasks])
    
    successes = [r for r in results if not isinstance(r, ParallelError)]
    failures = [r for r in results if isinstance(r, ParallelError)]
    
    if len(successes) < min_required:
        raise InsufficientResults(
            f"Got {len(successes)}/{min_required} required results. "
            f"Failures: {failures}"
        )
    
    return successes
```

---

### 2.7.2 Thread Pool and Worker Management

For CPU-bound operations or blocking I/O that cannot be made async, use thread pools:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)

async def run_in_thread(fn, *args):
    """Run a blocking function in a thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fn, *args)

# Usage: embedding computation (CPU-bound with local model)
async def parallel_embeddings(texts: list[str], model):
    tasks = [run_in_thread(model.encode, text) for text in texts]
    return await asyncio.gather(*tasks)
```

**Worker Pool Pattern for Agent Management:**

```python
class AgentWorkerPool:
    def __init__(self, agent_factory, pool_size: int = 5):
        self.pool = asyncio.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self.pool.put_nowait(agent_factory())
    
    async def execute(self, query: str):
        agent = await self.pool.get()
        try:
            return await agent.process(query)
        finally:
            await self.pool.put(agent)
    
    async def batch_execute(self, queries: list[str]):
        return await asyncio.gather(*[self.execute(q) for q in queries])
```

---

### 2.7.3 Distributed Execution Engines

For large-scale agentic systems exceeding single-machine capacity, distributed execution frameworks are necessary.

**Architecture:**

```
                    ┌──────────────────┐
                    │  Orchestrator    │
                    │  (DAG Scheduler) │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌────▼────┐ ┌───────▼──────┐
       │  Worker 1   │ │Worker 2 │ │  Worker 3    │
       │  (Machine A)│ │(Mach B) │ │  (Machine C) │
       │  Agent Pool │ │Agent Pl.│ │  Agent Pool  │
       └─────────────┘ └─────────┘ └──────────────┘
```

**Technologies:**

| Technology | Use Case | Characteristics |
|---|---|---|
| **Celery** | Task queue with worker distribution | Mature, Redis/RabbitMQ backend |
| **Ray** | Distributed Python execution | Actor model, shared memory |
| **Temporal** | Durable workflow orchestration | Fault-tolerant, replayable |
| **Dask** | Parallel data processing | Dynamic DAG scheduling |
| **Apache Airflow** | DAG-based workflow scheduling | Batch-oriented, cron-like |

**Ray Example for Distributed Agents:**

```python
import ray

@ray.remote
class DistributedAgent:
    def __init__(self, model_name):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model_name
    
    def process(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content

# Create distributed agent pool
ray.init()
agents = [DistributedAgent.remote("gpt-4o-mini") for _ in range(10)]

# Execute in parallel across cluster
futures = [agent.process.remote(query) for agent, query in zip(agents, queries)]
results = ray.get(futures)
```

**Temporal for Durable Workflows:**

```python
from temporalio import workflow, activity

@activity.defn
async def call_agent(agent_name: str, query: str) -> str:
    """Activity: a single agent call (durable, retryable)"""
    agent = get_agent(agent_name)
    return await agent.process(query)

@workflow.defn
class ParallelAgentWorkflow:
    @workflow.run
    async def run(self, query: str) -> dict:
        # Fan-out: parallel agent execution (durable)
        results = await asyncio.gather(
            workflow.execute_activity(
                call_agent, args=["search_agent", query],
                start_to_close_timeout=timedelta(seconds=60)
            ),
            workflow.execute_activity(
                call_agent, args=["code_agent", query],
                start_to_close_timeout=timedelta(seconds=60)
            ),
            workflow.execute_activity(
                call_agent, args=["math_agent", query],
                start_to_close_timeout=timedelta(seconds=60)
            ),
        )
        
        # Fan-in: aggregate
        return await workflow.execute_activity(
            call_agent, 
            args=["synthesis_agent", json.dumps(results)],
            start_to_close_timeout=timedelta(seconds=60)
        )
```

Temporal provides **durability**: if the orchestrator crashes mid-execution, it resumes from the last completed activity upon restart.

---

### 2.7.4 Observability in Parallel Pipelines

Observability in parallel systems is critical for debugging, performance optimization, and production monitoring.

**Three Pillars of Observability:**

**1. Distributed Tracing:**

Each request generates a **trace** — a tree of spans representing each operation:

```
Trace: request-abc-123
├── Span: router (12ms)
├── Span: fan-out
│   ├── Span: search_agent (2.3s)  [parallel]
│   ├── Span: code_agent (4.1s)    [parallel]
│   └── Span: math_agent (1.8s)    [parallel]
├── Span: barrier-wait (0ms, search+math already done)
├── Span: aggregator (1.2s)
└── Span: response-formatting (15ms)
Total: 5.33s (critical path: code_agent → aggregator)
```

**Implementation with OpenTelemetry:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("agentic-pipeline")

async def traced_parallel_execution(query: str, agents: dict):
    with tracer.start_as_current_span("parallel-execution") as parent:
        parent.set_attribute("query.length", len(query))
        parent.set_attribute("num_agents", len(agents))
        
        async def traced_agent(name, agent):
            with tracer.start_as_current_span(f"agent-{name}") as span:
                span.set_attribute("agent.name", name)
                start = time.monotonic()
                result = await agent.process(query)
                duration = time.monotonic() - start
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("output.length", len(result))
                return result
        
        results = await asyncio.gather(*[
            traced_agent(name, agent) 
            for name, agent in agents.items()
        ])
        
        parent.set_attribute("total_results", len(results))
        return results
```

**2. Structured Logging:**

```python
import structlog

logger = structlog.get_logger()

async def logged_parallel_execution(query, agents):
    execution_id = str(uuid.uuid4())
    
    logger.info("parallel_execution_start",
                execution_id=execution_id,
                num_agents=len(agents),
                query_hash=hashlib.md5(query.encode()).hexdigest())
    
    results = {}
    for name, agent in agents.items():
        start = time.monotonic()
        try:
            result = await agent.process(query)
            duration = time.monotonic() - start
            results[name] = result
            logger.info("agent_complete",
                       execution_id=execution_id,
                       agent=name,
                       duration_ms=round(duration * 1000, 2),
                       output_tokens=count_tokens(result),
                       status="success")
        except Exception as e:
            logger.error("agent_failed",
                        execution_id=execution_id,
                        agent=name,
                        error=str(e),
                        status="failure")
    
    return results
```

**3. Metrics and Dashboards:**

| Metric | Type | Description |
|---|---|---|
| `parallel.branches.count` | Histogram | Number of parallel branches per execution |
| `parallel.latency.per_branch` | Histogram | Latency distribution across branches |
| `parallel.straggler.ratio` | Gauge | $\frac{\max_i L(T_i)}{\text{mean}_i L(T_i)}$ — how much the slowest branch exceeds average |
| `parallel.cost.total` | Counter | Total token cost across all branches |
| `parallel.failure.rate` | Counter | Fraction of branches that fail |
| `parallel.utilization` | Gauge | $\frac{\sum_i L(T_i)}{k \cdot \max_i L(T_i)}$ — how well parallelism is utilized |
| `parallel.early_termination.rate` | Counter | Fraction of executions that terminate early |

**Straggler Detection Alert:**

$$
\text{Alert if } \frac{\max_i L(T_i)}{\text{median}_i L(T_i)} > 3.0
$$

This indicates one branch is taking 3× longer than the median, suggesting a problem (overloaded endpoint, complex sub-task, degraded model).

**Gantt Chart Visualization:**

For debugging parallel execution, generate Gantt charts showing the timeline of each branch:

```
Time →  0s    1s    2s    3s    4s    5s
search  |█████████████|
code    |██████████████████████████████|  ← straggler
math    |████████|
viz     |███████████████████|
                                        aggregator |████████|
```

This immediately reveals that `code_agent` is the bottleneck (critical path).

---

**Summary of Chapter 2:**

Parallelization in agentic systems transforms sequential LLM pipelines into concurrent execution graphs, reducing latency by $2\text{-}10\times$ while potentially increasing throughput by orders of magnitude. The theoretical speedup is bounded by Amdahl's Law: $S(n) = \frac{1}{(1-p) + p/n}$, where $p$ is the parallelizable fraction. Key patterns include fan-out/fan-in (decompose → parallel execute → aggregate), voting/ensembling (parallel generation → majority vote), pipeline parallelism (stage-level overlap), and speculative execution (pre-compute likely branches). The execution structure is formalized as a DAG with critical path analysis determining minimum latency. Practical implementation relies on `asyncio` for I/O-bound concurrency, distributed frameworks (Ray, Temporal) for cross-machine scaling, and comprehensive observability (tracing, logging, metrics) for production reliability. Cost management requires token budget allocation, early termination, and adaptive parallelism that dynamically adjusts branch count based on query complexity and budget constraints.