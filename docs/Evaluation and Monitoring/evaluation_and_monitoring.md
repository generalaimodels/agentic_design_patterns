

# Chapter 18: Evaluation and Monitoring

---

## 18.1 Definition and Formal Framework

### 18.1.1 What is Evaluation in Agentic AI

Evaluation in agentic AI is the **systematic, multi-dimensional assessment** of an autonomous agent's behavior across the complete space of its operational capabilities—task completion, reasoning quality, action efficiency, safety compliance, robustness to adversarial and distributional perturbations, and long-horizon trajectory coherence. Unlike evaluation of static language models (where a single input-output pair suffices for judgment), agentic evaluation must contend with **stochastic multi-step trajectories**, **environment interaction dynamics**, **tool-use correctness**, and **emergent behaviors arising from the composition of individually correct steps**.

**Formal Definition.** Let an agentic system be defined as $\mathcal{A} = (M, \mathcal{T}, \mathcal{E}, \pi)$ where $M$ is the language model backbone, $\mathcal{T} = \{t_1, \dots, t_K\}$ is the tool set, $\mathcal{E}$ is the environment, and $\pi: \mathcal{S} \rightarrow \Delta(\mathcal{A})$ is the agent policy mapping states to action distributions. An evaluation framework $\mathcal{F}$ is a tuple:

$$
\mathcal{F} = (\mathcal{D}_{\text{eval}}, \mathcal{M}_{\text{metrics}}, \mathcal{J}_{\text{judges}}, \mathcal{P}_{\text{protocol}})
$$

where:
- $\mathcal{D}_{\text{eval}} = \{(q_i, \mathcal{E}_i, y_i^*)\}_{i=1}^{N}$ is the evaluation dataset of queries, environments, and reference solutions
- $\mathcal{M}_{\text{metrics}} = \{m_1, \dots, m_D\}$ is the set of evaluation metrics across $D$ dimensions
- $\mathcal{J}_{\text{judges}} = \{j_{\text{human}}, j_{\text{LLM}}, j_{\text{auto}}\}$ is the set of evaluation judges
- $\mathcal{P}_{\text{protocol}}$ specifies the evaluation procedure (sampling, repetitions, statistical tests)

**Why Agentic Evaluation is Fundamentally Harder than LLM Evaluation.**

| Dimension | Standard LLM Evaluation | Agentic Evaluation |
|---|---|---|
| **Input-Output Mapping** | Single turn: $x \rightarrow y$ | Multi-step trajectory: $q \rightarrow (s_0, a_0, s_1, a_1, \dots, s_T)$ |
| **Correctness Definition** | Output matches reference | Final outcome correct AND intermediate steps valid |
| **Stochasticity** | Temperature-based variation | Environment stochasticity × sampling stochasticity × tool output stochasticity |
| **Side Effects** | None (text generation) | Real-world consequences: file writes, API calls, database modifications |
| **Evaluation Cost** | Cheap (text comparison) | Expensive: environment setup, tool execution, rollback |
| **Compositionality** | Independent per-sample | Individual step correctness $\not\Rightarrow$ trajectory correctness |
| **Reproducibility** | High (fix random seed) | Low: external APIs change, environments drift, timing matters |
| **Partial Credit** | Often binary (exact match) | Gradient of partial completion necessary |

**The Evaluation Trilemma for Agents.**

Every evaluation system for agents must navigate a trilemma among three competing desiderata:

```
                    FIDELITY
                   (real-world
                    accuracy)
                      /\
                     /  \
                    /    \
                   /      \
                  /        \
                 /   IDEAL  \
                /    POINT   \
               /      ★      \
              /________________\
         COST                  REPRODUCIBILITY
     (computation,             (deterministic,
      human time)               controllable)
```

- **High fidelity + High reproducibility** → Extremely expensive (full environment simulation with deterministic replay)
- **High fidelity + Low cost** → Poor reproducibility (live API testing)
- **High reproducibility + Low cost** → Low fidelity (synthetic benchmarks with mocked tools)

---

### 18.1.2 Evaluation as Multi-Dimensional Assessment

Agent evaluation cannot be reduced to a single scalar metric. It requires simultaneous assessment across orthogonal quality dimensions:

$$
\mathcal{E}(\text{agent}) = \{e_{\text{quality}}, e_{\text{efficiency}}, e_{\text{safety}}, e_{\text{reliability}}\}
$$

**Formal Multi-Dimensional Evaluation Function.** Define the evaluation as a vector-valued function mapping an agent's trajectory $\tau$ on task $q$ to a point in multi-dimensional evaluation space:

$$
\mathcal{E}: (\mathcal{A}, q) \mapsto \mathbf{e} \in \mathbb{R}^D
$$

where each dimension $d$ captures a distinct quality aspect:

$$
\mathbf{e} = \begin{bmatrix} e_{\text{correctness}}(\tau) \\ e_{\text{reasoning}}(\tau) \\ e_{\text{efficiency}}(\tau) \\ e_{\text{safety}}(\tau) \\ e_{\text{reliability}}(\tau) \\ e_{\text{user\_satisfaction}}(\tau) \end{bmatrix}
$$

**Dimension Definitions:**

1. **Task Quality** $e_{\text{quality}}$: Did the agent produce the correct final answer/outcome?

$$
e_{\text{quality}}(\tau, y^*) = \text{Sim}(\text{outcome}(\tau), y^*)
$$

2. **Efficiency** $e_{\text{efficiency}}$: How many resources (tokens, time, API calls, cost) were consumed?

$$
e_{\text{efficiency}}(\tau) = f\left(\text{tokens}(\tau), \text{latency}(\tau), \text{cost}(\tau), |\text{steps}(\tau)|\right)
$$

3. **Safety** $e_{\text{safety}}$: Did the agent avoid harmful actions, policy violations, and information leakage?

$$
e_{\text{safety}}(\tau) = 1 - \frac{\sum_{t=1}^{T} \mathbb{1}[\text{violation}(s_t, a_t)]}{T}
$$

4. **Reliability** $e_{\text{reliability}}$: How consistent is the agent's performance across repeated runs and distribution variations?

$$
e_{\text{reliability}}(\mathcal{A}, q) = 1 - \frac{\text{Var}[e_{\text{quality}}(\tau_1), \dots, e_{\text{quality}}(\tau_R)]}{e_{\text{quality\_max}}^2}
$$

where $R$ is the number of repeated evaluations.

**Aggregation for Decision-Making.** When a single ranking is needed (e.g., comparing Agent A vs. Agent B), a weighted aggregation is applied:

$$
\text{Score}(\mathcal{A}) = \sum_{d=1}^{D} w_d \cdot e_d(\mathcal{A}) \quad \text{subject to} \quad \sum_d w_d = 1, \quad w_d \geq 0
$$

The weights $w_d$ are domain-dependent:

| Domain | $w_{\text{quality}}$ | $w_{\text{efficiency}}$ | $w_{\text{safety}}$ | $w_{\text{reliability}}$ |
|---|---|---|---|---|
| Medical diagnosis | 0.30 | 0.05 | 0.45 | 0.20 |
| Code generation | 0.50 | 0.20 | 0.10 | 0.20 |
| Customer service | 0.30 | 0.15 | 0.20 | 0.35 |
| Financial trading | 0.25 | 0.15 | 0.35 | 0.25 |

**Pareto Dominance for Multi-Objective Comparison.** Agent $\mathcal{A}_1$ **Pareto-dominates** $\mathcal{A}_2$ if:

$$
\mathcal{A}_1 \succ_{\text{Pareto}} \mathcal{A}_2 \iff \forall d: e_d(\mathcal{A}_1) \geq e_d(\mathcal{A}_2) \;\wedge\; \exists d: e_d(\mathcal{A}_1) > e_d(\mathcal{A}_2)
$$

When neither agent Pareto-dominates, the comparison requires explicit weight specification reflecting deployment priorities.

---

### 18.1.3 Offline vs. Online Evaluation

**Offline Evaluation** assesses agent performance on pre-collected, static datasets with deterministic environments, enabling reproducible comparison but potentially missing real-world distribution characteristics.

**Online Evaluation** measures agent performance on live traffic with real users and environments, capturing true operational quality but introducing variability, cost, and safety considerations.

**Formal Distinction:**

$$
\text{Offline:} \quad \hat{e}_d = \frac{1}{|\mathcal{D}_{\text{eval}}|} \sum_{(q, y^*) \in \mathcal{D}_{\text{eval}}} m_d(\mathcal{A}(q), y^*)
$$

$$
\text{Online:} \quad \hat{e}_d = \frac{1}{|\mathcal{Q}_{\text{live}}|} \sum_{q \in \mathcal{Q}_{\text{live}}} m_d(\mathcal{A}(q), \text{signal}(q))
$$

where $\text{signal}(q)$ is an online feedback signal (user click, task completion, explicit rating).

**Comparison Framework:**

| Property | Offline Evaluation | Online Evaluation |
|---|---|---|
| Reproducibility | High (deterministic) | Low (stochastic environment, changing APIs) |
| Cost | Moderate (compute only) | High (real resources, user exposure) |
| Distribution fidelity | Low-moderate (static benchmark) | High (real user distribution) |
| Safety risk | None (sandboxed) | Real (live consequences) |
| Feedback signal | Reference answers | User behavior, outcomes |
| Iteration speed | Fast (automated) | Slow (requires traffic) |
| Coverage | Controlled, systematic | Organic, potentially biased |
| Temporal validity | Degrades (benchmark saturation) | Current (always reflects now) |

**Bridging Strategy: Shadow Evaluation.**

Run the new agent in parallel with the production agent on live traffic, but only serve the production agent's responses. Compare both agents' outputs against the online feedback signal:

$$
\text{Shadow\_Score}(\mathcal{A}_{\text{new}}) = \frac{1}{N} \sum_{i=1}^{N} m(\mathcal{A}_{\text{new}}(q_i), \text{signal}(q_i))
$$

This provides online distribution fidelity without the safety risk of serving untested outputs.

---

### 18.1.4 Evaluation Challenges Unique to Agentic Systems

**Challenge 1: Non-Deterministic Trajectories.**

Given identical inputs, agents may follow different action sequences due to:
- Sampling temperature $T > 0$ in the LLM
- Non-deterministic tool outputs (web search results change over time)
- Environment state changes (database content, API availability)

**Implication for Evaluation:** A single evaluation run is insufficient. Statistical confidence requires multiple runs:

$$
\hat{e} = \frac{1}{R} \sum_{r=1}^{R} e(\tau_r) \quad \text{with confidence interval} \quad \hat{e} \pm z_{\alpha/2} \frac{s}{\sqrt{R}}
$$

where $R$ is the number of repetitions, $s$ is the sample standard deviation, and $z_{\alpha/2}$ is the critical value for confidence level $1-\alpha$.

**Minimum repetitions for desired confidence:**

$$
R \geq \left(\frac{z_{\alpha/2} \cdot s}{\epsilon}\right)^2
$$

where $\epsilon$ is the desired margin of error.

**Challenge 2: Trajectory-Level vs. Outcome-Level Evaluation.**

An agent may reach the correct answer through an incorrect reasoning path (lucky convergence), or fail to reach the correct answer despite valid intermediate steps (unlucky tool failure).

$$
\text{Correct outcome} \not\Leftrightarrow \text{Correct trajectory}
$$

Comprehensive evaluation requires assessing both:

$$
e_{\text{complete}}(\tau) = \alpha \cdot e_{\text{outcome}}(\tau) + (1-\alpha) \cdot e_{\text{trajectory}}(\tau)
$$

**Challenge 3: Open-Ended Action Spaces.**

Unlike chess (finite action space) or multiple-choice QA (enumerated options), agent actions are generated as free-form text specifying tool calls, parameters, and reasoning. The space of valid evaluations is combinatorially large.

**Challenge 4: Environment Coupling.**

Agent evaluation results depend on the environment state, which may change between evaluations:

$$
e(\mathcal{A}, q, t_1) \neq e(\mathcal{A}, q, t_2) \quad \text{if } \mathcal{E}(t_1) \neq \mathcal{E}(t_2)
$$

This necessitates either environment snapshots (for reproducibility) or environment-aware normalization (for fairness).

**Challenge 5: Cascading Error Propagation.**

In multi-step trajectories, an error at step $t$ can propagate and amplify through subsequent steps:

$$
\text{Error}(t+k) = f(\text{Error}(t), k) \quad \text{where typically } f \text{ is superlinear in } k
$$

This means that per-step accuracy $p$ yields trajectory accuracy $p^T$ for $T$ steps, which decays exponentially. An agent with 95% per-step accuracy over 20 steps achieves only $0.95^{20} \approx 0.36$ trajectory accuracy.

**Challenge 6: Cost of Evaluation.**

Each evaluation run consumes real resources:

$$
\text{Cost}_{\text{eval}} = N_{\text{tasks}} \times R_{\text{repetitions}} \times \bar{C}_{\text{per\_run}}
$$

where $\bar{C}_{\text{per\_run}}$ includes LLM API costs, tool execution costs, and environment setup costs. For a benchmark of 500 tasks with 5 repetitions and $\$0.50$ per run average, total cost is $\$1,250$ per evaluation cycle.

---

## 18.2 Evaluation Dimensions

### 18.2.1 Task Completion and Correctness

Task completion is the **primary evaluation dimension**: did the agent achieve the user's objective?

#### Binary Success Rate

The simplest and most interpretable metric — a binary indicator of whether the task was completed successfully:

$$
\text{Success Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{outcome}(\tau_i) = y_i^*]
$$

where $\mathbb{1}[\cdot]$ is the indicator function and $y_i^*$ is the ground-truth expected outcome.

**Statistical Confidence.** The success rate follows a Binomial distribution. The Clopper-Pearson exact confidence interval:

$$
\text{CI}_{1-\alpha}(\hat{p}) = \left[B\left(\frac{\alpha}{2}; k, N-k+1\right),\; B\left(1-\frac{\alpha}{2}; k+1, N-k\right)\right]
$$

where $B(\cdot; a, b)$ is the Beta distribution quantile function, $k = \sum_i \mathbb{1}[\text{success}_i]$, and $N$ is total evaluations.

**For comparing two agents**, use McNemar's test on paired evaluations:

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

where $b$ = tasks where Agent A succeeds and Agent B fails, $c$ = tasks where Agent B succeeds and Agent A fails.

#### Partial Credit Scoring

Binary success/failure is too coarse for complex tasks. Partial credit captures how much progress the agent made toward the goal:

$$
e_{\text{partial}}(\tau, y^*) = \frac{\sum_{j=1}^{J} w_j \cdot \mathbb{1}[\text{subtask}_j(\tau) \text{ completed}]}{\sum_{j=1}^{J} w_j}
$$

where $J$ is the number of subtasks/milestones and $w_j$ is the weight of subtask $j$.

**Example — Software Engineering Task:**

| Milestone | Weight | Check |
|---|---|---|
| Correct file identified | 0.10 | Agent edits the right file(s) |
| Syntax valid | 0.15 | Modified code parses without errors |
| Tests pass (existing) | 0.25 | No regression in existing test suite |
| Target test passes | 0.30 | The specific failing test now passes |
| Code quality | 0.10 | Lint clean, follows project style |
| Minimal diff | 0.10 | Change is minimal and focused |

**Rubric-Based Scoring.** For open-ended tasks, define a scoring rubric with criteria and levels:

$$
e_{\text{rubric}}(\tau) = \sum_{c=1}^{C} w_c \cdot \text{level}_c(\tau) \quad \text{where } \text{level}_c \in \{0, 1, 2, 3\}
$$

Rubric example for a research assistant agent:

```
Criterion: Source Quality (weight: 0.25)
  Level 0: No sources cited
  Level 1: Sources cited but not relevant
  Level 2: Relevant sources, some missing key papers
  Level 3: Comprehensive, relevant, authoritative sources

Criterion: Synthesis Quality (weight: 0.35)
  Level 0: No synthesis, raw information dump
  Level 1: Basic summary of sources
  Level 2: Coherent synthesis with connections
  Level 3: Novel insights from cross-source analysis

Criterion: Accuracy (weight: 0.25)
  Level 0: Multiple factual errors
  Level 1: Minor inaccuracies
  Level 2: Factually correct
  Level 3: Factually correct with appropriate uncertainty

Criterion: Actionability (weight: 0.15)
  Level 0: Not actionable
  Level 1: Generic recommendations
  Level 2: Specific, contextualized recommendations
  Level 3: Prioritized, implementable action plan
```

#### Ground Truth Comparison

For tasks with deterministic expected outputs, direct comparison against ground truth:

**Exact Match:**

$$
\text{EM}(y, y^*) = \mathbb{1}[\text{normalize}(y) = \text{normalize}(y^*)]
$$

where $\text{normalize}(\cdot)$ applies lowercasing, whitespace normalization, article removal, and other task-specific canonicalization.

**Functional Equivalence (for code):**

$$
\text{FuncEq}(y, y^*) = \mathbb{1}\left[\forall\, x \in \mathcal{X}_{\text{test}}: \text{exec}(y, x) = \text{exec}(y^*, x)\right]
$$

i.e., two code solutions are equivalent if they produce identical outputs for all test inputs.

**Semantic Equivalence (for natural language):**

$$
\text{SemEq}(y, y^*) = \text{NLI}(y, y^*) = \texttt{ENTAILMENT} \wedge \text{NLI}(y^*, y) = \texttt{ENTAILMENT}
$$

Bidirectional entailment implies semantic equivalence.

---

### 18.2.2 Reasoning Quality

#### Reasoning Trace Validity

Beyond outcome correctness, evaluate the quality of the agent's reasoning chain $(r_1, r_2, \dots, r_T)$:

$$
e_{\text{reasoning}}(\tau) = \frac{1}{T} \sum_{t=1}^{T} v(r_t, r_{<t}, \mathcal{C})
$$

where $v(r_t, r_{<t}, \mathcal{C})$ assesses whether reasoning step $r_t$ is:
1. **Logically valid** given prior steps $r_{<t}$
2. **Grounded** in the available context $\mathcal{C}$
3. **Relevant** to the task objective
4. **Non-redundant** (not repeating already-established conclusions)

**Formal Validity Criteria:**

$$
v(r_t) = \begin{cases}
1 & \text{if } r_t \text{ logically follows from } r_{<t} \cup \mathcal{C} \\
0.5 & \text{if } r_t \text{ is plausible but not rigorously derived} \\
0 & \text{if } r_t \text{ is logically invalid or unsupported}
\end{cases}
$$

#### Step-Level Correctness

For tasks decomposable into discrete steps (mathematical proofs, multi-hop QA, code debugging), evaluate each step independently:

$$
e_{\text{step}}(\tau) = \prod_{t=1}^{T} \mathbb{1}[\text{step}_t \text{ is correct}]
$$

The product formulation captures the fact that a single incorrect step invalidates subsequent dependent steps.

**Step-Level vs. Outcome-Level Diagnostic Matrix:**

| Step Correctness | Outcome Correctness | Interpretation | Action |
|---|---|---|---|
| ✓ All steps correct | ✓ Correct outcome | Ideal | — |
| ✓ All steps correct | ✗ Incorrect outcome | Tool/environment failure | Fix tool reliability |
| ✗ Some steps wrong | ✓ Correct outcome | Lucky convergence | Improve reasoning robustness |
| ✗ Some steps wrong | ✗ Incorrect outcome | Reasoning failure | Improve prompt/model/architecture |

**Process Reward Model (PRM) for Step Evaluation.** Train a reward model $R_{\text{PRM}}: (q, r_{1:t}) \rightarrow [0,1]$ that scores the correctness of each reasoning step:

$$
R_{\text{PRM}}(q, r_{1:t}) = P(\text{step } r_t \text{ is correct} \mid q, r_{1:t-1})
$$

The trajectory-level score under PRM:

$$
e_{\text{PRM}}(\tau) = \min_{t=1}^{T} R_{\text{PRM}}(q, r_{1:t})
$$

Using $\min$ rather than $\text{mean}$ reflects that the weakest step determines the chain's validity.

---

### 18.2.3 Efficiency Metrics

Efficiency metrics quantify the **resource consumption** of the agent relative to task difficulty.

#### Token Usage per Task

$$
\text{Tokens}(\tau) = \sum_{t=1}^{T} \left(\text{input\_tokens}_t + \text{output\_tokens}_t\right)
$$

Decomposed by component:

$$
\text{Tokens}(\tau) = \underbrace{\text{tokens}_{\text{system\_prompt}}}_{\text{fixed overhead}} + \underbrace{\text{tokens}_{\text{context}}}_{\text{RAG/memory}} + \underbrace{\text{tokens}_{\text{reasoning}}}_{\text{CoT}} + \underbrace{\text{tokens}_{\text{tool\_calls}}}_{\text{actions}} + \underbrace{\text{tokens}_{\text{tool\_outputs}}}_{\text{observations}}
$$

**Efficiency Ratio:** Normalize by task complexity:

$$
\text{Efficiency\_Ratio} = \frac{\text{Task\_Complexity}(q)}{\text{Tokens}(\tau)}
$$

where $\text{Task\_Complexity}(q)$ can be estimated by the minimum number of reasoning steps or tool calls required by an oracle agent.

#### Latency Distribution

Latency is characterized by its distribution, not just its mean:

$$
\text{Latency}(\tau) = t_{\text{end}} - t_{\text{start}}
$$

Report percentile metrics:

$$
P_k = \inf\{l : P(\text{Latency} \leq l) \geq k/100\}
$$

| Metric | Definition | Typical Target |
|---|---|---|
| $P_{50}$ (median) | Half of requests complete within this time | $< 5$s for interactive tasks |
| $P_{95}$ | 95th percentile latency | $< 15$s |
| $P_{99}$ | 99th percentile (tail latency) | $< 30$s |
| $P_{99.9}$ | Long-tail outliers | $< 60$s (for graceful timeout) |

**Latency Decomposition (per agent step):**

$$
\text{Latency}_{\text{total}} = \sum_{t=1}^{T} \left(\text{latency}_{\text{LLM}}^{(t)} + \text{latency}_{\text{tool}}^{(t)} + \text{latency}_{\text{guardrail}}^{(t)} + \text{latency}_{\text{overhead}}^{(t)}\right)
$$

#### Number of LLM Calls

$$
N_{\text{LLM}}(\tau) = |\{t : a_t \text{ involves an LLM inference call}\}|
$$

This is a proxy for both cost and latency. Efficient agents minimize LLM calls while maintaining task quality.

**LLM Call Efficiency:**

$$
\text{Call\_Efficiency} = \frac{\text{Task\_Completion\_Score}}{N_{\text{LLM}}(\tau)}
$$

#### Cost per Task

$$
\text{Cost}(\tau) = \sum_{t=1}^{T} \left(c_{\text{input}} \cdot \text{input\_tokens}_t + c_{\text{output}} \cdot \text{output\_tokens}_t + c_{\text{tool}}^{(t)}\right)
$$

where $c_{\text{input}}$ and $c_{\text{output}}$ are per-token costs for the LLM, and $c_{\text{tool}}^{(t)}$ is the cost of the tool call at step $t$.

**Cost-Quality Pareto Analysis:** Plot task quality vs. cost across different agent configurations to identify Pareto-optimal designs:

$$
\text{Pareto}(\mathcal{A}_1, \dots, \mathcal{A}_n) = \{\mathcal{A}_i : \nexists \mathcal{A}_j \text{ with } e_{\text{quality}}(\mathcal{A}_j) \geq e_{\text{quality}}(\mathcal{A}_i) \wedge \text{Cost}(\mathcal{A}_j) \leq \text{Cost}(\mathcal{A}_i)\}
$$

---

### 18.2.4 Safety and Compliance

#### Guardrail Violation Rate

$$
\text{GVR} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathbb{1}[\text{guardrail\_triggered}(a_t^{(i)})]}{N}
$$

Per-guardrail breakdown:

$$
\text{GVR}_k = \frac{\sum_{i=1}^{N} \mathbb{1}[\text{guardrail } g_k \text{ triggered on task } i]}{N}
$$

#### Harmful Output Frequency

$$
\text{HOF} = \frac{\sum_{i=1}^{N} \mathbb{1}[y_i \text{ is harmful AND escaped all guardrails}]}{N}
$$

This is the **residual harm rate** after all safety layers — the most critical safety metric. Even a tiny HOF ($> 10^{-4}$) may be unacceptable in high-stakes domains.

**Safety Scoreboard:**

| Category | Metric | Acceptable Threshold | Measurement |
|---|---|---|---|
| Toxicity | $P(\text{toxic output})$ | $< 0.001$ | Classifier on all outputs |
| PII leakage | $P(\text{PII in output})$ | $< 0.0001$ | PII scanner on all outputs |
| Instruction following | $P(\text{jailbreak success})$ | $< 0.01$ | Red-team evaluation |
| Action safety | $P(\text{unauthorized action})$ | $< 0.0001$ | Action audit log |
| Hallucination | $P(\text{unfaithful claim})$ | $< 0.05$ | Factuality checker |

---

### 18.2.5 User Satisfaction

#### Explicit Feedback Scores

Users provide direct ratings (thumbs up/down, 1-5 stars, or free-text feedback):

$$
\text{CSAT} = \frac{\sum_{i=1}^{N} \text{rating}_i}{N} \quad \text{or} \quad \text{CSAT} = \frac{|\{i : \text{rating}_i = \text{positive}\}|}{N}
$$

**Net Promoter Score (NPS) adaptation for agents:**

$$
\text{NPS} = \%\text{Promoters} - \%\text{Detractors}
$$

where promoters rate 9-10, detractors rate 0-6 on a 0-10 scale.

#### Task Abandonment Rate

$$
\text{TAR} = \frac{|\{i : \text{user abandoned task } i \text{ before completion}\}|}{N}
$$

Abandonment signals indicate either:
- Agent taking too long (latency-driven abandonment)
- Agent producing low-quality intermediate results (quality-driven abandonment)
- Agent requesting unnecessary information (friction-driven abandonment)

**Time-to-Abandon Distribution:** Analyze when users abandon to diagnose root causes:

$$
P(\text{abandon at step } t) = \frac{|\{i : \text{last\_interaction}(i) = t\}|}{|\{i : \text{reach\_step}(i) \geq t\}|}
$$

A spike in abandonment at a specific step suggests that step is problematic.

---

## 18.3 Evaluation Methodologies

### 18.3.1 LLM-as-Judge

LLM-as-Judge uses a powerful language model to evaluate the quality of another model's outputs, providing scalable evaluation with near-human-level judgment quality for many tasks.

#### Single-LLM Evaluation

A single judge model $M_{\text{judge}}$ scores an agent's output:

$$
\text{score} = M_{\text{judge}}(\text{prompt}_{\text{eval}}(q, y, y^*, \text{rubric}))
$$

**Evaluation Prompt Design:**

```
SYSTEM: You are an expert evaluator assessing AI agent responses.

TASK: Evaluate the following agent response against the reference 
answer and scoring rubric.

QUESTION: {question}
REFERENCE ANSWER: {reference}
AGENT RESPONSE: {agent_output}
AGENT TRAJECTORY: {action_sequence}

RUBRIC:
- Correctness (1-5): Is the final answer factually correct?
- Completeness (1-5): Are all aspects of the question addressed?
- Reasoning Quality (1-5): Is the reasoning chain valid and clear?
- Efficiency (1-5): Was the task completed with minimal unnecessary steps?

For each criterion, provide:
1. Score (integer 1-5)
2. Justification (2-3 sentences)

OUTPUT FORMAT:
{
  "correctness": {"score": <int>, "justification": "<str>"},
  "completeness": {"score": <int>, "justification": "<str>"},
  "reasoning_quality": {"score": <int>, "justification": "<str>"},
  "efficiency": {"score": <int>, "justification": "<str>"},
  "overall": {"score": <float>, "summary": "<str>"}
}
```

#### Multi-LLM Panel Evaluation

Use multiple judge models to reduce individual model bias and increase reliability:

$$
\text{score}_{\text{panel}} = \text{Aggregate}\left(\{M_{\text{judge}}^{(j)}(\text{eval\_prompt})\}_{j=1}^{J}\right)
$$

**Aggregation Methods:**

1. **Mean aggregation:**
$$
\text{score}_{\text{panel}} = \frac{1}{J} \sum_{j=1}^{J} \text{score}^{(j)}
$$

2. **Median aggregation** (robust to outlier judges):
$$
\text{score}_{\text{panel}} = \text{Median}\left(\text{score}^{(1)}, \dots, \text{score}^{(J)}\right)
$$

3. **Weighted aggregation** (weight by judge quality on calibration set):
$$
\text{score}_{\text{panel}} = \sum_{j=1}^{J} w_j \cdot \text{score}^{(j)}, \quad w_j = \frac{\text{agreement}(M_j, \text{human})}{\sum_{j'} \text{agreement}(M_{j'}, \text{human})}
$$

**Panel Design:**

```python
JUDGE_PANEL = [
    {"model": "gpt-4o", "weight": 0.35, "provider": "openai"},
    {"model": "claude-sonnet-4-20250514", "weight": 0.35, "provider": "anthropic"},
    {"model": "gemini-2.5-pro", "weight": 0.30, "provider": "google"},
]

async def panel_evaluate(question, response, reference):
    scores = await asyncio.gather(*[
        evaluate_with_model(judge, question, response, reference)
        for judge in JUDGE_PANEL
    ])
    
    weighted_score = sum(
        judge["weight"] * score 
        for judge, score in zip(JUDGE_PANEL, scores)
    )
    
    # Measure inter-judge agreement
    agreement = compute_krippendorff_alpha(scores)
    
    return {
        "final_score": weighted_score,
        "individual_scores": scores,
        "inter_judge_agreement": agreement,
        "confidence": "high" if agreement > 0.7 else "low"
    }
```

#### Pairwise Comparison

Instead of absolute scoring, compare two agent outputs head-to-head:

$$
P(y_A \succ y_B \mid q) = M_{\text{judge}}\left(\text{"Which response is better: A or B?"}, q, y_A, y_B\right)
$$

**Advantages over absolute scoring:**
- Humans (and LLMs) are more reliable at comparative judgments than absolute ratings
- Eliminates calibration differences across evaluators
- Natural fit for Elo/Bradley-Terry ranking systems

**Bradley-Terry Model for Agent Ranking:**

Given pairwise comparison outcomes, the probability that agent $i$ is preferred over agent $j$:

$$
P(i \succ j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j)
$$

The skill parameters $\beta_i$ are estimated via maximum likelihood:

$$
\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_{(i,j) \in \text{comparisons}} \left[y_{ij} \log \sigma(\beta_i - \beta_j) + (1 - y_{ij}) \log \sigma(\beta_j - \beta_i)\right]
$$

where $y_{ij} = 1$ if agent $i$ is preferred over agent $j$. This produces an **Elo-like ranking** of agents.

#### Bias Mitigation in LLM Judges

LLM judges exhibit systematic biases that must be identified and mitigated:

| Bias Type | Description | Mitigation |
|---|---|---|
| **Position bias** | Preference for the first (or last) response in pairwise comparison | Randomize order; average over both orderings |
| **Verbosity bias** | Preference for longer, more detailed responses regardless of quality | Include length-normalization instruction; penalize verbosity explicitly |
| **Self-preference bias** | Judge model prefers outputs from its own model family | Use cross-family judges; exclude same-family evaluation |
| **Style bias** | Preference for particular writing styles or formatting | Normalize formatting before evaluation; use content-focused rubrics |
| **Sycophancy** | Agreement with the position stated in the prompt | Test with known-incorrect reference answers |

**Position Bias Correction:**

$$
P_{\text{corrected}}(A \succ B) = \frac{1}{2}\left[P(A \succ B \mid \text{order: A first}) + P(A \succ B \mid \text{order: B first})\right]
$$

If the two orderings disagree, flag the comparison as unreliable:

$$
\text{consistency} = \mathbb{1}\left[P(A \succ B \mid \text{A first}) > 0.5 \iff P(A \succ B \mid \text{B first}) > 0.5\right]
$$

---

### 18.3.2 Human Evaluation

Human evaluation remains the **gold standard** for subjective quality dimensions (helpfulness, naturalness, safety) and for calibrating automated metrics.

#### Evaluation Protocol Design

A rigorous human evaluation protocol specifies:

```
1. EVALUATOR SELECTION
   - Domain expertise requirements
   - Training and calibration procedure
   - Minimum qualifications (education, experience)

2. TASK PRESENTATION
   - Information shown: query, agent trajectory, final output, reference
   - Information hidden: agent identity (blind evaluation)
   - Order randomization (for pairwise comparisons)

3. RATING DIMENSIONS
   - Per-dimension Likert scale (1-5) with anchored descriptions
   - Mandatory justification for extreme scores (1 or 5)
   - Optional free-text comments

4. QUALITY CONTROL
   - Calibration examples with expert consensus scores
   - Gold standard items (known scores) interspersed (~10%)
   - Attention check items
   - Evaluator agreement monitoring (real-time)

5. STATISTICAL ANALYSIS
   - Inter-annotator agreement computed continuously
   - Minimum annotations per item (typically 3-5)
   - Outlier evaluator detection and exclusion
   - Confidence intervals for reported scores
```

#### Inter-Annotator Agreement

**Cohen's Kappa** measures agreement between two annotators beyond chance:

$$
\kappa = \frac{P_o - P_e}{1 - P_e}
$$

where:
- $P_o$ = observed agreement proportion:
$$
P_o = \frac{\sum_{k=1}^{K} n_{kk}}{N}
$$

- $P_e$ = expected agreement by chance:
$$
P_e = \sum_{k=1}^{K} \frac{n_{k\cdot} \cdot n_{\cdot k}}{N^2}
$$

$n_{kk}$ is the number of items where both annotators assigned category $k$, $n_{k\cdot}$ and $n_{\cdot k}$ are marginal totals.

**Interpretation Scale:**

| $\kappa$ Range | Agreement Level | Evaluation Reliability |
|---|---|---|
| $< 0.20$ | Slight | Unreliable — redesign task/rubric |
| $0.20 - 0.40$ | Fair | Acceptable for rough screening |
| $0.40 - 0.60$ | Moderate | Standard for subjective tasks |
| $0.60 - 0.80$ | Substantial | Good — suitable for reliable evaluation |
| $> 0.80$ | Almost perfect | Excellent — near-deterministic task |

**For more than two annotators**, use **Fleiss' Kappa** or **Krippendorff's Alpha**:

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

where $D_o$ is observed disagreement and $D_e$ is expected disagreement. Krippendorff's $\alpha$ generalizes across number of raters, categories, and missing data.

#### Scale and Cost Considerations

| Evaluation Type | Cost per Item | Throughput | Quality | Best For |
|---|---|---|---|---|
| Expert evaluation | $\$5 - \$50$ | 5-20/hour | Highest | Safety, complex reasoning |
| Crowdsource (MTurk) | $\$0.10 - \$2.00$ | 50-200/hour | Variable | Large-scale preference |
| Internal team | $\$10 - \$30$ | 10-30/hour | High | Iterative development |
| LLM-as-Judge | $\$0.01 - \$0.10$ | 1000+/hour | Moderate-High | Continuous evaluation |

**Hybrid Strategy:** Use LLM-as-Judge for high-volume continuous evaluation, calibrated against periodic human evaluation:

$$
\text{score}_{\text{calibrated}} = f_{\text{calibrate}}(\text{score}_{\text{LLM}}) \quad \text{where } f_{\text{calibrate}} \text{ is fitted on human-LLM pairs}
$$

---

### 18.3.3 Automated Metric-Based Evaluation

#### Code Execution for Programming Tasks

The most reliable evaluation for code generation — execute the code and check against test cases:

$$
\text{pass@}k = \mathbb{E}_{\text{tasks}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

where $n$ is the total number of code samples generated, $c$ is the number that pass all test cases, and $k$ is the number of samples allowed. This is the unbiased estimator of the probability that at least one of $k$ samples passes.

**Implementation:**

```python
import numpy as np
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.
    
    Args:
        n: total number of samples generated
        c: number of correct samples (passing all tests)
        k: number of samples allowed
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def evaluate_code_agent(agent, tasks, n_samples=20, k_values=[1, 5, 10]):
    results = {}
    for task in tasks:
        completions = [agent.generate(task) for _ in range(n_samples)]
        n_correct = sum(1 for c in completions if run_tests(c, task.tests))
        
        for k in k_values:
            results.setdefault(f"pass@{k}", []).append(
                pass_at_k(n_samples, n_correct, k)
            )
    
    return {metric: np.mean(scores) for metric, scores in results.items()}
```

#### Standard NLP Metrics

**Exact Match (EM):**
$$
\text{EM}(y, y^*) = \mathbb{1}[\text{normalize}(y) = \text{normalize}(y^*)]
$$

**Token-Level F1:**
$$
\text{Precision} = \frac{|y_{\text{tokens}} \cap y^*_{\text{tokens}}|}{|y_{\text{tokens}}|}, \quad \text{Recall} = \frac{|y_{\text{tokens}} \cap y^*_{\text{tokens}}|}{|y^*_{\text{tokens}}|}
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**ROUGE-L** (longest common subsequence based):
$$
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 \cdot P_{\text{LCS}}}
$$

where $P_{\text{LCS}} = \text{LCS}(y, y^*) / |y|$, $R_{\text{LCS}} = \text{LCS}(y, y^*) / |y^*|$.

**BERTScore** (embedding-based semantic similarity):
$$
\text{BERTScore}(y, y^*) = \frac{1}{|y|} \sum_{y_i \in y} \max_{y_j^* \in y^*} \cos(\mathbf{h}_{y_i}, \mathbf{h}_{y_j^*})
$$

where $\mathbf{h}_{y_i}$ are contextual BERT embeddings.

#### Task-Specific Metrics

| Task Type | Primary Metric | Secondary Metrics |
|---|---|---|
| Code generation | pass@k | Syntax validity, test coverage |
| Data analysis | Result accuracy | SQL correctness, chart quality |
| Web navigation | Task completion rate | Number of page visits, time |
| Email drafting | Human preference | Tone appropriateness, completeness |
| Research synthesis | Factual accuracy | Source coverage, coherence |
| API integration | Functional correctness | Latency, error handling |

---

### 18.3.4 Trajectory-Level Evaluation

#### Action Sequence Quality

Evaluate the entire trajectory, not just the final output:

$$
e_{\text{trajectory}}(\tau) = \frac{1}{T} \sum_{t=1}^{T} q(a_t, s_t, q)
$$

where $q(a_t, s_t, q)$ scores the quality of action $a_t$ given state $s_t$ and original query $q$.

**Action Quality Dimensions:**

1. **Relevance:** Is the action relevant to the task?
$$
\text{relevance}(a_t) = P(\text{relevant} \mid a_t, q, s_t)
$$

2. **Necessity:** Was this action necessary, or could it have been skipped?
$$
\text{necessity}(a_t) = 1 - P(\text{task completion without } a_t)
$$

3. **Ordering:** Was this action taken at the right time?
$$
\text{ordering}(a_t) = \text{Sim}(\text{position}(a_t, \tau), \text{optimal\_position}(a_t, \tau^*))
$$

4. **Correctness:** Was the action specification (tool, parameters) correct?
$$
\text{correctness}(a_t) = \mathbb{1}[\text{tool}(a_t) = \text{optimal\_tool}(s_t) \wedge \text{params}(a_t) \text{ are valid}]
$$

#### Recovery from Errors

A critical dimension of robust agents — the ability to detect and recover from errors:

$$
e_{\text{recovery}}(\tau) = \frac{|\{t : \text{error}(a_t) \wedge \text{recovered}(a_{t+1:T})\}|}{|\{t : \text{error}(a_t)\}|}
$$

**Error Recovery Taxonomy:**

| Recovery Type | Description | Example |
|---|---|---|
| **Self-correction** | Agent detects its own error and fixes it | "That search didn't work, let me try different keywords" |
| **Retry with backoff** | Agent retries failed tool call with modifications | API timeout → retry with exponential backoff |
| **Alternative strategy** | Agent switches to a different approach | Web search fails → try database lookup |
| **Graceful degradation** | Agent provides partial results with transparency | "I couldn't find X, but here is Y" |
| **Escalation** | Agent recognizes it cannot proceed and requests help | "I need human assistance for this step" |

#### Efficiency of Tool Use

$$
\text{Tool\_Efficiency}(\tau) = \frac{|\text{useful\_tool\_calls}(\tau)|}{|\text{total\_tool\_calls}(\tau)|}
$$

where a tool call is "useful" if it contributes information that is actually used in the final answer or in subsequent reasoning steps.

**Tool Use Anti-Patterns (penalized in evaluation):**

| Anti-Pattern | Description | Detection |
|---|---|---|
| Redundant calls | Same tool called with identical parameters | Duplicate detection in trajectory |
| Exploratory waste | Tool calls whose results are never used | Track information flow from tool output to answer |
| Wrong tool selection | Using a complex tool when a simpler one suffices | Compare against oracle tool selection |
| Parameter errors | Correct tool but incorrect parameters (then retry) | Count retries per tool call |

---

## 18.4 Benchmarking Agentic Systems

### 18.4.1 SWE-bench (Software Engineering)

**SWE-bench** evaluates agents on their ability to resolve real GitHub issues from popular open-source Python repositories.

**Benchmark Design:**

| Property | Specification |
|---|---|
| **Task** | Given a GitHub issue description + repository snapshot, produce a code patch |
| **Source** | Real issues from 12 popular Python repos (Django, Flask, scikit-learn, etc.) |
| **Ground Truth** | Developer-written patches + associated test suites |
| **Evaluation** | Automated: apply patch, run test suite, check pass/fail |
| **Size** | SWE-bench: 2,294 tasks; SWE-bench Lite: 300 tasks; SWE-bench Verified: 500 human-verified tasks |
| **Difficulty** | Ranges from single-line fixes to multi-file architectural changes |

**Evaluation Metric:**

$$
\text{Resolve Rate} = \frac{|\{i : \text{all\_tests\_pass}(\text{patch}_i)\}|}{N}
$$

**Why SWE-bench is important:**
- Tests real-world software engineering capability (not toy problems)
- Requires: reading code, understanding codebases, reasoning about bugs, generating correct fixes
- Agent must navigate large codebases (100K+ lines), identify relevant files, understand dependencies
- Ground truth is unambiguous (tests either pass or fail)

**Current SOTA Performance (as of 2025):** Top agents achieve 40-55% on SWE-bench Verified, demonstrating significant but still limited capability on real-world software engineering tasks.

---

### 18.4.2 WebArena (Web Navigation)

**WebArena** evaluates agents on realistic web-based tasks across self-hosted websites that mirror real services.

**Benchmark Design:**

| Property | Specification |
|---|---|
| **Task** | Complete complex web-based tasks (e.g., "Find the cheapest flight from NYC to London next month") |
| **Environment** | Self-hosted replicas of e-commerce, forums, CMS, maps, GitLab |
| **Action Space** | Browser actions: click, type, scroll, navigate, select |
| **Evaluation** | Functional correctness: final state matches expected outcome |
| **Size** | 812 tasks across 5 website categories |
| **Observation** | Accessibility tree, screenshot, or both |

**Evaluation:**

$$
\text{Task\_Success} = \mathbb{1}[\text{final\_state}(\mathcal{E}) \in \mathcal{S}_{\text{goal}}]
$$

where $\mathcal{S}_{\text{goal}}$ is the set of environment states that satisfy the task's success criteria (e.g., correct item in cart, correct page navigated to, correct form submitted).

---

### 18.4.3 GAIA (General AI Assistants)

**GAIA** (General AI Assistants benchmark) tests fundamental abilities that a general-purpose AI assistant should possess: web browsing, file manipulation, multi-step reasoning.

**Design Philosophy:** Tasks require multiple capabilities composed together but have **unambiguous, factual answers** that can be verified automatically.

| Property | Specification |
|---|---|
| **Task** | Multi-step questions requiring tool use (e.g., "What is the total area of all countries that the Danube flows through?") |
| **Difficulty Levels** | Level 1 (1-3 steps), Level 2 (5-10 steps), Level 3 (arbitrarily many steps with domain expertise) |
| **Evaluation** | Exact match against verified factual answers |
| **Size** | 466 questions (165 Level 1, 86 Level 2, 215 Level 3) |
| **Required Tools** | Web search, calculator, file reader, code executor |

**Metric:**

$$
\text{Accuracy@Level}(l) = \frac{|\{i : \text{level}(i) = l \wedge \text{normalize}(y_i) = \text{normalize}(y_i^*)\}|}{|\{i : \text{level}(i) = l\}|}
$$

---

### 18.4.4 AgentBench (Multi-Environment)

**AgentBench** provides a unified evaluation framework across 8 distinct environments, testing agent generalizability.

**Environments:**

| Environment | Task Type | Action Space | Evaluation |
|---|---|---|---|
| Operating System | Bash commands | Shell commands | Functional correctness |
| Database | SQL queries | SQL statements | Result matching |
| Knowledge Graph | SPARQL/traversal | Graph operations | Answer accuracy |
| Digital Card Game | Strategy | Game actions | Win rate |
| Lateral Thinking | Puzzles | Text answers | Exact match |
| House-Holding | Embodied tasks | Physical actions | Task completion |
| Web Shopping | E-commerce | Web actions | Purchase correctness |
| Web Browsing | Information retrieval | Browser actions | Answer accuracy |

**Aggregate Score:**

$$
\text{AgentBench\_Score} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \text{normalized\_score}(e)
$$

where normalization ensures comparability across environments with different score ranges.

---

### 18.4.5 ToolBench (Tool Use)

**ToolBench** evaluates agents' ability to select and use tools from a large, realistic tool library.

| Property | Specification |
|---|---|
| **Tool Library** | 16,000+ real-world REST APIs from RapidAPI |
| **Task Types** | Single-tool, multi-tool, multi-step |
| **Evaluation Dimensions** | Tool selection accuracy, argument correctness, task completion |
| **Challenge** | Tool discovery (finding the right API from thousands), API composition |

**Metrics:**

$$
\text{Tool\_Selection\_Acc} = \frac{|\text{correct\_tools\_selected}|}{|\text{required\_tools}|}
$$

$$
\text{Argument\_Acc} = \frac{|\text{correct\_arguments}|}{|\text{total\_arguments}|}
$$

$$
\text{Win\_Rate} = P(\text{ToolBench agent preferred over ChatGPT by LLM judge})
$$

---

### 18.4.6 OSWorld, AndroidWorld (GUI Agents)

**OSWorld** and **AndroidWorld** evaluate agents that interact with graphical user interfaces through pixel-level observations and mouse/keyboard actions.

**OSWorld:**

| Property | Specification |
|---|---|
| **Environment** | Full desktop OS (Ubuntu, Windows, macOS) in VM |
| **Observation** | Screenshots + accessibility tree |
| **Actions** | Mouse click, type, scroll, keyboard shortcuts |
| **Tasks** | 369 tasks across office, coding, system admin, creative applications |
| **Evaluation** | Custom scripts verify end state (file content, application state) |

**AndroidWorld:**

| Property | Specification |
|---|---|
| **Environment** | Android emulator |
| **Observation** | Screenshots + UI hierarchy |
| **Actions** | Tap, swipe, type, navigate |
| **Tasks** | Mobile app interactions (settings, messaging, browsing) |
| **Evaluation** | State-based verification |

**Key Challenge:** The observation space (pixels) is enormous and unstructured compared to text-based environments, requiring vision-language model integration.

---

### 18.4.7 Designing Custom Evaluation Suites

When existing benchmarks don't cover a specific deployment domain, design custom evaluation suites:

**Design Principles:**

1. **Representative sampling** — Tasks should reflect the actual distribution of user queries:

$$
P_{\text{eval}}(\text{task type}) \approx P_{\text{production}}(\text{task type})
$$

2. **Difficulty stratification** — Include easy, medium, and hard tasks to measure performance across the difficulty spectrum.

3. **Edge case inclusion** — Deliberately include adversarial, ambiguous, and boundary cases.

4. **Minimal data leakage** — Ensure evaluation tasks are not in the training data of the base model.

5. **Verifiable ground truth** — Every task must have an unambiguous, verifiable expected outcome.

**Custom Evaluation Suite Template:**

```python
@dataclass
class EvalTask:
    task_id: str
    query: str
    difficulty: str        # "easy", "medium", "hard"
    category: str          # domain-specific category
    expected_output: Any   # ground truth
    evaluation_fn: Callable # custom evaluation function
    required_tools: list[str]
    max_steps: int
    timeout_seconds: int
    metadata: dict = field(default_factory=dict)

class CustomEvalSuite:
    def __init__(self, tasks: list[EvalTask]):
        self.tasks = tasks
        self.validate_suite()
    
    def validate_suite(self):
        """Ensure suite meets quality criteria."""
        # Check difficulty distribution
        difficulty_dist = Counter(t.difficulty for t in self.tasks)
        assert difficulty_dist["hard"] >= 0.2 * len(self.tasks), \
            "At least 20% tasks should be hard"
        
        # Check category coverage
        categories = set(t.category for t in self.tasks)
        assert len(categories) >= 5, "Need at least 5 task categories"
        
        # Check all tasks have evaluation functions
        for task in self.tasks:
            assert callable(task.evaluation_fn), \
                f"Task {task.task_id} missing evaluation function"
    
    async def run(self, agent, n_repetitions: int = 3) -> EvalReport:
        results = []
        for task in self.tasks:
            task_results = []
            for rep in range(n_repetitions):
                result = await self._run_single(agent, task, rep)
                task_results.append(result)
            results.append(TaskResult(
                task=task,
                runs=task_results,
                mean_score=np.mean([r.score for r in task_results]),
                std_score=np.std([r.score for r in task_results]),
                success_rate=np.mean([r.success for r in task_results])
            ))
        return EvalReport(results=results)
```

---

## 18.5 Observability and Monitoring in Production

### 18.5.1 Tracing and Span-Based Logging

Production observability for agentic systems requires **distributed tracing** — capturing the full execution flow across multiple LLM calls, tool invocations, guardrail checks, and sub-agent interactions as a structured trace.

#### OpenTelemetry for Agents

**OpenTelemetry (OTel)** provides a vendor-neutral standard for distributed tracing. For agents, each trace represents a complete user interaction, and each span represents an individual operation within that interaction.

**Agent Trace Structure:**

```
Trace: user_request_abc123
├── Span: input_guardrails (12ms)
│   ├── Span: prompt_injection_check (5ms) [PASS]
│   ├── Span: pii_detection (4ms) [PASS]
│   └── Span: topic_filter (3ms) [PASS]
├── Span: planning_llm_call (1,823ms)
│   ├── Attribute: model = "gpt-4o"
│   ├── Attribute: input_tokens = 2,341
│   ├── Attribute: output_tokens = 456
│   ├── Attribute: temperature = 0.0
│   └── Attribute: cost = $0.038
├── Span: tool_execution (3,241ms)
│   ├── Span: web_search("quarterly earnings AAPL") (2,100ms)
│   │   ├── Attribute: results_count = 10
│   │   └── Attribute: top_result_relevance = 0.92
│   └── Span: calculator("revenue * 0.23") (41ms)
│       └── Attribute: result = 24_530_000
├── Span: synthesis_llm_call (2,102ms)
│   ├── Attribute: model = "gpt-4o"
│   ├── Attribute: input_tokens = 4,892
│   └── Attribute: output_tokens = 312
├── Span: output_guardrails (45ms)
│   ├── Span: toxicity_check (15ms) [PASS]
│   ├── Span: factuality_check (25ms) [PASS]
│   └── Span: pii_output_check (5ms) [PASS]
└── Span: response_delivery (3ms)
    └── Attribute: total_latency = 7,226ms
```

**Implementation:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize tracer
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agentic-system")

class TracedAgent:
    def __init__(self, agent):
        self.agent = agent
    
    async def process(self, query: str) -> str:
        with tracer.start_as_current_span("agent_request") as root_span:
            root_span.set_attribute("query", query)
            root_span.set_attribute("session_id", self.session_id)
            
            # Input guardrails
            with tracer.start_as_current_span("input_guardrails") as guard_span:
                guard_result = await self.run_input_guardrails(query)
                guard_span.set_attribute("result", guard_result.status)
                guard_span.set_attribute("guardrails_triggered", 
                                          guard_result.triggered)
                if guard_result.blocked:
                    root_span.set_attribute("blocked", True)
                    return guard_result.message
            
            # LLM reasoning
            with tracer.start_as_current_span("llm_reasoning") as llm_span:
                start = time.perf_counter()
                response = await self.agent.reason(query)
                latency = time.perf_counter() - start
                
                llm_span.set_attribute("model", response.model)
                llm_span.set_attribute("input_tokens", response.input_tokens)
                llm_span.set_attribute("output_tokens", response.output_tokens)
                llm_span.set_attribute("latency_ms", latency * 1000)
                llm_span.set_attribute("cost_usd", response.cost)
            
            # Tool execution
            for tool_call in response.tool_calls:
                with tracer.start_as_current_span(
                    f"tool_{tool_call.name}"
                ) as tool_span:
                    tool_span.set_attribute("tool", tool_call.name)
                    tool_span.set_attribute("params", str(tool_call.params))
                    result = await self.execute_tool(tool_call)
                    tool_span.set_attribute("success", result.success)
                    tool_span.set_attribute("latency_ms", result.latency_ms)
            
            # Output guardrails
            with tracer.start_as_current_span("output_guardrails"):
                final_output = await self.run_output_guardrails(
                    response.text, query
                )
            
            root_span.set_attribute("total_tokens", 
                                      response.total_tokens)
            root_span.set_attribute("total_cost_usd", 
                                      response.total_cost)
            root_span.set_attribute("total_steps", 
                                      len(response.tool_calls) + 1)
            
            return final_output
```

#### Trace Visualization Tools

| Tool | Key Features | Integration |
|---|---|---|
| **Langfuse** | Open-source, LLM-native tracing, prompt management, evaluation | Python SDK, OpenAI/Anthropic wrappers |
| **LangSmith** | LangChain-native, playground, dataset management, online eval | LangChain integration |
| **Arize Phoenix** | Open-source, embedding visualization, retrieval analysis | OpenTelemetry, LlamaIndex |
| **Braintrust** | Eval-focused, prompt playground, scoring | API-based, model-agnostic |
| **Weights & Biases Weave** | Experiment tracking, trace logging, evaluation | Python SDK |

---

### 18.5.2 Metrics Collection and Dashboards

#### Production Metrics Taxonomy

**Level 1: Infrastructure Metrics (collected per-second)**

| Metric | Unit | Collection Method |
|---|---|---|
| Request rate | req/s | Load balancer counter |
| Error rate | % | HTTP status code monitoring |
| CPU/GPU utilization | % | System monitoring agent |
| Memory usage | GB | System monitoring agent |
| Queue depth | count | Message queue metrics |

**Level 2: Application Metrics (collected per-request)**

| Metric | Unit | Collection Method |
|---|---|---|
| End-to-end latency | ms | Trace span duration |
| LLM call latency | ms | Per-call instrumentation |
| Tool call latency | ms | Per-call instrumentation |
| Token count (input/output) | tokens | API response metadata |
| Cost per request | USD | Token count × pricing |
| Steps per request | count | Trace span count |
| Guardrail trigger rate | % | Guardrail logging |

**Level 3: Quality Metrics (collected per-request, evaluated asynchronously)**

| Metric | Unit | Collection Method |
|---|---|---|
| LLM-as-Judge score | 1-5 | Async LLM evaluation |
| Factuality score | 0-1 | NLI-based checker |
| Hallucination rate | % | Source attribution check |
| User feedback (if available) | thumbs up/down | UI feedback widget |
| Task completion rate | % | Domain-specific checker |

**Dashboard Architecture:**

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter(
    'agent_requests_total', 
    'Total requests',
    ['status', 'agent_version']
)

REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Request latency',
    ['agent_version'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

TOKEN_USAGE = Counter(
    'agent_tokens_total',
    'Total tokens used',
    ['direction', 'model']  # direction: input/output
)

COST_TOTAL = Counter(
    'agent_cost_usd_total',
    'Total cost in USD',
    ['model']
)

GUARDRAIL_TRIGGERS = Counter(
    'agent_guardrail_triggers_total',
    'Guardrail trigger count',
    ['guardrail_type', 'result']  # result: pass/fail/modify
)

LLM_CALLS = Histogram(
    'agent_llm_calls_per_request',
    'Number of LLM calls per request',
    buckets=[1, 2, 3, 5, 8, 13, 21]
)

QUALITY_SCORE = Histogram(
    'agent_quality_score',
    'Quality score from LLM-as-Judge',
    buckets=[1.0, 2.0, 3.0, 4.0, 5.0]
)

class MetricsMiddleware:
    async def __call__(self, request, agent):
        start = time.perf_counter()
        
        try:
            result = await agent.process(request)
            status = "success"
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            
            REQUEST_COUNT.labels(
                status=status, 
                agent_version=agent.version
            ).inc()
            
            REQUEST_LATENCY.labels(
                agent_version=agent.version
            ).observe(duration)
            
            if hasattr(result, 'token_usage'):
                TOKEN_USAGE.labels(
                    direction="input", model=result.model
                ).inc(result.input_tokens)
                TOKEN_USAGE.labels(
                    direction="output", model=result.model
                ).inc(result.output_tokens)
            
            COST_TOTAL.labels(model=result.model).inc(result.cost)
            LLM_CALLS.observe(result.num_llm_calls)
        
        return result
```

---

### 18.5.3 Alerting and Anomaly Detection

#### Drift Detection in Agent Behavior

Agent behavior can drift over time due to model updates, environment changes, or adversarial activity. Detect drift by monitoring metric distributions:

**Statistical Process Control (SPC):**

For a metric $m_t$ measured over time, compute a rolling mean $\bar{m}$ and standard deviation $s$ from a baseline period. Alert when:

$$
|m_t - \bar{m}| > k \cdot s
$$

where $k = 3$ gives a $99.7\%$ confidence threshold (3-sigma rule) under normality.

**CUSUM (Cumulative Sum) for detecting sustained shifts:**

$$
S_t^+ = \max(0, S_{t-1}^+ + (m_t - \mu_0 - \delta)) \quad \text{(upward shift detection)}
$$

$$
S_t^- = \max(0, S_{t-1}^- + (\mu_0 - \delta - m_t)) \quad \text{(downward shift detection)}
$$

Alert when $S_t^+ > h$ or $S_t^- > h$, where $\mu_0$ is the target mean, $\delta$ is the minimum shift to detect, and $h$ is the decision threshold.

**Distribution-Level Drift Detection:**

For categorical metrics (tool selection distribution, error types), use the Population Stability Index (PSI):

$$
\text{PSI} = \sum_{i=1}^{B} (p_i^{\text{current}} - p_i^{\text{baseline}}) \cdot \ln\frac{p_i^{\text{current}}}{p_i^{\text{baseline}}}
$$

| PSI Value | Interpretation |
|---|---|
| $< 0.10$ | No significant drift |
| $0.10 - 0.25$ | Moderate drift — investigate |
| $> 0.25$ | Significant drift — action required |

#### Performance Regression Alerts

```python
class AlertManager:
    ALERT_RULES = [
        {
            "name": "high_error_rate",
            "metric": "error_rate_5min",
            "condition": lambda v: v > 0.05,
            "severity": "critical",
            "message": "Error rate exceeded 5% in 5-minute window"
        },
        {
            "name": "latency_regression",
            "metric": "p95_latency_5min",
            "condition": lambda v: v > 15.0,  # seconds
            "severity": "warning",
            "message": "P95 latency exceeded 15s"
        },
        {
            "name": "cost_spike",
            "metric": "cost_per_request_1hr_avg",
            "condition": lambda v, baseline: v > 2.0 * baseline,
            "severity": "warning",
            "message": "Average cost per request doubled vs. baseline"
        },
        {
            "name": "guardrail_spike",
            "metric": "guardrail_trigger_rate_15min",
            "condition": lambda v, baseline: v > baseline + 3 * baseline_std,
            "severity": "critical",
            "message": "Guardrail trigger rate anomalously high"
        },
        {
            "name": "quality_drop",
            "metric": "llm_judge_score_1hr_avg",
            "condition": lambda v, baseline: v < baseline - 0.5,
            "severity": "critical",
            "message": "LLM-as-Judge quality score dropped significantly"
        },
        {
            "name": "abandonment_spike",
            "metric": "task_abandonment_rate_1hr",
            "condition": lambda v: v > 0.20,
            "severity": "warning",
            "message": "Task abandonment rate exceeded 20%"
        }
    ]
```

---

### 18.5.4 Log Aggregation and Search

**Log Levels for Agentic Systems:**

| Level | Content | Storage Duration | Search Requirement |
|---|---|---|---|
| **DEBUG** | Full prompts, responses, intermediate states | 7 days | Ad-hoc debugging |
| **INFO** | Request IDs, tool calls, guardrail results, latencies | 30 days | Operational monitoring |
| **WARN** | Guardrail near-misses, elevated latency, retries | 90 days | Trend analysis |
| **ERROR** | Failed tool calls, guardrail blocks, exceptions | 1 year | Incident investigation |
| **AUDIT** | All inputs/outputs for compliance | 7 years (domain-dependent) | Regulatory compliance |

**Structured Logging Schema:**

```json
{
  "timestamp": "2025-01-15T10:23:45.123Z",
  "trace_id": "abc123def456",
  "span_id": "span_789",
  "level": "INFO",
  "event": "llm_call_complete",
  "agent_id": "agent_v2.3",
  "session_id": "sess_xyz",
  "user_id": "user_hash_abc",
  "model": "gpt-4o",
  "input_tokens": 2341,
  "output_tokens": 456,
  "latency_ms": 1823,
  "cost_usd": 0.038,
  "temperature": 0.0,
  "tool_calls": ["web_search", "calculator"],
  "guardrail_results": {
    "toxicity": "pass",
    "pii": "pass",
    "factuality": "pass"
  },
  "quality_score": null,
  "error": null
}
```

**Search and Analysis Capabilities:**

```
# Find all requests where guardrail blocked output
event:guardrail_block AND level:ERROR AND timestamp:[2025-01-14 TO 2025-01-15]

# Find high-latency requests for a specific agent version
agent_id:agent_v2.3 AND latency_ms:>10000

# Find requests with hallucination detection triggers
guardrail_results.factuality:fail AND quality_score:<3

# Analyze tool usage patterns
event:tool_call | stats count by tool_name | sort -count
```

---

## 18.6 Continuous Evaluation and Regression Testing

### 18.6.1 CI/CD for Agent Pipelines

Agentic systems require specialized CI/CD pipelines that treat **evaluation results as deployment gates**, not just unit test pass/fail.

**Agent CI/CD Pipeline:**

```
┌──────────────────────────────────────────────────────────────┐
│                    AGENT CI/CD PIPELINE                        │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  CODE    │  │  UNIT    │  │  EVAL    │  │  STAGING    │  │
│  │  CHANGE  │─▶│  TESTS   │─▶│  SUITE   │─▶│  CANARY     │  │
│  │          │  │          │  │          │  │  DEPLOY     │  │
│  └─────────┘  └──────────┘  └──────────┘  └─────────────┘  │
│       │            │             │               │           │
│  git push    pytest pass   eval scores     canary metrics    │
│              + lint         meet gates      pass thresholds  │
│              + type check                                    │
│                                                              │
│                                         ┌─────────────┐     │
│                                         │  PRODUCTION  │     │
│                                    ────▶│  DEPLOY      │     │
│                                         │  (gradual)   │     │
│                                         └─────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

**Evaluation Gates:**

```yaml
# .agent-ci/eval_gates.yaml
eval_suite: "core_eval_v3"
repetitions: 3

gates:
  - name: "task_success_rate"
    metric: "success_rate"
    threshold: 0.82       # Must meet or exceed
    comparison: "gte"
    baseline: "production_current"
    
  - name: "no_quality_regression"
    metric: "llm_judge_mean_score"
    threshold: -0.1       # No more than 0.1 point drop
    comparison: "regression_from_baseline"
    baseline: "production_current"
    
  - name: "safety_maintained"
    metric: "guardrail_violation_rate"
    threshold: 0.01       # Less than 1% violations
    comparison: "lte"
    
  - name: "cost_bounded"
    metric: "mean_cost_per_task"
    threshold: 1.5        # No more than 50% cost increase
    comparison: "ratio_to_baseline"
    baseline: "production_current"
    
  - name: "latency_bounded"
    metric: "p95_latency"
    threshold: 20.0       # P95 under 20 seconds
    comparison: "lte"

failure_action: "block_deploy"
notification: ["#agent-alerts", "agent-team@company.com"]
```

**Implementation:**

```python
class EvalGate:
    def __init__(self, config: dict):
        self.gates = config["gates"]
        self.eval_suite = config["eval_suite"]
        self.repetitions = config["repetitions"]
    
    async def evaluate(self, agent_candidate, agent_baseline) -> GateResult:
        # Run eval suite on candidate
        candidate_results = await self.run_eval(
            agent_candidate, self.repetitions
        )
        
        # Run eval suite on baseline (or use cached results)
        baseline_results = await self.get_baseline_results(agent_baseline)
        
        # Check each gate
        gate_results = []
        for gate in self.gates:
            passed = self.check_gate(
                gate, 
                candidate_results.metrics[gate["metric"]],
                baseline_results.metrics.get(gate["metric"])
            )
            gate_results.append(GateCheckResult(
                name=gate["name"],
                passed=passed,
                candidate_value=candidate_results.metrics[gate["metric"]],
                baseline_value=baseline_results.metrics.get(gate["metric"]),
                threshold=gate["threshold"]
            ))
        
        all_passed = all(g.passed for g in gate_results)
        
        return GateResult(
            passed=all_passed,
            gate_checks=gate_results,
            recommendation="DEPLOY" if all_passed else "BLOCK"
        )
```

---

### 18.6.2 Eval-Driven Development

Eval-driven development (EDD) is the methodology where **evaluation metrics drive all development decisions** — analogous to test-driven development (TDD) but for AI systems.

**EDD Workflow:**

```
1. DEFINE: Write evaluation criteria and expected metrics BEFORE 
   making changes
   
2. MEASURE: Run eval suite on current system (establish baseline)
   
3. HYPOTHESIZE: Formulate a specific, falsifiable hypothesis about 
   how a proposed change will affect metrics
   Example: "Switching to ReAct prompting will increase success_rate 
   by 5% with <10% latency increase"
   
4. IMPLEMENT: Make the change
   
5. EVALUATE: Run the same eval suite on the changed system
   
6. COMPARE: Statistically test whether the hypothesis is supported
   
7. DECIDE: Accept/reject the change based on evidence
   
8. ITERATE: Return to step 3 with new hypotheses
```

**Statistical Rigor in EDD:**

For comparing two agent versions, use paired bootstrap testing:

$$
\Delta = \hat{e}(\mathcal{A}_{\text{new}}) - \hat{e}(\mathcal{A}_{\text{baseline}})
$$

Bootstrap confidence interval:

$$
\text{CI}_{95\%}(\Delta) = [\Delta^*_{0.025}, \Delta^*_{0.975}]
$$

where $\Delta^*$ values are percentiles of the bootstrap distribution obtained by resampling evaluation results with replacement $B = 10,000$ times.

**Decision rule:** Accept the change if and only if:
1. The lower bound of the CI for the primary metric is $> 0$ (statistically significant improvement), OR
2. The CI overlaps zero but secondary metrics (cost, latency) improve without primary metric regression.

---

### 18.6.3 Canary Deployments for Agent Updates

Canary deployment gradually rolls out a new agent version to a small percentage of traffic, monitoring for regressions before full rollout.

**Canary Protocol:**

$$
\text{traffic\_split}(t) = \begin{cases}
1\% & t \in [0, T_1] \quad \text{(initial canary)} \\
5\% & t \in [T_1, T_2] \quad \text{(expanded canary)} \\
25\% & t \in [T_2, T_3] \quad \text{(broad canary)} \\
100\% & t > T_3 \quad \text{(full rollout)}
\end{cases}
$$

**Promotion Criteria at Each Stage:**

$$
\text{Promote}(t) = \mathbb{1}\left[\forall m \in \mathcal{M}: \frac{m_{\text{canary}}(t)}{m_{\text{baseline}}(t)} \geq \theta_m\right]
$$

| Metric | Threshold $\theta_m$ | Measurement Window |
|---|---|---|
| Success rate | $\geq 0.98$ (relative to baseline) | Rolling 1-hour |
| Error rate | $\leq 1.02$ (relative to baseline) | Rolling 15-min |
| P95 latency | $\leq 1.10$ (no more than 10% increase) | Rolling 1-hour |
| Cost per request | $\leq 1.20$ (no more than 20% increase) | Rolling 1-hour |
| Guardrail trigger rate | $\leq 1.05$ (no more than 5% increase) | Rolling 1-hour |
| User feedback | $\geq 0.95$ (relative to baseline) | Rolling 4-hour |

**Automatic Rollback:**

$$
\text{Rollback} = \mathbb{1}\left[\exists m \in \mathcal{M}_{\text{critical}}: m_{\text{canary}} < \theta_m^{\text{rollback}}\right]
$$

where $\theta_m^{\text{rollback}}$ is a stricter threshold that triggers immediate rollback without waiting for human review.

---

### 18.6.4 A/B Testing Agent Configurations

A/B testing compares two agent configurations on live traffic with statistical rigor.

**Experimental Design:**

$$
H_0: e(\mathcal{A}_A) = e(\mathcal{A}_B) \quad \text{vs.} \quad H_1: e(\mathcal{A}_A) \neq e(\mathcal{A}_B)
$$

**Sample Size Calculation:** For detecting a minimum effect size $\delta$ with power $1-\beta$ and significance $\alpha$:

$$
N = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}
$$

For binary success rate with baseline $p_0 = 0.80$, minimum detectable improvement $\delta = 0.03$, $\alpha = 0.05$, $\beta = 0.20$:

$$
N = \frac{2(1.96 + 0.84)^2 \cdot 0.80 \cdot 0.20}{0.03^2} \approx 2,\!239 \text{ per group}
$$

**Randomization Unit:** Randomize at the **user level** (not request level) to avoid within-user inconsistency:

```python
def assign_variant(user_id: str, experiment_id: str) -> str:
    """Deterministic, consistent assignment based on hash."""
    hash_input = f"{user_id}:{experiment_id}"
    hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
    
    # 50/50 split
    return "treatment" if hash_value % 100 < 50 else "control"
```

**Multi-Metric Analysis with Correction:** When testing multiple metrics simultaneously, apply Bonferroni or Benjamini-Hochberg correction:

$$
\alpha_{\text{adjusted}} = \frac{\alpha}{D} \quad \text{(Bonferroni)}
$$

where $D$ is the number of metrics being tested. This controls the family-wise error rate.

---

### 18.6.5 Golden Dataset Maintenance

The golden dataset is a curated, version-controlled set of evaluation examples that serves as the **ground truth reference** for regression testing.

**Golden Dataset Properties:**

1. **Representative:** Covers all important task types and difficulty levels.
2. **Verified:** Every example has human-verified ground truth.
3. **Versioned:** Changes are tracked with clear changelogs.
4. **Stable:** Core examples persist across versions for longitudinal comparison.
5. **Living:** Regularly updated with new edge cases discovered in production.

**Maintenance Protocol:**

```python
class GoldenDataset:
    def __init__(self, path: str, version: str):
        self.path = path
        self.version = version
        self.examples = self.load()
        self.changelog = self.load_changelog()
    
    def add_example(self, example: EvalTask, source: str, reason: str):
        """Add a new golden example with provenance."""
        # Verify ground truth
        assert self.verify_ground_truth(example), \
            "Ground truth must be verified by 2+ humans"
        
        # Check for duplicates
        assert not self.is_duplicate(example), \
            "Example too similar to existing golden example"
        
        # Add with metadata
        example.metadata.update({
            "added_version": self.version,
            "source": source,    # "production_failure", "red_team", "manual"
            "reason": reason,
            "verified_by": self.get_verifiers(),
            "added_date": datetime.utcnow().isoformat()
        })
        
        self.examples.append(example)
        self.changelog.append({
            "action": "add",
            "example_id": example.task_id,
            "reason": reason,
            "version": self.version
        })
    
    def retire_example(self, task_id: str, reason: str):
        """Retire an example that is no longer relevant."""
        # Don't delete — mark as retired for historical tracking
        example = self.get(task_id)
        example.metadata["retired"] = True
        example.metadata["retired_reason"] = reason
        example.metadata["retired_version"] = self.version
    
    def compute_coverage(self) -> dict:
        """Analyze coverage across dimensions."""
        return {
            "by_difficulty": Counter(e.difficulty for e in self.active),
            "by_category": Counter(e.category for e in self.active),
            "by_required_tools": Counter(
                tuple(sorted(e.required_tools)) for e in self.active
            ),
            "total_active": len(self.active),
            "total_retired": len(self.retired)
        }
```

**Golden Dataset Update Triggers:**

| Trigger | Action | Frequency |
|---|---|---|
| Production failure not covered by golden set | Add representative example | Per incident |
| Red-team discovers new attack vector | Add adversarial example | Per finding |
| New feature/tool added to agent | Add examples exercising new capability | Per feature |
| Benchmark saturation (>95% pass rate) | Add harder examples | Quarterly |
| Ground truth found to be incorrect | Fix or retire example | Immediately |
| External environment change (API deprecated) | Update or retire affected examples | As needed |

---

## 18.7 Debugging Agentic Systems

### 18.7.1 Replay and Simulation of Agent Trajectories

Trajectory replay enables **deterministic reproduction** of agent behavior for debugging, by capturing all inputs, intermediate states, and non-deterministic elements during execution and replaying them offline.

**Replay Architecture:**

```
RECORDING PHASE (Production):
┌───────────────────────────────────────────────┐
│  Agent Execution                              │
│                                               │
│  Step 1: query → LLM(prompt₁) → response₁    │──▶ Record
│  Step 2: response₁ → Tool(params) → result₁  │──▶ Record
│  Step 3: result₁ → LLM(prompt₂) → response₂  │──▶ Record
│  ...                                          │
│                                               │
│  Recorded: {                                  │
│    random_seed, temperature,                  │
│    all prompts, all responses,                │
│    all tool calls, all tool results,          │
│    all guardrail checks, timestamps           │
│  }                                            │
└───────────────────────────────────────────────┘

REPLAY PHASE (Debugging):
┌───────────────────────────────────────────────┐
│  Deterministic Replay                         │
│                                               │
│  Step 1: query → Mock LLM(prompt₁) → saved₁  │
│  Step 2: saved₁ → Mock Tool(params) → saved₂  │
│  Step 3: saved₂ → Mock LLM(prompt₂) → saved₃  │
│  ...                                          │
│                                               │
│  All external calls return recorded values    │
│  Internal logic executes with actual code     │
│  → Exact reproduction of the original run     │
└───────────────────────────────────────────────┘
```

**Implementation:**

```python
@dataclass
class ReplayRecord:
    trace_id: str
    steps: list[StepRecord]
    
@dataclass
class StepRecord:
    step_index: int
    step_type: str          # "llm_call", "tool_call", "guardrail"
    input_data: dict        # Full input to the step
    output_data: dict       # Full output from the step
    latency_ms: float
    metadata: dict

class TrajectoryRecorder:
    def __init__(self):
        self.current_trace = None
        self.steps = []
    
    def record_step(self, step_type, input_data, output_data, latency_ms):
        self.steps.append(StepRecord(
            step_index=len(self.steps),
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            metadata={"timestamp": time.time()}
        ))
    
    def save(self, trace_id: str) -> ReplayRecord:
        record = ReplayRecord(trace_id=trace_id, steps=self.steps)
        self.storage.save(record)
        return record

class TrajectoryReplayer:
    def __init__(self, record: ReplayRecord):
        self.record = record
        self.step_index = 0
    
    def mock_llm_call(self, prompt):
        """Return the recorded LLM response instead of calling the API."""
        step = self.record.steps[self.step_index]
        assert step.step_type == "llm_call"
        self.step_index += 1
        return step.output_data
    
    def mock_tool_call(self, tool_name, params):
        """Return the recorded tool result."""
        step = self.record.steps[self.step_index]
        assert step.step_type == "tool_call"
        assert step.input_data["tool"] == tool_name
        self.step_index += 1
        return step.output_data
    
    async def replay_with_modified_agent(self, agent):
        """Replay but with a modified agent, using recorded 
        external responses."""
        agent.llm = self.mock_llm_call
        agent.tool_executor = self.mock_tool_call
        
        result = await agent.process(
            self.record.steps[0].input_data["query"]
        )
        return result
```

---

### 18.7.2 Step-Through Debugging

Step-through debugging allows developers to **pause agent execution at each step**, inspect the state, and optionally modify inputs before proceeding.

**Interactive Debugger:**

```python
class AgentDebugger:
    def __init__(self, agent, breakpoints=None):
        self.agent = agent
        self.breakpoints = breakpoints or set()
        self.history = []
        self.paused = False
    
    async def debug(self, query: str):
        """Run agent with interactive debugging."""
        state = AgentState(query=query)
        
        while not state.is_terminal:
            step_info = {
                "step": len(self.history),
                "state": state.to_dict(),
                "proposed_action": None
            }
            
            # Get next action from agent
            action = await self.agent.plan_next_action(state)
            step_info["proposed_action"] = action.to_dict()
            
            # Check breakpoints
            if self.should_break(state, action):
                self.paused = True
                print(f"\n{'='*60}")
                print(f"BREAKPOINT at step {len(self.history)}")
                print(f"State: {state.summary()}")
                print(f"Proposed action: {action}")
                print(f"{'='*60}")
                
                command = await self.get_user_command()
                
                if command == "continue":
                    self.paused = False
                elif command == "step":
                    pass  # Execute one step, then break again
                elif command == "inspect":
                    self.inspect_state(state)
                    continue
                elif command == "modify":
                    action = self.modify_action(action)
                elif command == "abort":
                    return self.history
                elif command.startswith("eval"):
                    # Evaluate arbitrary expression in current context
                    expr = command[5:]
                    print(eval(expr, {"state": state, "action": action}))
                    continue
            
            # Execute action
            result = await self.agent.execute_action(action, state)
            
            step_info["result"] = result.to_dict()
            self.history.append(step_info)
            
            state = self.agent.update_state(state, action, result)
        
        return self.history
    
    def should_break(self, state, action) -> bool:
        """Check if any breakpoint condition is met."""
        return (
            "all" in self.breakpoints or
            action.tool in self.breakpoints or
            len(self.history) in self.breakpoints or
            (self.paused)  # Single-step mode
        )
```

---

### 18.7.3 Counterfactual Analysis: What If Different Tool/Route

Counterfactual analysis asks: **"What would have happened if the agent had made a different choice at step $t$?"** This isolates the impact of individual decisions.

**Formal Framework.** Given a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ and a counterfactual action $a_t'$ at step $t$:

$$
\tau' = (s_0, a_0, \dots, s_t, a_t', s_{t+1}', a_{t+1}', \dots, s_{T'}')
$$

The counterfactual impact:

$$
\Delta_{\text{CF}}(t, a_t') = e(\tau') - e(\tau)
$$

If $\Delta_{\text{CF}} > 0$, the alternative action would have led to a better outcome.

**Implementation:**

```python
class CounterfactualAnalyzer:
    def __init__(self, agent, replayer):
        self.agent = agent
        self.replayer = replayer
    
    async def analyze(self, trajectory: ReplayRecord, 
                       step: int, 
                       alternative_actions: list) -> list[CFResult]:
        """Test alternative actions at a specific step."""
        results = []
        
        for alt_action in alternative_actions:
            # Replay up to the target step using recorded data
            partial_state = self.replayer.replay_to_step(
                trajectory, step
            )
            
            # Execute the alternative action
            alt_result = await self.agent.execute_action(
                alt_action, partial_state
            )
            
            # Continue execution from the alternative state
            alt_trajectory = await self.agent.continue_from(
                partial_state, alt_action, alt_result
            )
            
            # Evaluate the counterfactual trajectory
            cf_score = self.evaluate(alt_trajectory)
            original_score = self.evaluate(trajectory)
            
            results.append(CFResult(
                step=step,
                original_action=trajectory.steps[step].input_data,
                alternative_action=alt_action,
                original_score=original_score,
                counterfactual_score=cf_score,
                impact=cf_score - original_score,
                explanation=self.explain_difference(
                    trajectory, alt_trajectory, step
                )
            ))
        
        return sorted(results, key=lambda r: r.impact, reverse=True)
```

**Automatic Counterfactual Generation:**

```python
def generate_counterfactuals(trajectory, step):
    """Generate alternative actions the agent could have taken."""
    original_action = trajectory.steps[step]
    alternatives = []
    
    if original_action.step_type == "tool_call":
        # What if we used a different tool?
        for tool in AVAILABLE_TOOLS:
            if tool != original_action.input_data["tool"]:
                alternatives.append(Action(
                    tool=tool,
                    params=adapt_params(original_action, tool)
                ))
        
        # What if we used different parameters?
        for param_variant in generate_param_variants(original_action):
            alternatives.append(param_variant)
    
    elif original_action.step_type == "llm_call":
        # What if we used a different temperature?
        for temp in [0.0, 0.3, 0.7, 1.0]:
            alternatives.append(original_action.with_temperature(temp))
        
        # What if we used a different model?
        for model in AVAILABLE_MODELS:
            alternatives.append(original_action.with_model(model))
    
    return alternatives
```

---

### 18.7.4 Attribution: Which Component Caused Failure

When an agent fails a task, attribution analysis determines **which component** (prompt, model, tool, guardrail, or architecture) was the root cause.

**Attribution Framework:**

$$
\text{Failure}(\tau) = \arg\max_{c \in \mathcal{C}} P(\text{root\_cause} = c \mid \tau, \text{failure\_type})
$$

where $\mathcal{C} = \{\text{prompt}, \text{model}, \text{tool}, \text{guardrail}, \text{architecture}, \text{data}\}$.

**Systematic Attribution Process:**

```
FAILURE TAXONOMY → DIAGNOSTIC PROCEDURE → ROOT CAUSE

1. WRONG FINAL ANSWER
   ├── Was the reasoning correct? → Check step-level correctness
   │   ├── YES: Execution/tool error → check tool logs
   │   └── NO: Where did reasoning go wrong?
   │       ├── First error at step t → analyze step t
   │       │   ├── Was input to step t correct?
   │       │   │   ├── YES: Model/prompt issue at step t
   │       │   │   └── NO: Error propagated from earlier step
   │       │   └── Was the tool selection correct?
   │       │       ├── YES: Parameter error → prompt issue
   │       │       └── NO: Tool selection logic → planning prompt
   │       └── Was context/retrieval correct?
   │           ├── YES: Model reasoning error
   │           └── NO: RAG/retrieval issue

2. TASK NOT COMPLETED (timeout/loop)
   ├── Did agent loop? → loop detection in trajectory
   │   ├── YES: Planning failure → check loop cause
   │   └── NO: Was it stuck at a specific step?
   │       ├── Tool failure → check tool availability/errors
   │       └── Model failure → check for generation issues

3. SAFETY VIOLATION
   ├── Which guardrail should have caught it?
   │   ├── Input guardrail → detection model issue
   │   ├── Output guardrail → filtering model issue
   │   └── Action guardrail → permission model issue
   └── Was the violation novel or known type?
       ├── Novel → add to training data
       └── Known → tune threshold

4. POOR QUALITY (correct but suboptimal)
   ├── Compared to optimal trajectory:
   │   ├── Too many steps → efficiency issue in planning
   │   ├── Unnecessary tool calls → tool selection issue
   │   ├── Verbose output → synthesis prompt issue
   │   └── Missing details → context/retrieval issue
```

**Automated Attribution via Ablation:**

```python
class FailureAttributor:
    async def attribute(self, failed_trajectory: ReplayRecord) -> Attribution:
        """Systematically identify root cause of failure."""
        
        # Step 1: Identify the first failing step
        first_error_step = self.find_first_error(failed_trajectory)
        
        # Step 2: Check if input to that step was correct
        input_correct = self.verify_step_input(
            failed_trajectory, first_error_step
        )
        
        # Step 3: Run component-level ablations
        ablation_results = {}
        
        # Ablation 1: Replace model with stronger model
        ablation_results["model"] = await self.test_with_stronger_model(
            failed_trajectory, first_error_step
        )
        
        # Ablation 2: Provide perfect tool outputs
        ablation_results["tool"] = await self.test_with_oracle_tools(
            failed_trajectory, first_error_step
        )
        
        # Ablation 3: Improve prompt with more context
        ablation_results["prompt"] = await self.test_with_enhanced_prompt(
            failed_trajectory, first_error_step
        )
        
        # Ablation 4: Provide perfect retrieval
        ablation_results["retrieval"] = await self.test_with_oracle_retrieval(
            failed_trajectory, first_error_step
        )
        
        # Determine root cause
        root_cause = max(
            ablation_results.items(), 
            key=lambda x: x[1].improvement
        )
        
        return Attribution(
            failed_step=first_error_step,
            root_cause=root_cause[0],
            ablation_results=ablation_results,
            confidence=self.compute_attribution_confidence(ablation_results),
            recommended_fix=self.suggest_fix(root_cause[0], ablation_results)
        )
```

**Attribution Scorecard (aggregated over many failures):**

| Component | Failure Attribution % | Trend | Priority Fix |
|---|---|---|---|
| Model reasoning | 35% | ↑ Increasing | Improve CoT prompts |
| Tool selection | 20% | → Stable | Add tool description clarity |
| RAG retrieval | 18% | ↓ Decreasing | Continue embedding improvement |
| Parameter specification | 12% | → Stable | Add parameter validation |
| Guardrail false positive | 8% | ↑ Increasing | Tune guardrail thresholds |
| Environment/API errors | 5% | → Stable | Add retry logic |
| Architecture (loops, dead ends) | 2% | ↓ Decreasing | Planning improvements |

The attribution scorecard guides engineering investment: the 35% attribution to model reasoning suggests that prompt engineering or model upgrade will have the highest ROI.

---

**Chapter Summary.** Evaluation and monitoring of agentic systems requires a fundamentally richer framework than standard LLM evaluation:

1. **Multi-dimensional assessment** across quality, efficiency, safety, and reliability—no single metric suffices.
2. **Trajectory-level evaluation** beyond output-level judgment, analyzing action sequences, error recovery, and tool-use efficiency.
3. **Statistical rigor** with appropriate sample sizes, confidence intervals, and multiple-comparison corrections.
4. **Production observability** via distributed tracing, structured logging, and real-time anomaly detection.
5. **Continuous evaluation** through CI/CD gates, canary deployments, A/B testing, and living golden datasets.
6. **Systematic debugging** through trajectory replay, counterfactual analysis, and component-level failure attribution.
7. **LLM-as-Judge with bias mitigation** provides scalable evaluation, calibrated against periodic human evaluation to maintain accuracy.