

# Chapter 5: Planning

---

## 5.1 Definition and Formal Framework

### 5.1.1 What is Planning in Agentic AI

Planning is the computational process by which an agent synthesizes a structured sequence or policy of actions that transforms the world from a current state to a desired goal state, subject to environmental constraints, action preconditions, resource budgets, and uncertainty. In the context of agentic AI powered by large language models (LLMs), planning transcends classical symbolic approaches—it becomes a hybrid cognitive capability where the LLM simultaneously serves as the **domain modeler**, the **heuristic evaluator**, the **plan generator**, and the **plan critic**.

Formally, planning in agentic AI can be characterized as the problem of computing a mapping:
![planning](./asserts/planning.png)
$$
\pi^*: \mathcal{S} \times \mathcal{G} \rightarrow \mathcal{A}^*
$$

where $\mathcal{S}$ is the perceivable state space, $\mathcal{G}$ is the goal specification space, and $\mathcal{A}^*$ denotes the Kleene closure over the action space (i.e., finite sequences of actions). The optimality criterion is defined over a cost or reward function:

$$
\pi^* = \arg\min_{\pi} \sum_{t=0}^{T} c(s_t, a_t) \quad \text{subject to} \quad s_T \in G
$$

**Key distinctions of planning in LLM-based agents:**

| Property | Classical AI Planning | LLM-Based Agent Planning |
|---|---|---|
| State representation | Symbolic propositions | Natural language + latent embeddings |
| Action space | Finite, enumerated | Open-ended, tool-augmented, API calls |
| Transition model | Explicit PDDL definitions | Implicit in LLM world knowledge |
| Search strategy | Systematic (A*, STRIPS) | Prompt-driven, sampling-based |
| Uncertainty handling | MDP / POMDP formalism | In-context stochastic reasoning |
| Plan verification | Theorem proving | Self-critique, execution feedback |

**Three fundamental requirements** distinguish genuine planning from mere next-token prediction:

1. **Anticipation**: The agent must reason about future states that do not yet exist, projecting the consequences of candidate actions before executing them.
2. **Commitment with flexibility**: The agent must produce an executable structure (a plan) while retaining the ability to revise it under new observations.
3. **Goal-directedness**: Every action in the plan must be justifiable as contributing to satisfying the goal specification $G$, either directly or instrumentally.

Planning is what separates a **reactive chatbot** (which merely responds to the current input) from an **autonomous agent** (which pursues multi-step objectives through deliberate coordination of actions over time).

---

### 5.1.2 Classical Planning Formalism

The canonical formal framework for deterministic planning is the **state-transition system**:

$$
\Pi = \langle S, A, \gamma, s_0, G \rangle
$$

**Component-by-component specification:**

**1. State Space $S$**

The state space $S$ is the set of all possible world configurations. In propositional planning, each state $s \in S$ is a truth assignment over a finite set of propositions $\mathcal{P} = \{p_1, p_2, \ldots, p_n\}$:

$$
s \subseteq \mathcal{P}, \quad |S| = 2^{|\mathcal{P}|}
$$

This exponential blowup is the fundamental source of computational intractability. In PDDL (Planning Domain Definition Language), states are represented as conjunctions of ground atoms:

```
(:init (on A B) (on B Table) (clear A) (arm-empty))
```

In LLM agents, the state is typically a **natural language description** augmented with structured observations:

$$
s_t = \text{NL-Description}(o_t) \oplus \text{Memory}_t \oplus \text{ToolOutputs}_t
$$

**2. Action Space $A$**

Each action $a \in A$ is defined by a triple:

$$
a = \langle \text{Pre}(a), \text{Eff}^+(a), \text{Eff}^-(a) \rangle
$$

- $\text{Pre}(a) \subseteq \mathcal{P}$: preconditions that must hold in the current state for $a$ to be applicable.
- $\text{Eff}^+(a) \subseteq \mathcal{P}$: add effects—propositions that become true after execution.
- $\text{Eff}^-(a) \subseteq \mathcal{P}$: delete effects—propositions that become false after execution.

An action $a$ is **applicable** in state $s$ if and only if:

$$
\text{Pre}(a) \subseteq s
$$

In LLM agents, actions include natural-language-specified tool calls, code generation, API invocations, and sub-agent delegations. The preconditions and effects are **implicitly encoded** in the LLM's parametric knowledge rather than explicitly declared.

**3. Transition Function $\gamma$**

The deterministic transition function specifies state evolution:

$$
\gamma: S \times A \rightarrow S
$$

$$
\gamma(s, a) = \begin{cases} (s \setminus \text{Eff}^-(a)) \cup \text{Eff}^+(a) & \text{if } \text{Pre}(a) \subseteq s \\ \text{undefined} & \text{otherwise} \end{cases}
$$

For LLM agents, $\gamma$ is approximated by the agent's **world model**—either the LLM's internal simulation of consequences or the actual environment execution followed by observation.

**4. Initial State $s_0$**

The initial state $s_0 \in S$ is the fully specified starting configuration of the world. In LLM agents, this is constructed from the user's task specification, the current environment observation, and any pre-loaded memory or context.

**5. Goal States $G$**

The goal is a partial specification $G \subseteq \mathcal{P}$ such that a state $s$ is a goal state if and only if:

$$
G \subseteq s
$$

In LLM agents, goal specifications are often **underspecified natural language instructions** that require the agent to infer the complete goal predicate set. For example, "book me a flight to Tokyo next Tuesday" implicitly requires satisfying constraints on dates, airports, user preferences, and payment.

**A solution (plan)** is a sequence of actions $\pi = \langle a_1, a_2, \ldots, a_n \rangle$ such that:

$$
s_0 \xrightarrow{a_1} s_1 \xrightarrow{a_2} s_2 \xrightarrow{} \cdots \xrightarrow{a_n} s_n \quad \text{where} \quad G \subseteq s_n
$$

**Complexity results:**

| Problem | Complexity |
|---|---|
| Plan existence (STRIPS) | PSPACE-complete |
| Optimal plan (STRIPS) | PSPACE-complete |
| Bounded-length plan | NP-complete |
| Plan verification | P (polynomial) |

These complexity results motivate the use of LLMs as **learned heuristic generators** that can bypass systematic search through informed plan proposal, dramatically reducing the effective search space.

---

### 5.1.3 Planning Under Uncertainty in LLM Agents

Real-world agentic tasks introduce multiple forms of uncertainty that the classical deterministic formalism cannot capture. LLM agents must plan under at least **four distinct uncertainty types**:

**Type 1: Outcome Uncertainty (Stochastic Transitions)**

Actions may produce different outcomes probabilistically. This extends the planning problem to a **Markov Decision Process (MDP)**:

$$
\Pi_{\text{MDP}} = \langle S, A, T, R, \gamma_d, s_0 \rangle
$$

where $T: S \times A \times S \rightarrow [0, 1]$ is the transition probability function:

$$
T(s' | s, a) = P(s_{t+1} = s' | s_t = s, a_t = a)
$$

and $R: S \times A \rightarrow \mathbb{R}$ is the reward function, $\gamma_d \in [0,1)$ is the discount factor. The optimal policy maximizes the expected discounted return:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma_d^t R(s_t, a_t)\right]
$$

In LLM agents, outcome uncertainty manifests when tool calls may fail, web pages may have changed, or API responses are nondeterministic.

**Type 2: Observational Uncertainty (Partial Observability)**

The agent does not have full access to the true state. This extends to a **Partially Observable MDP (POMDP)**:

$$
\Pi_{\text{POMDP}} = \langle S, A, T, R, \Omega, O, \gamma_d, b_0 \rangle
$$

where $\Omega$ is the observation space, $O: S \times A \times \Omega \rightarrow [0,1]$ is the observation function, and $b_0$ is the initial belief state (a probability distribution over $S$):

$$
b_0(s) = P(s_0 = s)
$$

The belief update upon observing $o$ after taking action $a$:

$$
b'(s') = \eta \cdot O(o | s', a) \sum_{s \in S} T(s' | s, a) \cdot b(s)
$$

where $\eta$ is a normalizing constant. LLM agents operate in partially observable environments by default—they cannot see the full state of a website, a codebase, or a user's intent.

**Type 3: Model Uncertainty (Unknown Dynamics)**

The agent does not know the true transition function $T$ or reward function $R$. This is the domain of **model-based reinforcement learning** and **Bayesian planning**, where the agent maintains a posterior over possible world models:

$$
P(\mathcal{M} | \mathcal{D}) \propto P(\mathcal{D} | \mathcal{M}) P(\mathcal{M})
$$

LLMs encode an approximate world model in their parameters, but this model is trained on a static corpus and may be inaccurate for novel environments. The agent must therefore balance **exploitation** of its existing world model with **exploration** to refine it.

**Type 4: Goal Uncertainty (Ambiguous Specifications)**

The user's goal may be ambiguous, incomplete, or evolving. The agent must plan over a **distribution of possible goals**:

$$
P(G | \text{instruction}) = \text{LLM-Inference}(\text{instruction})
$$

This motivates **information-gathering actions**—clarification questions, exploratory probes—that reduce goal uncertainty before committing to expensive plan execution.

**LLM-Specific Uncertainty: Sampling Stochasticity**

A unique source of uncertainty in LLM agents is the **stochastic nature of autoregressive generation**. Given the same prompt, temperature-based sampling produces different plans:

$$
P(a_t | s_t, \text{prompt}) = \text{softmax}\left(\frac{\text{logits}(a_t)}{\tau}\right)
$$

where $\tau$ is the temperature. This means the planning process itself is a random variable, necessitating strategies such as majority voting, self-consistency, or best-of-$N$ sampling.

---

### 5.1.4 Planning vs. Reasoning vs. Acting

These three cognitive capabilities are frequently conflated. Rigorous disambiguation is essential:

**Formal Definitions:**

| Capability | Definition | Formal Characterization |
|---|---|---|
| **Reasoning** | Deriving new conclusions from existing information using logical or probabilistic inference | $\mathcal{R}: \text{Premises} \rightarrow \text{Conclusions}$ |
| **Planning** | Constructing a sequence of actions to achieve a goal in a stateful environment | $\mathcal{P}: (s_0, G) \rightarrow \langle a_1, \ldots, a_n \rangle$ |
| **Acting** | Executing a specific action in the environment and observing the result | $\mathcal{E}: (s, a) \rightarrow (s', o)$ |

**Relationship diagram:**

```
                    ┌──────────────┐
                    │   REASONING  │ (inference over knowledge)
                    │  "What is    │
                    │   true?"     │
                    └──────┬───────┘
                           │ informs
                           ▼
                    ┌──────────────┐
                    │   PLANNING   │ (action sequence synthesis)
                    │  "What to    │
                    │   do?"       │
                    └──────┬───────┘
                           │ produces
                           ▼
                    ┌──────────────┐
                    │    ACTING    │ (environment interaction)
                    │  "Do it."   │
                    └──────┬───────┘
                           │ generates
                           ▼
                    ┌──────────────┐
                    │ OBSERVATION  │ (feedback)
                    │  "What       │
                    │ happened?"   │
                    └──────────────┘
```

**Critical distinctions:**

1. **Reasoning without Planning**: Answering a factual question ("What is the capital of France?") involves reasoning but no planning—there is no state to transform, no actions to sequence.

2. **Planning without Deep Reasoning**: A lookup-table planner that matches state patterns to pre-computed plans does planning without the kind of inferential reasoning LLMs perform.

3. **Acting without Planning**: A purely reactive agent (e.g., a thermostat) acts based on current stimulus without constructing any future-oriented plan.

4. **Planning requires Reasoning about Action Consequences**: For each candidate action $a$, the planner must reason about the resulting state $\gamma(s, a)$ and evaluate whether it progresses toward $G$. Planning thus **subsumes** a specific form of reasoning—reasoning about hypothetical future states.

5. **The ReAct Insight**: The ReAct framework (Yao et al., 2023) demonstrates that interleaving reasoning traces with acting steps produces superior performance compared to either in isolation. The reasoning trace serves as a **micro-planning** step that contextualizes each action:

$$
\underbrace{\text{Thought}_t}_{\text{Reasoning}} \rightarrow \underbrace{\text{Action}_t}_{\text{Acting}} \rightarrow \underbrace{\text{Observation}_t}_{\text{Perceiving}} \rightarrow \text{Thought}_{t+1} \rightarrow \cdots
$$

**The planning spectrum in LLM agents:**

$$
\underbrace{\text{Reactive}}_{\text{No planning}} \longleftrightarrow \underbrace{\text{One-step lookahead}}_{\text{Minimal planning}} \longleftrightarrow \underbrace{\text{Full plan generation}}_{\text{Open-loop planning}} \longleftrightarrow \underbrace{\text{Search + Replanning}}_{\text{Closed-loop planning}}
$$

---

## 5.2 Planning Paradigms for LLM Agents

### 5.2.1 Task Decomposition (Top-Down Planning)

Task decomposition is the hierarchical process of breaking a complex task $T$ into simpler, independently solvable subtasks. This is the most natural planning paradigm for LLM agents because it mirrors how humans approach complex problems and aligns well with the compositional nature of language.

#### Hierarchical Task Networks (HTN)

HTN planning operates by recursively decomposing **compound tasks** into **primitive tasks** using a library of **decomposition methods**. Formally:

$$
\text{HTN} = \langle S, A, M, s_0, T_{\text{root}} \rangle
$$

- $S$: State space (as before)
- $A$: Primitive actions (directly executable)
- $M$: Set of decomposition methods, where each method $m \in M$ maps a compound task to an ordered set of subtasks:

$$
m: T_{\text{compound}} \times S \rightarrow \langle T_1, T_2, \ldots, T_k \rangle
$$

- $s_0$: Initial state
- $T_{\text{root}}$: The top-level task to accomplish

**Decomposition proceeds recursively until all tasks are primitive:**

```
T_root
├── T_1 (compound)
│   ├── T_1.1 (primitive: action a₁)
│   └── T_1.2 (compound)
│       ├── T_1.2.1 (primitive: action a₂)
│       └── T_1.2.2 (primitive: action a₃)
├── T_2 (primitive: action a₄)
└── T_3 (compound)
    ├── T_3.1 (primitive: action a₅)
    └── T_3.2 (primitive: action a₆)
```

**Final plan (reading leaves left-to-right):** $\pi = \langle a_1, a_2, a_3, a_4, a_5, a_6 \rangle$

In LLM agents, the decomposition methods $M$ are not pre-specified—they are **generated by the LLM** conditioned on the task description and the current state. This is the fundamental insight: the LLM serves as a **universal decomposition oracle**.

**Prompt pattern for HTN-style decomposition:**

```
Given the task: "{T_root}"
Current state: "{s_0}"
Available tools: [tool_1, tool_2, ..., tool_n]

Break this task into a sequence of subtasks. For each subtask,
specify whether it can be executed directly (primitive) or needs
further decomposition (compound).
```

#### Recursive Decomposition

The recursive decomposition process is formally:

$$
T \rightarrow \{T_1, T_2, \ldots, T_k\}
$$

with the **composability constraint** that successful execution of all subtasks in sequence must accomplish the parent task:

$$
\text{Execute}(T) \Leftrightarrow \text{Execute}(T_1) \wedge \text{Execute}(T_2) \wedge \cdots \wedge \text{Execute}(T_k)
$$

**Recursive depth control** is critical. Without termination conditions, the LLM may decompose indefinitely. Practical strategies:

1. **Depth limit**: Stop decomposing after $d_{\max}$ levels.
2. **Primitive detection**: The LLM judges whether a subtask is directly executable by available tools.
3. **Complexity threshold**: Subtasks below a token-estimated complexity threshold are treated as primitive.

The recursive decomposition function:

$$
\text{Decompose}(T, d) = \begin{cases}
\{T\} & \text{if } \text{IsPrimitive}(T) \vee d \geq d_{\max} \\
\bigcup_{i=1}^{k} \text{Decompose}(T_i, d+1) & \text{where } \{T_1, \ldots, T_k\} = \text{LLM}(T)
\end{cases}
$$

**Example:**

Task: "Write a research paper on transformer efficiency."

```
Level 0: Write a research paper on transformer efficiency
Level 1: ├── Conduct literature survey
         ├── Design experiments  
         ├── Run experiments
         ├── Analyze results
         └── Write manuscript
Level 2: ├── Conduct literature survey
         │   ├── Search arxiv for relevant papers (primitive: search_tool)
         │   ├── Read and summarize top 20 papers (primitive: read + LLM)
         │   └── Identify research gaps (primitive: LLM reasoning)
         ├── Design experiments
         │   ├── Select baseline models (primitive: LLM reasoning)
         │   ├── Define evaluation metrics (primitive: LLM reasoning)
         │   └── Write experiment configuration (primitive: code_gen)
         ...
```

#### Plan-and-Execute Pattern

The Plan-and-Execute pattern separates plan construction from plan execution into distinct phases with distinct LLM invocations:

**Phase 1 — Planning (single LLM call):**

$$
\text{Plan} = \text{LLM}_{\text{planner}}(T, s_0, \text{Tools})
$$

Output: An ordered list of steps $\langle \text{step}_1, \text{step}_2, \ldots, \text{step}_n \rangle$

**Phase 2 — Execution (per-step LLM calls):**

For each step $i = 1, \ldots, n$:

$$
(a_i, o_i) = \text{LLM}_{\text{executor}}(\text{step}_i, s_i, \text{history}_{<i})
$$

$$
s_{i+1} = \text{Environment}(s_i, a_i)
$$

**Advantages over monolithic generation:**

1. **Separation of concerns**: The planner can focus on high-level strategy without being burdened by execution details.
2. **Reusability**: The same plan structure can be executed with different executor models or tools.
3. **Debuggability**: The plan can be inspected and validated before any execution occurs.
4. **Efficiency**: If execution of step $i$ fails, only step $i$ needs to be retried or the plan modified from step $i$ onward, not the entire plan regenerated.

**LangGraph implementation pattern:**

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, task: str, state: dict) -> str:
        # Phase 1: Generate plan
        plan = self.planner.invoke(
            f"Create a step-by-step plan for: {task}\n"
            f"Available tools: {self.tools}\n"
            f"Current state: {state}"
        )
        steps = parse_plan(plan)

        # Phase 2: Execute each step
        results = []
        for step in steps:
            action = self.executor.invoke(
                f"Execute this step: {step}\n"
                f"Previous results: {results}\n"
                f"Available tools: {self.tools}"
            )
            observation = execute_action(action, self.tools)
            results.append((step, action, observation))

            # Optional: replan on failure
            if is_failure(observation):
                remaining = self.replan(task, results, steps)
                steps = remaining

        return synthesize_results(results)
```

---

### 5.2.2 Sequential Planning

Sequential planning constructs a **linear chain of actions** without hierarchical decomposition. The plan is a flat sequence:

$$
\pi = \langle a_1, a_2, \ldots, a_n \rangle
$$

where each action $a_i$ is conditioned on the initial state and the assumed effects of all preceding actions.

#### Chain-of-Actions Generation

In chain-of-actions generation, the LLM produces the entire action sequence in a single autoregressive pass:

$$
P(\pi | s_0, G) = \prod_{i=1}^{n} P(a_i | a_{<i}, s_0, G)
$$

This is the simplest planning approach but has significant limitations:

1. **No intermediate feedback**: The entire plan is generated before any action is executed, so environmental dynamics cannot be incorporated.
2. **Error compounding**: If action $a_i$ is incorrect, all subsequent actions $a_{i+1}, \ldots, a_n$ are conditioned on a flawed premise.
3. **Length limitations**: The plan length is bounded by the LLM's context window.

**Mitigation: Plan Conditioning on Intermediate Observations**

To address error compounding, the plan can be regenerated or continued after each action execution:

$$
a_{i+1} = \text{LLM}(s_0, G, a_1, o_1, a_2, o_2, \ldots, a_i, o_i)
$$

This transforms sequential planning into **iterative planning** (Section 5.2.3).

#### Step-by-Step Plan Construction

A more structured approach prompts the LLM to generate each action one at a time, validating preconditions before proceeding:

```
Step 1: [Action] → Check: Is the precondition satisfied? 
        If yes → record effect, proceed to Step 2
        If no  → modify action or insert prerequisite action
Step 2: ...
```

**Formal algorithm:**

$$
\textbf{Algorithm: StepwisePlanConstruction}
$$

```
Input: s₀, G, LLM, max_steps
Output: plan π

s ← s₀
π ← []
for i = 1 to max_steps:
    aᵢ ← LLM.propose_action(s, G, π)
    if LLM.check_preconditions(aᵢ, s):
        s ← LLM.predict_effects(aᵢ, s)
        π.append(aᵢ)
        if G ⊆ s:
            return π  // Goal achieved
    else:
        aᵢ' ← LLM.fix_action(aᵢ, s)  // Repair
        ...
return FAILURE
```

---

### 5.2.3 Iterative Planning (Closed-Loop)

Iterative planning is the **closed-loop** paradigm where the agent continuously interleaves planning with execution and observation, adjusting its plan based on real environmental feedback.

#### Observe → Plan → Act → Observe Cycle

The core loop is:

$$
\text{for } t = 0, 1, 2, \ldots: \quad o_t \xrightarrow{\text{plan}} \pi_t \xrightarrow{\text{act}} a_t \xrightarrow{\text{env}} o_{t+1}
$$

At each time step $t$, the agent:

1. **Observes**: Receives observation $o_t$ from the environment.
2. **Plans**: Generates or updates the current plan $\pi_t$ conditioned on all history $h_t = (o_0, a_0, o_1, a_1, \ldots, o_t)$.
3. **Acts**: Executes the first action $a_t = \pi_t[0]$ from the current plan.
4. **Receives feedback**: The environment transitions to a new state, yielding observation $o_{t+1}$.

**Formal closed-loop planning:**

$$
\pi_t = \text{LLM}(h_t, G) = \text{LLM}(o_0, a_0, o_1, \ldots, o_t, G)
$$

$$
a_t = \text{head}(\pi_t)
$$

$$
o_{t+1} = \text{Env}(a_t)
$$

This is the dominant paradigm in production agent systems because it provides **robustness to unexpected outcomes**. If a tool call fails, the agent observes the failure and replans accordingly.

#### Replanning on Observation Changes

The decision to replan can be formalized as a **plan validity check**:

$$
\text{Replan}(t) = \begin{cases}
\text{True} & \text{if } o_t \neq \hat{o}_t \text{ (observation diverges from prediction)} \\
\text{True} & \text{if } \exists a \in \pi_{t-1}[1:] : \text{Pre}(a) \not\subseteq \hat{s}(o_t) \\
\text{False} & \text{otherwise (plan remains valid)}
\end{cases}
$$

where $\hat{o}_t$ is the observation the agent expected and $\hat{s}(o_t)$ is the state estimated from the actual observation $o_t$.

**Selective replanning** avoids the cost of full plan regeneration. The agent can:

1. Continue with the existing plan if observations match expectations.
2. Modify only the affected portion of the plan.
3. Fully regenerate the plan from scratch if the deviation is too large.

The replanning decision involves a **cost-benefit trade-off**:

$$
\text{Replan if } \quad \text{Cost}_{\text{suboptimal\_plan}} > \text{Cost}_{\text{replanning}} + \text{Cost}_{\text{delay}}
$$

---

### 5.2.4 Reactive Planning

Reactive planning operates **without explicit lookahead**—the agent selects actions based solely on the current observation and internal reasoning state, without constructing a multi-step plan.

#### Stimulus-Response Without Lookahead

A purely reactive agent implements a policy mapping:

$$
\pi_{\text{reactive}}: \mathcal{O} \rightarrow \mathcal{A}
$$

No future states are simulated, no plan is stored. Each action is selected greedily based on the current observation. This is computationally cheap but can fail on tasks requiring **coordination of multiple dependent actions**.

**When reactive planning suffices:**
- Tasks where each step is largely independent of future steps.
- Environments with immediate, informative feedback.
- Simple tool-use patterns (e.g., single API call per user request).

**When reactive planning fails:**
- Tasks requiring resource allocation across steps.
- Tasks with irreversible actions where premature commitment is costly.
- Long-horizon tasks where early actions constrain later possibilities.

#### ReAct Framework: Interleaved Reasoning + Acting

The **ReAct** framework (Yao et al., 2023) augments reactive planning with explicit reasoning traces. The agent alternates between:

1. **Thought** (reasoning): The LLM generates a natural language reasoning trace that analyzes the current observation, assesses progress toward the goal, and decides what action to take next.
2. **Action**: The agent executes a tool call or environment interaction.
3. **Observation**: The environment returns a result.

**Formal trace structure:**

$$
\text{Thought}_1 \rightarrow \text{Action}_1 \rightarrow \text{Observation}_1 \rightarrow \text{Thought}_2 \rightarrow \text{Action}_2 \rightarrow \text{Observation}_2 \rightarrow \cdots
$$

**Mathematical characterization:**

At each step $t$, the LLM generates:

$$
(\text{thought}_t, a_t) = \text{LLM}(\text{prompt}, \text{thought}_{<t}, a_{<t}, o_{<t})
$$

$$
o_t = \text{Env}(a_t)
$$

The thought trace $\text{thought}_t$ serves as a **lightweight planning step**—it does not produce a full plan but reasons about the immediate next action in the context of the overall goal.

**ReAct example:**

```
Task: What is the elevation range for the area that the 
      eastern sector of the Colorado orogeny extends into?

Thought 1: I need to find the eastern sector of the Colorado 
           orogeny and what area it extends into.
Action 1:  Search("Colorado orogeny eastern sector")
Obs 1:     The Colorado orogeny extended into the High Plains...

Thought 2: The eastern sector extends into the High Plains. 
           Now I need the elevation range of the High Plains.
Action 2:  Search("High Plains elevation range")
Obs 2:     The High Plains rise from around 1,800 ft to 7,000 ft.

Thought 3: The elevation range is 1,800 to 7,000 feet.
Action 3:  Finish("1,800 to 7,000 feet")
```

**ReAct vs. Pure CoT vs. Pure Acting:**

| Method | Reasoning | Acting | Grounding | Planning Depth |
|---|---|---|---|---|
| Chain-of-Thought (CoT) | ✓ | ✗ | ✗ (hallucination risk) | Implicit |
| Act-only | ✗ | ✓ | ✓ | None |
| ReAct | ✓ | ✓ | ✓ | One-step lookahead |

ReAct achieves a balance: reasoning provides the **why** behind each action, while acting provides **grounding** in real observations that prevent hallucination drift.

---

## 5.3 Search-Based Planning

Search-based planning treats the planning problem as a **graph search** problem, systematically exploring the space of possible thought sequences, action sequences, or partial plans to find optimal or satisficing solutions. LLMs serve as both the **node generator** (expanding the search frontier) and the **evaluation function** (assessing the quality of each node).

### 5.3.1 Tree of Thoughts (ToT)

The Tree of Thoughts framework (Yao et al., 2024) generalizes Chain-of-Thought (CoT) prompting from a single linear chain into a **tree-structured exploration** over intermediate reasoning steps (thoughts).

**Formal definition:**

Let a **thought** $z$ be a coherent unit of reasoning (a sentence, a paragraph, a partial solution). A thought tree is:

$$
\mathcal{T} = (V, E)
$$

where each node $v \in V$ is a state $s = [x, z_1, \ldots, z_i]$ consisting of the input $x$ and a sequence of thoughts, and each edge represents a thought extension $z_{i+1}$.

The three key operations:

**1. Thought Generator** $G(p_\theta, s, k)$: Given the current state $s$ and the LLM $p_\theta$, generate $k$ candidate next thoughts:

$$
z_{i+1}^{(j)} \sim p_\theta(z_{i+1} | s), \quad j = 1, \ldots, k
$$

Two strategies:
- **Sample**: Draw $k$ i.i.d. samples from the LLM with temperature $\tau > 0$.
- **Propose**: Prompt the LLM to generate $k$ distinct candidates in a single call using a prompt like "Generate $k$ different approaches."

**2. State Evaluator** $V(p_\theta, s)$: Assess the promise of a partial reasoning state:

$$
V(s) = p_\theta(\text{"this reasoning path leads to a correct solution"} | s)
$$

Or quantitatively via voting:

$$
V(s) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{LLM}_i(s) = \text{"good"}]
$$

This evaluator serves as the **heuristic function** that guides search.

**3. Search Algorithm**: Apply standard graph search over the thought tree.

#### BFS/DFS Over Thought Branches

**BFS (Breadth-First Search) over ToT:**

```
Input: x, LLM p_θ, breadth b, depth d
Initialize: S₀ = {[x]}  (root state)

For each depth level i = 1, ..., d:
    S_candidates = {}
    For each state s ∈ S_{i-1}:
        Generate b thoughts: z₁, ..., z_b ~ G(p_θ, s, b)
        S_candidates ∪= {[s, z_j] for j=1,...,b}
    
    Evaluate: V(s') for each s' ∈ S_candidates
    S_i = top-b states from S_candidates by V(s')

Return: best state in S_d
```

**DFS (Depth-First Search) over ToT:**

```
Input: x, LLM p_θ, depth limit d, threshold V_thresh

Function DFS(s, depth):
    if depth == d:
        return evaluate_final(s)
    
    Generate thoughts: z₁, ..., z_b ~ G(p_θ, s, b)
    For each zⱼ:
        s' = [s, zⱼ]
        if V(s') ≥ V_thresh:
            result = DFS(s', depth + 1)
            if result is solution:
                return result
    
    return FAILURE (backtrack)
```

**Comparison:**

| Strategy | Completeness | Memory | Best For |
|---|---|---|---|
| BFS | Yes (if $b$ sufficient) | $O(b^d)$ | Shallow trees, need diversity |
| DFS | No (can miss branches) | $O(b \cdot d)$ | Deep reasoning, memory-constrained |

#### Value Function for Thought Evaluation

The state value function estimates the expected cumulative reward achievable from a given thought state:

$$
V(s) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t \,\Big|\, s_0 = s\right]
$$

In the ToT context, $r_t$ is the reward signal for the quality of reasoning at step $t$, and $\gamma \in (0, 1]$ is the discount factor.

**Practical implementations of $V(s)$:**

1. **LLM-as-Judge**: Prompt the LLM to rate the partial solution on a scale of 1-10.

$$
V(s) = \frac{1}{10} \cdot \text{LLM}(\text{"Rate this partial solution 1-10: "} + s)
$$

2. **Classification-based**: Prompt the LLM to classify the thought as "sure/likely/unlikely" to lead to a correct solution:

$$
V(s) = P_\theta(\text{"sure"} | s) + 0.5 \cdot P_\theta(\text{"likely"} | s)
$$

3. **Outcome-based simulation**: Complete the reasoning from $s$ multiple times and count successes:

$$
V(s) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{Complete}(s, i) = \text{correct}]
$$

This is essentially a **Monte Carlo estimate** of the value function.

---

### 5.3.2 Graph of Thoughts (GoT)

The Graph of Thoughts framework (Besta et al., 2024) generalizes ToT by allowing **non-tree structures**—specifically, thought nodes can have multiple parents, enabling **thought merging**, **refinement**, and **loop** operations.

#### Non-Linear Thought Structures

Formally, a GoT is a directed acyclic graph (DAG):

$$
\mathcal{G} = (V, E, \mathcal{O})
$$

where:
- $V$: Set of thought vertices, each representing a partial solution or reasoning step.
- $E \subseteq V \times V$: Directed edges representing derivation relationships.
- $\mathcal{O}$: Set of graph operations (transformations).

**Four fundamental graph operations:**

| Operation | Description | Graph Effect |
|---|---|---|
| **Generate** | Create new thoughts from existing ones | Add child nodes |
| **Aggregate** | Merge multiple thoughts into one | Create node with multiple parents |
| **Refine** | Improve a thought based on feedback | Add self-loop or replacement node |
| **Score** | Evaluate thought quality | Assign weights to nodes |

#### Thought Merging and Refinement

**Aggregation (Merging):** Given thoughts $z_1, z_2, \ldots, z_m$ that represent partial solutions to complementary subproblems, produce a merged thought:

$$
z_{\text{merged}} = \text{LLM}(\text{"Combine these partial solutions: "} z_1, z_2, \ldots, z_m)
$$

This is not possible in tree structures where each node has exactly one parent.

**Example — Sorting with GoT:**

To sort a list $[5, 3, 8, 1, 9, 2, 7, 4, 6]$:

```
Phase 1 (Generate - Decompose):
  Node A: Sort [5, 3, 8]    → [3, 5, 8]
  Node B: Sort [1, 9, 2]    → [1, 2, 9]
  Node C: Sort [7, 4, 6]    → [4, 6, 7]

Phase 2 (Aggregate - Merge):
  Node D: Merge [3,5,8] + [1,2,9]  → [1, 2, 3, 5, 8, 9]
  
Phase 3 (Aggregate - Merge):
  Node E: Merge [1,2,3,5,8,9] + [4,6,7]  → [1,2,3,4,5,6,7,8,9]
  
Phase 4 (Refine - Verify):
  Node F: Verify Node E is correctly sorted → Confirmed
```

The graph structure:
```
A ──┐
    ├──→ D ──┐
B ──┘        ├──→ E ──→ F
C ───────────┘
```

This DAG structure enables **divide-and-conquer** strategies that trees cannot naturally represent.

**Refinement:** A thought $z$ is iteratively improved:

$$
z^{(k+1)} = \text{LLM}(\text{"Improve this solution: "} z^{(k)}, \text{"Feedback: "} \text{Critique}(z^{(k)}))
$$

This creates a chain:

$$
z^{(0)} \rightarrow z^{(1)} \rightarrow \cdots \rightarrow z^{(K)}
$$

where each refinement step reduces errors or improves quality, and $K$ is determined by a convergence criterion:

$$
\|z^{(k+1)} - z^{(k)}\|_{\text{semantic}} < \epsilon
$$

---

### 5.3.3 Monte Carlo Tree Search (MCTS) for Planning

MCTS is a **best-first search** algorithm that builds a search tree incrementally using random simulations (rollouts) to estimate the value of each node. When combined with LLMs, MCTS provides a principled framework for balancing **exploration** (trying new approaches) with **exploitation** (deepening promising approaches).

**The four phases of MCTS:**

```
        ┌────────────┐
        │  SELECTION  │──→ Traverse tree using UCB1
        └──────┬─────┘
               ▼
        ┌────────────┐
        │  EXPANSION  │──→ Add new child node(s)
        └──────┬─────┘
               ▼
        ┌────────────┐
        │  SIMULATION │──→ Rollout to terminal state
        │  (Rollout)  │
        └──────┬─────┘
               ▼
        ┌──────────────┐
        │ BACKPROPAGATION│──→ Update ancestor values
        └──────────────┘
```

#### UCB1 Selection

The Upper Confidence Bound for Trees (UCT) formula balances exploration and exploitation:

$$
\text{UCT}(s, a) = \bar{Q}(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}
$$

where:
- $\bar{Q}(s, a)$: Mean value (reward) of action $a$ from state $s$, estimated from previous rollouts.
- $N(s)$: Total visit count of state $s$.
- $N(s, a)$: Visit count of action $a$ from state $s$.
- $c$: Exploration constant controlling the exploration-exploitation trade-off. Commonly $c = \sqrt{2}$ for theoretical guarantees, tuned empirically in practice.

The first term exploits high-value actions; the second term explores under-visited actions. As $N(s, a) \rightarrow \infty$, the exploration bonus vanishes and the algorithm converges to the optimal action.

**Selection phase:** Starting from the root, repeatedly select the child with the highest UCT value until reaching a leaf node:

$$
a^* = \arg\max_{a \in A(s)} \text{UCT}(s, a)
$$

**Expansion phase:** At the leaf node, generate one or more child nodes using the LLM:

$$
\{z_1, z_2, \ldots, z_k\} = \text{LLM}(s_{\text{leaf}}, \text{"propose next steps"})
$$

**Simulation phase:** From the newly expanded node, simulate a complete trajectory to a terminal state using a rollout policy (see below).

**Backpropagation phase:** Update the value estimates of all ancestors:

$$
\bar{Q}(s, a) \leftarrow \bar{Q}(s, a) + \frac{1}{N(s, a)}\left(r - \bar{Q}(s, a)\right)
$$

$$
N(s, a) \leftarrow N(s, a) + 1, \quad N(s) \leftarrow N(s) + 1
$$

where $r$ is the reward obtained from the rollout.

#### Rollout Policies Using LLMs

The rollout policy $\pi_{\text{rollout}}$ determines how the simulation phase plays out from an expanded node to a terminal state. Traditional MCTS uses random rollouts, but LLMs enable **informed rollouts**:

**1. LLM-Greedy Rollout:**

$$
a_t = \arg\max_{a} P_\theta(a | s_t) \quad \text{(greedy decoding)}
$$

The LLM greedily completes the reasoning/plan from the current node. This provides high-quality value estimates but is expensive (one full LLM inference per rollout).

**2. LLM Value Network (No Rollout):**

Replace the rollout entirely with a learned value function:

$$
V_\phi(s) \approx \mathbb{E}_{\pi_{\text{rollout}}}[R | s_0 = s]
$$

This can be the LLM itself prompted to evaluate the state, or a separate smaller model trained to predict outcome quality.

**3. Hybrid Rollout:**

Use the LLM for the first $k$ steps of the rollout (where quality matters most), then switch to a cheaper heuristic or random policy:

$$
\pi_{\text{hybrid}}(s, t) = \begin{cases}
\text{LLM}(s) & \text{if } t \leq k \\
\text{random}(s) & \text{if } t > k
\end{cases}
$$

**MCTS for LLM Agent Planning — Algorithm:**

```python
class MCTSPlanner:
    def __init__(self, llm, c_explore=1.414, n_simulations=100):
        self.llm = llm
        self.c = c_explore
        self.n_sims = n_simulations
        self.Q = defaultdict(float)    # Q(s,a) values
        self.N_sa = defaultdict(int)   # Visit counts N(s,a)
        self.N_s = defaultdict(int)    # Visit counts N(s)
        self.children = {}             # Children of each node

    def search(self, root_state):
        for _ in range(self.n_sims):
            path = self._select(root_state)
            leaf = path[-1]
            child = self._expand(leaf)
            reward = self._simulate(child)
            self._backpropagate(path + [child], reward)
        
        # Return best action from root
        return max(self.children[root_state],
                   key=lambda a: self.Q[(root_state, a)])

    def _uct(self, state, action):
        if self.N_sa[(state, action)] == 0:
            return float('inf')
        exploit = self.Q[(state, action)] / self.N_sa[(state, action)]
        explore = self.c * sqrt(log(self.N_s[state]) / 
                                self.N_sa[(state, action)])
        return exploit + explore

    def _simulate(self, state):
        """LLM-based rollout"""
        completion = self.llm(f"Complete this plan and evaluate: {state}")
        reward = self.llm(f"Rate the quality 0-1: {completion}")
        return float(reward)
```

---

### 5.3.4 Beam Search Over Plan Space

Beam search maintains a fixed-size set of $B$ candidate plans (the **beam**) at each step, expanding only the most promising candidates:

$$
\textbf{Algorithm: Beam Search Planning}
$$

```
Input: s₀, G, LLM, beam_width B, max_depth D
Initialize: Beam₀ = {(s₀, [])}  // (state, plan_so_far)

For depth d = 1, ..., D:
    Candidates = {}
    For each (s, π) ∈ Beam_{d-1}:
        Generate k candidate actions: a₁, ..., aₖ ~ LLM(s, G)
        For each aⱼ:
            s' = predict_next_state(s, aⱼ)  // LLM-simulated
            score = evaluate(s', G)          // LLM-scored
            Candidates ∪= {(s', π + [aⱼ], score)}
    
    Beam_d = top-B candidates from Candidates by score
    
    If any (s, π) ∈ Beam_d satisfies G:
        return π

Return: best plan in Beam_D
```

**Scoring function for beam elements:**

$$
\text{Score}(s, \pi, G) = \alpha \cdot \text{GoalProximity}(s, G) + \beta \cdot \text{PlanCoherence}(\pi) - \lambda \cdot \text{Cost}(\pi)
$$

where:
- $\text{GoalProximity}(s, G)$: How close the current state is to the goal (LLM-estimated).
- $\text{PlanCoherence}(\pi)$: Internal consistency and logical flow of the plan.
- $\text{Cost}(\pi)$: Number of steps, token usage, or expected execution time.

**Trade-offs:**

| Beam Width $B$ | Computation | Plan Quality | Diversity |
|---|---|---|---|
| $B = 1$ (greedy) | Minimal | Often suboptimal | None |
| $B = 5$-$10$ | Moderate | Good | Moderate |
| $B = 50$+ | Expensive | Near-optimal | High |

---

### 5.3.5 A* Search with LLM Heuristics

A* search finds the **optimal plan** (shortest, lowest-cost) by maintaining a priority queue ordered by:

$$
f(n) = g(n) + h(n)
$$

where:
- $g(n)$: Actual cost from the start state to node $n$.
- $h(n)$: **Heuristic estimate** of the cost from $n$ to the nearest goal state.

**A* guarantees optimality** if $h(n)$ is **admissible** (never overestimates the true cost):

$$
h(n) \leq h^*(n) \quad \forall n
$$

where $h^*(n)$ is the true optimal cost-to-go.

**LLM as Heuristic Function:**

The key innovation in LLM-based A* is using the LLM to estimate $h(n)$:

$$
h_{\text{LLM}}(n) = \text{LLM}(\text{"Estimate the number of steps remaining to reach goal } G \text{ from state } n\text{"})
$$

**Challenge: Admissibility is not guaranteed.** LLMs may overestimate or underestimate the remaining cost. Strategies to address this:

1. **Conservative estimation**: Prompt the LLM to provide a lower bound: "What is the minimum number of steps needed?"
2. **Ensemble averaging**: Use multiple LLM calls and take the minimum estimate:

$$
h_{\text{ensemble}}(n) = \min_{i=1}^{K} h_{\text{LLM}}^{(i)}(n)
$$

3. **Weighted A***: Relax optimality for speed by using:

$$
f(n) = g(n) + w \cdot h(n), \quad w \geq 1
$$

With $w > 1$, the search is faster but the solution is at most $w$ times the optimal cost.

**Algorithm:**

```python
def a_star_llm(s0, goal, llm, action_generator):
    open_set = PriorityQueue()
    open_set.put((0, s0, []))      # (f-value, state, plan)
    g_score = {s0: 0}
    
    while not open_set.empty():
        f, current, plan = open_set.get()
        
        if is_goal(current, goal):
            return plan
        
        actions = llm.generate_actions(current, goal)
        for action in actions:
            next_state = llm.predict_state(current, action)
            tentative_g = g_score[current] + cost(action)
            
            if next_state not in g_score or tentative_g < g_score[next_state]:
                g_score[next_state] = tentative_g
                h = llm.estimate_cost_to_goal(next_state, goal)
                f_new = tentative_g + h
                open_set.put((f_new, next_state, plan + [action]))
    
    return FAILURE
```

**Comparison of search-based planning methods:**

| Method | Completeness | Optimality | Time Complexity | LLM Calls |
|---|---|---|---|---|
| ToT-BFS | Yes (if $b$ sufficient) | No guarantee | $O(b^d)$ | $O(b^d)$ |
| ToT-DFS | No (may miss) | No | $O(b \cdot d)$ | $O(b \cdot d)$ |
| MCTS | Asymptotically | Asymptotically | $O(n_{\text{sim}} \cdot d)$ | $O(n_{\text{sim}} \cdot d)$ |
| Beam Search | No | No | $O(B \cdot k \cdot d)$ | $O(B \cdot k \cdot d)$ |
| A* | Yes (if $h$ admissible) | Yes (if $h$ admissible) | Depends on $h$ | Variable |

---

## 5.4 LLM-Native Planning Techniques

These techniques leverage the LLM's generative and in-context learning capabilities directly for planning, without requiring explicit search procedures.

### 5.4.1 Zero-Shot Plan Generation

Zero-shot planning generates a plan using only the task description and the LLM's pre-trained knowledge, without any examples:

$$
\pi = \text{LLM}(\text{task\_description}, \text{available\_tools}, \text{constraints})
$$

**Prompt template:**

```
You are an autonomous agent. Given the following task, create a 
detailed step-by-step plan to accomplish it.

Task: {task_description}
Available tools: {tool_list_with_descriptions}
Constraints: {time_limit, resource_limit, safety_constraints}

Output your plan as a numbered list of steps. For each step, specify:
1. The action to take
2. The tool to use (if any)
3. The expected outcome
4. Dependencies on previous steps
```

**Strengths:**
- No example curation required.
- Generalizes to novel task domains.
- Fast—single LLM inference.

**Weaknesses:**
- Quality depends entirely on the LLM's pre-trained knowledge.
- Plans may be vague, infeasible, or miss edge cases.
- No grounding in the actual environment state.

**Quality depends on task novelty vs. pre-training data overlap:**

$$
\text{PlanQuality}_{\text{zero-shot}} \propto P_{\text{pretrain}}(\text{task\_domain})
$$

For tasks well-represented in training data (e.g., "deploy a web app"), zero-shot plans are often serviceable. For novel or domain-specific tasks, quality degrades significantly.

---

### 5.4.2 Few-Shot Plan Generation with Exemplars

Few-shot planning provides $k$ example (task, plan) pairs in the prompt to guide the LLM's plan generation via in-context learning:

$$
\pi = \text{LLM}(\underbrace{(T_1, \pi_1), (T_2, \pi_2), \ldots, (T_k, \pi_k)}_{\text{exemplars}}, T_{\text{new}})
$$

**The quality of few-shot planning depends on:**

1. **Exemplar relevance**: Selected examples should be structurally similar to the target task. Formally, we want exemplars $T_i$ that minimize the distributional distance:

$$
T_i^* = \arg\min_{T_i \in \mathcal{E}} D_{\text{embed}}(T_i, T_{\text{new}})
$$

where $D_{\text{embed}}$ is a distance metric in the LLM's embedding space.

2. **Exemplar diversity**: Examples should cover different planning patterns (sequential, conditional, parallel) to expose the LLM to the full range of plan structures.

3. **Exemplar ordering**: Due to recency bias in transformers, the most relevant exemplar should be placed closest to the target task (last in the prompt).

**Dynamic exemplar selection:**

```python
def select_exemplars(task_new, exemplar_pool, k=3):
    embeddings = embed(exemplar_pool)
    query_embed = embed(task_new)
    similarities = cosine_similarity(query_embed, embeddings)
    top_k_indices = argsort(similarities)[-k:]
    # Order: least similar first, most similar last (recency bias)
    return [exemplar_pool[i] for i in top_k_indices]
```

---

### 5.4.3 Chain-of-Thought as Implicit Planning

Chain-of-Thought (CoT) prompting (Wei et al., 2022) induces the LLM to generate intermediate reasoning steps before producing a final answer. While originally designed for reasoning tasks, CoT functions as **implicit planning** when applied to action-oriented tasks.

**The mechanism:**

Standard prompting:

$$
P(a | x) = \text{LLM}(x)
$$

CoT prompting:

$$
P(a | x) = \sum_{z} P(a | x, z) \cdot P(z | x)
$$

where $z$ represents the chain of intermediate thoughts. By marginalizing over thought chains, CoT effectively searches over a richer space of reasoning trajectories.

**CoT as planning — the isomorphism:**

| CoT Component | Planning Analog |
|---|---|
| Input $x$ | Initial state $s_0$ + goal $G$ |
| Each thought $z_i$ | Intermediate state $s_i$ or action $a_i$ |
| Final answer | Goal state reached |
| Thought chain $z_1, \ldots, z_n$ | Plan $\pi = \langle a_1, \ldots, a_n \rangle$ |

**Why CoT is "implicit" planning:**

1. The "plan" is not separated from execution—the thought chain simultaneously constructs and evaluates the plan.
2. There is no explicit state representation—states are implicitly encoded in the evolving text.
3. There is no backtracking—the autoregressive nature means thoughts flow forward only (unless self-consistency or ToT is used).

**Self-Consistency CoT** (Wang et al., 2023) adds a weak form of search:

$$
\pi^* = \arg\max_{\pi} \sum_{i=1}^{N} \mathbb{1}[\text{CoT}_i(x) = \pi]
$$

Sample $N$ independent CoT chains and select the most frequently occurring plan/answer by majority vote. This provides robustness through diversity.

---

### 5.4.4 Structured Plan Output (JSON/YAML Plans)

Forcing the LLM to output plans in structured formats (JSON, YAML, XML) provides several benefits:

1. **Machine-parseable**: Downstream execution engines can directly consume the plan.
2. **Schema validation**: Plan structure can be validated against a predefined schema.
3. **Explicit dependencies**: Structured formats naturally represent ordering, parallelism, and conditional logic.
4. **Reduced ambiguity**: Structured output constrains the LLM's generation, reducing vague or ambiguous steps.

**JSON plan schema example:**

```json
{
  "plan": {
    "goal": "Deploy ML model to production",
    "steps": [
      {
        "id": 1,
        "action": "run_tests",
        "tool": "pytest",
        "args": {"test_dir": "./tests"},
        "preconditions": [],
        "expected_outcome": "All tests pass",
        "fallback": {"action": "fix_tests", "then_retry": true}
      },
      {
        "id": 2,
        "action": "build_docker_image",
        "tool": "docker",
        "args": {"dockerfile": "./Dockerfile", "tag": "model:v2"},
        "preconditions": [1],
        "expected_outcome": "Docker image built successfully"
      },
      {
        "id": 3,
        "action": "push_to_registry",
        "tool": "docker",
        "args": {"image": "model:v2", "registry": "gcr.io/project"},
        "preconditions": [2]
      },
      {
        "id": 4,
        "action": "deploy_to_k8s",
        "tool": "kubectl",
        "args": {"manifest": "./k8s/deployment.yaml"},
        "preconditions": [3],
        "expected_outcome": "Pods running and healthy"
      }
    ],
    "parallelizable_groups": [[1], [2], [3], [4]],
    "estimated_duration_minutes": 15,
    "rollback_plan": "kubectl rollout undo deployment/model"
  }
}
```

**Constrained decoding for structured output:**

To guarantee valid JSON/YAML output, use **grammar-constrained decoding**:

$$
P(t_{i+1} | t_{\leq i}) = \begin{cases}
\frac{P_\theta(t_{i+1} | t_{\leq i})}{\sum_{t' \in \mathcal{V}_{\text{valid}}} P_\theta(t' | t_{\leq i})} & \text{if } t_{i+1} \in \mathcal{V}_{\text{valid}}(t_{\leq i}) \\
0 & \text{otherwise}
\end{cases}
$$

where $\mathcal{V}_{\text{valid}}(t_{\leq i})$ is the set of tokens that maintain valid JSON according to a JSON grammar parser tracking the current parse state.

Libraries like **Outlines**, **Guidance**, and **Instructor** implement this by maintaining a finite-state automaton synchronized with the generation process.

---

### 5.4.5 Plan Verification and Validation

Before executing a plan, it should be verified for correctness and validated for feasibility. This is a critical safety layer in agentic systems.

#### Precondition/Postcondition Checking

For each action $a_i$ in the plan $\pi = \langle a_1, a_2, \ldots, a_n \rangle$, verify:

**Precondition satisfaction:**

$$
\forall i \in \{1, \ldots, n\}: \text{Pre}(a_i) \subseteq \hat{s}_{i-1}
$$

where $\hat{s}_{i-1}$ is the predicted state after executing actions $a_1, \ldots, a_{i-1}$.

**Postcondition consistency:**

$$
\forall i \in \{1, \ldots, n\}: \text{Post}(a_i, \hat{s}_{i-1}) = \hat{s}_i
$$

In LLM agents, preconditions and postconditions are not formally specified. Instead, the LLM itself acts as the verifier:

```
Given the plan:
Step 1: {action_1} → Expected result: {result_1}
Step 2: {action_2} → Expected result: {result_2}
...

For each step, verify:
1. Can this step be executed given the results of previous steps?
2. Are there any missing prerequisites?
3. Does the expected result logically follow from the action?
4. Are there any potential failure modes not addressed?
```

**Formal verification pipeline:**

```python
def verify_plan(plan, initial_state, goal, llm):
    state = initial_state
    issues = []
    
    for i, step in enumerate(plan.steps):
        # Check preconditions
        pre_check = llm(f"""
            Current state: {state}
            Proposed action: {step.action}
            Can this action be executed in the current state?
            If not, what is missing?
        """)
        if not pre_check.is_valid:
            issues.append(f"Step {i}: precondition violation - {pre_check.reason}")
        
        # Simulate state transition
        state = llm(f"""
            Current state: {state}
            Action executed: {step.action}
            What is the new state after this action?
        """)
        
        # Check for side effects
        side_effects = llm(f"""
            Does action '{step.action}' have any unintended side effects
            that could interfere with later steps?
        """)
        if side_effects.found:
            issues.append(f"Step {i}: side effect risk - {side_effects.description}")
    
    # Check goal satisfaction
    goal_check = llm(f"Does final state {state} satisfy goal {goal}?")
    if not goal_check.satisfied:
        issues.append(f"Plan does not achieve goal: {goal_check.reason}")
    
    return PlanVerificationResult(valid=len(issues)==0, issues=issues)
```

#### Plan Soundness and Completeness

**Soundness:** A plan $\pi$ is sound if executing it from $s_0$ is guaranteed to reach a state satisfying $G$, assuming the transition model is correct:

$$
\text{Sound}(\pi) \iff \gamma(s_0, \pi) \models G
$$

**Completeness:** A planning algorithm is complete if it finds a plan whenever one exists:

$$
\text{Complete}(\mathcal{A}) \iff (\exists \pi : \gamma(s_0, \pi) \models G) \implies \mathcal{A}(s_0, G) \neq \text{FAILURE}
$$

LLM-based planners are **neither sound nor complete** in general:

- **Not sound**: The LLM may generate plans with logical errors, violated preconditions, or incorrect effect predictions.
- **Not complete**: The LLM may fail to find a valid plan even when one exists, due to limited context, knowledge gaps, or generation biases.

**Improving soundness — multi-agent verification:**

```
Planner Agent → generates plan π
Critic Agent  → identifies flaws in π
Planner Agent → revises π based on critique
Verifier Agent → formally checks revised π
```

This adversarial setup can significantly improve plan quality through iterative refinement.

---

## 5.5 Adaptive and Dynamic Planning

Real-world environments are non-stationary: unexpected events occur, actions fail, new information emerges, and goals evolve. Adaptive planning encompasses all strategies for modifying plans in response to environmental dynamics.

### 5.5.1 Replanning on Failure

When an action fails during plan execution, the agent must decide how to proceed. The replanning decision depends on:

**Failure taxonomy:**

| Failure Type | Example | Recovery Strategy |
|---|---|---|
| **Transient** | API timeout, rate limit | Retry with backoff |
| **Correctable** | Wrong parameters | Fix parameters, retry |
| **Blocking** | Required tool unavailable | Substitute alternative tool |
| **Fatal** | Goal provably unachievable | Abort or modify goal |
| **Partial** | Action partially succeeded | Continue from partial state |

**Replanning algorithm:**

$$
\textbf{Algorithm: FailureAwareExecution}
$$

```
Input: plan π = [a₁, ..., aₙ], state s₀, max_retries R
State: s ← s₀

For i = 1, ..., n:
    For retry = 1, ..., R:
        o ← Execute(aᵢ, s)
        If o.success:
            s ← UpdateState(s, o)
            Break
        Else if o.failure_type == TRANSIENT:
            Wait(exponential_backoff(retry))
            Continue  // Retry same action
        Else if o.failure_type == CORRECTABLE:
            aᵢ ← LLM.fix_action(aᵢ, o.error_message, s)
            Continue  // Retry fixed action
        Else:  // BLOCKING or FATAL
            π_remaining ← LLM.replan(s, G, history)
            Return FailureAwareExecution(π_remaining, s, R)
    
    If all retries exhausted:
        π_remaining ← LLM.replan(s, G, history)
        Return FailureAwareExecution(π_remaining, s, R)

Return SUCCESS
```

---

### 5.5.2 Plan Repair vs. Plan Regeneration

When a plan becomes invalid, two recovery strategies exist:

**Plan Repair:** Modify the minimum number of steps in the existing plan to restore validity:

$$
\pi_{\text{repaired}} = \arg\min_{\pi'} \text{EditDistance}(\pi, \pi') \quad \text{s.t.} \quad \gamma(s_{\text{current}}, \pi') \models G
$$

**Plan Regeneration:** Discard the entire remaining plan and generate a new one from scratch:

$$
\pi_{\text{new}} = \text{LLM}(s_{\text{current}}, G, \text{history})
$$

**Decision criterion:**

$$
\text{Strategy} = \begin{cases}
\text{Repair} & \text{if } \frac{|\text{affected steps}|}{|\text{remaining steps}|} < \theta_{\text{repair}} \\
\text{Regenerate} & \text{otherwise}
\end{cases}
$$

where $\theta_{\text{repair}} \in (0, 1)$ is a threshold (typically $0.3$–$0.5$). If fewer than $\theta_{\text{repair}}$ fraction of remaining steps are affected by the failure, repair is more efficient; otherwise, full regeneration is preferable.

**Comparative analysis:**

| Dimension | Plan Repair | Plan Regeneration |
|---|---|---|
| Computational cost | Lower (modify subset) | Higher (generate full plan) |
| Plan coherence | Risk of Frankenstein plans | Fresh, internally consistent |
| Context utilization | Preserves good decisions | May lose useful partial work |
| Applicable when | Localized failures | Fundamental assumption changes |
| Implementation | Harder (need failure localization) | Easier (same as initial planning) |

**LLM-based plan repair prompt:**

```
The following plan was being executed:
{original_plan}

Execution succeeded through Step {k}. Step {k+1} failed because:
{failure_reason}

Current state: {current_state}
Original goal: {goal}

Modify the remaining plan (Steps {k+1} to {n}) to account for 
this failure. Keep as many original steps as possible. Only change 
what is necessary.
```

---

### 5.5.3 Conditional Branching in Plans

Real plans often require **conditional logic**—different actions depending on observed outcomes. This extends flat plan sequences into **plan programs** with control flow.

**Formal representation:**

A conditional plan is a directed acyclic graph (or tree) where edges are labeled with conditions:

$$
\pi_{\text{conditional}} = \begin{cases}
\langle a_1, a_2, a_3 \rangle & \text{if } \text{Condition}_A \text{ holds after } a_1 \\
\langle a_1, a_4, a_5 \rangle & \text{if } \text{Condition}_B \text{ holds after } a_1 \\
\langle a_1, a_6 \rangle & \text{otherwise}
\end{cases}
$$

**Structured representation:**

```yaml
plan:
  - step: 1
    action: "check_api_status"
    branches:
      - condition: "api_status == 'available'"
        next_steps:
          - step: 2a
            action: "call_api_directly"
      - condition: "api_status == 'rate_limited'"
        next_steps:
          - step: 2b
            action: "wait_and_retry"
            args: {delay: 60}
          - step: 3b
            action: "call_api_directly"
      - condition: "api_status == 'down'"
        next_steps:
          - step: 2c
            action: "use_cached_data"
          - step: 3c
            action: "flag_for_manual_review"
```

**Implementation as a state machine:**

```python
class ConditionalPlanExecutor:
    def execute(self, plan_node, state):
        result = execute_action(plan_node.action, state)
        new_state = update_state(state, result)
        
        if plan_node.is_terminal:
            return new_state
        
        # Evaluate conditions
        for branch in plan_node.branches:
            if evaluate_condition(branch.condition, new_state):
                return self.execute(branch.next_node, new_state)
        
        # Default branch
        if plan_node.default_branch:
            return self.execute(plan_node.default_branch, new_state)
        
        raise PlanError("No matching branch condition")
```

---

### 5.5.4 Contingency Planning (Plan B Generation)

Contingency planning proactively generates alternative plans for anticipated failure modes **before** failures occur:

$$
\Pi = \{(\pi_1, C_1), (\pi_2, C_2), \ldots, (\pi_m, C_m)\}
$$

where $\pi_i$ is an alternative plan and $C_i$ is the triggering condition for switching to $\pi_i$.

**Formally, the contingency planning problem:**

$$
\Pi^* = \arg\min_{\Pi} \sum_{i=1}^{m} P(C_i) \cdot \text{Cost}(\pi_i) + (1 - \sum_{i=1}^{m} P(C_i)) \cdot \text{Cost}(\pi_1)
$$

subject to:

$$
\forall i: \gamma(s_{C_i}, \pi_i) \models G
$$

where $s_{C_i}$ is the state at which contingency condition $C_i$ triggers, and $P(C_i)$ is the estimated probability of that contingency arising.

**Prompt for contingency plan generation:**

```
Primary Plan:
{primary_plan}

For each step in the primary plan, identify the most likely 
failure mode and generate a contingency plan:

Step N: {action}
  - Most likely failure: {failure_mode}
  - Probability estimate: {probability}
  - Contingency plan if this fails: {alternative_actions}
  - Resources needed for contingency: {resources}
```

**Decision to pre-compute contingencies:**

Not all steps warrant contingency plans. The expected value of contingency planning for step $i$:

$$
\text{EV}_{\text{contingency}}(i) = P(\text{fail}_i) \cdot [\text{Cost}_{\text{replan}} - \text{Cost}_{\text{contingency\_switch}}] - \text{Cost}_{\text{contingency\_generation}}
$$

Only generate contingencies when $\text{EV}_{\text{contingency}}(i) > 0$.

---

### 5.5.5 Real-Time Planning Under Time Constraints

In time-critical applications (robotics, live customer service, trading), the agent must produce an actionable plan within a strict time budget $\Delta t$:

$$
\pi = \text{Plan}(s_0, G, \Delta t) \quad \text{such that} \quad \text{Time}(\text{Plan}) \leq \Delta t
$$

**Anytime planning algorithms** address this by producing incrementally improving plans:

$$
\pi_1, \pi_2, \pi_3, \ldots \quad \text{where} \quad \text{Quality}(\pi_1) \leq \text{Quality}(\pi_2) \leq \text{Quality}(\pi_3) \leq \cdots
$$

The agent returns the best plan found when the time budget expires.

**Strategies for real-time LLM planning:**

1. **Budget-aware prompt design**: Use smaller prompts and request concise plans when time is limited.

2. **Progressive deepening**: Start with a coarse plan, then refine:
   - $t < \Delta t / 3$: Generate a high-level 3-step plan.
   - $t < 2\Delta t / 3$: Decompose each step into sub-steps.
   - $t < \Delta t$: Verify and repair the detailed plan.

3. **Parallel plan generation**: Generate multiple candidate plans simultaneously across multiple LLM calls, select the best:

$$
\pi^* = \arg\max_{\pi \in \{\pi_1, \ldots, \pi_k\}} \text{Quality}(\pi)
$$

4. **Cached plan templates**: Pre-compute plan templates for common task types, then adapt at runtime:

$$
\pi = \text{Adapt}(\pi_{\text{template}}, s_0, G)
$$

Adaptation is faster than from-scratch generation.

5. **Computation-quality trade-off curve:**

$$
\text{Quality}(\Delta t) = Q_{\max} \cdot (1 - e^{-\lambda \Delta t})
$$

where $Q_{\max}$ is the asymptotic quality and $\lambda$ controls the rate of improvement. This exponential saturation means most quality is achieved early, with diminishing returns for additional computation.

---

## 5.6 Multi-Step Planning with World Models

### 5.6.1 Internal World Models for Simulation

A **world model** is an internal representation that allows the agent to simulate the effects of actions without actually executing them in the environment:

$$
\hat{s}_{t+1} = \mathcal{W}(s_t, a_t)
$$

where $\mathcal{W}$ is the world model that predicts the next state given the current state and action.

**LLMs as implicit world models:**

Large language models encode a vast amount of world knowledge in their parameters. When prompted appropriately, they can simulate action consequences:

$$
\hat{s}_{t+1} = \text{LLM}(\text{"Current state: } s_t \text{. Action taken: } a_t \text{. What is the new state?"})
$$

**Formal properties of world models:**

1. **Accuracy**: $\mathbb{E}[\|\hat{s}_{t+1} - s_{t+1}\|] < \epsilon$ — the predicted state should be close to the true state.
2. **Consistency**: Multi-step rollouts should not accumulate excessive error:

$$
\text{Error}(T) = \sum_{t=0}^{T} \|\hat{s}_t - s_t\| \leq C \cdot T^\alpha, \quad \alpha < 2
$$

Sub-quadratic error growth is desirable; quadratic or worse indicates the model is unreliable for long-horizon planning.

3. **Calibration**: The model's uncertainty about state predictions should correlate with actual prediction errors.

**Types of world models in LLM agents:**

| Type | Description | Accuracy | Speed |
|---|---|---|---|
| **LLM-implicit** | LLM simulates effects via prompting | Moderate | Slow (full LLM inference) |
| **Learned neural** | Separate neural network trained on trajectories | High (in distribution) | Fast |
| **Symbolic** | Rule-based transition system (PDDL) | Perfect (if rules correct) | Very fast |
| **Hybrid** | Neural backbone + symbolic constraints | High | Moderate |

**World model architecture for LLM agents:**

```python
class WorldModel:
    def __init__(self, llm, environment_description):
        self.llm = llm
        self.env_desc = environment_description
        self.state_history = []
    
    def predict(self, state, action):
        """Predict next state without executing action."""
        prompt = f"""
        Environment: {self.env_desc}
        Current state: {state}
        Action: {action}
        
        Predict the resulting state after this action is executed.
        Consider:
        1. Direct effects of the action
        2. Side effects and environmental reactions
        3. Any constraints or physical laws that apply
        
        Output the new state description.
        """
        predicted_state = self.llm(prompt)
        confidence = self.estimate_confidence(state, action, predicted_state)
        return predicted_state, confidence
    
    def simulate_trajectory(self, state, action_sequence):
        """Simulate multiple steps forward."""
        trajectory = [state]
        cumulative_uncertainty = 0
        
        for action in action_sequence:
            next_state, confidence = self.predict(trajectory[-1], action)
            cumulative_uncertainty += (1 - confidence)
            trajectory.append(next_state)
            
            # Abort if uncertainty too high
            if cumulative_uncertainty > self.uncertainty_threshold:
                return trajectory, False  # Unreliable
        
        return trajectory, True  # Reliable
```

---

### 5.6.2 Forward Simulation and Outcome Prediction

Forward simulation uses the world model to **project the consequences** of a candidate plan before executing it:

$$
\hat{\tau} = \langle s_0, a_1, \hat{s}_1, a_2, \hat{s}_2, \ldots, a_n, \hat{s}_n \rangle
$$

where each $\hat{s}_i = \mathcal{W}(\hat{s}_{i-1}, a_i)$.

**Uses of forward simulation:**

1. **Plan evaluation**: Score candidate plans by their simulated outcomes:

$$
\text{Score}(\pi) = R(\hat{s}_n) + \sum_{t=0}^{n-1} r(\hat{s}_t, a_{t+1})
$$

2. **Failure anticipation**: Detect potential failures before they occur:

$$
\text{FailureRisk}(\pi) = \max_{t \in [0,n]} P(\text{failure} | \hat{s}_t, a_{t+1})
$$

3. **Plan comparison**: Choose among multiple candidate plans:

$$
\pi^* = \arg\max_{\pi \in \{\pi_1, \ldots, \pi_k\}} \text{Score}(\pi) - \lambda \cdot \text{FailureRisk}(\pi)
$$

4. **Rollout horizon determination**: Determine how far ahead to plan:

$$
H^* = \arg\max_H \text{Score}(\pi_{1:H}) - \beta \cdot \text{CumulativeUncertainty}(H)
$$

**Error accumulation in multi-step simulation:**

A critical concern is that prediction errors compound over multiple steps. If the per-step prediction error is $\epsilon$, the error after $T$ steps can be modeled as:

$$
\text{Error}(T) \approx \epsilon \cdot \frac{(1 + L)^T - 1}{L}
$$

where $L$ is the Lipschitz constant of the transition dynamics. For $L > 0$, errors grow exponentially, making long-horizon simulations unreliable.

**Mitigation strategies:**

1. **Short-horizon planning with replanning**: Plan only $H$ steps ahead, execute, observe, replan.
2. **Ensemble world models**: Average predictions from multiple models to reduce variance:

$$
\hat{s}_{t+1} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{W}_k(s_t, a_t)
$$

3. **Uncertainty-aware planning**: Weight future states by confidence:

$$
\text{Score}(\pi) = \sum_{t=0}^{n} \gamma^t \cdot \text{Confidence}_t \cdot r_t
$$

---

### 5.6.3 Counterfactual Planning

Counterfactual planning asks: "What would have happened if a different action had been taken?" This enables the agent to learn from past decisions and improve future plans.

**Formal definition:**

Given a trajectory $\tau = \langle s_0, a_1, s_1, \ldots, a_T, s_T \rangle$, a counterfactual query at time $t$ with alternative action $a'_t$ asks:

$$
\hat{s}'_{t+1:T} = \text{Simulate}(s_t, a'_t, a_{t+1}, \ldots, a_T) \quad \text{using } \mathcal{W}
$$

**Counterfactual value difference:**

$$
\Delta V(t, a'_t) = V(\hat{\tau}'_{t:T}) - V(\tau_{t:T})
$$

If $\Delta V > 0$, the alternative action $a'_t$ would have been better.

**Applications in LLM agents:**

1. **Retrospective plan improvement**: After task completion, identify suboptimal actions and store improved strategies for future tasks.

2. **Regret-based learning**: Compute regret:

$$
\text{Regret}(T) = \sum_{t=1}^{T} \max_{a'_t} \Delta V(t, a'_t)
$$

High regret indicates the agent made many suboptimal decisions and should update its planning strategy.

3. **Experience replay with counterfactuals**: When storing experiences in memory, augment with counterfactual analyses for richer learning signals.

**Counterfactual reasoning prompt:**

```
The agent took the following trajectory:
{trajectory}

At Step {t}, the agent chose action: {a_t}
Alternative actions considered: {alternatives}

For each alternative action, simulate:
1. What would the immediate outcome have been?
2. How would subsequent steps have been affected?
3. Would the overall result have been better or worse?
4. What is the key lesson for future planning?
```

---

### 5.6.4 Model-Based vs. Model-Free Planning in Agents

**Model-based planning** uses an explicit world model to simulate and evaluate plans before execution:

$$
\pi^* = \arg\max_{\pi} \sum_{t=0}^{T} \gamma^t r(\hat{s}_t, a_t) \quad \text{where } \hat{s}_{t+1} = \mathcal{W}(\hat{s}_t, a_t)
$$

**Model-free planning** selects actions based directly on learned value functions or policies without simulating future states:

$$
a_t = \arg\max_a Q(s_t, a) \quad \text{(value-based)}
$$

$$
a_t \sim \pi_\theta(a | s_t) \quad \text{(policy-based)}
$$

**Comprehensive comparison:**

| Dimension | Model-Based | Model-Free |
|---|---|---|
| **Data efficiency** | High (can plan in imagination) | Low (needs real interactions) |
| **Computational cost** | High (simulation is expensive) | Low per action (policy lookup) |
| **Adaptability** | High (update model, replan) | Low (must retrain policy) |
| **Error source** | Model error (compounding) | Estimation error (value function) |
| **Long-horizon** | Degrades due to model error | Degrades due to credit assignment |
| **In LLM agents** | LLM predicts next state | LLM directly predicts best action |

**Dyna-style hybrid architecture:**

The Dyna framework (Sutton, 1991) combines both approaches:

```
Real Experience:    s, a, r, s' (from environment)
                    ↓
                 Update World Model: W(s,a) → s'
                 Update Policy/Value: Q(s,a) ← r + γ max Q(s',a')
                    ↓
Simulated Experience: For k iterations:
                      Sample s̃ from visited states
                      Sample ã from available actions
                      s̃' = W(s̃, ã), r̃ = R(s̃, ã)
                      Update Q(s̃, ã) ← r̃ + γ max Q(s̃', a')
```

In LLM agents, this manifests as:
- **Model-based component**: The LLM simulates "what would happen if I do X" before deciding.
- **Model-free component**: The LLM directly outputs the best action based on pattern matching from its training data (cached policies).

Most production LLM agents operate in a **soft hybrid**: the LLM implicitly simulates consequences during CoT reasoning (model-based) while also relying on learned action patterns (model-free).

---

## 5.7 Planning Evaluation

### 5.7.1 Plan Quality Metrics

Evaluating plans requires multiple complementary metrics that capture different aspects of plan quality:

**1. Optimality**

The degree to which the plan minimizes cost (or maximizes reward) relative to the best possible plan:

$$
\text{Optimality}(\pi) = \frac{\text{Cost}(\pi^*)}{\text{Cost}(\pi)} \in (0, 1]
$$

where $\pi^*$ is the optimal plan. An optimality of 1.0 means the plan is optimal.

For tasks where the optimal plan is unknown, **relative optimality** compares against a reference planner:

$$
\text{RelativeOptimality}(\pi) = \frac{\text{Cost}(\pi_{\text{reference}})}{\text{Cost}(\pi)}
$$

**2. Feasibility**

Whether the plan can be executed without violating constraints:

$$
\text{Feasible}(\pi) = \bigwedge_{i=1}^{n} \left[\text{Pre}(a_i) \subseteq \hat{s}_{i-1}\right] \in \{\text{True}, \text{False}\}
$$

For soft feasibility (some constraints are preferences, not hard requirements):

$$
\text{SoftFeasibility}(\pi) = 1 - \frac{\text{Number of violated constraints}}{\text{Total constraints}}
$$

**3. Completeness**

Whether the plan achieves the full goal specification:

$$
\text{Completeness}(\pi) = \frac{|G \cap \hat{s}_n|}{|G|}
$$

where $\hat{s}_n$ is the predicted final state and $G$ is the set of goal propositions. A completeness of 1.0 means all goal conditions are achieved.

**4. Robustness**

The plan's resilience to perturbations and unexpected outcomes:

$$
\text{Robustness}(\pi) = \mathbb{E}_{\delta \sim \mathcal{D}_{\text{perturbation}}} \left[\text{Success}(\pi, s_0 + \delta)\right]
$$

A robust plan succeeds even when the initial state or transition dynamics are slightly perturbed.

**5. Parsimony (Plan Length)**

Shorter plans are generally preferred (fewer steps = fewer failure points):

$$
\text{Parsimony}(\pi) = \frac{1}{|\pi|}
$$

**Composite plan quality score:**

$$
Q(\pi) = w_1 \cdot \text{Optimality}(\pi) + w_2 \cdot \text{Completeness}(\pi) + w_3 \cdot \text{Robustness}(\pi) + w_4 \cdot \text{Parsimony}(\pi)
$$

where $\sum_i w_i = 1$ and weights are task-dependent.

---

### 5.7.2 Plan Execution Success Rate

The most direct evaluation metric: does the plan actually work when executed?

$$
\text{ExecutionSuccessRate} = \frac{\text{Number of tasks successfully completed}}{\text{Total number of tasks attempted}}
$$

**Granular success metrics:**

1. **Full success rate**: The entire task is completed correctly:

$$
\text{SR}_{\text{full}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{TaskComplete}(i)]
$$

2. **Partial success rate**: Fraction of subtasks completed:

$$
\text{SR}_{\text{partial}} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{SubtasksCompleted}(i)}{\text{TotalSubtasks}(i)}
$$

3. **Progress rate**: How far toward the goal the agent progresses before failure:

$$
\text{Progress}(i) = \frac{\text{StepsSucceeded}(i)}{\text{TotalStepsRequired}(i)}
$$

4. **Conditional success rate**: Success rate given that the plan is feasible:

$$
\text{SR}_{\text{conditional}} = P(\text{Success} | \text{Feasible plan})
$$

This isolates execution failures from planning failures.

**Statistical considerations:**

For $N$ task evaluations, the 95% confidence interval for the success rate is:

$$
\text{SR} \pm 1.96 \sqrt{\frac{\text{SR}(1 - \text{SR})}{N}}
$$

For reliable comparisons between methods, ensure $N$ is large enough that confidence intervals do not overlap. A minimum of $N = 100$ tasks is typically required for meaningful statistical comparisons.

---

### 5.7.3 Planning Efficiency

Efficiency metrics capture the **cost** of planning, not just the quality of the plan:

**1. Step Efficiency**

The ratio of necessary actions to total actions taken:

$$
\eta_{\text{steps}} = \frac{|\pi_{\text{optimal}}|}{|\pi_{\text{actual}}|}
$$

Where $|\pi_{\text{actual}}|$ includes retries, failed actions, and exploratory actions.

**2. Token Cost**

The total number of LLM tokens consumed during planning and execution:

$$
C_{\text{tokens}} = \sum_{t=1}^{T} (\text{InputTokens}_t + \text{OutputTokens}_t)
$$

This directly maps to monetary cost:

$$
C_{\text{monetary}} = C_{\text{input}} \cdot p_{\text{input}} + C_{\text{output}} \cdot p_{\text{output}}
$$

where $p_{\text{input}}$ and $p_{\text{output}}$ are per-token prices.

**3. Time Efficiency**

$$
\eta_{\text{time}} = \frac{T_{\text{optimal}}}{T_{\text{actual}}}
$$

**Wall-clock time breakdown:**

$$
T_{\text{total}} = T_{\text{planning}} + T_{\text{execution}} + T_{\text{replanning}} + T_{\text{waiting}}
$$

**4. LLM Call Efficiency**

$$
\eta_{\text{calls}} = \frac{\text{Minimum required LLM calls}}{\text{Actual LLM calls}}
$$

**5. Cost-effectiveness**

The ratio of plan quality to total cost:

$$
\text{CostEffectiveness} = \frac{Q(\pi)}{C_{\text{total}}}
$$

**Efficiency frontier analysis:**

Plot plan quality vs. computational cost for multiple methods. The **Pareto frontier** identifies methods that are not dominated:

$$
\text{Pareto-optimal}(\mathcal{M}_i) \iff \nexists \mathcal{M}_j : Q(\mathcal{M}_j) \geq Q(\mathcal{M}_i) \wedge C(\mathcal{M}_j) \leq C(\mathcal{M}_i)
$$

```
Plan Quality
    │     ★ ─ ─ ─ ─ ─ Pareto Frontier
    │   ★       ★
    │ ★    ○          ○: dominated methods
    │○     ○       ★: Pareto-optimal methods
    │○  ○
    └──────────────── Computational Cost
```

---

### 5.7.4 Benchmarks: ALFWorld, WebArena, SWE-bench

**1. ALFWorld (Shridhar et al., 2021)**

| Dimension | Description |
|---|---|
| **Domain** | Embodied household tasks in text-based environments |
| **Task types** | Pick & place, clean, heat, cool, examine, put objects |
| **Interface** | Text observations and text actions |
| **State space** | Textual descriptions of rooms, objects, and their states |
| **Action space** | Natural language commands (e.g., "go to desk 1", "pick up pen") |
| **Evaluation** | Task success rate (binary: goal conditions met or not) |
| **Number of tasks** | 134 test tasks across 6 task types |
| **Planning challenges** | Multi-step object manipulation, spatial navigation, state tracking |

**Example ALFWorld task:**
```
Task: Put a clean mug in the coffee maker.
Required plan:
  1. Find a mug (explore rooms)
  2. Pick up the mug
  3. Go to the sink
  4. Clean the mug
  5. Go to the coffee maker
  6. Put the mug in the coffee maker
```

**Key results (state of the art):**

| Method | Success Rate |
|---|---|
| BUTLER (supervised) | 26% |
| ReAct (few-shot) | 71% |
| Reflexion | 97% |
| AutoGen (multi-agent) | 85% |

**2. WebArena (Zhou et al., 2023)**

| Dimension | Description |
|---|---|
| **Domain** | Realistic web browsing tasks on live websites |
| **Task types** | Information retrieval, form filling, e-commerce, content management |
| **Interface** | Browser actions (click, type, scroll, navigate) on real web pages |
| **State space** | DOM trees, accessibility trees, screenshots |
| **Action space** | Click(element), Type(element, text), Scroll, Navigate(URL), etc. |
| **Evaluation** | Functional correctness (URL match, element presence, string match) |
| **Number of tasks** | 812 tasks across 5 website categories |
| **Planning challenges** | Long-horizon navigation, dynamic web content, complex form interactions |

**Example WebArena task:**
```
Task: "Find the cheapest one-way flight from Pittsburgh to 
       Los Angeles on Dec 25 on the booking site."
Required plan:
  1. Navigate to booking site
  2. Select "Flights" tab
  3. Set departure: Pittsburgh
  4. Set destination: Los Angeles
  5. Set date: Dec 25
  6. Select "One-way"
  7. Click "Search"
  8. Sort by price (ascending)
  9. Return the cheapest option
```

**Key results:**

| Method | Success Rate |
|---|---|
| GPT-4 (direct) | 14.4% |
| GPT-4 + CoT | 12.7% |
| GPT-4 + Set-of-Marks | 26.4% |
| Agent-E | 33.3% |
| Human | 78.2% |

The large gap between AI and human performance on WebArena highlights the difficulty of long-horizon planning in complex, dynamic environments.

**3. SWE-bench (Jimenez et al., 2024)**

| Dimension | Description |
|---|---|
| **Domain** | Real-world software engineering: resolving GitHub issues |
| **Task types** | Bug fixes, feature implementations, test additions |
| **Interface** | File system, code editor, terminal, test runner |
| **State space** | Repository state (files, tests, git history) |
| **Action space** | Edit files, run commands, search code, read files |
| **Evaluation** | Test pass rate (does the patch pass the issue's test cases?) |
| **Number of tasks** | 2,294 tasks (SWE-bench full), 300 (SWE-bench Lite), 500 (SWE-bench Verified) |
| **Planning challenges** | Code understanding, fault localization, multi-file editing, test awareness |

**Example SWE-bench task:**
```
Repository: django/django
Issue: "DateTimeField doesn't handle timezone-aware datetimes 
        correctly when USE_TZ=True"
Required plan:
  1. Understand the issue (read issue description, reproduce bug)
  2. Locate relevant code (search for DateTimeField implementation)
  3. Identify the root cause (incorrect timezone conversion logic)
  4. Design a fix (modify the conversion method)
  5. Implement the fix (edit the source file)
  6. Verify the fix (run existing tests + write new test)
  7. Ensure no regressions (run full test suite)
```

**Key results (SWE-bench Verified):**

| Method | Resolved (%) |
|---|---|
| Claude 3.5 Sonnet (direct) | 49.0% |
| OpenAI o1-preview | 28.6% |
| SWE-Agent + GPT-4 | 23.0% |
| Devin | 41.7% |
| Agentless | 30.7% |
| CodeStory Aide | 43.0% |

**Cross-benchmark planning complexity analysis:**

| Benchmark | Avg. Steps | Branching Factor | Horizon | Key Planning Challenge |
|---|---|---|---|---|
| ALFWorld | 5-15 | ~10 | Short-Medium | State tracking, spatial reasoning |
| WebArena | 10-30 | ~50 | Long | Dynamic content, error recovery |
| SWE-bench | 20-100+ | ~100+ | Very Long | Code understanding, fault localization |

**Benchmark selection guidance:**

$$
\text{Benchmark Choice} = f(\text{Planning Paradigm}, \text{Action Space Complexity}, \text{Horizon Length})
$$

- **ALFWorld**: Best for evaluating basic planning and grounding capabilities.
- **WebArena**: Best for evaluating adaptive planning in dynamic, partially observable environments.
- **SWE-bench**: Best for evaluating complex, long-horizon planning requiring deep domain knowledge and multi-step reasoning.

---

**Chapter Summary — Key Takeaways:**

1. Planning in LLM agents is fundamentally different from classical AI planning: state spaces are textual, action spaces are open-ended, and transition models are implicit.

2. The choice of planning paradigm (decomposition, sequential, iterative, reactive) depends on task horizon, feedback availability, and computational budget.

3. Search-based methods (ToT, GoT, MCTS, A*) provide systematic exploration of plan space but incur significant computational cost. The LLM serves dual roles as node expander and heuristic evaluator.

4. LLM-native techniques (zero-shot, few-shot, CoT) are efficient but lack formal guarantees. Structured output formats and verification layers partially address this gap.

5. Adaptive planning (replanning, repair, contingencies) is essential for robustness in real-world deployments where actions can fail and environments are non-stationary.

6. World models enable forward simulation and counterfactual reasoning but suffer from compounding prediction errors over long horizons.

7. Evaluation must be multi-dimensional: plan quality (optimality, feasibility, completeness), execution success rate, and efficiency (steps, tokens, time) must all be measured on standardized benchmarks.