

# Chapter 10: Goal Setting and Monitoring

---

## 10.1 Definition and Formal Framework

### 10.1.1 What is Goal Setting in Agentic Systems

Goal setting in agentic systems is the formal process by which an autonomous agent acquires, represents, decomposes, pursues, monitors, and adaptively revises desired end-states that govern its decision-making across temporal horizons. Unlike reactive systems that respond to stimuli without overarching purpose, goal-directed agents maintain persistent representations of desired world states and systematically orchestrate actions to transform the current world state into one satisfying the goal specification.

**Formal Definition.** A goal-directed agentic system is defined as:

$$
\mathcal{A}_g = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O}, \pi_\theta, \mathcal{G}, \mathcal{M}, \mathcal{E} \rangle
$$

where:
- $\mathcal{S}$ is the state space (environment + internal agent state)
- $\mathcal{A}$ is the action space (tool calls, generations, API invocations)
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ is the stochastic transition function
- $\mathcal{O}$ is the observation space
- $\pi_\theta: \mathcal{S} \times \mathcal{G} \rightarrow \Delta(\mathcal{A})$ is the goal-conditioned policy
- $\mathcal{G}$ is the goal space (set of all representable goals)
- $\mathcal{M}: \mathcal{S} \times \mathcal{G} \rightarrow [0, 1]$ is the monitoring function (progress measurement)
- $\mathcal{E}: \mathcal{S} \times \mathcal{G} \rightarrow \{\text{success}, \text{failure}, \text{partial}, \text{in\_progress}\}$ is the evaluation function

The agent's behavior is fundamentally governed by the interaction between its current state $s_t$, its active goal set $G_{\text{active}} \subseteq \mathcal{G}$, and its policy $\pi_\theta$:

$$
a_t = \pi_\theta(s_t, G_{\text{active}}) = \arg\max_{a \in \mathcal{A}} \mathbb{E}\left[\sum_{k=0}^{T} \gamma^k \cdot \text{GoalProgress}(s_{t+k}, G_{\text{active}}) \;\Big|\; s_t, a\right]
$$

**Goal Setting vs. Task Execution.** A critical distinction separates goal setting from task execution:

| Aspect | Goal Setting | Task Execution |
|---|---|---|
| **Abstraction level** | What to achieve | How to achieve it |
| **Temporality** | Persistent across interactions | Transient per action |
| **Representation** | Declarative (desired end-state) | Procedural (action sequence) |
| **Flexibility** | Revisable, re-prioritizable | Fixed once initiated |
| **Evaluation** | Success criteria on world state | Correctness of individual steps |

**The Goal-Setting Lifecycle.** Goal setting in agentic systems follows a cyclic process:

```
┌─────────────────────────────────────────────────────────┐
│                  Goal-Setting Lifecycle                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. ACQUISITION ──► 2. REPRESENTATION ──► 3. DECOMPOSE  │
│       ▲                                        │        │
│       │                                        ▼        │
│  7. REFLECT ◄── 6. EVALUATE ◄── 5. MONITOR ◄── 4. PLAN │
│       │                                                 │
│       ▼                                                 │
│  8. ADAPT (revise/abandon/spawn new goals)              │
│       │                                                 │
│       └──────────────► Return to Step 2 ───────────►    │
└─────────────────────────────────────────────────────────┘
```

**Multi-Level Goal Architecture.** Real-world agents maintain goals at multiple abstraction levels simultaneously:

$$
\mathcal{G}_{\text{hierarchy}} = \underbrace{\mathcal{G}_{\text{mission}}}_{\text{overarching purpose}} \supset \underbrace{\mathcal{G}_{\text{strategic}}}_{\text{long-term objectives}} \supset \underbrace{\mathcal{G}_{\text{tactical}}}_{\text{medium-term plans}} \supset \underbrace{\mathcal{G}_{\text{operational}}}_{\text{immediate actions}}
$$

Each level constrains and guides the level below it:

$$
\pi_\theta(a | s, g_{\text{operational}}) \text{ subject to } g_{\text{operational}} \models g_{\text{tactical}} \models g_{\text{strategic}} \models g_{\text{mission}}
$$

where $g_i \models g_j$ denotes that satisfying $g_i$ contributes to satisfying $g_j$.

---

### 10.1.2 Goal as a Formal Specification

A goal is not merely a string of natural language—it is a structured specification with well-defined components:

$$
g = \langle \text{objective}, \text{constraints}, \text{success\_criteria}, \text{deadline} \rangle
$$

**Component 1: Objective ($\text{objective}$)**

The objective specifies the desired end-state or outcome. Formally, it is a predicate over world states:

$$
\text{objective}: \mathcal{S} \rightarrow \{0, 1\}
$$

A state $s$ satisfies the objective if $\text{objective}(s) = 1$. For complex objectives, this generalizes to a real-valued satisfaction function:

$$
\text{objective}: \mathcal{S} \rightarrow [0, 1]
$$

where values between 0 and 1 represent partial satisfaction.

**Examples:**
- Binary: $\text{objective}(s) = \mathbb{1}[\text{``all unit tests pass in state } s\text{''}]$
- Graded: $\text{objective}(s) = \frac{|\text{passing tests}(s)|}{|\text{total tests}|}$

**Component 2: Constraints ($\text{constraints}$)**

Constraints define boundaries on acceptable behavior during goal pursuit. They are invariants that must hold throughout the trajectory, not just at the end:

$$
\text{constraints} = \{c_1, c_2, \ldots, c_m\}, \quad c_i: \mathcal{S} \times \mathcal{A} \rightarrow \{0, 1\}
$$

A trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ is **constraint-compliant** if:

$$
\text{Compliant}(\tau) = \bigwedge_{t=0}^{T} \bigwedge_{i=1}^{m} c_i(s_t, a_t) = 1
$$

**Constraint Types:**

| Type | Formal Definition | Example |
|---|---|---|
| **Safety** | $c_{\text{safe}}(s, a) = \mathbb{1}[a \notin \mathcal{A}_{\text{dangerous}}]$ | "Do not delete production data" |
| **Resource** | $c_{\text{cost}}(\tau) = \mathbb{1}[\sum_t \text{cost}(a_t) \leq B]$ | "Use at most 1000 API calls" |
| **Temporal** | $c_{\text{time}}(\tau) = \mathbb{1}[T \leq T_{\text{max}}]$ | "Complete within 30 minutes" |
| **Quality** | $c_{\text{qual}}(s_T) = \mathbb{1}[\text{quality}(s_T) \geq \tau_q]$ | "Code coverage must exceed 80%" |
| **Ethical** | $c_{\text{eth}}(a) = \mathbb{1}[a \models \text{ethical\_policy}]$ | "Do not impersonate humans" |

**Component 3: Success Criteria ($\text{success\_criteria}$)**

Success criteria operationalize the objective into measurable, verifiable conditions:

$$
\text{success\_criteria} = \{(\text{metric}_i, \text{threshold}_i, \text{comparator}_i)\}_{i=1}^{k}
$$

where $\text{comparator}_i \in \{\geq, \leq, =, \in\}$. Goal $g$ is achieved when:

$$
\text{Achieved}(g, s) = \bigwedge_{i=1}^{k} \text{metric}_i(s) \;\text{comparator}_i\; \text{threshold}_i
$$

**Example:** For a goal "Deploy a web application":
$$
\text{success\_criteria} = \begin{cases}
(\text{HTTP\_status}(\text{endpoint}), 200, =) \\
(\text{response\_time\_p99}, 500\text{ms}, \leq) \\
(\text{test\_coverage}, 0.85, \geq) \\
(\text{security\_scan\_issues}, 0, =)
\end{cases}
$$

**Component 4: Deadline ($\text{deadline}$)**

The temporal bound on goal achievement:

$$
\text{deadline} \in \mathbb{R}^+ \cup \{\infty\}
$$

A goal with $\text{deadline} = \infty$ has no temporal constraint. The deadline induces a **urgency function**:

$$
\text{urgency}(g, t) = \begin{cases}
\frac{t}{\text{deadline}(g)} & \text{if } t < \text{deadline}(g) \\
1 + \alpha \cdot (t - \text{deadline}(g)) & \text{if } t \geq \text{deadline}(g)
\end{cases}
$$

where $\alpha > 0$ penalizes overdue goals, driving the agent to prioritize them.

**Complete Goal Representation Example:**

$$
g_{\text{deploy}} = \left\langle \begin{array}{l}
\text{objective}: \text{``Production deployment of v2.0 is live and healthy''} \\
\text{constraints}: \{\text{no downtime}, \text{budget} \leq \$100, \text{no data loss}\} \\
\text{success\_criteria}: \{(\text{status}, 200, =), (\text{latency}_{p99}, 500\text{ms}, \leq)\} \\
\text{deadline}: 2024\text{-}12\text{-}31T23:59:59Z
\end{array} \right\rangle
$$

---

### 10.1.3 Goal-Directed vs. Reactive Behavior

The distinction between goal-directed and reactive behavior is fundamental to agent architecture and determines the agent's capacity for autonomous, purposeful action.

**Reactive Behavior.** A purely reactive agent maps observations directly to actions without maintaining goal state:

$$
\pi_{\text{reactive}}: \mathcal{O} \rightarrow \mathcal{A}
$$

The agent responds to stimuli in real-time but has no concept of "working toward" anything. Each action is determined solely by the current observation. There is no internal representation of desired future states.

**Properties of reactive systems:**
- No persistent state across decisions
- $O(1)$ memory requirements
- Cannot plan multi-step strategies
- Fast response time but limited task complexity
- Example: a chatbot that simply answers the current question

**Goal-Directed Behavior.** A goal-directed agent maintains an explicit goal representation and conditions its policy on both the current state and the active goal:

$$
\pi_{\text{goal}}: \mathcal{S} \times \mathcal{G} \rightarrow \mathcal{A}
$$

**Properties of goal-directed systems:**
- Persistent goal state across decisions
- Memory proportional to goal complexity
- Can decompose and pursue multi-step strategies
- Slower per-decision (planning overhead) but capable of complex tasks
- Example: a coding agent that plans, implements, tests, and deploys

**Formal Comparison via Controllability:**

The **controllability** of an agent measures the set of states reachable under its policy within a given time horizon $T$:

$$
\text{Reach}_T(\pi, s_0) = \{s \in \mathcal{S} : \exists (a_0, \ldots, a_{T-1}), \; s_T = s \text{ under } \pi\}
$$

For reactive agents:

$$
|\text{Reach}_T(\pi_{\text{reactive}}, s_0)| \leq |\mathcal{A}|^T \quad \text{(but typically much smaller due to no planning)}
$$

For goal-directed agents:

$$
|\text{Reach}_T(\pi_{\text{goal}}, s_0)| \approx |\mathcal{A}|^T \quad \text{(can systematically explore toward any reachable state)}
$$

**Hybrid Architecture.** Modern agentic systems combine both modes:

$$
\pi_{\text{hybrid}}(s, G) = \begin{cases}
\pi_{\text{reactive}}(o_t) & \text{if interrupt detected or safety trigger} \\
\pi_{\text{goal}}(s, G) & \text{otherwise (deliberative mode)}
\end{cases}
$$

The reactive layer handles urgent, safety-critical responses (e.g., stopping a destructive operation immediately), while the goal-directed layer handles strategic planning and execution.

---

### 10.1.4 Relationship to Planning and Evaluation

Goal setting occupies a specific position in the agent's cognitive architecture, mediating between user intent and executable plans.

**The Agent Cognitive Pipeline:**

$$
\underbrace{\text{User Intent}}_{\text{raw}} \xrightarrow{\text{Interpret}} \underbrace{\text{Goal Set}}_{\text{structured}} \xrightarrow{\text{Decompose}} \underbrace{\text{Sub-goals}}_{\text{hierarchy}} \xrightarrow{\text{Plan}} \underbrace{\text{Action Sequence}}_{\text{executable}} \xrightarrow{\text{Execute}} \underbrace{\text{Outcomes}}_{\text{observed}} \xrightarrow{\text{Evaluate}} \underbrace{\text{Goal Status}}_{\text{assessed}}
$$

**Relationship to Planning.** Planning is the process of determining *how* to achieve goals; goal setting determines *what* to achieve. Formally:

$$
\text{Goal Setting}: \text{Intent} \rightarrow \mathcal{G} \quad \text{(specifies the destination)}
$$

$$
\text{Planning}: \mathcal{S} \times \mathcal{G} \rightarrow \mathcal{A}^* \quad \text{(computes the route)}
$$

The quality of planning is fundamentally bounded by the quality of goal specification. A well-defined goal with clear success criteria enables focused, efficient planning. An ambiguous goal forces the planner into exploration, hedging, or clarification-seeking behavior.

**Goal-Plan Consistency Requirement:**

$$
\text{Plan}(s_0, g) = (a_0, a_1, \ldots, a_{T-1}) \implies \text{objective}(s_T) = 1 \quad \text{(under deterministic transitions)}
$$

Under stochastic transitions:

$$
P\left(\text{objective}(s_T) = 1 \;\Big|\; \text{Plan}(s_0, g)\right) \geq 1 - \delta
$$

for confidence level $1 - \delta$.

**Relationship to Evaluation.** Evaluation closes the loop by assessing whether the goal was achieved:

$$
\text{Evaluation}: \mathcal{S}_{\text{final}} \times \mathcal{G} \rightarrow \text{Verdict}
$$

Evaluation feeds back into goal setting through three channels:

1. **Goal completion**: Mark goal as achieved and potentially spawn successor goals
2. **Goal revision**: Modify the goal based on new information encountered during execution
3. **Goal abandonment**: Determine the goal is infeasible and reallocate resources

$$
\text{Feedback loop}: \text{Evaluation}(s_T, g) \xrightarrow{\text{inform}} \text{GoalAdapter}(G_{\text{active}}) \xrightarrow{\text{update}} G_{\text{active}}'
$$

---

## 10.2 Goal Representation

### 10.2.1 Natural Language Goals

The most common goal input modality for LLM-based agents is natural language, provided by the user or generated by the agent itself during self-directed exploration.

**Formal Characterization.** A natural language goal $g_{\text{NL}} \in \Sigma^*$ is an utterance in a natural language $\Sigma$ that encodes the user's intent:

$$
g_{\text{NL}}: \text{``Build a REST API for a todo application with authentication''}
$$

**Advantages:**
- Maximally expressive (natural language is Turing-complete in descriptive power)
- Zero barrier to entry for non-technical users
- Captures nuanced, context-dependent intent
- Allows ambiguity that can be resolved through clarification

**Challenges:**

**Challenge 1: Ambiguity.** Natural language is inherently ambiguous. The same utterance can encode multiple distinct goals:

$$
P(g_{\text{formal}} | g_{\text{NL}}) = \sum_{i} P(g_{\text{formal}} = g_i | g_{\text{NL}}) \quad \text{where } |\{g_i\}| > 1
$$

For the example above, "authentication" could mean OAuth 2.0, API keys, JWT tokens, session-based authentication, or any combination. The entropy of the goal interpretation is:

$$
H[g_{\text{formal}} | g_{\text{NL}}] = -\sum_{i} P(g_i | g_{\text{NL}}) \log P(g_i | g_{\text{NL}})
$$

High entropy indicates ambiguity requiring clarification.

**Challenge 2: Incompleteness.** Users often omit critical specifications, leaving implicit requirements:

$$
g_{\text{explicit}} \subset g_{\text{intended}} \quad \text{(explicit specifications are a subset of intended ones)}
$$

The agent must infer implicit requirements from context, domain knowledge, and conventions:

$$
g_{\text{inferred}} = g_{\text{explicit}} \cup \text{Infer}(g_{\text{explicit}}, \text{context}, \text{domain\_knowledge})
$$

**Challenge 3: Under-Specification of Success Criteria.** "Build a good API" lacks measurable criteria. The agent must operationalize qualitative intent into quantitative thresholds.

**NL Goal Parsing Pipeline:**

$$
g_{\text{NL}} \xrightarrow{\text{LLM Parse}} g_{\text{structured}} = \langle \text{obj}, \text{constraints}, \text{criteria}, \text{deadline} \rangle
$$

```
Prompt Template for Goal Parsing:

Given the user's request, extract a structured goal specification:

User Request: "{g_NL}"

Extract:
1. Primary Objective: [What is the desired end-state?]
2. Constraints: [What limitations or requirements exist?]
3. Success Criteria: [How will we know it's done?]
4. Implicit Assumptions: [What is the user likely assuming?]
5. Clarification Needed: [What is ambiguous?]
6. Deadline: [Any time constraints?]
```

---

### 10.2.2 Formal Goal Specifications (PDDL-Style)

For environments requiring precise, machine-verifiable goal specifications, formal languages like PDDL (Planning Domain Definition Language) provide unambiguous representations.

**PDDL Goal Specification.** In PDDL, goals are expressed as logical formulas over predicates:

$$
\text{Goal}_{\text{PDDL}} = \phi(P_1, P_2, \ldots, P_n)
$$

where $\phi$ is a first-order logic formula and $P_i$ are domain predicates.

**Example: File Organization Goal**

```pddl
(define (problem organize-project)
  (:domain filesystem)
  
  (:objects
    main-py utils-py config-json - file
    src tests docs - directory
  )
  
  (:init
    (in-root main-py) (in-root utils-py) (in-root config-json)
    (is-source main-py) (is-source utils-py) (is-config config-json)
  )
  
  (:goal
    (and
      (in-directory main-py src)
      (in-directory utils-py src)
      (in-directory config-json docs)
      (exists-file "tests/__init__.py" tests)
      (forall (?f - file)
        (imply (is-source ?f) (in-directory ?f src)))
    )
  )
)
```

**Formal Semantics.** A PDDL goal $\phi$ is satisfied in state $s$ iff:

$$
s \models \phi \iff \text{every ground instance of } \phi \text{ evaluates to true under the interpretation induced by } s
$$

**Advantages of Formal Specifications:**
- **Verifiability**: Goal satisfaction is decidable (can be checked automatically)
- **Composability**: Goals combine via logical connectives ($\wedge, \vee, \neg, \rightarrow, \forall, \exists$)
- **Planner compatibility**: Directly usable by classical AI planners (STRIPS, GraphPlan, FF)
- **No ambiguity**: Each formula has exactly one meaning

**Limitations:**
- Requires domain modeling expertise
- Cannot express qualitative, aesthetic, or subjective goals
- Becomes unwieldy for complex real-world domains
- Poor handling of uncertainty and stochastic environments

**Translation: NL → Formal Specification.** LLMs can serve as translators:

$$
\text{LLM}: g_{\text{NL}} \xrightarrow{\text{translate}} g_{\text{formal}}
$$

The translation quality is measured by **semantic fidelity**:

$$
\text{Fidelity}(g_{\text{NL}}, g_{\text{formal}}) = P(\text{user\_satisfied} | \text{Execute}(g_{\text{formal}}))
$$

---

### 10.2.3 Goal Decomposition Hierarchies

Complex goals are rarely achievable in a single action. They must be decomposed into a hierarchy of sub-goals:

$$
G \rightarrow \{G_1, G_2, \ldots, G_k\} \rightarrow \{G_{1.1}, G_{1.2}, \ldots\}
$$

**Formal Definition.** A goal decomposition hierarchy is a tree $\mathcal{T}_G = (V, E)$ where:
- The root $v_0 \in V$ represents the top-level goal $G$
- Each internal node $v_i$ represents a sub-goal $G_i$
- Leaf nodes represent **atomic goals** (directly achievable via a single action or short action sequence)
- Edges $e = (v_{\text{parent}}, v_{\text{child}})$ represent the decomposition relation

**Correctness Criterion.** A decomposition is **correct** if achieving all children guarantees achieving the parent:

$$
\text{Correct}(G \rightarrow \{G_1, \ldots, G_k\}) \iff \left(\bigwedge_{i=1}^{k} \text{Achieved}(G_i)\right) \implies \text{Achieved}(G)
$$

**Completeness Criterion.** A decomposition is **complete** if no necessary sub-goal is missing:

$$
\text{Complete}(G \rightarrow \{G_1, \ldots, G_k\}) \iff \neg\exists G' \notin \{G_1, \ldots, G_k\}: \left(\bigwedge_{i} \text{Achieved}(G_i) \wedge \neg\text{Achieved}(G')\right) \implies \neg\text{Achieved}(G)
$$

**Depth-Bounded Decomposition.** To ensure tractability, decompositions are bounded:

$$
\text{depth}(\mathcal{T}_G) \leq D_{\text{max}}, \quad \text{branching factor}(\mathcal{T}_G) \leq B_{\text{max}}
$$

The total number of leaf goals is bounded by:

$$
|\text{Leaves}(\mathcal{T}_G)| \leq B_{\text{max}}^{D_{\text{max}}}
$$

**Example Decomposition:**

```
G: "Build a web application with user authentication"
├── G1: "Set up project structure"
│   ├── G1.1: "Initialize repository"
│   ├── G1.2: "Configure build tools"
│   └── G1.3: "Set up development environment"
├── G2: "Implement backend API"
│   ├── G2.1: "Design database schema"
│   ├── G2.2: "Implement user model"
│   ├── G2.3: "Implement auth endpoints"
│   │   ├── G2.3.1: "POST /register"
│   │   ├── G2.3.2: "POST /login"
│   │   └── G2.3.3: "POST /logout"
│   └── G2.4: "Implement middleware"
├── G3: "Implement frontend"
│   ├── G3.1: "Create login page"
│   ├── G3.2: "Create registration page"
│   └── G3.3: "Create dashboard"
├── G4: "Write tests"
│   ├── G4.1: "Unit tests"
│   └── G4.2: "Integration tests"
└── G5: "Deploy"
    ├── G5.1: "Configure CI/CD"
    └── G5.2: "Deploy to production"
```

---

### 10.2.4 SMART Goals Framework for Agents

The SMART framework, adapted for agentic systems, ensures goals are well-specified enough for autonomous pursuit.

**SMART Criteria Formalization:**

$$
\text{SMART}(g) = \text{Specific}(g) \wedge \text{Measurable}(g) \wedge \text{Achievable}(g) \wedge \text{Relevant}(g) \wedge \text{Time-bound}(g)
$$

**Specific ($S$).** The goal must unambiguously specify the desired outcome:

$$
S(g) = \mathbb{1}\left[H[g_{\text{formal}} | g_{\text{NL}}] < \epsilon_S\right]
$$

A goal is specific when the entropy of its interpretation is below a threshold $\epsilon_S$. Operationally, this means there is minimal ambiguity about what constitutes success.

**Quantification:** Count the number of distinct valid interpretations $|\mathcal{I}(g)|$. A goal is specific if $|\mathcal{I}(g)| = 1$ (ideally) or $|\mathcal{I}(g)| \leq k_S$ (pragmatically).

- ✗ "Make the code better" → $|\mathcal{I}| \gg 1$ (not specific)
- ✓ "Reduce cyclomatic complexity of function `process_data` to below 10" → $|\mathcal{I}| = 1$ (specific)

**Measurable ($M$).** The goal must have quantifiable success criteria:

$$
M(g) = \mathbb{1}\left[\exists \text{metric}: \mathcal{S} \rightarrow \mathbb{R}, \; \exists \tau \in \mathbb{R}: \; \text{Achieved}(g, s) \equiv \text{metric}(s) \geq \tau\right]
$$

Every goal must be reducible to at least one measurable condition. The measurement function must be computable by the agent or its tools.

- ✗ "Write high-quality documentation" → no metric defined
- ✓ "Write documentation covering all public functions, with readability score > 60 (Flesch-Kincaid)" → measurable

**Achievable ($A$).** The goal must be within the agent's capability space:

$$
A(g) = \mathbb{1}\left[\exists \tau = (a_0, \ldots, a_T): \; \tau \text{ is feasible under } \pi_\theta, \; s_T \models g\right]
$$

Achievability depends on:
1. The agent's action space $\mathcal{A}$ (does it have the necessary tools?)
2. The agent's knowledge (does it know how?)
3. The environment's constraints (is the goal physically/logically possible?)
4. Resource budgets (is there enough time/compute/cost budget?)

**Feasibility Check:**

$$
P(\text{Achieved}(g) | \pi_\theta, s_0, \text{resources}) \geq p_{\text{min}}
$$

If estimated probability falls below $p_{\text{min}}$, the agent should request clarification or resources, or decline the goal.

**Relevant ($R$).** The goal must align with the user's broader intent and the agent's role:

$$
R(g) = \mathbb{1}\left[\text{Contributes}(g, G_{\text{mission}})\right]
$$

An irrelevant goal wastes resources. The agent should validate relevance by checking alignment with the overarching mission.

**Time-bound ($T$).** The goal must have a defined temporal boundary:

$$
T(g) = \mathbb{1}\left[\text{deadline}(g) < \infty\right]
$$

Without a deadline, goals can drift indefinitely, consuming resources without resolution.

**SMART Validation Pipeline:**

$$
g_{\text{raw}} \xrightarrow{\text{SMART check}} \begin{cases}
g_{\text{validated}} & \text{if } \text{SMART}(g) = 1 \\
g_{\text{clarification\_needed}} & \text{if } \text{SMART}(g) = 0
\end{cases}
$$

When a goal fails SMART validation, the agent generates targeted clarification questions:

$$
\text{Questions}(g) = \{q_i : q_i \text{ resolves failure in SMART criterion } c_i\}
$$

---

### 10.2.5 Quantitative vs. Qualitative Goals

**Quantitative Goals.** Goals with numerically measurable success criteria:

$$
g_{\text{quant}} = \{(\text{metric}_i, \text{threshold}_i, \text{comparator}_i)\}_{i=1}^{k}
$$

$$
\text{Achieved}(g_{\text{quant}}, s) = \bigwedge_{i=1}^{k} \text{metric}_i(s) \;\text{comparator}_i\; \text{threshold}_i
$$

**Examples:**
- "Achieve test coverage $\geq 90\%$": $\text{metric} = \text{coverage}$, $\tau = 0.9$, $\text{comp} = \geq$
- "Reduce API latency to $< 100\text{ms}$": $\text{metric} = \text{latency}_{p99}$, $\tau = 100$, $\text{comp} = <$
- "Generate exactly 5 blog post titles": $\text{metric} = |\text{titles}|$, $\tau = 5$, $\text{comp} = =$

**Qualitative Goals.** Goals involving subjective, aesthetic, or contextual judgments that resist direct quantification:

$$
g_{\text{qual}} = \langle \text{description}, \text{evaluation\_rubric} \rangle
$$

$$
\text{Achieved}(g_{\text{qual}}, s) = \text{Judge}(s, \text{rubric}) \geq \tau_{\text{quality}}
$$

where $\text{Judge}$ is either a human evaluator or an LLM-as-judge.

**Examples:**
- "Write an engaging blog post" → requires subjective quality assessment
- "Design an intuitive user interface" → requires usability judgment
- "Produce a professional-looking report" → requires aesthetic evaluation

**Operationalizing Qualitative Goals.** Transform qualitative goals into measurable proxies:

$$
g_{\text{qual}} \xrightarrow{\text{operationalize}} \{g_{\text{quant}_1}, g_{\text{quant}_2}, \ldots, g_{\text{quant}_m}\}
$$

For "Write an engaging blog post":

$$
\text{Proxies} = \begin{cases}
\text{Readability score (Flesch-Kincaid)} \geq 60 \\
\text{Word count} \in [800, 1500] \\
\text{Contains at least 3 section headers} \\
\text{LLM-judge engagement score} \geq 7/10
\end{cases}
$$

The operationalization introduces an **alignment gap**:

$$
\Delta_{\text{align}} = |P(\text{user\_satisfied} | \text{proxies\_met}) - 1|
$$

Perfect proxy alignment ($\Delta_{\text{align}} = 0$) is rare; there is always some Goodhart's Law risk where optimizing proxies diverges from the true qualitative intent.

---

## 10.3 Goal Decomposition Strategies

### 10.3.1 Top-Down Recursive Decomposition

Top-down decomposition starts with the top-level goal and recursively breaks it into sub-goals until atomic (directly executable) goals are reached.

**Algorithm:**

$$
\text{Decompose}(G, d) = \begin{cases}
\{G\} & \text{if } \text{IsAtomic}(G) \text{ or } d \geq D_{\text{max}} \\
\bigcup_{i=1}^{k} \text{Decompose}(G_i, d+1) & \text{where } \{G_1, \ldots, G_k\} = \text{Split}(G)
\end{cases}
$$

The $\text{Split}$ function is implemented by the LLM:

```
Prompt:
Break down the following goal into 3-7 sub-goals.
Each sub-goal should be:
- Necessary for achieving the parent goal
- As independent as possible from other sub-goals
- Roughly similar in complexity

Goal: {G}
Context: {current_state, available_tools, constraints}

Output format:
1. Sub-goal 1: [description] | Dependencies: [list] | Estimated effort: [low/medium/high]
2. Sub-goal 2: ...
```

**Termination Criteria for Atomic Goals:**

$$
\text{IsAtomic}(G) = \begin{cases}
1 & \text{if } G \text{ can be achieved by a single tool call} \\
1 & \text{if estimated steps}(G) \leq k_{\text{atomic}} \\
1 & \text{if } G \text{ matches a known skill in the library} \\
0 & \text{otherwise}
\end{cases}
$$

**Correctness Verification.** After decomposition, verify the decomposition's logical soundness:

$$
\text{Verify}(G \rightarrow \{G_1, \ldots, G_k\}) = \text{LLM}\left(\text{``Does completing } G_1, \ldots, G_k \text{ guarantee achieving } G\text{?''}\right)
$$

If verification fails, the decomposition is revised with additional sub-goals or restructured.

**Complexity Analysis:**

$$
\text{Total sub-goals} = \sum_{d=0}^{D_{\text{max}}} B^d = \frac{B^{D_{\text{max}}+1} - 1}{B - 1} = O(B^{D_{\text{max}}})
$$

For $B = 5, D_{\text{max}} = 3$: approximately 156 sub-goals. Practical decompositions typically have $B \in [2, 7]$ and $D \in [2, 4]$.

---

### 10.3.2 Bottom-Up Goal Assembly

Bottom-up assembly starts with known atomic capabilities and assembles them into higher-level goals.

**Algorithm.** Given the agent's skill library $\mathcal{S} = \{s_1, \ldots, s_n\}$:

1. **Identify relevant skills**: $\mathcal{S}_{\text{rel}} = \{s_i \in \mathcal{S} : \text{Relevant}(s_i, G)\}$
2. **Construct sub-goal DAG**: Create a dependency graph over relevant skills
3. **Verify coverage**: Check if the assembled skills cover the full goal

$$
\text{Coverage}(G, \mathcal{S}_{\text{rel}}) = \frac{|\text{GoalAspects covered by } \mathcal{S}_{\text{rel}}|}{|\text{Total GoalAspects of } G|}
$$

4. **Identify gaps**: If coverage $< 1$, identify missing capabilities:

$$
\text{Gaps}(G, \mathcal{S}_{\text{rel}}) = \text{GoalAspects}(G) \setminus \bigcup_{s_i \in \mathcal{S}_{\text{rel}}} \text{Covers}(s_i)
$$

**Advantages:**
- Grounded in actual capabilities (avoids planning for impossible actions)
- Naturally produces feasible plans
- Leverages existing skill libraries (e.g., Voyager-style)

**Limitations:**
- Cannot discover novel decompositions beyond known skills
- May miss creative problem-solving approaches
- Biased toward familiar strategies

---

### 10.3.3 AND/OR Goal Trees

AND/OR trees provide a richer decomposition structure that captures both mandatory and alternative sub-goal relationships.

**Formal Definition.** An AND/OR goal tree is a tree $\mathcal{T} = (V, E, \text{type})$ where:
- $\text{type}: V \rightarrow \{\text{AND}, \text{OR}, \text{LEAF}\}$
- AND nodes: **All** children must be achieved
- OR nodes: **At least one** child must be achieved
- LEAF nodes: Atomic, directly achievable goals

**Evaluation Semantics:**

$$
\text{Eval}(v) = \begin{cases}
\text{Achieved}(v) & \text{if type}(v) = \text{LEAF} \\
\bigwedge_{\text{child } c \text{ of } v} \text{Eval}(c) & \text{if type}(v) = \text{AND} \\
\bigvee_{\text{child } c \text{ of } v} \text{Eval}(c) & \text{if type}(v) = \text{OR}
\end{cases}
$$

**Example:**

```
G: "Authenticate the user" [AND]
├── G1: "Receive credentials" [AND]
│   ├── G1.1: "Get username" [LEAF]
│   └── G1.2: "Get password" [LEAF]
├── G2: "Validate credentials" [OR]          ← Alternative methods
│   ├── G2.1: "Check against database" [LEAF]
│   ├── G2.2: "Validate via OAuth provider" [LEAF]
│   └── G2.3: "Verify LDAP credentials" [LEAF]
└── G3: "Issue session token" [LEAF]
```

**Optimal Strategy Selection.** For OR nodes, the agent selects the child with the highest expected success probability and lowest cost:

$$
c^* = \arg\max_{c \in \text{children}(v)} \frac{P(\text{Eval}(c) = 1)}{\text{Cost}(c)}
$$

**Probability Computation in AND/OR Trees.** Assuming independent sub-goals:

$$
P(\text{AND node}) = \prod_{i} P(\text{child}_i)
$$

$$
P(\text{OR node}) = 1 - \prod_{i} (1 - P(\text{child}_i))
$$

This enables the agent to estimate the overall success probability of a complex goal and identify the most vulnerable sub-goals (those with the lowest individual success probability).

---

### 10.3.4 Dependency-Aware Decomposition

Sub-goals often have temporal, causal, or resource dependencies that constrain execution order.

**Dependency Graph.** A dependency structure is a DAG $\mathcal{D} = (V_G, E_D)$ where:
- $V_G = \{G_1, \ldots, G_k\}$ is the set of sub-goals
- $(G_i, G_j) \in E_D$ means "$G_i$ must be completed before $G_j$ can begin"

**Dependency Types:**

| Type | Notation | Meaning | Example |
|---|---|---|---|
| **Strict temporal** | $G_i \prec G_j$ | $G_i$ must finish before $G_j$ starts | "Design schema" before "Implement models" |
| **Data dependency** | $G_i \xrightarrow{d} G_j$ | $G_j$ requires output of $G_i$ | "Generate code" before "Write tests for code" |
| **Resource conflict** | $G_i \perp_R G_j$ | Cannot execute simultaneously (shared resource) | Two DB-heavy tasks on same server |
| **Soft preference** | $G_i \rightsquigarrow G_j$ | Preferred but not required ordering | "Write docs" after "Stabilize API" |

**Topological Execution Order.** The valid execution orders are all topological sorts of $\mathcal{D}$:

$$
\text{ValidOrders}(\mathcal{D}) = \{\sigma \in S_k : \forall (G_i, G_j) \in E_D, \; \sigma^{-1}(i) < \sigma^{-1}(j)\}
$$

**Critical Path Analysis.** The critical path determines the minimum total execution time:

$$
T_{\text{min}} = \max_{\text{path } p \in \mathcal{D}} \sum_{G_i \in p} t(G_i)
$$

where $t(G_i)$ is the estimated duration of sub-goal $G_i$. Sub-goals on the critical path cannot be delayed without delaying the overall goal.

**Parallelism Identification.** Sub-goals without dependency relationships can execute in parallel:

$$
\text{Parallelizable}(G_i, G_j) \iff (G_i, G_j) \notin E_D \wedge (G_j, G_i) \notin E_D \wedge G_i \not\perp_R G_j
$$

The maximum parallelism at any point is:

$$
\text{MaxParallel}(\mathcal{D}) = \max_{\text{antichain } A \text{ in } \mathcal{D}} |A|
$$

where an antichain is a set of mutually unordered elements.

---

### 10.3.5 LLM-Based Goal Parsing and Structuring

LLMs serve as the primary mechanism for converting natural language goals into structured, decomposed representations.

**Goal Parsing Pipeline:**

$$
g_{\text{NL}} \xrightarrow{\text{Phase 1: Parse}} g_{\text{structured}} \xrightarrow{\text{Phase 2: Decompose}} \mathcal{T}_G \xrightarrow{\text{Phase 3: Validate}} \mathcal{T}_G^{\text{validated}} \xrightarrow{\text{Phase 4: Enrich}} \mathcal{T}_G^{\text{final}}
$$

**Phase 1: Structured Parsing.** Extract the four-component goal specification:

```
System: You are a goal specification engine. Convert user goals into structured format.

User: "I need to migrate our MySQL database to PostgreSQL by next Friday, 
       without any data loss and with less than 1 hour of downtime."