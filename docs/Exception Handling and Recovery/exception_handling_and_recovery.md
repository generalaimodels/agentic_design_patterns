

# Chapter 11: Exception Handling and Recovery

---

## 11.1 Definition and Formal Framework

### 11.1.1 What is Exception Handling in Agentic Systems

Exception handling in agentic systems is the disciplined practice of detecting, classifying, responding to, and recovering from deviations between the agent's expected execution trajectory and its actual observed behavior—such that the agent either returns to a valid goal-pursuing state or degrades gracefully with full diagnostic transparency.

Unlike traditional software systems where exceptions are well-typed, deterministic, and caught at compile or runtime through structured language constructs (`try/catch`), agentic systems face a fundamentally broader and more insidious class of exceptions. The stochastic nature of LLM outputs, the open-ended action spaces, the multi-step temporal dependencies, and the coupling with external environments create failure modes that are **partially observable, semantically ambiguous, and compositionally emergent**.

**Formal Definition.** An exception handling system for an agent $\mathcal{A}$ is defined as a tuple:

$$
\mathcal{E}\mathcal{H} = \langle \mathcal{D}, \mathcal{C}, \mathcal{R}, \mathcal{S}_{\text{check}}, \mathcal{L} \rangle
$$

where:
- $\mathcal{D}: \mathcal{S} \times \mathcal{O} \times \mathcal{G} \rightarrow \{0, 1\} \times \mathcal{E}$ is the **detection function** that identifies whether an exception has occurred and characterizes it
- $\mathcal{C}: \mathcal{E} \rightarrow \mathcal{T}_{\text{exception}}$ is the **classification function** mapping detected exceptions to a type taxonomy
- $\mathcal{R}: \mathcal{E} \times \mathcal{S} \times \mathcal{G} \rightarrow \mathcal{A}^*$ is the **recovery function** producing a recovery action sequence
- $\mathcal{S}_{\text{check}}: \mathcal{S} \rightarrow \{0, 1\}$ is the **state validation function** (checkpoint integrity)
- $\mathcal{L}: \mathcal{E} \times \mathcal{S} \times \tau \rightarrow \text{LogEntry}$ is the **logging function** for diagnostics and post-mortem analysis

**The Exception Handling Loop.** During agent execution, exception handling operates as a continuous supervisory process:

```
┌──────────────────────────────────────────────────────────┐
│              Agent Execution with Exception Handling     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  for each step t:                                        │
│    1. Agent selects action: a_t = π_θ(s_t, G)            │
│    2. Execute action: o_t = Env(a_t)                     │
│    3. DETECT: (is_exception, e) = D(s_t, o_t, G)        │
│    4. if is_exception:                                   │
│       a. CLASSIFY: type = C(e)                           │
│       b. LOG: L(e, s_t, trajectory)                      │
│       c. RECOVER: recovery_plan = R(e, s_t, G)           │
│       d. Execute recovery_plan                           │
│       e. VALIDATE: assert S_check(s_{t+1})               │
│       f. if not valid: ESCALATE or ABORT                 │
│    5. else: update state s_{t+1}, continue               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Why Agentic Exception Handling is Fundamentally Harder.** The following properties distinguish agentic exceptions from traditional software exceptions:

| Property | Traditional Software | Agentic Systems |
|---|---|---|
| **Exception source** | Deterministic code paths | Stochastic LLM + external world |
| **Detectability** | Explicit (thrown by runtime) | Often implicit (semantic errors, hallucinations) |
| **Type system** | Well-typed exception hierarchy | Open-ended, novel failure modes |
| **Reproducibility** | Deterministic (same input → same error) | Non-deterministic (temperature, randomness) |
| **Blast radius** | Local to call stack | Cascading through multi-step plans |
| **Recovery** | Programmatic (catch/retry) | May require replanning, model switching, human escalation |
| **Partial failure** | Rare (usually all-or-nothing) | Common (partial results, degraded quality) |

---

### 11.1.2 Exception as State Deviation

An exception is formally defined as a measurable deviation between the expected and actual system states:

$$
e = (s_{\text{expected}}, s_{\text{actual}}, \Delta)
$$

where $\Delta$ quantifies the nature and magnitude of the deviation.

**Expected State Model.** The expected state $s_{\text{expected}}$ is derived from the agent's internal model of how its actions should affect the environment:

$$
s_{\text{expected}}(t+1) = \hat{\mathcal{T}}(s_t, a_t)
$$

where $\hat{\mathcal{T}}$ is the agent's **learned or assumed transition model**. This may be:
- An explicit world model (model-based RL)
- An implicit expectation encoded in the LLM's reasoning ("I expect this API call to return JSON")
- A specification-derived postcondition ("After `write_file(path, content)`, `read_file(path)` should return `content`")

**Actual State.** The actual state $s_{\text{actual}}$ is the observed outcome after action execution:

$$
s_{\text{actual}}(t+1) = \mathcal{T}(s_t, a_t) \quad \text{(true environment transition)}
$$

**Deviation Quantification.** The deviation $\Delta$ is a structured representation of the difference:

$$
\Delta = s_{\text{expected}} \ominus s_{\text{actual}} = \langle \Delta_{\text{type}}, \Delta_{\text{magnitude}}, \Delta_{\text{dimensions}} \rangle
$$

**Deviation Type ($\Delta_{\text{type}}$):**

$$
\Delta_{\text{type}} \in \{\text{missing}, \text{unexpected}, \text{incorrect}, \text{timeout}, \text{format\_error}, \text{semantic\_error}\}
$$

| $\Delta_{\text{type}}$ | Description | Example |
|---|---|---|
| $\text{missing}$ | Expected output absent | API returned empty response |
| $\text{unexpected}$ | Received output not predicted by model | Unexpected error code |
| $\text{incorrect}$ | Output present but wrong value | Hallucinated fact in response |
| $\text{timeout}$ | No response within expected time | Tool hung for 60s |
| $\text{format\_error}$ | Output structure invalid | JSON parse error |
| $\text{semantic\_error}$ | Output syntactically valid but semantically wrong | Correct JSON but wrong data |

**Deviation Magnitude ($\Delta_{\text{magnitude}}$):** Quantifies severity on a continuous scale:

$$
\Delta_{\text{magnitude}} = d(s_{\text{expected}}, s_{\text{actual}}) \in [0, \infty)
$$

where $d$ is a domain-appropriate distance metric. For structured states:

$$
d(s_e, s_a) = \sum_{i} w_i \cdot d_i(s_e^{(i)}, s_a^{(i)})
$$

where $s^{(i)}$ are state components (e.g., file content, database state, API response) and $w_i$ are importance weights.

**Deviation Dimensions ($\Delta_{\text{dimensions}}$):** Identifies which state components deviate:

$$
\Delta_{\text{dimensions}} = \{i : d_i(s_{\text{expected}}^{(i)}, s_{\text{actual}}^{(i)}) > \epsilon_i\}
$$

**Exception Severity Classification.** Based on deviation magnitude and impact on goal achievement:

$$
\text{Severity}(e) = f(\Delta_{\text{magnitude}}, \text{GoalImpact}(e))
$$

$$
\text{GoalImpact}(e) = P(\text{GoalAchieved} | e) - P(\text{GoalAchieved} | \neg e)
$$

| Severity Level | $\Delta_{\text{magnitude}}$ | Goal Impact | Response |
|---|---|---|---|
| **TRACE** | Negligible | None | Log only |
| **WARNING** | Small | Minimal degradation | Log + monitor |
| **ERROR** | Moderate | Partial goal failure | Attempt recovery |
| **CRITICAL** | Large | Goal failure likely | Immediate recovery or escalation |
| **FATAL** | Total | Goal impossible | Abort + escalate + rollback |

---

### 11.1.3 Error Taxonomy for Agentic Systems

A comprehensive taxonomy classifies all failure modes along orthogonal axes.

**Axis 1: Source Layer**

$$
\text{Source} \in \{\text{LLM}, \text{Tool}, \text{Workflow}, \text{Environment}, \text{User}, \text{Infrastructure}\}
$$

**Axis 2: Temporal Extent**

$$
\text{Temporal} \in \{\text{Transient}, \text{Intermittent}, \text{Persistent}\}
$$

- **Transient**: Occurs once, unlikely to recur (e.g., network glitch)
- **Intermittent**: Occurs sporadically, non-deterministic (e.g., rate limiting)
- **Persistent**: Occurs consistently until root cause is resolved (e.g., wrong API key)

**Axis 3: Observability**

$$
\text{Observability} \in \{\text{Explicit}, \text{Silent}, \text{Delayed}\}
$$

- **Explicit**: Error is immediately signaled (HTTP 500, exception thrown)
- **Silent**: Error occurs without any signal (hallucinated answer accepted as correct)
- **Delayed**: Error effects manifest later in the pipeline (corrupted intermediate state)

**Axis 4: Recoverability**

$$
\text{Recoverability} \in \{\text{Auto-recoverable}, \text{Retry-recoverable}, \text{Intervention-required}, \text{Irrecoverable}\}
$$

**Combined Taxonomy Table:**

| Error Class | Source | Temporal | Observability | Recoverability | Example |
|---|---|---|---|---|---|
| Hallucination | LLM | Intermittent | Silent | Retry-recoverable | Agent cites nonexistent paper |
| Format violation | LLM | Transient | Explicit | Auto-recoverable | JSON output malformed |
| Tool timeout | Tool | Transient | Explicit | Retry-recoverable | API call hangs |
| Infinite loop | Workflow | Persistent | Delayed | Intervention-required | Agent repeats same action |
| API deprecation | Environment | Persistent | Explicit | Intervention-required | Endpoint returns 410 Gone |
| Data corruption | Workflow | Persistent | Silent/Delayed | Irrecoverable (without backup) | Overwritten file with wrong content |

---

### 11.1.4 Difference from Traditional Software Exception Handling

**Traditional Exception Handling (Java/Python/C++ paradigm):**

```python
try:
    result = risky_operation()
except SpecificError as e:
    handle_specific(e)
except GenericError as e:
    handle_generic(e)
finally:
    cleanup()
```

Properties:
- Exceptions are **typed** objects in a well-defined class hierarchy
- Control flow is **deterministic**: same input triggers same exception
- Recovery is **local**: catch block handles the exception at the call site
- State is **explicit**: stack trace provides full execution context
- Semantics are **well-defined**: language specification governs behavior

**Agentic Exception Handling.** Fundamentally different across every dimension:

**Difference 1: Semantic Exceptions Have No Type.** An LLM hallucinating a function name that doesn't exist isn't a `NameError`—it's a semantically valid output that happens to be factually wrong. There is no exception class for "this plan step is subtly flawed."

$$
\text{Traditional}: e \in \text{ExceptionTypeHierarchy} \quad \text{(finite, enumerable)}
$$

$$
\text{Agentic}: e \in \text{SemanticDeviationSpace} \quad \text{(infinite, open-ended)}
$$

**Difference 2: Non-Deterministic Occurrence.** The same prompt can produce correct output on one run and an exception on the next, due to sampling temperature:

$$
P(e | \text{prompt}, \theta, T) > 0 \quad \text{even when } P(\text{correct} | \text{prompt}, \theta, T) > 0.9
$$

**Difference 3: Cascading Multi-Step Failures.** An error in step 3 of a 10-step plan can corrupt the state in ways that only manifest at step 8:

$$
e_3 \xrightarrow{\text{latent corruption}} s_4 \rightarrow s_5 \rightarrow \ldots \rightarrow e_8^{\text{manifest}}
$$

The root cause ($e_3$) and the symptom ($e_8$) are temporally separated, making diagnosis difficult.

**Difference 4: Recovery May Require Replanning.** In traditional software, recovery is typically a local operation (retry, fallback value, rethrow). In agentic systems, recovery may require completely revising the execution plan:

$$
\text{Traditional}: \text{catch}(e) \rightarrow \text{local fix}
$$

$$
\text{Agentic}: \text{detect}(e) \rightarrow \text{diagnose}(e) \rightarrow \text{replan}(G, s_{\text{current}}) \rightarrow \text{execute(new\_plan)}
$$

**Difference 5: Partial Success is the Norm.** Traditional exceptions are binary (operation succeeds or fails). Agentic operations frequently produce partial results:

$$
\text{Traditional}: \text{Result} \in \{\text{success}, \text{failure}\}
$$

$$
\text{Agentic}: \text{Result} \in [0, 1] \quad \text{(continuous quality spectrum)}
$$

An agent tasked with writing 5 functions might correctly implement 3, partially implement 1, and fail on 1.

---

## 11.2 Types of Exceptions in Agent Workflows

### 11.2.1 LLM-Level Exceptions

LLM-level exceptions originate from the language model itself—the core reasoning engine of the agentic system. These are the most insidious because they can be semantically invisible.

**Hallucination Detection**

Hallucination is the generation of content that is fluent, confident, and structurally valid but factually incorrect, logically inconsistent, or unsupported by the available context.

**Formal Definition.** Let $y = \text{LLM}_\theta(x, c)$ be the model's output given input $x$ and context $c$. A hallucination occurs when:

$$
\text{Hallucination}(y, x, c) = \mathbb{1}\left[\exists y_i \in y : y_i \notin \text{Entailed}(c) \wedge y_i \notin \text{WorldKnowledge}(\theta)\right]
$$

More precisely, we distinguish:

**Intrinsic Hallucination:** Output contradicts the provided context:

$$
\exists y_i \in y : c \models \neg y_i
$$

**Extrinsic Hallucination:** Output cannot be verified or refuted from context:

$$
\exists y_i \in y : c \not\models y_i \wedge c \not\models \neg y_i
$$

**Detection Mechanisms:**

1. **Self-Consistency Check:** Generate $n$ responses and check agreement:
$$
\text{Consistency}(x) = \frac{1}{\binom{n}{2}} \sum_{i<j} \text{sim}(y_i, y_j)
$$
Low consistency indicates potential hallucination:
$$
\text{Hallucination suspected} \iff \text{Consistency}(x) < \tau_{\text{consistency}}
$$

2. **Retrieval-Augmented Verification:** Cross-reference generated claims against retrieved evidence:
$$
\text{Support}(y_i, \text{Evidence}) = \max_{e \in \text{Evidence}} \text{NLI}(e, y_i)
$$
where $\text{NLI}$ is a natural language inference model returning $\{\text{entail}, \text{contradict}, \text{neutral}\}$.

3. **Token-Level Uncertainty:** High entropy in the model's output distribution correlates with hallucination:
$$
H[p_\theta(\cdot | x, y_{<t})] = -\sum_{v \in \mathcal{V}} p_\theta(v | x, y_{<t}) \log p_\theta(v | x, y_{<t})
$$
$$
\text{Hallucination risk at position } t \propto H[p_\theta(\cdot | x, y_{<t})]
$$

**Output Format Violation**

The LLM fails to adhere to the specified output schema.

**Formal Definition.** Given an expected schema $\mathcal{F}$ (e.g., JSON Schema), a format violation occurs when:

$$
\text{FormatViolation}(y, \mathcal{F}) = \mathbb{1}\left[\neg \text{Validate}(y, \mathcal{F})\right]
$$

**Common Format Violations:**

| Violation | Description | Detection |
|---|---|---|
| Invalid JSON | Missing brackets, trailing commas | JSON.parse() fails |
| Schema mismatch | Wrong field names, types, or structure | JSON Schema validation |
| Incomplete output | Response truncated mid-generation | Missing closing delimiters |
| Extra content | Markdown wrapping, explanatory text around JSON | Regex extraction fails |
| Encoding errors | Invalid Unicode, escape sequences | Encoding validation |

**Detection Implementation:**

```python
def detect_format_violation(output: str, schema: dict) -> Exception | None:
    # Step 1: Extract structured content from potentially noisy output
    extracted = extract_json(output)  # handles markdown fences, etc.
    if extracted is None:
        return FormatError("No valid JSON found in output")
    
    # Step 2: Parse
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as e:
        return FormatError(f"JSON parse error: {e}")
    
    # Step 3: Validate against schema
    errors = jsonschema.validate(parsed, schema)
    if errors:
        return SchemaError(f"Schema validation failed: {errors}")
    
    return None  # No violation
```

**Refusal and Safety Triggers**

The LLM declines to perform a requested action due to safety filters, content policies, or alignment training.

$$
\text{Refusal}(x) = \mathbb{1}\left[\text{LLM}_\theta(x) \in \mathcal{R}_{\text{refusal\_patterns}}\right]
$$

**Classification of Refusals:**

| Type | Cause | Appropriate Response |
|---|---|---|
| **True positive** | Request genuinely violates policy | Respect refusal, inform user |
| **False positive (overrefusal)** | Benign request misclassified | Rephrase, add context, use system prompt |
| **Capability limit** | Model lacks ability, not policy | Switch model or approach |

**Overrefusal Detection:**

$$
\text{Overrefusal}(x, y) = \mathbb{1}\left[\text{IsRefusal}(y) \wedge \neg\text{ViolatesPolicy}(x)\right]
$$

Recovery strategies for overrefusal:
1. Rephrase the request to avoid triggering safety classifiers
2. Add explicit context clarifying legitimate use case
3. Break the request into smaller, less suspicious sub-requests
4. Fall back to a different model with different safety thresholds

**Context Length Overflow**

The accumulated context exceeds the model's maximum context window $C_{\max}$:

$$
\text{Overflow}(s_t) = \mathbb{1}\left[|\text{TokenCount}(s_t)| > C_{\max}\right]
$$

**Detection:** Monitor token count before each LLM call:

$$
\text{TokensRequired}(s_t, a_t) = |\text{system\_prompt}| + |\text{history}| + |\text{tools}| + |\text{query}| + |\text{generation\_budget}|
$$

**Recovery Strategies:**

$$
\text{Recovery}_{\text{overflow}} = \begin{cases}
\text{Truncate}(\text{history}, C_{\max} - \text{margin}) & \text{(lossy, simple)} \\
\text{Summarize}(\text{history}) & \text{(lossy, semantic preservation)} \\
\text{SlidingWindow}(\text{history}, W) & \text{(keep recent, drop old)} \\
\text{RAG}(\text{history}) \rightarrow \text{retrieve relevant} & \text{(selective recall)} \\
\text{UpgradeModel}(C_{\max}' > C_{\max}) & \text{(if available)}
\end{cases}
$$

**Rate Limiting and API Errors**

External LLM APIs impose rate limits that produce HTTP 429 (Too Many Requests) errors:

$$
\text{RateLimit}(t) = \mathbb{1}\left[\text{RequestCount}(t - W, t) > R_{\max}\right]
$$

where $W$ is the rate window and $R_{\max}$ is the maximum requests per window.

---

### 11.2.2 Tool-Level Exceptions

Tool-level exceptions arise from the agent's interactions with external tools, APIs, and services.

**Tool Invocation Failure**

The tool fails to execute the requested operation:

$$
\text{ToolFailure}(t, \text{args}) = \mathbb{1}\left[\text{Tool}(t, \text{args}) \rightarrow \text{Error}\right]
$$

**Failure Modes:**

| Mode | HTTP Status | Recovery |
|---|---|---|
| Tool not found | 404 | Check tool name, update tool list |
| Invalid arguments | 400/422 | Validate args against schema, retry with corrected args |
| Server error | 500 | Retry with backoff |
| Authentication failure | 401/403 | Refresh credentials, escalate |
| Resource not found | 404 | Verify resource existence, adjust path |

**Timeout and Unresponsive Tools**

$$
\text{Timeout}(t, \text{args}, T_{\max}) = \mathbb{1}\left[\text{Duration}(\text{Tool}(t, \text{args})) > T_{\max}\right]
$$

**Timeout Strategy:**

$$
T_{\max}(\text{tool}) = \begin{cases}
5\text{s} & \text{if tool is a fast lookup (search, read)} \\
30\text{s} & \text{if tool is a computation (compile, analyze)} \\
300\text{s} & \text{if tool is a long operation (deploy, train)} \\
\text{custom} & \text{if tool declares expected duration}
\end{cases}
$$

**Invalid Tool Arguments**

The LLM generates arguments that do not conform to the tool's input schema:

$$
\text{InvalidArgs}(\text{args}, \text{schema}) = \neg\text{JSONSchema.validate}(\text{args}, \text{schema})
$$

**Common LLM Argument Errors:**

| Error | Example | Root Cause |
|---|---|---|
| Wrong type | `"count": "five"` instead of `"count": 5` | LLM generates string instead of integer |
| Missing required field | `{"query": "..."}` missing `"database"` | LLM omits required parameter |
| Extra fields | `{"query": "...", "verbose": true}` | LLM hallucinates parameter |
| Out-of-range values | `{"limit": -1}` | LLM ignores constraints |
| Injection attempt | `{"path": "../../etc/passwd"}` | Prompt injection attack |

**Permission Denied**

The tool refuses the operation due to insufficient permissions:

$$
\text{PermDenied}(\text{tool}, \text{args}, \text{context}) = \mathbb{1}\left[\text{RequiredPerms}(\text{tool}, \text{args}) \not\subseteq \text{GrantedPerms}(\text{context})\right]
$$

---

### 11.2.3 Workflow-Level Exceptions

Workflow-level exceptions emerge from the interaction patterns between multiple steps, tools, and agent components.

**Infinite Loops**

The agent enters a repeating cycle of actions without making progress:

$$
\text{InfiniteLoop}(\tau, k) = \mathbb{1}\left[\exists (a_t, a_{t+1}, \ldots, a_{t+k-1}) : (a_{t+ik}, \ldots, a_{t+(i+1)k-1}) \cong (a_t, \ldots, a_{t+k-1}) \; \forall i \leq N_{\text{repeat}}\right]
$$

where $\cong$ denotes semantic equivalence (not just string equality) and $N_{\text{repeat}}$ is the repetition threshold.

**Detection Algorithm:**

```python
def detect_loop(trajectory: list[Action], 
                window: int = 3, 
                threshold: int = 3) -> bool:
    """Detect repeating action patterns."""
    for cycle_len in range(1, window + 1):
        if len(trajectory) < cycle_len * threshold:
            continue
        recent = trajectory[-cycle_len * threshold:]
        cycles = [recent[i*cycle_len:(i+1)*cycle_len] 
                  for i in range(threshold)]
        if all(semantic_equal(cycles[0], c) for c in cycles[1:]):
            return True
    return False

def semantic_equal(seq1: list[Action], seq2: list[Action]) -> bool:
    """Check if two action sequences are semantically equivalent."""
    if len(seq1) != len(seq2):
        return False
    return all(
        a1.tool == a2.tool and 
        similarity(a1.args, a2.args) > 0.95
        for a1, a2 in zip(seq1, seq2)
    )
```

**Recovery from Loops:**

$$
\text{Recovery}_{\text{loop}} = \begin{cases}
\text{Inject ``you are repeating'' warning} & \text{(soft break)} \\
\text{Force different action} & \text{(hard break)} \\
\text{Add loop detection context to prompt} & \text{(preventive)} \\
\text{Reset to last non-looping state} & \text{(rollback)} \\
\text{Escalate to human} & \text{(if other strategies fail)}
\end{cases}
$$

**Deadlocks in Multi-Agent Systems**

In systems with multiple agents sharing resources, deadlocks occur when two or more agents wait for each other indefinitely:

$$
\text{Deadlock} \iff \exists \text{cycle in wait-for graph } \mathcal{W} = (V_{\text{agents}}, E_{\text{waits}})
$$

**Formal Conditions (Coffman et al.):** All four must hold simultaneously:
1. **Mutual Exclusion**: At least one resource is held in a non-sharable mode
2. **Hold and Wait**: An agent holds resources while waiting for additional ones
3. **No Preemption**: Resources cannot be forcibly taken from agents
4. **Circular Wait**: $A_1 \rightarrow A_2 \rightarrow \ldots \rightarrow A_n \rightarrow A_1$ in the wait-for graph

**Detection:** Periodically construct the wait-for graph and check for cycles using DFS:

$$
\text{HasCycle}(\mathcal{W}) = \exists v \in V : v \text{ is reachable from itself via directed edges in } \mathcal{W}
$$

**Plan Infeasibility**

The agent discovers mid-execution that its plan cannot achieve the goal:

$$
\text{Infeasible}(G, s_t, \pi) = \mathbb{1}\left[\nexists (a_t, \ldots, a_T) : s_T \models G \text{ starting from } s_t\right]
$$

**Causes:**
- Preconditions for a required action are not satisfiable
- Required resource is unavailable
- Contradictory constraints discovered during execution
- Environment changed since plan was created

**State Corruption**

The agent's internal state or the environment state becomes inconsistent:

$$
\text{Corrupted}(s) = \mathbb{1}\left[\neg \text{Invariant}(s)\right]
$$

where $\text{Invariant}(s)$ encodes domain-specific consistency constraints.

---

### 11.2.4 Environment-Level Exceptions

Environment-level exceptions arise from changes in the external world that invalidate the agent's assumptions.

**External API Changes**

$$
\text{APIChanged}(t) = \mathbb{1}\left[\text{Spec}(\text{API}, t) \neq \text{Spec}(\text{API}, t_{\text{last\_known}})\right]
$$

| Change Type | Detection Signal | Impact |
|---|---|---|
| Endpoint removed | HTTP 404/410 | Tool completely broken |
| Schema changed | Unexpected response structure | Parsing failure |
| Auth method changed | HTTP 401 | All requests fail |
| Rate limits tightened | HTTP 429 more frequent | Throughput degraded |
| Behavioral change | Same schema, different semantics | Silent errors |

**Data Source Unavailability**

$$
\text{Unavailable}(\text{source}) = \mathbb{1}\left[\text{HealthCheck}(\text{source}) = \text{FAIL}\right]
$$

**Network Failures**

$$
\text{NetworkFailure} = \mathbb{1}\left[\text{Latency} > T_{\text{timeout}} \vee \text{PacketLoss} > \rho_{\max}\right]
$$

---

## 11.3 Detection Mechanisms

### 11.3.1 Output Validators and Schema Enforcement

Output validation is the first line of defense against LLM-level exceptions.

**Multi-Layer Validation Pipeline:**

$$
y_{\text{raw}} \xrightarrow{\text{L1: Syntax}} y_{\text{parsed}} \xrightarrow{\text{L2: Schema}} y_{\text{validated}} \xrightarrow{\text{L3: Semantic}} y_{\text{verified}} \xrightarrow{\text{L4: Domain}} y_{\text{accepted}}
$$

**Layer 1: Syntactic Validation.** Verify that the output is well-formed according to the expected format:

$$
\text{SyntaxValid}(y, \text{format}) = \begin{cases}
\text{JSON.parse}(y) \neq \text{error} & \text{if format} = \text{JSON} \\
\text{YAML.parse}(y) \neq \text{error} & \text{if format} = \text{YAML} \\
\text{Regex.match}(y, \text{pattern}) & \text{if format} = \text{regex} \\
\text{XML.parse}(y) \neq \text{error} & \text{if format} = \text{XML}
\end{cases}
$$

**Layer 2: Schema Validation.** Verify structural conformance to the expected schema:

```python
from pydantic import BaseModel, validator
from typing import Literal

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    reasoning: str
    
    @validator('tool_name')
    def tool_must_exist(cls, v):
        if v not in AVAILABLE_TOOLS:
            raise ValueError(f"Unknown tool: {v}")
        return v
    
    @validator('arguments')
    def args_must_match_schema(cls, v, values):
        tool = AVAILABLE_TOOLS.get(values.get('tool_name'))
        if tool:
            jsonschema.validate(v, tool.input_schema)
        return v
```

**Layer 3: Semantic Validation.** Verify that the content is semantically meaningful:

$$
\text{SemanticValid}(y, \text{context}) = \begin{cases}
\text{Coherent}(y) & \text{(no internal contradictions)} \\
\text{Relevant}(y, \text{query}) & \text{(addresses the question)} \\
\text{Grounded}(y, \text{context}) & \text{(supported by provided information)}
\end{cases}
$$

**Layer 4: Domain Validation.** Apply domain-specific business rules:

$$
\text{DomainValid}(y) = \bigwedge_{r \in \text{Rules}} r(y) = \texttt{true}
$$

Example domain rules:
- "File paths must be within the project directory"
- "SQL queries must be SELECT-only (no mutations)"
- "Monetary amounts must be positive"
- "Email addresses must match RFC 5322 format"

---

### 11.3.2 Assertion-Based Checking

Assertions are explicit invariant checks inserted at critical points in the agent's execution pipeline.

**Formal Definition.** An assertion is a predicate that must hold at a specific execution point:

$$
\text{Assert}(P, s_t, \text{message}) = \begin{cases}
\text{continue} & \text{if } P(s_t) = \texttt{true} \\
\text{raise AssertionError}(\text{message}) & \text{if } P(s_t) = \texttt{false}
\end{cases}
$$

**Assertion Categories for Agents:**

**Pre-Condition Assertions (before action execution):**

$$
\text{PreAssert}(a_t, s_t): \text{``State } s_t \text{ satisfies preconditions for action } a_t\text{''}
$$

```python
def pre_assert_tool_call(tool_name: str, args: dict, state: AgentState):
    assert tool_name in state.available_tools, \
        f"Tool '{tool_name}' not available"
    assert state.budget_remaining > 0, \
        "Budget exhausted, cannot make tool calls"
    assert state.step_count < MAX_STEPS, \
        f"Maximum steps ({MAX_STEPS}) exceeded"
    assert not state.is_terminated, \
        "Agent has already terminated"
```

**Post-Condition Assertions (after action execution):**

$$
\text{PostAssert}(a_t, s_t, s_{t+1}): \text{``State } s_{t+1} \text{ reflects expected effect of } a_t\text{''}
$$

```python
def post_assert_file_write(path: str, content: str):
    assert os.path.exists(path), \
        f"File {path} should exist after write"
    actual = open(path).read()
    assert actual == content, \
        f"File content mismatch: expected {len(content)} chars, got {len(actual)}"
```

**Invariant Assertions (must hold at all times):**

$$
\text{InvAssert}(s_t): \text{``Global invariants hold in state } s_t\text{''}
$$

```python
def invariant_check(state: AgentState):
    assert len(state.conversation_history) <= MAX_HISTORY, \
        "Conversation history exceeds safe limit"
    assert state.total_cost <= state.cost_budget, \
        f"Cost {state.total_cost} exceeds budget {state.cost_budget}"
    assert not detect_loop(state.action_history), \
        "Agent appears to be in an infinite loop"
    assert state.goal is not None, \
        "Agent is operating without an active goal"
```

---

### 11.3.3 Anomaly Detection in Agent Behavior

Anomaly detection identifies statistically unusual agent behavior patterns that may indicate subtle errors not caught by explicit validators.

**Formal Framework.** Define a behavioral profile $\mathcal{B}$ as the distribution over agent actions and outputs during normal operation:

$$
\mathcal{B} = P(a_t, o_t, \text{duration}_t, \text{tokens}_t | s_t, G)
$$

An anomaly is detected when the observed behavior deviates significantly from this profile:

$$
P(\text{anomaly} | o_t) = P(o_t \notin \text{NormalRegion}(\mathcal{B})) > \tau
$$

**Anomaly Detection Methods:**

**Method 1: Statistical Process Control.** Monitor key metrics over a sliding window and flag deviations beyond control limits:

$$
\text{Anomaly}(x_t) = \mathbb{1}\left[|x_t - \bar{x}| > k \cdot \sigma_x\right]
$$

where $\bar{x}$ and $\sigma_x$ are computed over a recent window of normal operation, and $k$ is typically set to 3 (three-sigma rule).

**Metrics to monitor:**

| Metric | Normal Range | Anomaly Indicates |
|---|---|---|
| Response length (tokens) | $\mu \pm 3\sigma$ | Hallucination (too long) or failure (too short) |
| Tool call frequency | $[f_{\min}, f_{\max}]$ per episode | Loop (too high) or stall (too low) |
| Action diversity | $H[\text{action dist}] > \epsilon$ | Loop (low entropy) |
| Execution time per step | $[t_{\min}, t_{\max}]$ | Timeout (too long) or shortcut (too short) |
| Error rate | $< \rho_{\max}$ | Systematic failure |

**Method 2: Sequence-Level Anomaly Detection.** Use a trained model to score the likelihood of action sequences:

$$
\text{AnomalyScore}(\tau) = -\frac{1}{T} \sum_{t=1}^{T} \log P_{\text{model}}(a_t | a_{<t}, s_t)
$$

High anomaly scores indicate sequences that are unlikely under the learned behavioral model.

**Method 3: Goal-Progress Anomaly.** Monitor whether progress toward the goal is consistent with expected trajectory:

$$
\text{ExpectedProgress}(t) = f_{\text{plan}}(t) \quad \text{(derived from the plan)}
$$

$$
\text{ProgressAnomaly}(t) = |\text{ActualProgress}(t) - \text{ExpectedProgress}(t)| > \delta
$$

---

### 11.3.4 Watchdog Timers and Heartbeats

**Watchdog Timer.** A countdown timer that must be periodically reset ("fed") by the agent. If the timer expires, the system assumes the agent is stuck:

$$
\text{Watchdog}(T_{\text{wd}}) = \begin{cases}
\text{Reset timer to } T_{\text{wd}} & \text{when agent reports progress} \\
\text{Trigger recovery} & \text{when timer reaches 0}
\end{cases}
$$

**Implementation:**

```python
class WatchdogTimer:
    def __init__(self, timeout: float, on_expire: Callable):
        self.timeout = timeout
        self.on_expire = on_expire
        self._last_reset = time.time()
        self._task = asyncio.create_task(self._monitor())
    
    def feed(self):
        """Reset the watchdog. Call this on every meaningful progress."""
        self._last_reset = time.time()
    
    async def _monitor(self):
        while True:
            await asyncio.sleep(1.0)
            elapsed = time.time() - self._last_reset
            if elapsed > self.timeout:
                await self.on_expire()
                break  # or reset for recurring monitoring
```

**Heartbeat Mechanism.** The agent periodically emits heartbeat signals confirming it is alive and making progress:

$$
\text{Heartbeat}(t) = (\text{timestamp}, \text{step\_count}, \text{current\_action}, \text{progress\_pct})
$$

Missing heartbeats trigger escalation:

$$
\text{MissedHeartbeats}(t) = \left\lfloor\frac{t - t_{\text{last\_heartbeat}}}{\text{heartbeat\_interval}}\right\rfloor
$$

$$
\text{Action}(\text{missed}) = \begin{cases}
\text{none} & \text{if missed} = 0 \\
\text{log warning} & \text{if missed} = 1 \\
\text{send probe} & \text{if missed} = 2 \\
\text{force restart} & \text{if missed} \geq 3
\end{cases}
$$

---

### 11.3.5 LLM-Based Error Detection

The LLM itself can serve as an error detector, evaluating its own outputs or the outputs of other agents and tools for correctness.

**Self-Verification Pattern:**

$$
\text{Verify}(y, x, c) = \text{LLM}_\theta\left(\text{``Is the following output correct given the input and context?''} \oplus x \oplus c \oplus y\right)
$$

**Implementation:**

```python
async def llm_verify(output: str, query: str, context: str) -> VerificationResult:
    verification_prompt = f"""
    Analyze the following output for errors, inconsistencies, or issues.
    
    Original Query: {query}
    Context: {context}
    Output to Verify: {output}
    
    Check for:
    1. Factual accuracy (does it contradict known facts or provided context?)
    2. Logical consistency (does it contradict itself?)
    3. Completeness (does it fully address the query?)
    4. Format correctness (does it follow the expected format?)
    5. Safety (does it contain harmful content?)
    
    Respond with JSON:
    {{
        "is_valid": true/false,
        "issues": ["list of issues found"],
        "confidence": 0.0-1.0,
        "severity": "none|low|medium|high|critical"
    }}
    """
    return await llm.generate(verification_prompt, format="json")
```

**Limitation: Self-Verification Reliability.** An LLM verifying its own output is susceptible to correlated errors—the same knowledge gaps that caused the original error may cause verification to miss it:

$$
P(\text{Verify catches error}) = 1 - P(\text{same blind spot})
$$

**Mitigation:** Use a different model, different prompt framing, or different temperature for verification:

$$
\text{Verify}(y) = \text{LLM}_{\theta_2}(y) \quad \text{where } \theta_2 \neq \theta_1 \text{ (model diversity)}
$$

Or use an ensemble:

$$
\text{Verified}(y) = \mathbb{1}\left[\frac{1}{K}\sum_{k=1}^{K} \text{LLM}_{\theta_k}(\text{``Is } y \text{ correct?''}) > \tau_{\text{verify}}\right]
$$

---

## 11.4 Recovery Strategies

### 11.4.1 Retry with Backoff

The simplest recovery strategy: re-attempt the failed operation after a delay, with exponentially increasing wait times to avoid overwhelming the failing resource.

$$
t_{\text{wait}}(n) = \min(t_{\text{base}} \cdot 2^n + \text{jitter}(n), \; t_{\text{max}})
$$

where:
- $n$ is the retry attempt number (starting from 0)
- $t_{\text{base}}$ is the initial wait time (e.g., 1 second)
- $\text{jitter}(n) \sim \text{Uniform}(0, t_{\text{base}} \cdot 2^n)$ prevents thundering herd
- $t_{\text{max}}$ is the maximum wait time cap

**Retry Budget.** Each operation has a maximum retry count $N_{\max}$:

$$
\text{TotalWaitTime} = \sum_{n=0}^{N_{\max}-1} t_{\text{wait}}(n) \leq \sum_{n=0}^{N_{\max}-1} t_{\text{base}} \cdot 2^{n+1} = t_{\text{base}} \cdot (2^{N_{\max}+1} - 2)
$$

For $t_{\text{base}} = 1\text{s}$, $N_{\max} = 5$: maximum total wait $= 62\text{s}$.

**Retry Decision Function:**

$$
\text{ShouldRetry}(e, n) = \begin{cases}
\texttt{true} & \text{if } n < N_{\max} \wedge \text{IsTransient}(e) \wedge \text{BudgetRemaining} > 0 \\
\texttt{false} & \text{otherwise}
\end{cases}
$$

**Transient Error Classification:**

$$
\text{IsTransient}(e) = \begin{cases}
\texttt{true} & \text{if } e.\text{code} \in \{408, 429, 500, 502, 503, 504\} \\
\texttt{true} & \text{if } e.\text{type} \in \{\text{timeout}, \text{connection\_reset}, \text{rate\_limit}\} \\
\texttt{false} & \text{if } e.\text{code} \in \{400, 401, 403, 404, 422\} \\
\texttt{false} & \text{if } e.\text{type} \in \{\text{invalid\_args}, \text{auth\_failure}, \text{not\_found}\}
\end{cases}
$$

Non-transient errors should not be retried without modification—they require a different recovery strategy.

---

### 11.4.2 Retry with Modified Prompt/Strategy

When a retry with identical input is unlikely to succeed (e.g., the LLM consistently produces a format violation), modify the prompt or strategy before retrying.

**Modification Strategies:**

**Strategy 1: Prompt Augmentation (add error feedback):**

$$
\text{prompt}_{n+1} = \text{prompt}_n \oplus \text{``Previous attempt failed with error: ''} \oplus e_n \oplus \text{``Please correct and try again.''}
$$

**Strategy 2: Increased Specificity:**

$$
\text{prompt}_{n+1} = \text{prompt}_n \oplus \text{``IMPORTANT: Output MUST be valid JSON. Do not include any text outside the JSON object.''}
$$

**Strategy 3: Example Injection (few-shot recovery):**

$$
\text{prompt}_{n+1} = \text{prompt}_n \oplus \text{``Here is an example of the expected output format:''} \oplus \text{example}
$$

**Strategy 4: Temperature Adjustment:**

$$
T_{n+1} = \begin{cases}
T_n \cdot 0.5 & \text{if error was format violation (reduce randomness)} \\
T_n \cdot 1.5 & \text{if error was content repetition (increase randomness)} \\
T_n & \text{otherwise}
\end{cases}
$$

**Strategy 5: Decomposition (break the request into simpler parts):**

$$
\text{FailedRequest}(x) \rightarrow \text{SubRequest}_1(x_1), \text{SubRequest}_2(x_2), \ldots
$$

**Implementation:**

```python
async def retry_with_modification(
    operation: Callable,
    max_retries: int = 3,
    modifiers: list[PromptModifier] = None
) -> Result:
    errors = []
    for attempt in range(max_retries):
        try:
            prompt = base_prompt
            # Apply modifications based on previous errors
            for modifier in (modifiers or []):
                prompt = modifier.apply(prompt, errors)
            result = await operation(prompt)
            validate(result)
            return result
        except Exception as e:
            errors.append(e)
            if not should_retry(e, attempt):
                raise
            await asyncio.sleep(backoff(attempt))
    raise MaxRetriesExceeded(errors)
```

---

### 11.4.3 Fallback to Alternative Model/Tool

When the primary model or tool consistently fails, switch to an alternative:

$$
\text{Fallback}(e, M_{\text{primary}}) = \begin{cases}
M_{\text{secondary}} & \text{if } M_{\text{secondary}} \text{ available and capable} \\
M_{\text{tertiary}} & \text{if } M_{\text{secondary}} \text{ also fails} \\
\text{escalate} & \text{if no alternatives remain}
\end{cases}
$$

**Model Fallback Chain:**

$$
\text{Claude 3.5 Sonnet} \xrightarrow{\text{fail}} \text{GPT-4o} \xrightarrow{\text{fail}} \text{Claude 3 Haiku (simpler task)} \xrightarrow{\text{fail}} \text{Escalate}
$$

**Tool Fallback Chain:**

$$
\text{Primary API} \xrightarrow{\text{fail}} \text{Backup API} \xrightarrow{\text{fail}} \text{Cached results} \xrightarrow{\text{fail}} \text{Manual workaround}
$$

**Fallback Selection Criteria:**

$$
M_{\text{fallback}} = \arg\max_{M \in \mathcal{M}_{\text{available}}} \frac{P(\text{success} | M, \text{task}) \cdot \text{Quality}(M, \text{task})}{\text{Cost}(M) \cdot \text{Latency}(M)}
$$

---

### 11.4.4 Graceful Degradation

When full recovery is impossible, the agent delivers a reduced-quality result rather than complete failure.

**Degradation Hierarchy:**

$$
\text{Degradation levels} = [\text{Full}, \text{Reduced}, \text{Minimal}, \text{Informative Failure}]
$$

| Level | Description | Example |
|---|---|---|
| **Full** | Complete goal achievement | All 5 functions implemented and tested |
| **Reduced** | Partial goal with acknowledged gaps | 3/5 functions implemented, 2 stubbed |
| **Minimal** | Core functionality only | Main function implemented, no tests |
| **Informative Failure** | No output but useful diagnostics | "Failed because API X is down; here's what I completed so far" |

**Formal Degradation Function:**

$$
\text{Output}(G, s_t, e) = \begin{cases}
\text{FullResult}(G) & \text{if no exception} \\
\text{PartialResult}(G_{\text{completed}}) \oplus \text{Report}(G_{\text{remaining}}, e) & \text{if partial completion} \\
\text{DiagnosticReport}(e, \text{trajectory}) & \text{if total failure}
\end{cases}
$$

The agent explicitly communicates what was achieved, what remains, and why full completion failed.

---

### 11.4.5 Rollback to Last Known Good State

When an error corrupts the agent's state or the environment, restore from the most recent checkpoint.

**Rollback Operation:**

$$
\text{Rollback}(s_t, \text{checkpoint}_k) = s_t \leftarrow s_{\text{checkpoint}_k}
$$

where $\text{checkpoint}_k$ is the most recent state satisfying:

$$
k^* = \arg\max_{k} \{k : \text{Invariant}(s_{\text{checkpoint}_k}) = \texttt{true} \wedge k \leq t\}
$$

**Rollback Completeness.** A rollback is complete only if **all side effects** since the checkpoint are reversed:

$$
\text{CompleteRollback}(s_t, s_k) = \text{Undo}(\text{SideEffects}(a_k, a_{k+1}, \ldots, a_{t-1}))
$$

For irreversible side effects (sent emails, deleted files without backup, deployed code), complete rollback is impossible. This motivates **compensation actions** (Section 11.6.4).

---

### 11.4.6 Escalation to Human Operator

When automated recovery strategies are exhausted or the situation requires judgment beyond the agent's capability:

**Escalation Triggers:**

$$
\text{Escalate}(e, s, G) = \begin{cases}
\texttt{true} & \text{if retry\_count} \geq N_{\max} \text{ and all fallbacks failed} \\
\texttt{true} & \text{if } e.\text{severity} = \text{CRITICAL or FATAL} \\
\texttt{true} & \text{if action requires authorization not granted to agent} \\
\texttt{true} & \text{if ambiguity cannot be resolved programmatically} \\
\texttt{true} & \text{if safety constraint may be violated} \\
\texttt{false} & \text{otherwise}
\end{cases}
$$

**Escalation Message Structure:**

$$
\text{Escalation} = \langle \text{context}, \text{error}, \text{actions\_taken}, \text{options}, \text{recommendation}, \text{urgency} \rangle
$$

```
ESCALATION TO HUMAN OPERATOR

Context: Building REST API for todo application (Step 4/7)
Error: Database migration failed - PostgreSQL version incompatible
Actions Taken:
  1. Attempted migration with psycopg2 → Version mismatch error
  2. Attempted fallback to SQLite → Schema incompatibility
  3. Searched documentation → No clear resolution found

Options:
  A. Upgrade PostgreSQL to version 15+ (requires admin access)
  B. Rewrite schema to be compatible with current version
  C. Switch to a different database entirely

Recommendation: Option A (minimal code changes, best long-term solution)
Urgency: MEDIUM (blocking further progress but no data loss)

Awaiting your decision to proceed.
```

---

## 11.5 Checkpoint and State Management

### 11.5.1 Workflow Checkpointing

Checkpointing periodically saves the agent's complete state, enabling recovery without re-executing the entire workflow.

**Formal Definition.** A checkpoint is a serialized snapshot of the agent's state at a specific execution point:

$$
\text{Checkpoint}_k = \text{Serialize}(s_{t_k}) = (\text{conversation}, \text{plan}, \text{progress}, \text{memory}, \text{env\_state}, \text{metadata})
$$

**Checkpointing Policy.** Determine when to create checkpoints:

$$
\text{ShouldCheckpoint}(s_t, t) = \begin{cases}
\texttt{true} & \text{if } t \mod T_{\text{interval}} = 0 \quad \text{(periodic)} \\
\texttt{true} & \text{if sub-goal completed} \quad \text{(milestone)} \\
\texttt{true} & \text{if about to execute destructive action} \quad \text{(pre-caution)} \\
\texttt{true} & \text{if state significantly changed} \quad \text{(delta-based)} \\
\texttt{false} & \text{otherwise}
\end{cases}
$$

**Checkpoint Storage Requirements:**

$$
\text{Storage}(\text{Checkpoint}) = |\text{conversation}| + |\text{plan}| + |\text{memory}| + |\text{env\_snapshot}|
$$

For long-running agents, checkpoint storage can grow significantly. **Incremental checkpointing** reduces this:

$$
\text{IncrementalCheckpoint}_k = \text{Delta}(s_{t_k}, s_{t_{k-1}})
$$

Only the state changes since the last checkpoint are stored.

**Checkpoint Integrity Verification:**

$$
\text{VerifyCheckpoint}(\text{cp}) = \begin{cases}
\text{Hash}(\text{cp.data}) = \text{cp.hash} & \text{(data integrity)} \\
\text{Deserialize}(\text{cp.data}) \neq \text{error} & \text{(format validity)} \\
\text{Invariant}(\text{Deserialize}(\text{cp.data})) & \text{(state consistency)}
\end{cases}
$$

---

### 11.5.2 Idempotent Operation Design

An operation is **idempotent** if executing it multiple times produces the same result as executing it once:

$$
f(f(x)) = f(x) \quad \forall x
$$

In the agentic context:

$$
\text{Idempotent}(a) \iff \forall s: \mathcal{T}(s, a) = \mathcal{T}(\mathcal{T}(s, a), a)
$$

**Why Idempotency Matters.** During recovery, the agent may re-execute operations that partially completed before the failure. Idempotent operations are safe to retry without side-effect duplication.

**Idempotent vs. Non-Idempotent Operations:**

| Operation | Idempotent? | Why |
|---|---|---|
| `read_file(path)` | ✓ | Reading doesn't change state |
| `write_file(path, content)` | ✓ | Same content written regardless of prior state |
| `append_file(path, content)` | ✗ | Each call adds more content |
| `PUT /users/42 {name: "Alice"}` | ✓ | Same result on repeat |
| `POST /users {name: "Alice"}` | ✗ | Creates new user each time |
| `DELETE /users/42` | ✓* | First call deletes; subsequent calls are no-op (or 404) |
| `send_email(to, subject, body)` | ✗ | Each call sends another email |

**Making Non-Idempotent Operations Safe:**

**Technique 1: Idempotency Keys.** Assign a unique key to each operation; the server deduplicates:

$$
\text{Execute}(a, \text{key}) = \begin{cases}
\text{Execute}(a) \text{ and store result for key} & \text{if key not seen} \\
\text{Return stored result for key} & \text{if key already seen}
\end{cases}
$$

**Technique 2: Check-Before-Act.** Before executing, check if the desired state already exists:

```python
async def idempotent_create_file(path: str, content: str):
    if os.path.exists(path):
        existing = open(path).read()
        if existing == content:
            return  # Already done, skip
    open(path, 'w').write(content)
```

**Technique 3: Conditional Execution.** Use precondition checks (ETags, version numbers):

$$
\text{Execute}(a) \text{ only if } \text{version}(s) = v_{\text{expected}}
$$

---

### 11.5.3 Transaction Semantics in Agent Workflows

Agent workflows often need to execute multiple steps as an atomic unit—either all succeed or all are rolled back.

**ACID Properties Adapted for Agents:**

| ACID Property | Traditional DB | Agentic Workflow Adaptation |
|---|---|---|
| **Atomicity** | Transaction commits or rolls back entirely | Workflow step group succeeds or all side effects are undone |
| **Consistency** | DB moves from one valid state to another | Agent state satisfies invariants after each step group |
| **Isolation** | Concurrent transactions don't interfere | Concurrent agents don't corrupt shared resources |
| **Durability** | Committed data persists | Completed sub-goals persist through failures |

**Transaction Model for Agent Workflows:**

$$
\text{Transaction}(\text{steps}) = \begin{cases}
\text{Commit} & \text{if all steps succeed and postconditions hold} \\
\text{Rollback} & \text{if any step fails and compensation is possible} \\
\text{Compensate} & \text{if partial completion with irreversible steps}
\end{cases}
$$

**Implementation using the Unit-of-Work Pattern:**

```python
class AgentTransaction:
    def __init__(self):
        self.operations = []
        self.compensations = []
        self.committed = False
    
    async def execute(self, operation, compensation):
        """Execute operation; register compensation for rollback."""
        try:
            result = await operation()
            self.operations.append((operation, result))
            self.compensations.append(compensation)
            return result
        except Exception as e:
            await self.rollback()
            raise
    
    async def rollback(self):
        """Undo all completed operations in reverse order."""
        for compensation in reversed(self.compensations):
            try:
                await compensation()
            except Exception as e:
                log.error(f"Compensation failed: {e}")
                # Compensation failure is logged but doesn't stop rollback
    
    async def commit(self):
        """Mark transaction as committed; discard compensations."""
        self.committed = True
        self.compensations.clear()
```

---

### 11.5.4 State Serialization and Deserialization

Robust checkpoint and recovery requires reliable state serialization.

**Serialization Requirements:**

$$
\text{Deserialize}(\text{Serialize}(s)) = s \quad \forall s \in \mathcal{S} \quad \text{(round-trip fidelity)}
$$

**State Components and Serialization Strategies:**

| Component | Type | Serialization Method |
|---|---|---|
| Conversation history | List of messages | JSON array |
| Plan state | Tree/DAG of sub-goals | JSON with references |
| Tool results cache | Dict of results | JSON with type annotations |
| File modifications | Set of file diffs | Unified diff format |
| Environment state | External system state | Snapshot or reference (URI) |
| Model state | Token count, context window | Numeric metadata |
| Execution counters | Step count, retry count, cost | Integer fields |

**Serialization Format:**

```json
{
    "version": "1.0",
    "timestamp": "2024-12-15T10:30:00Z",
    "checkpoint_id": "cp_abc123",
    "agent_state": {
        "goal": {
            "objective": "Build REST API",
            "progress": 0.45,
            "sub_goals_completed": ["G1", "G2.1", "G2.2"],
            "sub_goals_pending": ["G2.3", "G3", "G4", "G5"]
        },
        "conversation": [...],
        "plan": {...},
        "memory": {...},
        "metrics": {
            "steps_executed": 23,
            "total_tokens": 45000,
            "total_cost_usd": 0.87,
            "errors_encountered": 2,
            "errors_recovered": 2
        }
    },
    "environment_snapshot": {
        "files_created": ["src/models.py", "src/routes.py"],
        "files_modified": ["requirements.txt"],
        "external_state_refs": ["db://migration_v3"]
    },
    "integrity": {
        "hash": "sha256:a1b2c3...",
        "schema_version": "1.0"
    }
}
```

---

## 11.6 Fault Tolerance Patterns

### 11.6.1 Circuit Breaker Pattern

The circuit breaker prevents an agent from repeatedly calling a failing service, avoiding resource waste and cascading failures.

**State Machine:**

$$
\text{CircuitBreaker} \in \{\text{CLOSED}, \text{OPEN}, \text{HALF-OPEN}\}
$$

$$
\text{CLOSED} \xrightarrow{\text{failure count} \geq N_{\text{threshold}}} \text{OPEN} \xrightarrow{\text{timeout } T_{\text{reset}}} \text{HALF-OPEN} \xrightarrow{\text{success}} \text{CLOSED}
$$

$$
\text{HALF-OPEN} \xrightarrow{\text{failure}} \text{OPEN}
$$

**Formal Definition:**

$$
\text{CircuitBreaker.call}(f, \text{args}) = \begin{cases}
f(\text{args}) & \text{if state} = \text{CLOSED} \\
\text{raise CircuitOpen} & \text{if state} = \text{OPEN} \\
f(\text{args}) \text{ (probe)} & \text{if state} = \text{HALF-OPEN}
\end{cases}
$$

**Implementation:**

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, 
                 reset_timeout: float = 60.0,
                 half_open_max_calls: int = 1):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.half_open_calls = 0
        self.half_open_max = half_open_max_calls
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
            else:
                raise CircuitOpenError(
                    f"Circuit open. Retry after {self.reset_timeout}s"
                )
        
        if self.state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max:
            raise CircuitOpenError("Half-open call limit reached")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
```

**Per-Service Circuit Breakers in Agents:**

$$
\text{Agent} \rightarrow \begin{cases}
\text{CB}_{\text{LLM API}} & \text{(protects against LLM provider outage)} \\
\text{CB}_{\text{Search API}} & \text{(protects against search service failure)} \\
\text{CB}_{\text{Database}} & \text{(protects against database overload)} \\
\text{CB}_{\text{File System}} & \text{(protects against disk issues)}
\end{cases}
$$

---

### 11.6.2 Bulkhead Pattern (Failure Isolation)

The bulkhead pattern isolates failures to prevent them from cascading across the entire system—inspired by ship bulkheads that contain flooding to one compartment.

**Formal Definition.** Partition the agent's resources into isolated pools:

$$
\mathcal{R}_{\text{total}} = \bigsqcup_{i=1}^{k} \mathcal{R}_i \quad \text{(disjoint partition)}
$$

A failure in partition $\mathcal{R}_i$ cannot consume resources from partition $\mathcal{R}_j$ ($i \neq j$):

$$
\text{Failure}(\mathcal{R}_i) \implies \mathcal{R}_j \text{ unaffected} \quad \forall j \neq i
$$

**Implementation for Agents:**

| Bulkhead | Isolated Resource | Protection Against |
|---|---|---|
| LLM call pool | Max concurrent LLM requests | One task monopolizing LLM quota |
| Tool execution pool | Max concurrent tool calls | Runaway tool invocations |
| Memory pool | Max memory per task | Memory leak in one task |
| Cost budget per sub-task | Dollar allocation per sub-goal | One sub-goal consuming entire budget |

```python
class BulkheadPool:
    def __init__(self, name: str, max_concurrent: int, max_queue: int = 100):
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue(maxsize=max_queue)
    
    async def execute(self, func, *args, timeout: float = 30.0):
        try:
            async with asyncio.timeout(timeout):
                async with self.semaphore:
                    return await func(*args)
        except asyncio.TimeoutError:
            raise BulkheadTimeoutError(
                f"Bulkhead '{self.name}' timeout after {timeout}s"
            )

# Usage: Separate pools for different service types
llm_pool = BulkheadPool("llm", max_concurrent=5)
tool_pool = BulkheadPool("tools", max_concurrent=10)
db_pool = BulkheadPool("database", max_concurrent=3)
```

---

### 11.6.3 Saga Pattern for Distributed Agent Workflows

The Saga pattern manages long-running, distributed transactions by breaking them into a sequence of local transactions, each with a compensating action.

**Formal Definition.** A saga is a sequence of transactions $T_1, T_2, \ldots, T_n$ with corresponding compensations $C_1, C_2, \ldots, C_n$:

$$
\text{Saga} = [(T_1, C_1), (T_2, C_2), \ldots, (T_n, C_n)]
$$

**Execution Semantics:**

$$
\text{Execute}(\text{Saga}) = \begin{cases}
T_1 \rightarrow T_2 \rightarrow \ldots \rightarrow T_n & \text{if all succeed} \\
T_1 \rightarrow \ldots \rightarrow T_k \rightarrow C_k \rightarrow C_{k-1} \rightarrow \ldots \rightarrow C_1 & \text{if } T_{k+1} \text{ fails}
\end{cases}
$$

**Example: Agent Deploying a Feature**

$$
\text{Saga}_{\text{deploy}} = \begin{bmatrix}
(T_1: \text{CreateBranch}, & C_1: \text{DeleteBranch}) \\
(T_2: \text{WriteCode}, & C_2: \text{RevertCommits}) \\
(T_3: \text{RunTests}, & C_3: \text{no-op}) \\
(T_4: \text{CreatePR}, & C_4: \text{ClosePR}) \\
(T_5: \text{MergePR}, & C_5: \text{RevertMerge}) \\
(T_6: \text{Deploy}, & C_6: \text{Rollback Deployment})
\end{bmatrix}
$$

If $T_3$ (RunTests) fails:
$$
T_1 \rightarrow T_2 \rightarrow T_3 \text{ (FAIL)} \rightarrow C_2 \text{ (RevertCommits)} \rightarrow C_1 \text{ (DeleteBranch)}
$$

**Orchestration vs. Choreography:**

**Orchestrator-based Saga:** A central coordinator directs the execution:

$$
\text{Orchestrator}: T_1 \xrightarrow{\text{success}} T_2 \xrightarrow{\text{success}} \ldots \xrightarrow{\text{fail}} C_k \xrightarrow{} C_{k-1} \xrightarrow{} \ldots
$$

**Choreography-based Saga:** Each step triggers the next via events:

$$
T_1 \xrightarrow{\text{event: T1\_completed}} T_2 \xrightarrow{\text{event: T2\_completed}} \ldots
$$

For LLM agents, orchestrator-based sagas are preferred because the central agent (LLM) naturally serves as the coordinator.

---

### 11.6.4 Compensation Actions and Undo Operations

Compensation actions are the inverse operations that restore system state when a forward transaction must be reversed.

**Formal Definition.** For a transaction $T$ that transforms state $s$ to $s'$, the compensation $C$ satisfies:

$$
C(\mathcal{T}(s, T)) \approx s \quad \text{(approximate inverse)}
$$

Note: Exact inverse ($C(T(s)) = s$) is often impossible for real-world operations. Compensations restore **semantic equivalence**, not bitwise identity.

**Compensation Categories:**

| Category | Description | Example |
|---|---|---|
| **Perfect undo** | Exact state restoration | Delete created file |
| **Semantic undo** | Equivalent but not identical state | Refund payment (may differ in timing) |
| **Best-effort undo** | Partial state restoration | Send correction email (original already sent) |
| **No undo** | Irreversible; must accept or escalate | Physical action performed, data sent externally |

**Compensation Table for Common Agent Operations:**

| Operation | Compensation | Perfect? |
|---|---|---|
| `create_file(path, content)` | `delete_file(path)` | ✓ |
| `modify_file(path, new_content)` | `write_file(path, old_content)` | ✓ (if old content saved) |
| `delete_file(path)` | `create_file(path, saved_content)` | ✓ (if content was backed up) |
| `send_message(channel, text)` | `delete_message(channel, msg_id)` | Partial (recipients may have seen it) |
| `create_pr(repo, ...)` | `close_pr(repo, pr_id)` | ✓ |
| `deploy(service, version)` | `rollback(service, prev_version)` | ✓ (if stateless) |
| `send_email(to, subject, body)` | `send_correction_email(to, ...)` | ✗ (best-effort) |

---

## 11.7 Logging, Diagnostics, and Post-Mortem

### 11.7.1 Structured Error Logging

Structured logging captures exception data in a machine-parseable format for automated analysis.

**Log Entry Schema:**

$$
\text{LogEntry} = \langle \text{timestamp}, \text{level}, \text{source}, \text{error}, \text{context}, \text{trace}, \text{recovery}, \text{metadata} \rangle
$$

```json
{
    "timestamp": "2024-12-15T10:30:45.123Z",
    "level": "ERROR",
    "source": {
        "component": "tool_executor",
        "tool": "execute_query",
        "step": 5,
        "sub_goal": "G2.3"
    },
    "error": {
        "type": "ToolTimeoutError",
        "message": "Tool 'execute_query' timed out after 30s",
        "code": "TOOL_TIMEOUT",
        "severity": "ERROR",
        "is_transient": true
    },
    "context": {
        "goal": "Build REST API with authentication",
        "current_plan_step": "Create user table",
        "attempt_number": 2,
        "total_steps_completed": 4,
        "elapsed_time_s": 145.3
    },
    "trace": {
        "conversation_id": "conv_xyz789",
        "trajectory_hash": "sha256:def456...",
        "preceding_actions": [
            {"action": "read_schema", "result": "success"},
            {"action": "execute_query", "result": "timeout"}
        ]
    },
    "recovery": {
        "strategy": "retry_with_backoff",
        "attempt": 2,
        "next_wait_s": 4.0,
        "max_retries": 3,
        "fallback_available": true
    },
    "metadata": {
        "model": "claude-3-5-sonnet",
        "tokens_used": 1250,
        "cost_usd": 0.019,
        "session_id": "sess_abc123"
    }
}
```

---

### 11.7.2 Trace Reconstruction

Trace reconstruction builds a complete, navigable execution history from logged data, enabling temporal debugging of agent behavior.

**Trace Structure.** An execution trace is an ordered sequence of annotated events:

$$
\text{Trace} = [(t_0, \text{event}_0, s_0), (t_1, \text{event}_1, s_1), \ldots, (t_T, \text{event}_T, s_T)]
$$

**Event Types:**

| Event Type | Data Captured |
|---|---|
| `GOAL_SET` | Goal specification, source (user/agent) |
| `PLAN_CREATED` | Plan structure, sub-goals, dependencies |
| `ACTION_START` | Action name, arguments, expected outcome |
| `ACTION_RESULT` | Result content, duration, tokens used |
| `LLM_CALL` | Prompt (or hash), response, model, temperature |
| `TOOL_CALL` | Tool name, arguments, response, latency |
| `EXCEPTION` | Error type, message, stack trace |
| `RECOVERY_START` | Strategy chosen, reason |
| `RECOVERY_RESULT` | Success/failure, new state |
| `CHECKPOINT` | Checkpoint ID, state hash |
| `GOAL_COMPLETE` | Achievement status, metrics |

**Trace Visualization:**

```
t=0.0s  [GOAL_SET]      "Build REST API with auth"
t=0.1s  [PLAN_CREATED]  5 sub-goals, 12 leaf tasks
t=0.5s  [ACTION_START]  init_project(name="todo-api")
t=1.2s  [TOOL_CALL]     create_directory("/project") → success (0.7s)
t=1.3s  [ACTION_RESULT] init_project → success
t=1.5s  [CHECKPOINT]    cp_001 (state hash: a1b2c3)
t=2.0s  [ACTION_START]  create_database_schema()
t=2.5s  [LLM_CALL]      "Generate PostgreSQL schema..." → 450 tokens (0.5s)
t=3.0s  [TOOL_CALL]     execute_query(CREATE TABLE...) → success (0.5s)
...
t=45.0s [TOOL_CALL]     execute_query(INSERT...) → TIMEOUT (30.0s) ⚠️
t=45.0s [EXCEPTION]     ToolTimeoutError at step 5
t=45.1s [RECOVERY_START] retry_with_backoff (attempt 1, wait 2s)
t=47.1s [TOOL_CALL]     execute_query(INSERT...) → success (0.8s) ✓
t=47.2s [RECOVERY_RESULT] success
```

---

### 11.7.3 Root Cause Analysis

Root cause analysis (RCA) traces the causal chain from observed symptoms to the underlying cause.

**Causal Chain Model:**

$$
\text{RootCause} \xrightarrow{\text{causes}} \text{Intermediate}_1 \xrightarrow{\text{causes}} \ldots \xrightarrow{\text{causes}} \text{Symptom}
$$

**RCA Methodology for Agents:**

**Step 1: Symptom Identification.** What observable failure occurred?

$$
\text{Symptom}: \text{``Tool call returned incorrect data at step 8''}
$$

**Step 2: Timeline Reconstruction.** Using the trace, reconstruct the sequence of events leading to the symptom.

**Step 3: Hypothesis Generation.** For each step in the causal chain, generate candidate causes:

$$
\text{Hypotheses}(\text{symptom}) = \{h_1, h_2, \ldots, h_m\}
$$

**Step 4: Hypothesis Testing.** Check each hypothesis against the trace data:

$$
P(h_i | \text{trace}) \propto P(\text{trace} | h_i) \cdot P(h_i)
$$

**Step 5: Root Cause Identification.**

$$
\text{RootCause} = \arg\max_{h_i} P(h_i | \text{trace})
$$

**LLM-Assisted RCA:**

```
Prompt:
Analyze this agent execution trace and identify the root cause of the failure.

Trace: {trace_data}

Exception: {exception_details}

Perform root cause analysis:
1. What was the immediate cause of the failure?
2. What earlier events contributed to this failure?
3. What is the fundamental root cause?
4. How could this be prevented in the future?
5. What monitoring would detect this earlier?

Respond in structured format with confidence levels.
```

**The Five Whys for Agents:**

$$
\text{Why}_1: \text{``Database query failed''} \rightarrow \text{Timeout}
$$
$$
\text{Why}_2: \text{``Why timeout?''} \rightarrow \text{Query too slow}
$$
$$
\text{Why}_3: \text{``Why slow?''} \rightarrow \text{Missing index on queried column}
$$
$$
\text{Why}_4: \text{``Why no index?''} \rightarrow \text{Schema migration didn't include it}
$$
$$
\text{Why}_5: \text{``Why wasn't it in migration?''} \rightarrow \text{LLM generated schema without performance considerations}
$$

Root cause: The LLM's schema generation prompt lacks instructions about indexing strategy. Fix: Add "Include appropriate indexes for common query patterns" to the schema generation prompt.

---

### 11.7.4 Error Aggregation and Pattern Detection

Over time, error logs accumulate and reveal systematic patterns that indicate structural issues rather than random failures.

**Error Aggregation Pipeline:**

$$
\text{RawErrors} \xrightarrow{\text{Normalize}} \text{Canonical Errors} \xrightarrow{\text{Group}} \text{Error Clusters} \xrightarrow{\text{Analyze}} \text{Patterns} \xrightarrow{\text{Action}} \text{Systemic Fixes}
$$

**Error Normalization.** Map diverse error messages to canonical categories:

$$
\text{Normalize}(\text{``Connection refused: port 5432''}) = \text{DB\_CONNECTION\_FAILURE}
$$

$$
\text{Normalize}(\text{``psycopg2.OperationalError: could not connect''}) = \text{DB\_CONNECTION\_FAILURE}
$$

**Clustering Algorithm.** Group errors by similarity:

$$
\text{Cluster}(\{e_1, \ldots, e_N\}) = \{C_1, \ldots, C_k\} \quad \text{where } \forall e_i, e_j \in C_l: \text{sim}(e_i, e_j) > \tau_{\text{cluster}}
$$

**Pattern Detection Metrics:**

| Metric | Formula | Interpretation |
|---|---|---|
| Error rate | $\frac{|\text{errors}|}{|\text{total operations}|}$ | Overall reliability |
| Error concentration | $\frac{|\text{errors from top-1 cluster}|}{|\text{total errors}|}$ | Dominance of one failure mode |
| Mean time between failures | $\frac{\text{total time}}{|\text{failures}|}$ | Reliability interval |
| Failure recurrence | $P(\text{error}_{t+1} = e | \text{error}_t = e)$ | Whether specific errors repeat |
| Cascade coefficient | $\frac{|\text{secondary errors caused by } e|}{1}$ | Blast radius of each error type |

**Trend Detection.** Monitor error rates over time using change-point detection:

$$
\text{Alert if } \frac{\text{ErrorRate}(t, t+W)}{\text{ErrorRate}(t-W, t)} > \alpha_{\text{threshold}}
$$

An increasing error rate signals systemic degradation (model drift, API changes, data distribution shift).

**Automated Pattern Response:**

$$
\text{Response}(\text{pattern}) = \begin{cases}
\text{Auto-fix (prompt update)} & \text{if pattern is LLM format error with known fix} \\
\text{Auto-fix (circuit breaker)} & \text{if pattern is repeated service failure} \\
\text{Alert engineer} & \text{if pattern is new or severity is high} \\
\text{Trigger model retraining} & \text{if pattern indicates capability regression} \\
\text{Update documentation} & \text{if pattern reveals unclear tool API}
\end{cases}
$$

---

**Chapter Summary.** Exception handling in agentic systems extends far beyond traditional `try/catch` paradigms to encompass a multi-layered detection, classification, recovery, and learning framework. The stochastic nature of LLM outputs, the open-ended action spaces, and the coupling with external environments create failure modes that are partially observable, semantically complex, and temporally distributed. Robust agentic systems require: (1) multi-layer validation pipelines catching syntactic through semantic errors; (2) differentiated recovery strategies from simple retry through full replanning and human escalation; (3) transactional state management with checkpointing, idempotency, and compensation actions; (4) fault tolerance patterns (circuit breaker, bulkhead, saga) preventing cascading failures; and (5) comprehensive observability infrastructure enabling root cause analysis and systemic pattern detection. The formal framework $e = (s_{\text{expected}}, s_{\text{actual}}, \Delta)$ provides the mathematical foundation for quantifying, classifying, and responding to the full spectrum of agentic failures.