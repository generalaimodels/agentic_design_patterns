

# Chapter 3: Reflection

---

## 3.1 Definition and Formal Framework

### 3.1.1 What is Reflection in Agentic AI

Reflection in agentic AI is the **computational capacity of a system to evaluate, critique, and iteratively improve its own outputs, reasoning processes, and decision strategies**. It is the mechanism by which an agent transitions from open-loop generation (produce output, deliver) to closed-loop generation (produce output, evaluate, revise, deliver), introducing a self-referential feedback signal that operates entirely within the agent's own inference pipeline — without external human feedback at runtime.

In cognitive science, reflection corresponds to **metacognition** — the process of "thinking about thinking." In the context of LLM-based agents, reflection operationalizes metacognition as an additional inference pass where the model evaluates its own prior output against explicit or implicit quality criteria, identifies deficiencies, and generates an improved version.

**Why Reflection is Fundamental:**

LLMs are autoregressive generators: they produce tokens sequentially, committing to each token before generating the next. This creates a structural limitation — the model cannot revise earlier tokens in light of later reasoning. Reflection addresses this by introducing an **outer loop** around the generation process:

```
┌─────────────────────────────────────────────────┐
│                 Reflection Loop                 │
│                                                 │
│  ┌──────────┐      ┌──────────┐     ┌──────────┐│
│  │ Generate  │───▶ │ Evaluate│───▶│  Refine   │|
│  │  y_t      │     │  r_t     │    │  y_{t+1}  ││
│  └──────────┘      └──────────┘     └──────────┘│
│       ▲                                │        │
│       └────────────────────────────────┘        │
│                                                 │
│       Repeat until convergence or budget        │
└─────────────────────────────────────────────────┘
```

**Fundamental Properties of Reflection:**

| Property | Description |
|---|---|
| **Self-Referential** | The evaluating system examines outputs produced by (a version of) itself |
| **Iterative** | Reflection naturally induces a loop: generate → evaluate → revise → re-evaluate |
| **Goal-Conditioned** | Evaluation is relative to an explicit or implicit objective |
| **Non-Monotonic** | Reflection does not guarantee improvement; it can degrade output quality |
| **Resource-Consuming** | Each reflection pass costs additional tokens, latency, and compute |
| **Bounded** | Practical systems impose iteration limits to prevent infinite loops |

**Reflection vs. Single-Pass Generation:**

Consider a coding task where the model must produce a correct Python function.

- **Single-pass**: Generate the function once. If there is a bug, it is delivered to the user.
- **With reflection**: Generate the function, then critique it ("Does this handle edge cases? Are there off-by-one errors? Does it satisfy all constraints?"), then revise based on the critique.

Empirically, reflection improves pass rates on HumanEval from ~65% (single-pass GPT-4) to ~88% (GPT-4 with self-refinement), demonstrating the substantial quality gains achievable.

---

### 3.1.2 Reflection as Self-Evaluation

We formalize reflection as a **self-evaluation function** that produces a structured assessment:

$$
r = \mathcal{E}(a, y, g)
$$

where:

- $a \in \mathcal{A}$: the **action** taken by the agent (the prompt strategy, tool calls, reasoning steps)
- $y \in \mathcal{Y}$: the **output** produced by the agent (generated text, code, plan)
- $g \in \mathcal{G}$: the **goal** or specification against which the output is evaluated (user query, task requirements, quality rubric)
- $r \in \mathcal{R}$: the **reflection** — a structured evaluation containing identified issues, quality scores, and improvement suggestions

**Structure of a Reflection $r$:**

$$
r = (\text{score}, \text{issues}, \text{suggestions}, \text{verdict})
$$

where:

- $\text{score} \in [0, 1]$: overall quality score
- $\text{issues} = \{(d_1, \text{severity}_1), \ldots, (d_m, \text{severity}_m)\}$: set of identified deficiencies with severity levels
- $\text{suggestions} = \{s_1, \ldots, s_m\}$: corresponding improvement suggestions for each issue
- $\text{verdict} \in \{\texttt{ACCEPT}, \texttt{REVISE}, \texttt{REJECT}\}$: overall decision

**Formal Evaluation Function Decomposition:**

The evaluation function $\mathcal{E}$ decomposes into multiple **aspect evaluators**:

$$
\mathcal{E}(a, y, g) = \bigoplus_{j=1}^{J} \mathcal{E}_j(a, y, g)
$$

where $\mathcal{E}_j$ evaluates aspect $j$ (correctness, completeness, style, safety, etc.) and $\bigoplus$ is a combination operator (weighted sum, conjunction, or structured merge).

**Example Aspects:**

| Aspect $j$ | Evaluator $\mathcal{E}_j$ | Output |
|---|---|---|
| **Correctness** | Does $y$ produce the right answer/behavior? | Boolean + explanation |
| **Completeness** | Does $y$ address all parts of $g$? | Fraction addressed |
| **Consistency** | Is $y$ internally consistent (no contradictions)? | Boolean + conflicts |
| **Efficiency** | Is $y$ computationally/linguistically efficient? | Score $\in [0, 1]$ |
| **Safety** | Does $y$ comply with safety constraints? | Boolean + violations |
| **Style** | Does $y$ match the required format/tone? | Score $\in [0, 1]$ |

**The Reflection-Refinement Cycle:**

Given the evaluation $r_t = \mathcal{E}(a, y_t, g)$, the agent produces a refined output:

$$
y_{t+1} = \mathcal{F}(y_t, r_t, g)
$$

where $\mathcal{F}$ is the **refinement function** that takes the current output, the reflection, and the original goal, producing an improved output.

The complete iterative process is:

$$
y_0 = \text{LLM}(g) \quad \text{(initial generation)}
$$

$$
r_t = \mathcal{E}(a, y_t, g) \quad \text{(reflection)}
$$

$$
y_{t+1} = \mathcal{F}(y_t, r_t, g) \quad \text{(refinement)}
$$

$$
\text{Terminate when } r_t.\text{verdict} = \texttt{ACCEPT} \text{ or } t > T_{\max}
$$

**Fixed-Point Interpretation:**

The reflection-refinement cycle seeks a **fixed point** of the evaluation-refinement map:

$$
y^* = \mathcal{F}(y^*, \mathcal{E}(a, y^*, g), g)
$$

At the fixed point, reflection identifies no further issues, and refinement makes no further changes. Whether this fixed point exists, is unique, and is reachable depends on the properties of $\mathcal{E}$ and $\mathcal{F}$.

---

### 3.1.3 Meta-Cognition in LLM-Based Agents

Meta-cognition is **cognition about cognition** — the ability of an agent to reason about its own knowledge, capabilities, limitations, and reasoning processes.

**Levels of Meta-Cognition:**

| Level | Description | LLM Capability |
|---|---|---|
| **Level 0: No meta-cognition** | Agent generates without any self-awareness | Standard prompting |
| **Level 1: Output monitoring** | Agent checks if its output satisfies basic criteria | Self-evaluation prompts |
| **Level 2: Process monitoring** | Agent evaluates its reasoning steps, not just the final output | Chain-of-thought critique |
| **Level 3: Strategy monitoring** | Agent evaluates whether its approach/strategy is appropriate | Meta-strategic reasoning |
| **Level 4: Capability awareness** | Agent knows what it can and cannot do | Calibrated uncertainty |

**Meta-Cognitive Prompting:**

At each level, meta-cognition is elicited through carefully designed prompts:

**Level 1 — Output Monitoring:**
```
Review your answer above. Is it correct? Does it fully address the question?
If there are any errors, identify and correct them.
```

**Level 2 — Process Monitoring:**
```
Review your reasoning step by step. For each step:
1. Is the logic valid?
2. Are there any unjustified assumptions?
3. Could an alternative step lead to a different conclusion?
```

**Level 3 — Strategy Monitoring:**
```
Before continuing, consider:
1. Is your current approach the best strategy for this problem?
2. Are there alternative approaches you should consider?
3. What are the risks of your current approach failing?
```

**Level 4 — Capability Awareness:**
```
Before attempting this task, assess:
1. Do you have sufficient knowledge to answer this accurately?
2. What is your confidence level (0-100%)?
3. What specific aspects are you uncertain about?
4. Should you recommend that the user consult an external source?
```

**Formal Model of Meta-Cognitive Processing:**

Let $M$ be a meta-cognitive module. The agent's processing becomes:

$$
\text{Meta-Augmented Output} = M(\text{LLM}(g)) = (\hat{y}, \hat{c}, \hat{u})
$$

where:
- $\hat{y}$: the potentially revised output
- $\hat{c} \in [0, 1]$: calibrated confidence estimate
- $\hat{u}$: structured uncertainty description (what the agent is uncertain about and why)

---

### 3.1.4 Reflection vs. Self-Correction vs. Critique

These three concepts are related but distinct:

**Reflection:**

The broad process of self-evaluation encompassing identification of issues, quality assessment, and reasoning about one's own reasoning. Reflection is **diagnostic** — it identifies what is wrong and why.

$$
\text{Reflection}: (y, g) \rightarrow r = (\text{issues}, \text{scores}, \text{explanations})
$$

**Critique:**

A specific form of reflection focused on producing a **textual evaluation** of the output. Critique is the natural-language expression of reflection. It is the intermediate artifact between reflection and correction.

$$
\text{Critique}: (y, g) \rightarrow c_{\text{text}} = \text{"The output has the following issues: ..."}
$$

**Self-Correction:**

The act of **modifying the output** based on reflection or critique. Self-correction is **therapeutic** — it fixes the identified issues.

$$
\text{Self-Correction}: (y, c_{\text{text}}, g) \rightarrow y' \text{ where } \text{Quality}(y') > \text{Quality}(y)
$$

**The Full Pipeline:**

$$
y \xrightarrow{\text{Reflection}} r \xrightarrow{\text{Critique}} c_{\text{text}} \xrightarrow{\text{Self-Correction}} y'
$$

**Critical Distinction:**

Self-correction is **not guaranteed to improve quality**. Research by Huang et al. (2023) demonstrated that LLMs often cannot self-correct reasoning without external feedback. The model that made the error in the first place may not reliably identify the error upon re-examination. This finding has profound implications:

1. Self-correction works best when the critique can leverage **external signals** (test execution, tool output, retrieval) that provide ground-truth feedback
2. Self-correction without external grounding risks **sycophantic self-evaluation** (the model approves its own incorrect output) or **hallucinated corrections** (the model "fixes" correct content, introducing new errors)
3. The effectiveness of self-correction depends heavily on the **correlation between generation errors and evaluation errors** — if the same model makes the same mistakes in both modes, reflection adds cost without improvement

**When Each Strategy is Appropriate:**

| Scenario | Best Approach | Why |
|---|---|---|
| Code generation | Self-correction with test execution | External signal (test pass/fail) grounds the critique |
| Mathematical proof | Self-correction with verification | Can check logical validity step-by-step |
| Creative writing | Critique without forced correction | Quality is subjective; forcing changes may degrade style |
| Factual QA | Reflection with retrieval | External knowledge provides ground truth |
| Safety review | Dual-agent critique | Independent evaluator avoids self-bias |

---

## 3.2 Types of Reflection

### 3.2.1 Output-Level Reflection

Output-level reflection evaluates the **final product** — the generated text, code, plan, or answer — against quality criteria.

#### Self-Evaluation of Generated Responses

The agent re-reads its own output and assesses it holistically:

$$
r_{\text{output}} = \text{LLM}\left(\text{"Evaluate this response: "} \| y \| \text{" for the question: "} \| g\right)
$$

**Structured Self-Evaluation Template:**

```
You produced the following response to the user's question.

Question: {goal}
Your Response: {output}

Evaluate your response on the following dimensions (1-5 scale):
1. **Accuracy**: Are all facts and claims correct?
2. **Completeness**: Does the response address every part of the question?
3. **Clarity**: Is the response well-organized and easy to understand?
4. **Relevance**: Does every part of the response relate to the question?
5. **Conciseness**: Is the response appropriately concise without unnecessary content?

For each dimension, provide:
- Score (1-5)
- Specific issues found (if any)
- Suggested improvements (if score < 4)

Overall verdict: ACCEPT / REVISE / REJECT
```

**Implementation:**

```python
class OutputReflector:
    def __init__(self, evaluator_llm):
        self.evaluator = evaluator_llm
    
    async def reflect(self, output: str, goal: str) -> ReflectionResult:
        prompt = SELF_EVALUATION_TEMPLATE.format(
            goal=goal, output=output
        )
        reflection_text = await self.evaluator.generate(prompt)
        return self._parse_reflection(reflection_text)
    
    def _parse_reflection(self, text: str) -> ReflectionResult:
        # Extract scores, issues, suggestions, verdict
        scores = extract_scores(text)
        issues = extract_issues(text)
        suggestions = extract_suggestions(text)
        verdict = extract_verdict(text)
        return ReflectionResult(
            scores=scores,
            issues=issues,
            suggestions=suggestions,
            verdict=verdict,
            overall_score=sum(scores.values()) / len(scores)
        )
```

#### Correctness Checking

For tasks with verifiable correctness (math, code, logic), the agent explicitly checks whether its output is correct.

**Mathematical Correctness:**

$$
\text{Check}: \text{LLM}\left(\text{"Verify: Is } f(x) = 2x + 3 \text{ the derivative of } F(x) = x^2 + 3x + 1\text{?"}\right)
$$

The agent should compute $F'(x) = 2x + 3$ and compare with the claimed answer.

**Code Correctness via Mental Execution:**

```
Trace through the code with these test cases:

Input: [3, 1, 4, 1, 5]
Expected output: [1, 1, 3, 4, 5]

Step 1: ...
Step 2: ...
Actual output: ...
Match: YES / NO
```

**Code Correctness via Actual Execution (External Signal):**

The most reliable form of correctness checking uses an external code executor:

```python
class CodeCorrectnessChecker:
    async def check(self, code: str, test_cases: list[dict]) -> CheckResult:
        results = []
        for tc in test_cases:
            try:
                output = await self.sandbox.execute(code, tc["input"])
                passed = output == tc["expected_output"]
                results.append(TestResult(
                    input=tc["input"],
                    expected=tc["expected_output"],
                    actual=output,
                    passed=passed
                ))
            except Exception as e:
                results.append(TestResult(
                    input=tc["input"],
                    expected=tc["expected_output"],
                    actual=None,
                    passed=False,
                    error=str(e)
                ))
        
        return CheckResult(
            all_passed=all(r.passed for r in results),
            pass_rate=sum(r.passed for r in results) / len(results),
            failures=[r for r in results if not r.passed]
        )
```

This is the **gold standard** for code reflection because it provides a ground-truth signal, not a model-estimated one.

#### Consistency Verification

The agent checks its output for **internal consistency** — do different parts of the response contradict each other?

**Types of Inconsistency:**

1. **Factual contradiction**: "The population is 5 million" in one paragraph and "With its 3 million inhabitants" in another.

2. **Logical contradiction**: "X implies Y" in one step and "Y is false, but X is true" in another.

3. **Numerical inconsistency**: A financial report where individual line items don't sum to the reported total.

4. **Temporal inconsistency**: "Event A happened before B" in one place and "B preceded A" in another.

**Formal Consistency Check:**

Extract propositions $\{p_1, p_2, \ldots, p_n\}$ from the output. Verify:

$$
\nexists \; i, j : p_i \land p_j \text{ is a contradiction}
$$

**Practical Implementation:**

```
Review your response for internal consistency:

1. List all factual claims made in the response
2. Check if any two claims contradict each other
3. Check if all numerical values are consistent (sums, percentages, ratios)
4. Check if the temporal ordering of events is consistent throughout
5. Check if the conclusion logically follows from the premises stated

Report any inconsistencies found.
```

---

### 3.2.2 Process-Level Reflection

Process-level reflection evaluates the **reasoning process** itself, not just the final output. It asks: "Was my reasoning sound?" rather than "Is my answer correct?"

#### Step-by-Step Reasoning Audit

The agent reviews each step of its chain-of-thought reasoning:

$$
\text{For each step } s_i \text{ in the reasoning chain } (s_1, s_2, \ldots, s_n):
$$

$$
\text{Valid}(s_i) = \begin{cases}
\text{True} & \text{if } s_i \text{ follows logically from } s_1, \ldots, s_{i-1} \text{ and given facts} \\
\text{False} & \text{otherwise}
\end{cases}
$$

**Audit Template:**

```
Review your reasoning step by step. For each step, evaluate:

Step 1: {step_1}
- Is this step logically valid? 
- Does it follow from the given information?
- Are there any hidden assumptions?

Step 2: {step_2}
- Is this step logically valid?
- Does it correctly build on Step 1?
- Could an error in Step 1 have propagated here?

...

Overall: Is the chain of reasoning sound from premises to conclusion?
If not, identify the first step where an error occurs.
```

**Error Propagation in Reasoning Chains:**

A critical insight: errors in early steps propagate and compound through the chain. If step $s_i$ is incorrect, all subsequent steps $s_{i+1}, \ldots, s_n$ that depend on $s_i$ are potentially invalid, regardless of their individual logical soundness.

$$
P(\text{chain correct}) = \prod_{i=1}^{n} P(s_i \text{ correct} \mid s_1, \ldots, s_{i-1} \text{ correct})
$$

For a 10-step chain with 95% per-step accuracy:

$$
P(\text{chain correct}) = 0.95^{10} \approx 0.60
$$

This motivates **per-step reflection** rather than only final-answer reflection.

#### Plan Validity Assessment

Before executing a multi-step plan, the agent evaluates whether the plan is sound:

$$
\text{PlanValid}(\pi) = \bigwedge_{i=1}^{n} \text{StepFeasible}(\pi_i, \text{State}_i) \land \text{Achieves}(\pi, g)
$$

**Plan Evaluation Dimensions:**

| Dimension | Question |
|---|---|
| **Feasibility** | Can each step actually be executed with available tools/resources? |
| **Sufficiency** | Does the plan, if executed correctly, achieve the goal? |
| **Efficiency** | Is the plan reasonably efficient, or are there unnecessary steps? |
| **Ordering** | Are the steps in the correct order (respecting dependencies)? |
| **Robustness** | What happens if a step fails? Is there a fallback? |

**Implementation:**

```python
class PlanReflector:
    async def evaluate_plan(self, plan: list[dict], goal: str, 
                            available_tools: list[str]) -> PlanEvaluation:
        prompt = f"""
Evaluate this plan for achieving the goal.

Goal: {goal}
Available tools: {available_tools}

Plan:
{self._format_plan(plan)}

For each step, assess:
1. Is this step feasible with the available tools?
2. Does it logically follow from previous steps?
3. Does its output provide what subsequent steps need?

Overall assessment:
1. Does this plan fully achieve the goal?
2. Are there missing steps?
3. Are there unnecessary steps?
4. What could go wrong, and how would we recover?

Output your evaluation as structured JSON.
"""
        evaluation = await self.llm.generate(prompt)
        return self._parse_evaluation(evaluation)
```

#### Strategy Effectiveness Evaluation

Beyond evaluating the plan, the agent evaluates whether its **overall strategy** is appropriate:

```
Before proceeding, reflect on your approach:

1. STRATEGY CHOICE: You chose a {approach_type} approach. 
   - Why is this the best approach for this problem?
   - What alternative strategies exist?
   - What are the risks of this strategy?

2. RESOURCE USAGE: Your plan requires {n_steps} steps and {n_tools} tools.
   - Is this proportionate to the problem complexity?
   - Can any steps be parallelized or eliminated?

3. INFORMATION SUFFICIENCY: 
   - Do you have all the information needed to execute this strategy?
   - What information is missing? How would you obtain it?

4. EXIT CRITERIA:
   - How will you know when the task is complete?
   - What does "good enough" look like for this task?
```

---

### 3.2.3 Meta-Level Reflection

Meta-level reflection operates on the agent's **self-model** — its understanding of its own capabilities, limitations, and epistemic state.

#### Capability Awareness: What Can I vs. Cannot I Do

The agent explicitly reasons about whether a task falls within its competence boundary:

$$
\text{CanDo}(q) = P(\text{correct response} \mid q, \text{model capabilities})
$$

**Capability Boundary Estimation:**

```
Before answering, assess whether this question is within your capabilities:

1. KNOWLEDGE DOMAIN: Does this require specialized knowledge you may lack?
   - Medical diagnosis → Limited capability, recommend professional
   - Recent events (after training cutoff) → Cannot answer reliably
   - Highly technical niche → May have incomplete knowledge

2. TASK TYPE: Does this require capabilities you don't have?
   - Real-time data → Cannot access
   - Personal information → Do not have
   - Physical world interaction → Cannot perform

3. PRECISION REQUIREMENT: Does this require a level of precision you cannot guarantee?
   - Legal advice → Cannot guarantee accuracy
   - Financial calculations → Should be verified independently

Confidence assessment: HIGH / MEDIUM / LOW / CANNOT_DO
```

**Formal Capability Model:**

Define a capability function $\kappa: \mathcal{Q} \rightarrow [0, 1]$ that estimates the probability of producing a correct response:

$$
\kappa(q) = P(\text{correct} \mid q) = \sigma\left(\mathbf{w}^\top \phi(q) + b\right)
$$

where $\phi(q)$ is a feature vector encoding task characteristics (domain, complexity, recency, precision requirements) and $\sigma$ is the sigmoid function. This can be trained on historical performance data.

#### Confidence Calibration

A well-calibrated agent's confidence scores should match its actual accuracy:

$$
P(\text{correct} \mid \text{confidence} = c) \approx c
$$

**Calibration Measurement — Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|
$$

where $B$ is the number of bins, $n_b$ is the count in bin $b$, $\text{acc}(b)$ is the actual accuracy in bin $b$, and $\text{conf}(b)$ is the average predicted confidence in bin $b$.

**Why LLMs are Poorly Calibrated:**

1. **Overconfidence**: Models trained on human-curated text develop a bias toward confident-sounding language, even when uncertain
2. **Verbalized confidence ≠ token probability**: An LLM may say "I'm 90% confident" while its token probabilities tell a different story
3. **Sycophantic pressure**: RLHF training encourages models to be helpful, which biases toward appearing confident

**Calibration Techniques:**

**Technique 1: Verbalized Probability Elicitation**

```
Answer the following question, and then provide your confidence level.

Question: {question}

Answer: {your answer}

How confident are you that this answer is correct?
Express your confidence as a percentage (0-100%), where:
- 0% means you're guessing randomly
- 50% means it could go either way
- 90% means you're very confident but acknowledge some uncertainty
- 100% means you are certain (use very rarely)

Confidence: __%
```

**Technique 2: Consistency-Based Calibration**

Generate $k$ responses at temperature $T > 0$ and measure agreement:

$$
\hat{c}(q) = \frac{|\{i : a_i = a_{\text{mode}}\}|}{k}
$$

where $a_{\text{mode}}$ is the most common answer across $k$ samples. This self-consistency-based confidence is empirically better calibrated than verbalized confidence.

**Technique 3: Token-Probability Calibration**

Use the model's token-level log-probabilities as a confidence signal:

$$
\hat{c}(q) = \exp\left(\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, q)\right)
$$

Apply Platt scaling or isotonic regression to calibrate:

$$
c_{\text{calibrated}} = \sigma(w \cdot \hat{c}(q) + b)
$$

where $w, b$ are learned on a calibration set.

#### Uncertainty Quantification

Beyond a single confidence score, the agent provides structured uncertainty information:

**Aleatoric Uncertainty** (irreducible, inherent to the problem):

$$
\text{Aleatoric}: \text{"This question has genuinely ambiguous answers because..."}
$$

**Epistemic Uncertainty** (reducible, due to the model's limited knowledge):

$$
\text{Epistemic}: \text{"I'm uncertain because this topic was underrepresented in my training data..."}
$$

**Structured Uncertainty Output:**

```python
class UncertaintyReport:
    overall_confidence: float           # 0 to 1
    uncertainty_type: str               # "aleatoric" | "epistemic" | "mixed"
    uncertain_aspects: list[str]        # What specifically is uncertain
    evidence_strength: str              # "strong" | "moderate" | "weak" | "none"
    alternative_answers: list[dict]     # [{answer, probability, reasoning}]
    recommended_action: str             # "trust" | "verify" | "seek_expert"
```

**Implementation:**

```python
UNCERTAINTY_PROMPT = """
For your answer to the question below, provide a detailed uncertainty analysis.

Question: {question}
Your Answer: {answer}

Uncertainty Analysis:
1. Overall confidence (0-100%): 
2. What are you most certain about in your answer?
3. What are you least certain about?
4. Are there plausible alternative answers? If so, list them with estimated probabilities.
5. What type of uncertainty is this?
   - Inherent ambiguity in the question (aleatoric)
   - Gaps in your knowledge (epistemic)
   - Both
6. What additional information would reduce your uncertainty?
7. Should the user independently verify this answer? Why or why not?
"""
```

---

## 3.3 Reflection Architectures

### 3.3.1 Self-Refine Loop

The Self-Refine framework (Madaan et al., 2023) implements reflection as a three-phase iterative loop using a single LLM for all three roles.

**Architecture:**

$$
y_0 \xrightarrow{\text{Feedback}} r_0 \xrightarrow{\text{Refine}} y_1 \xrightarrow{\text{Feedback}} r_1 \xrightarrow{\text{Refine}} y_2 \xrightarrow{\cdots} y_T
$$

#### Generate → Critique → Refine Cycle

**Phase 1 — Generate:**

$$
y_0 = \text{LLM}(g)
$$

The LLM produces an initial output given the goal $g$.

**Phase 2 — Critique (Feedback):**

$$
r_t = \text{LLM}(\text{FeedbackPrompt}(y_t, g))
$$

The same LLM critiques its own output, producing structured feedback.

**Phase 3 — Refine:**

$$
y_{t+1} = \text{LLM}(\text{RefinePrompt}(y_t, r_t, g))
$$

The same LLM uses the feedback to produce an improved output.

**Feedback Prompt Template:**

```
I produced the following {output_type} for the task below.

Task: {goal}
My Output: {y_t}

Please provide specific, actionable feedback on this output:
1. What are the strengths of this output?
2. What are the specific weaknesses or errors?
3. For each weakness, suggest a concrete improvement.
4. Rate the overall quality (1-10).
5. Would you accept this output as-is, or does it need revision?
```

**Refine Prompt Template:**

```
I need to improve my {output_type} based on the feedback received.

Task: {goal}
My Previous Output: {y_t}
Feedback: {r_t}

Please produce an improved version that addresses all the feedback points.
Maintain the strengths identified in the feedback while fixing the weaknesses.

Improved Output:
```

**Implementation:**

```python
class SelfRefine:
    def __init__(self, llm, max_iterations=3, quality_threshold=8):
        self.llm = llm
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    async def run(self, goal: str) -> RefinementResult:
        # Phase 1: Initial generation
        output = await self.llm.generate(
            f"Complete the following task:\n{goal}"
        )
        history = [{"iteration": 0, "output": output, "feedback": None}]
        
        for t in range(self.max_iterations):
            # Phase 2: Critique
            feedback = await self.llm.generate(
                FEEDBACK_TEMPLATE.format(goal=goal, output=output)
            )
            quality_score = extract_quality_score(feedback)
            
            history.append({
                "iteration": t + 1,
                "feedback": feedback,
                "quality_score": quality_score
            })
            
            # Check termination
            if quality_score >= self.quality_threshold:
                break
            
            # Phase 3: Refine
            output = await self.llm.generate(
                REFINE_TEMPLATE.format(
                    goal=goal, output=output, feedback=feedback
                )
            )
            history[-1]["output"] = output
        
        return RefinementResult(
            final_output=output,
            iterations=len(history) - 1,
            history=history
        )
```

#### Convergence Criteria

The iterative loop must terminate. Three convergence criteria are used:

**Criterion 1 — Quality Threshold:**

$$
\text{Stop if } \text{Score}(y_t) \geq \tau_{\text{quality}}
$$

**Criterion 2 — Diminishing Change:**

$$
\text{Stop if } \|y_{t+1} - y_t\| < \epsilon
$$

where $\|\cdot\|$ is a semantic distance metric. In practice, this is measured as:

$$
\Delta_t = 1 - \text{sim}(\text{Embed}(y_t), \text{Embed}(y_{t+1}))
$$

If $\Delta_t < \epsilon$, the output is converging (changes are minimal), and further iterations are unlikely to yield substantial improvement.

**Criterion 3 — No New Issues:**

$$
\text{Stop if } r_t.\text{issues} = \emptyset
$$

The critique identifies no further issues.

#### Maximum Iteration Bounds

Regardless of convergence criteria, a hard limit prevents infinite loops:

$$
t \leq T_{\max} \quad \text{(typically } T_{\max} \in \{2, 3, 5\}\text{)}
$$

**Why Small $T_{\max}$:**

1. Empirically, most improvement occurs in the first 1-2 iterations
2. Marginal quality gains diminish rapidly:

$$
\Delta Q(t) = Q(y_{t+1}) - Q(y_t) \approx c \cdot e^{-\lambda t}
$$

where $c$ and $\lambda$ are positive constants. Improvement decays exponentially.

3. Cost scales linearly: each iteration costs approximately the same number of tokens:

$$
C_{\text{total}} = C_0 + T \cdot (C_{\text{critique}} + C_{\text{refine}})
$$

4. Risk of degradation increases with iterations (Section 3.5)

---

### 3.3.2 Reflexion Framework

Reflexion (Shinn et al., 2023) extends self-refinement to **multi-episode learning** where the agent maintains a persistent memory of past failures and their reflections.

#### Episodic Memory of Past Failures

Unlike Self-Refine (which operates within a single generation cycle), Reflexion operates **across episodes**:

**Episode 1:**
- Agent attempts task $g$, produces output $y_1$
- Environment provides feedback (test failure, incorrect answer)
- Agent reflects: "I failed because I didn't handle edge case X"
- Reflection is stored in memory

**Episode 2:**
- Agent attempts the same task $g$ again
- Agent reads its stored reflections from Episode 1
- Agent produces improved output $y_2$, informed by past failure analysis

**Memory Structure:**

$$
\mathcal{M} = \{(g_i, y_i, \text{outcome}_i, \text{reflection}_i)\}_{i=1}^{N}
$$

Each memory entry contains the task, the attempted output, the outcome (success/failure/score), and the textual reflection about what went wrong.

**Memory-Augmented Generation:**

$$
y_t = \text{LLM}(g, \mathcal{M}_{\text{relevant}})
$$

where $\mathcal{M}_{\text{relevant}} \subseteq \mathcal{M}$ are the reflections relevant to the current task (retrieved by semantic similarity or exact match).

```python
class ReflexionAgent:
    def __init__(self, llm, evaluator, max_episodes=5):
        self.llm = llm
        self.evaluator = evaluator
        self.max_episodes = max_episodes
        self.memory = []  # List of reflection strings
    
    async def solve(self, task: str) -> str:
        for episode in range(self.max_episodes):
            # Generate with memory context
            memory_context = "\n".join(
                f"Previous attempt reflection: {m}" for m in self.memory
            )
            
            prompt = f"""
Task: {task}

{f"Lessons from previous attempts:{chr(10)}{memory_context}" if self.memory else ""}

Produce your solution, taking into account any lessons from previous attempts.
"""
            output = await self.llm.generate(prompt)
            
            # Evaluate
            result = await self.evaluator.evaluate(task, output)
            
            if result.success:
                return output
            
            # Reflect on failure
            reflection = await self.llm.generate(f"""
Task: {task}
Your attempt: {output}
Result: {result.feedback}

Reflect on why your attempt failed. What specific mistakes did you make?
What should you do differently next time? Be concrete and specific.
""")
            self.memory.append(reflection)
        
        return output  # Return best attempt after max episodes
```

#### Verbal Reinforcement Learning

Reflexion can be understood as a form of **reinforcement learning where the policy update is expressed in natural language** rather than as gradient-based parameter updates.

In standard RL:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

where $\theta$ are model parameters and $\nabla_\theta J$ is the policy gradient.

#### Reflection as Textual Gradient

In Reflexion, the "parameters" are the textual instructions in the prompt, and the "gradient" is the textual reflection:

$$
\theta_{t+1}^{\text{verbal}} = \theta_t^{\text{verbal}} + \nabla_{\text{text}} \mathcal{L}(y_t, y^*)
$$

where:
- $\theta_t^{\text{verbal}}$: the accumulated reflections/instructions that condition generation at step $t$
- $\nabla_{\text{text}} \mathcal{L}(y_t, y^*)$: the textual reflection describing how the output $y_t$ differs from the desired output $y^*$ and how to correct it
- The "$+$" operator is string concatenation (appending to the memory)

**Analogy to Gradient Descent:**

| RL/Optimization | Reflexion |
|---|---|
| Parameters $\theta$ | Accumulated textual reflections in prompt |
| Loss $\mathcal{L}(y, y^*)$ | Task failure/error description |
| Gradient $\nabla_\theta \mathcal{L}$ | Textual reflection on why the failure occurred |
| Learning rate $\alpha$ | How much the model attends to reflections |
| Parameter update | Appending reflection to memory |
| Convergence | Task solved correctly |

**Critical difference**: In standard RL, the gradient modifies the model's internal parameters (weights). In Reflexion, the model weights are frozen; only the **input context** changes. The "learning" happens entirely through in-context conditioning, not weight updates.

**Formal Analysis of Reflexion Convergence:**

Let $\pi_t$ be the policy at episode $t$, conditioned on accumulated reflections $\mathcal{M}_t$:

$$
\pi_t(\cdot \mid g) = \text{LLM}(\cdot \mid g, \mathcal{M}_t)
$$

The expected performance improves if reflections are informative:

$$
V(\pi_{t+1}) \geq V(\pi_t) \iff \text{Reflections provide actionable, correct diagnostic information}
$$

This is not guaranteed — reflections can be incorrect, irrelevant, or misleading.

---

### 3.3.3 Constitutional AI-Style Self-Critique

Constitutional AI (Bai et al., 2022) introduces a framework where the model critiques its own outputs against a set of **explicit principles** (a "constitution").

#### Principle-Based Evaluation

A constitution $\mathcal{C} = \{c_1, c_2, \ldots, c_P\}$ is a set of principles that define desirable output properties:

**Example Constitutional Principles:**

```
c_1: "The response should be helpful and directly address the user's question."
c_2: "The response should not contain harmful, unethical, or illegal content."
c_3: "The response should be factually accurate to the best of available knowledge."
c_4: "The response should acknowledge uncertainty rather than presenting speculation as fact."
c_5: "The response should be respectful and not demean any individual or group."
```

**Principle-Based Critique:**

For each principle $c_j$, the model evaluates the output:

$$
\text{score}_j = \text{LLM}(\text{"Does this output satisfy principle: "} c_j \text{"? Output: "} y)
$$

**Critique Prompt:**

```
Review the following response against this principle:

Principle: {c_j}
Response: {y}

Does the response satisfy this principle? 
- If yes, explain briefly why.
- If no, explain specifically how it violates the principle 
  and suggest how to fix the violation.

Assessment: SATISFIES / VIOLATES
Explanation: ...
Suggested Fix: ...
```

#### Multi-Aspect Scoring

The constitutional evaluation produces a **multi-dimensional score vector**:

$$
\mathbf{s} = (s_1, s_2, \ldots, s_P) \in [0, 1]^P
$$

where $s_j$ is the compliance score for principle $c_j$.

**Overall Compliance:**

$$
S_{\text{overall}} = \min_j s_j \quad \text{(strictest: must satisfy ALL principles)}
$$

or

$$
S_{\text{overall}} = \sum_j w_j \cdot s_j \quad \text{(weighted: some principles more important)}
$$

The "min" formulation is appropriate for safety-critical applications where **any** violation is unacceptable.

**Implementation:**

```python
class ConstitutionalCritic:
    def __init__(self, llm, constitution: list[str]):
        self.llm = llm
        self.constitution = constitution
    
    async def critique(self, output: str, goal: str) -> ConstitutionalReview:
        reviews = []
        
        # Evaluate against each principle in parallel
        tasks = [
            self._evaluate_principle(output, goal, principle, idx)
            for idx, principle in enumerate(self.constitution)
        ]
        reviews = await asyncio.gather(*tasks)
        
        # Aggregate
        all_satisfied = all(r.satisfies for r in reviews)
        violations = [r for r in reviews if not r.satisfies]
        overall_score = sum(r.score for r in reviews) / len(reviews)
        
        return ConstitutionalReview(
            principle_reviews=reviews,
            all_satisfied=all_satisfied,
            violations=violations,
            overall_score=overall_score,
            verdict="ACCEPT" if all_satisfied else "REVISE"
        )
    
    async def _evaluate_principle(self, output, goal, principle, idx):
        prompt = PRINCIPLE_EVAL_TEMPLATE.format(
            principle=principle,
            output=output,
            goal=goal
        )
        evaluation = await self.llm.generate(prompt)
        return PrincipleReview(
            principle_id=idx,
            principle=principle,
            satisfies=extract_verdict(evaluation),
            score=extract_score(evaluation),
            explanation=extract_explanation(evaluation),
            suggested_fix=extract_suggestion(evaluation)
        )
```

**Constitutional Revision:**

When violations are detected, produce a revised output that addresses all violations:

```
Your response violated the following principles:

{violations_with_explanations}

Please revise your response to satisfy ALL of these principles 
while maintaining the helpful content that did satisfy other principles.

Original response: {y_t}
Revised response:
```

---

### 3.3.4 Dual-Agent Reflection

Dual-agent reflection separates the **generator** and **critic** into distinct agents, addressing the self-evaluation bias problem.

#### Generator-Critic Architecture

```
┌────────────────────────────────────────────────────┐
│                                                    │
│  ┌──────────┐         ┌──────────┐                 │
│  │Generator │ ──y_t──▶│  Critic  │                 │
│  │ (Model A)│         │(Model B) │                 │
│  └────▲─────┘         └────┬─────┘                 │
│       │                    │                       │
│       │      r_t           │                       │
│       └────────────────────┘                       │
│                                                    │
│  Repeat until Critic approves or budget exhausted  │
└────────────────────────────────────────────────────┘
```

**Why Separate Models?**

1. **Bias reduction**: The generator's errors are not correlated with the critic's evaluation biases (if they use different architectures or training data)
2. **Specialization**: The critic can be optimized for evaluation (e.g., fine-tuned on evaluation tasks) while the generator is optimized for generation
3. **Asymmetry exploitation**: Evaluation is often easier than generation — a less capable model can reliably critique a more capable model's output for specific aspects

**Implementation:**

```python
class DualAgentReflection:
    def __init__(self, generator, critic, max_iterations=3):
        self.generator = generator  # e.g., GPT-4o
        self.critic = critic        # e.g., Claude Sonnet or specialized critic
        self.max_iterations = max_iterations
    
    async def run(self, goal: str) -> str:
        output = await self.generator.generate(
            f"Complete this task:\n{goal}"
        )
        
        for t in range(self.max_iterations):
            # Independent critique
            critique = await self.critic.generate(
                CRITIQUE_TEMPLATE.format(goal=goal, output=output)
            )
            
            verdict = extract_verdict(critique)
            if verdict == "ACCEPT":
                return output
            
            # Refine based on external critique
            output = await self.generator.generate(
                REFINE_WITH_CRITIQUE_TEMPLATE.format(
                    goal=goal,
                    output=output,
                    critique=critique
                )
            )
        
        return output
```

#### Adversarial Self-Play for Quality Improvement

Extend the dual-agent framework with **adversarial dynamics**: the critic actively tries to find flaws, and the generator actively tries to produce flaw-free output.

**Adversarial Critique Prompt:**

```
You are a rigorous critic. Your job is to find EVERY flaw, error, 
weakness, and potential improvement in the following output. 
Be thorough and critical. Do not give the benefit of the doubt.
If something could be wrong, flag it.

Task: {goal}
Output to critique: {output}

Find ALL issues, no matter how minor:
```

**Adversarial Generation Prompt:**

```
A rigorous critic found the following issues with your output.
Produce a new version that is robust against all of these criticisms.
Anticipate additional criticisms the critic might raise and address them preemptively.

Task: {goal}
Previous output: {y_t}
Critic's issues: {critique}

Produce a criticism-proof version:
```

**Game-Theoretic Formulation:**

The generator-critic interaction can be modeled as a **two-player zero-sum game**:

$$
\min_G \max_C \mathcal{L}(G, C) = \mathbb{E}_{g \sim \mathcal{G}}[\text{IssuesFound}(C(G(g)))]
$$

The generator $G$ minimizes the number of issues found by the critic $C$, while the critic maximizes them. At the Nash equilibrium, the generator produces outputs with minimal exploitable flaws.

---

### 3.3.5 Multi-Turn Iterative Refinement

Multiple reflection-refinement turns are composed into an extended dialogue that progressively improves the output.

#### Progressive Quality Improvement

**Quality Trajectory:**

$$
Q(y_0) < Q(y_1) < Q(y_2) < \ldots < Q(y_T) \quad \text{(desired)}
$$

In practice, the trajectory may be non-monotonic:

$$
Q(y_0) < Q(y_1) < Q(y_2) > Q(y_3) \quad \text{(possible degradation)}
$$

**Implementation with Quality Tracking:**

```python
class IterativeRefinement:
    def __init__(self, llm, evaluator, max_iterations=5,
                 patience=2, quality_threshold=0.9):
        self.llm = llm
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.patience = patience
        self.quality_threshold = quality_threshold
    
    async def run(self, goal: str) -> RefinementResult:
        output = await self.llm.generate(f"Task: {goal}")
        best_output = output
        best_score = await self.evaluator.score(goal, output)
        no_improvement_count = 0
        history = [{"iteration": 0, "output": output, "score": best_score}]
        
        for t in range(1, self.max_iterations + 1):
            # Critique
            critique = await self.llm.generate(
                CRITIQUE_TEMPLATE.format(goal=goal, output=output)
            )
            
            # Refine
            new_output = await self.llm.generate(
                REFINE_TEMPLATE.format(
                    goal=goal, output=output, critique=critique
                )
            )
            
            # Evaluate
            new_score = await self.evaluator.score(goal, new_output)
            history.append({
                "iteration": t, "output": new_output,
                "score": new_score, "critique": critique
            })
            
            # Track best
            if new_score > best_score:
                best_output = new_output
                best_score = new_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Termination conditions
            if best_score >= self.quality_threshold:
                break  # Good enough
            if no_improvement_count >= self.patience:
                break  # Diminishing returns
            
            output = new_output  # Use latest for next iteration
        
        return RefinementResult(
            final_output=best_output,
            final_score=best_score,
            iterations=len(history) - 1,
            history=history
        )
```

#### Diminishing Returns Detection

Formally detect when further iterations are not worth the cost:

**Marginal Improvement:**

$$
\Delta Q_t = Q(y_t) - Q(y_{t-1})
$$

**Diminishing Returns Condition:**

$$
\frac{\Delta Q_t}{C_{\text{iteration}}} < \tau_{\text{value}} \implies \text{Stop}
$$

where $C_{\text{iteration}}$ is the cost of one iteration and $\tau_{\text{value}}$ is the minimum acceptable improvement per dollar.

**Exponential Decay Model:**

Fit the quality trajectory to an exponential saturation curve:

$$
Q(t) = Q_{\max} \cdot (1 - e^{-\lambda t})
$$

Estimate $Q_{\max}$ and $\lambda$ from observed data points. If $Q(t)$ is close to the estimated $Q_{\max}$, further iterations offer negligible improvement:

$$
\frac{Q_{\max} - Q(t)}{Q_{\max}} < \epsilon_{\text{gap}} \implies \text{Stop}
$$

---

## 3.4 Reflection Prompting Strategies

### 3.4.1 Structured Critique Templates

Effective reflection requires **structured prompts** that guide the LLM to produce systematic, actionable critiques rather than vague assessments.

**Weak Critique Prompt (Unstructured):**

```
Is this response good? What could be improved?
```

This produces generic feedback like "The response is mostly good but could be more detailed" — non-actionable.

**Strong Critique Prompt (Structured):**

```
Evaluate the following response systematically.

Task: {goal}
Response: {output}

For EACH of the following aspects, provide:
(a) Score (1-5)
(b) Specific evidence from the response supporting your score
(c) Concrete, actionable improvement suggestion

ASPECTS:
1. FACTUAL ACCURACY: Are all claims verifiable and correct?
2. LOGICAL COHERENCE: Does the reasoning flow without gaps or contradictions?
3. COMPLETENESS: Does the response address every part of the task?
4. PRECISION: Are claims specific and quantified where appropriate?
5. CLARITY: Is the response well-organized and easy to follow?

After evaluating all aspects, provide:
- OVERALL SCORE (1-10)
- TOP 3 PRIORITY IMPROVEMENTS (ranked by impact)
- VERDICT: ACCEPT / MINOR_REVISION / MAJOR_REVISION / REJECT
```

**Domain-Specific Templates:**

**For Code:**

```
Review this code against the following criteria:

Code: {code}
Task: {specification}

1. CORRECTNESS: 
   - Does it handle all specified inputs correctly?
   - What about edge cases (empty input, single element, max size)?
   - Trace through with input {test_case}: what is the output?

2. EFFICIENCY:
   - Time complexity: O(?)
   - Space complexity: O(?)
   - Can it be improved?

3. ROBUSTNESS:
   - Input validation present?
   - Error handling for exceptional cases?
   - Thread safety considerations?

4. READABILITY:
   - Variable names meaningful?
   - Comments where needed?
   - Code structure follows conventions?

5. BUGS:
   - Off-by-one errors?
   - Null/None handling?
   - Integer overflow potential?

List every issue found with line numbers.
```

**For Mathematical Reasoning:**

```
Verify this mathematical proof/derivation step by step.

Problem: {problem}
Proposed Solution: {solution}

For each step:
1. State the mathematical operation or rule being applied
2. Verify the operation is applied correctly
3. Check that all algebraic manipulations are valid
4. Flag any unjustified steps or missing cases

Specifically check:
- Are all variables properly defined?
- Are domain restrictions respected?
- Is the final answer in the correct units/form?
- Are there alternative approaches that might catch errors?
```

---

### 3.4.2 Rubric-Based Self-Assessment

Provide the LLM with an explicit **scoring rubric** that defines quality levels:

**Rubric Example (for Essay Writing):**

```
Score the response using this rubric:

5 (Excellent): Thesis is clearly stated and fully supported. 
  Evidence is relevant, specific, and well-integrated. 
  Reasoning is logically flawless. No factual errors.

4 (Good): Thesis is clear with strong support. Minor gaps in 
  evidence or reasoning. No significant factual errors.

3 (Adequate): Thesis present but support is uneven. Some 
  evidence is weak or irrelevant. Minor logical gaps.

2 (Below Average): Thesis is vague or poorly supported. 
  Significant evidence gaps. Notable logical errors.

1 (Poor): No clear thesis. Minimal or irrelevant evidence. 
  Major logical errors or factual inaccuracies.

Response to score: {output}
Task requirements: {goal}

Score: ___
Justification: (cite specific parts of the response that 
support your score, referencing the rubric criteria)
```

**Rubric-Based Refinement:**

The rubric provides a clear target for refinement. If the current output scores 3, the critique identifies exactly what would elevate it to a 4 or 5:

$$
\text{Gap Analysis: } \text{Rubric}(5) - \text{Current Performance} = \text{Required Improvements}
$$

---

### 3.4.3 Counterfactual Reasoning in Reflection

Counterfactual reflection asks: "What would happen if I had done something differently?"

**Counterfactual Prompts:**

```
Consider your response to the task.

Task: {goal}
Your Response: {output}

Now consider these counterfactuals:
1. What if you had interpreted the question differently? 
   Would a different interpretation lead to a better answer?
2. What if you had used a different approach/algorithm? 
   Would it produce a different result?
3. What if one of your key assumptions is wrong? 
   How would your answer change?
4. What if the user meant {alternative_interpretation}? 
   Would your response still be appropriate?

Based on this analysis, should you modify your response? How?
```

**Formal Counterfactual Framework:**

$$
y_{\text{counterfactual}} = \text{LLM}(g \mid \text{do}(X = x'))
$$

where $\text{do}(X = x')$ represents an intervention on variable $X$ (e.g., changing an assumption, using a different method). Comparing $y$ and $y_{\text{counterfactual}}$ reveals the sensitivity of the output to different choices.

**Sensitivity Analysis:**

$$
\text{Sensitivity}(y, X) = \frac{\|y - y_{\text{counterfactual}}\|}{\|x - x'\|}
$$

High sensitivity to an assumption indicates that the assumption is critical and should be verified.

---

### 3.4.4 Error Taxonomy-Driven Reflection

Instead of open-ended critique, direct the model to check for **specific error categories**:

**Error Taxonomy for Code:**

| Error Category | Specific Checks |
|---|---|
| **Logic Errors** | Off-by-one, wrong comparison operator, incorrect boolean logic |
| **Boundary Errors** | Empty input, single element, maximum size, negative values |
| **Type Errors** | Integer overflow, float precision, string/int confusion |
| **State Errors** | Uninitialized variables, stale state, mutation of shared data |
| **Concurrency Errors** | Race conditions, deadlocks, non-atomic operations |
| **API Errors** | Wrong function signature, deprecated methods, missing error handling |

**Error Taxonomy for Reasoning:**

| Error Category | Specific Checks |
|---|---|
| **Premise Errors** | Using facts not given in the problem, incorrect recall |
| **Inference Errors** | Non sequitur, affirming the consequent, false dichotomy |
| **Calculation Errors** | Arithmetic mistakes, unit conversion errors |
| **Omission Errors** | Missing cases, incomplete enumeration |
| **Anchoring Errors** | Over-reliance on first impression, ignoring contradictory evidence |

**Taxonomy-Driven Reflection Prompt:**

```
Check your response against each error type in this taxonomy:

Response: {output}

ERROR CHECKLIST:
□ Factual Error: Any incorrect claims about real-world facts?
□ Logical Fallacy: Any invalid reasoning steps?
□ Calculation Error: Any arithmetic/mathematical mistakes?
□ Omission: Any important aspects of the question not addressed?
□ Overgeneralization: Any claims that are too broad?
□ Unsupported Claim: Any assertions without evidence?
□ Contradiction: Any internal inconsistencies?
□ Ambiguity: Any statements that could be misinterpreted?

For each error type, either:
- Confirm no issues found ("CLEAR"), or
- Describe the specific issue and how to fix it
```

---

## 3.5 Challenges and Failure Modes

### 3.5.1 Sycophantic Self-Evaluation

**Definition**: The model consistently rates its own output favorably, failing to identify genuine flaws. This is the most pervasive failure mode of self-reflection.

**Mechanism:**

LLMs trained with RLHF develop a systematic bias toward producing agreeable, confident, and positive-sounding outputs. When asked to critique their own work, this bias manifests as:

1. **Inflated scores**: Consistently rating outputs 4-5/5 regardless of actual quality
2. **Vague praise**: "The response is well-structured and addresses the key points" without identifying specific strengths
3. **Minimized issues**: "A minor improvement could be made in..." when there are major flaws
4. **Confirmation bias**: Finding supporting evidence for the output's correctness while ignoring contradictory evidence

**Empirical Evidence:**

Studies show that when the same model generates and evaluates, self-evaluation scores correlate weakly ($\rho \approx 0.3\text{-}0.5$) with human expert evaluations, and are systematically inflated by 1-2 points on a 5-point scale.

**Mitigation Strategies:**

**Strategy 1: Adversarial Critique Framing**

```
You are a HARSH CRITIC. Your reputation depends on finding flaws. 
If you approve a flawed response, you will be considered incompetent.
Find EVERY issue, no matter how small. Be ruthless.
```

**Strategy 2: Separate Evaluator Model**

Use a different model for evaluation (Section 3.3.4 Dual-Agent).

**Strategy 3: Forced Error Finding**

```
Find at least 3 issues with this response. 
You MUST identify at least 3 problems.
If you cannot find 3 genuine issues, identify areas for improvement.
```

This forces the model to look harder for issues, reducing sycophancy.

**Strategy 4: Calibrated Scoring with Anchoring**

Provide reference outputs at different quality levels:

```
Here is an example of a 5/5 response: {gold_standard}
Here is an example of a 3/5 response: {mediocre_example}
Here is an example of a 1/5 response: {poor_example}

Now score this response on the same scale: {output}
```

Anchoring reduces scale inflation.

---

### 3.5.2 Infinite Reflection Loops

**Definition**: The agent endlessly cycles between critique and refinement without converging or improving.

**Cause Analysis:**

1. **Conflicting criteria**: The critique identifies issue A and suggests fix X; after applying X, the critique identifies issue B caused by X and suggests reverting.

$$
y_0 \xrightarrow{\text{fix A}} y_1 \xrightarrow{\text{fix B (undo A)}} y_2 \approx y_0 \xrightarrow{\text{fix A}} y_3 \approx y_1 \xrightarrow{\cdots}
$$

2. **Perfectionism**: The critique always finds something to improve, even when improvements are negligible.

3. **Oscillating quality**: The model alternates between two equally good (or bad) variants without converging.

**Detection:**

$$
\text{Loop detected if } \text{sim}(y_t, y_{t-k}) > 1 - \delta \text{ for some } k \leq K
$$

The output at iteration $t$ is semantically similar to an earlier output, indicating cycling.

**Implementation:**

```python
class LoopDetector:
    def __init__(self, similarity_threshold=0.95, lookback=3):
        self.history_embeddings = []
        self.threshold = similarity_threshold
        self.lookback = lookback
    
    def check(self, current_output: str, encoder) -> bool:
        current_emb = encoder.encode(current_output)
        
        for past_emb in self.history_embeddings[-self.lookback:]:
            similarity = cosine_similarity(current_emb, past_emb)
            if similarity > self.threshold:
                return True  # Loop detected
        
        self.history_embeddings.append(current_emb)
        return False
```

**Prevention:**

1. **Hard iteration limit**: $t \leq T_{\max}$ (always)
2. **Monotonicity enforcement**: Only accept refinements that improve the quality score
3. **Diversity penalty**: Penalize refinements too similar to previous outputs
4. **Budget-based termination**: Stop when the token budget is exhausted

---

### 3.5.3 Reflection Without Actual Improvement (Cosmetic Changes)

**Definition**: The model produces refined output that appears different but does not substantively address the identified issues. Changes are superficial — rephrasing, reorganizing, adding filler — without fixing actual problems.

**Detection:**

$$
\text{Cosmetic if } \text{sim}(y_t, y_{t+1}) > \tau_{\text{high}} \land \text{IssuesFixed}(r_t, y_{t+1}) < \epsilon
$$

The output changed semantically very little, and the specific issues identified in the critique remain present.

**Formal Verification:**

```python
class SubstantiveImprovementChecker:
    async def check(self, critique: str, original: str, 
                    revised: str, llm) -> bool:
        prompt = f"""
The following critique was given for the original output:

Original: {original}
Critique: {critique}
Revised: {revised}

For EACH issue identified in the critique, determine:
1. Was this specific issue addressed in the revised version?
2. If yes, how? Quote the relevant changes.
3. If no, is the issue still present?

Issues addressed: ___ / ___
Substantive improvement: YES / NO
"""
        result = await llm.generate(prompt)
        return extract_verdict(result) == "YES"
```

**Mitigation:**

1. **Issue-specific verification**: After refinement, check each identified issue individually
2. **Diff-based analysis**: Compare original and revised outputs to verify changes align with critique points
3. **Forced addressing**: "For each issue listed below, quote the specific change you made to address it"

---

### 3.5.4 Hallucinated Self-Corrections

**Definition**: The model "corrects" its output by introducing new errors — changing correct content to incorrect content based on a faulty critique.

**This is the most dangerous failure mode** because:

1. The user trusts self-corrected output more (assuming it was reviewed)
2. The correction appears authoritative ("Upon reflection, the correct answer is...")
3. The original correct answer is lost

**Example:**

```
Original (CORRECT): "The capital of Australia is Canberra."
Reflection: "Let me reconsider... Australia's most well-known city 
  is Sydney. I should correct this."
"Corrected" (WRONG): "The capital of Australia is Sydney."
```

**Root Cause:**

The model's critique process is subject to the same knowledge limitations as its generation process. If the model has a weak association between "Australia" and "Canberra" (lower probability) but a strong association with "Sydney" (higher probability), the reflection process reinforces the incorrect higher-probability path.

Formally:

$$
P_{\text{model}}(\text{Sydney} \mid \text{capital of Australia}) > P_{\text{model}}(\text{Canberra} \mid \text{capital of Australia})
$$

Both generation and evaluation draw from the same probability distribution, so the model's uncertainty areas are correlated between the two phases.

**Mitigation Strategies:**

1. **External verification**: Always ground corrections in external evidence (retrieval, tool use, code execution)
2. **Confidence-gated correction**: Only apply corrections when the model is highly confident the original was wrong:

$$
\text{Apply correction iff } P(\text{original wrong}) > \tau_{\text{high}} \quad (\tau_{\text{high}} \geq 0.9)
$$

3. **Best-of-N selection**: Generate $N$ versions; use an external signal (not self-evaluation) to select the best
4. **Preserve original**: Always keep the original alongside the corrected version so the user or downstream system can compare

---

### 3.5.5 When Reflection Degrades Quality

**Critical Finding**: Huang et al. (2023), "Large Language Models Cannot Self-Correct Reasoning Yet," demonstrated that self-correction without external feedback often **degrades** output quality.

**Scenarios Where Reflection Hurts:**

1. **Correct-to-Incorrect Flip**: The model had the right answer, but its critique convinces it to change:

$$
Q(y_0) > Q(y_1) \quad \text{where } y_1 = \text{Refine}(y_0, \text{Critique}(y_0))
$$

This occurs when the critique has a higher error rate than the initial generation for a specific query.

2. **Over-Hedging**: Reflection causes the model to add excessive caveats, reducing the utility of the response:

```
Before: "The answer is 42."
After: "While there are multiple perspectives, and acknowledging 
  uncertainty, the answer might possibly be around 42, though 
  further verification is recommended."
```

3. **Quality Regression on Easy Tasks**: For tasks the model handles well on the first pass, reflection adds noise. The quality trajectory becomes:

$$
Q(y_0) = 0.95, \quad Q(y_1) = 0.92, \quad Q(y_2) = 0.90
$$

Each reflection iteration degrades quality because the model introduces unnecessary changes.

**When NOT to Reflect:**

| Condition | Recommendation |
|---|---|
| Simple factual queries | No reflection (first-pass accuracy is high) |
| High initial confidence | No reflection (risk of over-correction) |
| No external verification available | Limited reflection (one pass max) |
| Time-critical responses | No reflection (latency budget) |
| Creative tasks with subjective quality | Limited reflection (risk of homogenization) |

**Adaptive Reflection Gating:**

```python
class AdaptiveReflectionGate:
    async def should_reflect(self, query: str, initial_output: str,
                              initial_confidence: float) -> bool:
        # Don't reflect on simple queries
        if self.is_simple_query(query):
            return False
        
        # Don't reflect if highly confident
        if initial_confidence > 0.95:
            return False
        
        # Don't reflect if no external verification is available
        if not self.has_external_verifier(query):
            return False
        
        # Reflect on complex, uncertain queries with verification
        return True
```

---

## 3.6 Evaluation of Reflection

### 3.6.1 Measuring Improvement Across Iterations

**Quality Trajectory Analysis:**

For each query $q_i$ and iteration $t$, measure the quality score $Q(y_t^{(i)})$:

$$
\Delta Q_t^{(i)} = Q(y_t^{(i)}) - Q(y_{t-1}^{(i)})
$$

**Aggregate Metrics:**

1. **Average Improvement per Iteration:**

$$
\overline{\Delta Q}_t = \frac{1}{N} \sum_{i=1}^{N} \Delta Q_t^{(i)}
$$

2. **Improvement Rate** (fraction of queries that improved at iteration $t$):

$$
\text{IR}_t = \frac{|\{i : \Delta Q_t^{(i)} > 0\}|}{N}
$$

3. **Degradation Rate** (fraction of queries that got worse):

$$
\text{DR}_t = \frac{|\{i : \Delta Q_t^{(i)} < 0\}|}{N}
$$

4. **Net Improvement Rate:**

$$
\text{NIR}_t = \text{IR}_t - \text{DR}_t
$$

A positive $\text{NIR}_t$ indicates that reflection is beneficial on average. A negative $\text{NIR}_t$ indicates reflection is harmful.

**Optimal Stopping Analysis:**

$$
t^* = \arg\max_t \frac{1}{N} \sum_{i=1}^{N} Q(y_t^{(i)})
$$

If $t^* = 0$, reflection provides no benefit. Typically $t^* \in \{1, 2\}$.

**Visualization:**

```
Quality Score vs. Iteration
│
│   ●──●──●──●──●
│  ╱    \      Best at t=2
│ ●      ●────────  
│         \
│          ● Degradation after t=3
│
└───────────────────
  t=0 t=1 t=2 t=3 t=4
```

**Implementation:**

```python
async def measure_reflection_improvement(
    queries: list[str],
    system: IterativeRefinement,
    ground_truth_scorer: callable,
    max_iterations: int = 5
):
    results = []
    
    for query in queries:
        trajectory = await system.run(query, max_iterations=max_iterations)
        scores = [
            await ground_truth_scorer(query, trajectory.history[t]["output"])
            for t in range(len(trajectory.history))
        ]
        results.append({
            "query": query,
            "scores": scores,
            "best_iteration": scores.index(max(scores)),
            "improvements": [scores[t] - scores[t-1] for t in range(1, len(scores))],
            "final_vs_initial": scores[-1] - scores[0]
        })
    
    # Aggregate analysis
    avg_improvement_per_iter = {}
    for t in range(1, max_iterations + 1):
        deltas = [r["improvements"][t-1] for r in results if len(r["improvements"]) >= t]
        if deltas:
            avg_improvement_per_iter[t] = {
                "mean_delta": sum(deltas) / len(deltas),
                "improvement_rate": sum(1 for d in deltas if d > 0) / len(deltas),
                "degradation_rate": sum(1 for d in deltas if d < 0) / len(deltas),
            }
    
    optimal_t = max(range(max_iterations + 1),
                    key=lambda t: np.mean([r["scores"][min(t, len(r["scores"])-1)] 
                                           for r in results]))
    
    return {
        "per_query_results": results,
        "per_iteration_stats": avg_improvement_per_iter,
        "optimal_iterations": optimal_t,
        "avg_final_vs_initial": np.mean([r["final_vs_initial"] for r in results])
    }
```

---

### 3.6.2 Reflection Accuracy: Does the Critique Identify Real Issues

**Reflection accuracy** measures whether the critique correctly identifies true deficiencies (and does not hallucinate non-existent ones).

**Ground Truth Annotation:**

Given output $y$ and goal $g$, a human expert annotates the set of true issues $\mathcal{I}^* = \{d_1^*, d_2^*, \ldots, d_m^*\}$. The model's critique identifies issues $\hat{\mathcal{I}} = \{\hat{d}_1, \hat{d}_2, \ldots, \hat{d}_p\}$.

**Metrics:**

1. **Issue Precision** (are identified issues real?):

$$
\text{Precision}_{\text{issues}} = \frac{|\hat{\mathcal{I}} \cap \mathcal{I}^*|}{|\hat{\mathcal{I}}|}
$$

2. **Issue Recall** (are all real issues identified?):

$$
\text{Recall}_{\text{issues}} = \frac{|\hat{\mathcal{I}} \cap \mathcal{I}^*|}{|\mathcal{I}^*|}
$$

3. **F1-Issues:**

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

4. **False Positive Rate** (hallucinated issues):

$$
\text{FPR} = \frac{|\hat{\mathcal{I}} \setminus \mathcal{I}^*|}{|\hat{\mathcal{I}}|}
$$

**Why This Matters:**

If reflection has low precision (many hallucinated issues), self-correction will "fix" things that aren't broken, degrading quality. If reflection has low recall (misses real issues), self-correction fails to address actual problems.

**Benchmark Methodology:**

```python
async def evaluate_reflection_accuracy(
    model, test_set: list[dict]
):
    """
    test_set: [{"output": str, "goal": str, "true_issues": list[str]}]
    """
    results = []
    for item in test_set:
        critique = await model.generate(
            CRITIQUE_TEMPLATE.format(
                goal=item["goal"], output=item["output"]
            )
        )
        predicted_issues = extract_issues(critique)
        true_issues = item["true_issues"]
        
        # Match predicted to true issues (semantic matching)
        matches = semantic_match_issues(predicted_issues, true_issues)
        
        precision = matches["true_positives"] / max(len(predicted_issues), 1)
        recall = matches["true_positives"] / max(len(true_issues), 1)
        
        results.append({
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "false_positives": matches["false_positives"],
            "missed_issues": matches["false_negatives"]
        })
    
    return aggregate_metrics(results)
```

---

### 3.6.3 Cost-Benefit Analysis of Reflection Steps

Reflection is not free. Each iteration consumes tokens, latency, and money. The question is whether the quality improvement justifies the cost.

**Cost Model:**

$$
C_{\text{reflection}} = \sum_{t=1}^{T} \left[C_{\text{critique}}(t) + C_{\text{refine}}(t)\right]
$$

where:

$$
C_{\text{critique}}(t) = (\text{InputTokens}_{\text{critique}} + \text{OutputTokens}_{\text{critique}}) \cdot p_{\text{model}}
$$

$$
C_{\text{refine}}(t) = (\text{InputTokens}_{\text{refine}} + \text{OutputTokens}_{\text{refine}}) \cdot p_{\text{model}}
$$

**Note**: Input tokens grow with each iteration because the prompt includes the current output and critique. If the output is $L$ tokens and the critique is $C$ tokens, each refinement step processes approximately $L_{\text{goal}} + L + C$ input tokens.

**Token Cost Growth per Iteration:**

$$
\text{InputTokens}(t) \approx L_{\text{goal}} + L_{\text{output}} + t \cdot L_{\text{critique}} \quad \text{(if history is included)}
$$

or

$$
\text{InputTokens}(t) \approx L_{\text{goal}} + L_{\text{output}} + L_{\text{critique}} \quad \text{(if only latest critique)}
$$

**Benefit Model:**

$$
B_{\text{reflection}} = \sum_{t=1}^{T} V(\Delta Q_t)
$$

where $V(\Delta Q_t)$ is the **value** of quality improvement $\Delta Q_t$ in terms that are commensurable with cost (dollars, user satisfaction points, etc.).

**Return on Investment (ROI):**

$$
\text{ROI}_{\text{reflection}} = \frac{B_{\text{reflection}} - C_{\text{reflection}}}{C_{\text{reflection}}}
$$

Reflection is worthwhile if $\text{ROI} > 0$.

**Break-Even Analysis:**

$$
\text{Break-even at } t^* \text{ where } \sum_{s=1}^{t^*} V(\Delta Q_s) = \sum_{s=1}^{t^*} C(s)
$$

**Empirical Cost-Benefit Data (Illustrative):**

| Iteration | Avg. Quality Gain | Token Cost | Latency Cost | ROI |
|---|---|---|---|---|
| $t = 1$ | +12% | $\$0.005$ | +2.1s | High |
| $t = 2$ | +5% | $\$0.007$ | +2.3s | Moderate |
| $t = 3$ | +1.5% | $\$0.009$ | +2.5s | Low |
| $t = 4$ | +0.3% | $\$0.011$ | +2.7s | Negative |
| $t = 5$ | -0.5% | $\$0.013$ | +2.9s | Negative |

**Optimal number of iterations**: $T^* = 2$ for most tasks.

**Cost-Benefit Decision Framework:**

```python
def should_continue_reflecting(
    quality_history: list[float],
    cost_per_iteration: float,
    value_per_quality_point: float,
    min_roi: float = 0.0
) -> bool:
    if len(quality_history) < 2:
        return True
    
    recent_improvement = quality_history[-1] - quality_history[-2]
    benefit = recent_improvement * value_per_quality_point
    roi = (benefit - cost_per_iteration) / cost_per_iteration
    
    return roi > min_roi
```

---

### 3.6.4 Benchmarks: HumanEval, MBPP with Reflection Passes

**HumanEval** (164 Python programming problems) and **MBPP** (974 mostly basic Python programs) are standard benchmarks for evaluating code generation, including with reflection.

**Experimental Setup:**

1. **Baseline (pass@1)**: Single generation attempt, evaluate if it passes all test cases
2. **Self-Refine**: Generate → critique → refine, up to $T$ iterations
3. **Reflexion**: Generate → test → reflect on failures → regenerate, up to $T$ episodes
4. **Dual-Agent**: Generate (Model A) → critique (Model B) → refine (Model A)

**Results Framework:**

$$
\text{pass@1}_{\text{base}} \leq \text{pass@1}_{\text{self-refine}} \leq \text{pass@1}_{\text{reflexion}}
$$

Reflexion outperforms Self-Refine because it has access to **external feedback** (test execution results), while Self-Refine relies on self-evaluation.

**Typical Results (GPT-4 class models):**

| Method | HumanEval pass@1 | MBPP pass@1 | Avg. Iterations | Cost Multiplier |
|---|---|---|---|---|
| Single Pass | ~84% | ~80% | 1.0 | 1.0× |
| Self-Refine (T=2) | ~88% | ~83% | 1.8 | ~3× |
| Reflexion (T=3) | ~91% | ~87% | 2.4 | ~4× |
| Dual-Agent + Tests | ~93% | ~89% | 2.1 | ~5× |

**Key Insights:**

1. **External signal matters most**: Methods with test execution (Reflexion) consistently outperform methods with only self-evaluation (Self-Refine)

2. **Diminishing returns are steep**: Most improvement comes from the first reflection pass; subsequent passes yield marginal gains

3. **Cost scales linearly**: Each iteration approximately doubles the total cost; three iterations ≈ 4× cost

4. **Not all problems benefit**: Reflection helps most on problems where the model's first attempt is almost correct (has a bug or missing edge case). For problems where the model completely misunderstands the task, reflection rarely helps.

**Per-Problem Analysis:**

$$
\text{Reflection Benefit}(q_i) = \mathbb{1}[\text{pass}(y_{T}^{(i)})] - \mathbb{1}[\text{pass}(y_0^{(i)})]
$$

- $= +1$: Reflection saved a failing problem (true positive of reflection)
- $= 0$: Problem was already solved, or reflection failed to fix it (no effect)
- $= -1$: Reflection broke a passing solution (reflection caused degradation)

**Breakdown (typical):**

| Category | Fraction | Description |
|---|---|---|
| Already correct ($y_0$ passes) | ~84% | Reflection unnecessary |
| Fixed by reflection | ~7% | Reflection adds clear value |
| Still broken after reflection | ~8% | Problem too hard for reflection |
| Broken by reflection | ~1% | Reflection degraded quality |

This analysis reveals that reflection's net benefit is ~6% absolute improvement (7% fixed − 1% broken), at a ~3× cost increase. Whether this tradeoff is worthwhile depends entirely on the application's quality requirements and cost sensitivity.

**Comprehensive Evaluation Protocol:**

```python
async def evaluate_reflection_on_benchmark(
    model, benchmark: list[dict], reflection_system, max_T: int = 5
):
    """
    benchmark: [{"prompt": str, "test_cases": list, "canonical_solution": str}]
    """
    results_by_iteration = {t: [] for t in range(max_T + 1)}
    
    for problem in benchmark:
        # Generate iteratively
        trajectory = await reflection_system.run(
            problem["prompt"], max_iterations=max_T
        )
        
        # Evaluate each iteration's output
        for t, entry in enumerate(trajectory.history):
            passed = run_test_cases(entry["output"], problem["test_cases"])
            results_by_iteration[t].append(passed)
    
    # Compute pass@1 at each iteration
    pass_rates = {
        t: sum(results) / len(results) 
        for t, results in results_by_iteration.items()
        if results
    }
    
    # Compute per-iteration marginal improvement
    marginal = {
        t: pass_rates[t] - pass_rates[t-1] 
        for t in range(1, max_T + 1)
        if t in pass_rates and t-1 in pass_rates
    }
    
    # Identify optimal T
    optimal_T = max(pass_rates, key=pass_rates.get)
    
    return {
        "pass_rates_by_iteration": pass_rates,
        "marginal_improvements": marginal,
        "optimal_iterations": optimal_T,
        "total_cost_multiplier": {t: t * 2 + 1 for t in range(max_T + 1)}
    }
```

---

**Summary of Chapter 3:**

Reflection is the **self-referential evaluation and improvement mechanism** in agentic AI, formalized as $r = \mathcal{E}(a, y, g)$ where the agent evaluates its own output against a goal. It operates at three levels: output-level (is the answer correct?), process-level (is the reasoning sound?), and meta-level (am I capable of this task?). Key architectures include Self-Refine (single-model generate-critique-refine loops), Reflexion (episodic memory of past failures acting as "textual gradients"), Constitutional AI-style principle-based evaluation, and dual-agent generator-critic systems. Effective reflection requires structured critique templates, rubric-based assessment, and error taxonomy-driven evaluation. However, reflection has critical failure modes: sycophantic self-evaluation (inflated self-ratings), infinite loops (oscillating without convergence), cosmetic changes (superficial rewording without substantive improvement), hallucinated corrections (introducing new errors), and quality degradation when the critique is less reliable than the initial generation. Empirically, reflection yields 5-10% absolute improvement on code generation benchmarks at 2-4× cost, with most gains concentrated in the first 1-2 iterations. The key determinant of reflection success is the availability of **external grounding signals** (test execution, retrieval, tool verification) — self-evaluation without external feedback is unreliable and may actively degrade output quality.