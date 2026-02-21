<p align="center">
  <h1 align="center">Agentic Design Patterns</h1>
  <p align="center">
    A rigorous, engineer-oriented reference for building production-grade autonomous AI agents.
  </p>
</p>

<p align="center">
  <a href="https://generalaimodels.github.io/agentic_design_patterns/"><img alt="Documentation" src="https://img.shields.io/badge/docs-live-blue?style=flat-square&logo=github-pages"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
  <a href="https://github.com/generalaimodels/agentic_design_patterns/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/generalaimodels/agentic_design_patterns/nextjs.yml?branch=main&style=flat-square&label=deploy"></a>
  <a href="https://github.com/generalaimodels/agentic_design_patterns/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/generalaimodels/agentic_design_patterns?style=flat-square"></a>
</p>

---

## Why This Repository Exists

Most agentic AI resources teach you *what* agents are.  
This repository teaches you **how to architect them** — with formal decision frameworks, battle-tested patterns, and the kind of structural rigour that separates reliable systems from fragile demos.

Every chapter grounds its ideas in **mathematical formalism** where it matters (routing as decision functions, memory as scored retrieval, planning as constrained search) and pairs them with **concrete architectural patterns** drawn from real-world deployment experience.

**📖  Read the full documentation →  [generalaimodels.github.io/agentic_design_patterns](https://generalaimodels.github.io/agentic_design_patterns/)**

---

## What's Inside

The reference is organised into **seven parts, 26 chapters, and six appendices**, progressing from foundational primitives through production infrastructure to advanced reasoning engines.

### Part I — Foundations of Agentic Systems

| # | Chapter | Core Idea |
|---|---------|-----------|
| 1 | **Routing** | Decision functions for dispatching queries — static, semantic, LLM-based, adaptive (contextual bandits, RL). |
| 2 | **Parallelization** | Fan-out/fan-in, voting, pipeline & data parallelism, dependency DAGs, speculative execution. |
| 3 | **Reflection** | Self-Refine loops, Reflexion-style episodic learning, Constitutional self-critique, convergence criteria. |
| 4 | **Tool Use** | Specification ↔ selection ↔ execution pipeline, security models, Toolformer-style training. |
| 5 | **Planning** | Classical planning, Tree/Graph of Thoughts, MCTS, A* with LLM heuristics, adaptive replanning. |
| 6 | **Multi-Agent Systems** | Hub-spoke, peer-to-peer, blackboard, market-based architectures; debate & adversarial patterns. |

### Part II — Core Infrastructure Patterns

| # | Chapter | Core Idea |
|---|---------|-----------|
| 7 | **Memory Management** | Short/long-term memory taxonomies, vector/graph storage, consolidation, context-window packing. |
| 8 | **Learning & Adaptation** | In-context learning, prompt-level optimisation (DSPy), experience banks, RL for agents. |
| 9 | **Model Context Protocol (MCP)** | Standardised host ↔ client ↔ server protocol — resources, tools, prompts, sampling. |
| 10 | **Goal Setting & Monitoring** | Goal decomposition, progress tracking, deviation detection, dynamic re-prioritisation. |

### Part III — Robustness & Safety

| # | Chapter | Core Idea |
|---|---------|-----------|
| 11 | **Exception Handling & Recovery** | Error taxonomy, circuit breakers, bulkhead isolation, saga patterns, checkpoint/rollback. |
| 12 | **Human-in-the-Loop (HITL)** | Approval gates, escalation, trust calibration, progressive autonomy. |
| 13 | **Knowledge Retrieval (RAG)** | Indexing pipelines, hybrid retrieval, Graph RAG, agentic RAG, evaluation (RAGAS). |

### Part IV — Communication & Coordination

| # | Chapter | Core Idea |
|---|---------|-----------|
| 14 | **Inter-Agent Communication (A2A)** | Agent cards, task lifecycle, pub/sub, structured message semantics, mutual auth. |
| 15 | **Resource-Aware Optimisation** | Token budgets, latency profiling, cost modelling, Pareto-optimal quality-cost frontiers. |

### Part V — Reasoning & Decision-Making

| # | Chapter | Core Idea |
|---|---------|-----------|
| 16 | **Reasoning Techniques** | Chain-of-Thought, Tree/Graph/Algorithm of Thoughts, self-consistency, process reward models. |
| 17 | **Guardrails & Safety** | Prompt injection defence, hallucination detection, Constitutional AI, red-teaming methodology. |
| 18 | **Evaluation & Monitoring** | LLM-as-Judge, trajectory evaluation, benchmarks (SWE-bench, WebArena, GAIA), observability. |
| 19 | **Prioritization** | Static/dynamic priority, multi-objective ranking, starvation prevention, inter-agent negotiation. |

### Part VI — Advanced Topics

| # | Chapter | Core Idea |
|---|---------|-----------|
| 20 | **Exploration & Discovery** | Curiosity-driven exploration, novelty search, structured strategy exploration. |
| 21 | **Advanced Prompt Engineering** | Prompt chaining, state management, task decomposition, security in prompt pipelines. |
| 22 | **GUI & Real-World Agents** | Vision-language grounding, action spaces, sim-to-real transfer. |
| 23 | **Agentic Frameworks Taxonomy** | LangGraph, CrewAI, AutoGen, OpenAI Assistants — architecture deep-dives. |
| 24 | **AgentSpace** | Multi-agent collaboration environments and shared workspaces. |
| 25 | **CLI Agents** | Terminal-native agent design, command execution safety, shell integration. |
| 26 | **Reasoning Engine Internals** | Tokenization paths, KV-cache, speculative decoding, attention mechanics. |

### Appendices

| Appendix | Topic |
|----------|-------|
| A | Mathematical Foundations (probability, information theory, MDPs, game theory) |
| B | System Design Patterns (microservices, event-driven, serverless, container orchestration) |
| C | Prompt Libraries & Templates |
| D | Benchmarks & Evaluation Suites |
| E | Security & Compliance Reference (GDPR, HIPAA, SOC 2, threat modelling) |
| F | Glossary of Terms |

---

## Documentation Site

The full reference is published as a searchable, statically generated site powered by **Next.js** and deployed via **GitHub Pages**:

**🌐  [generalaimodels.github.io/agentic_design_patterns](https://generalaimodels.github.io/agentic_design_patterns/)**

The site is built automatically on every push to `main` through a CI/CD pipeline (GitHub Actions). It includes:

- **Full-text search** across all 26 chapters
- **KaTeX-rendered mathematics** for formal definitions and proofs
- **Mermaid diagram support** for architecture visuals
- **Auto-generated navigation** with chapter hierarchy
- **Responsive layout** optimised for both desktop and mobile reading

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/generalaimodels/agentic_design_patterns.git
cd agentic_design_patterns

# Install dependencies
npm install

# Run the documentation site locally
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to browse the documentation locally.

To build the static site for production:

```bash
npm run build   # Runs content pipeline + Next.js export
```

The output is written to the `out/` directory.

---

## Repository Structure

```
agentic_design_patterns/
├── docs/                    # Source markdown — all 26 chapters + appendices
│   ├── README.md            # Master index with full table of contents
│   ├── Routing/             # Chapter 1
│   ├── Parallelization/     # Chapter 2
│   ├── Reflection/          # Chapter 3
│   ├── ...                  # Chapters 4–19
│   └── Agentic_ai/          # Advanced topics (prompt chaining, frameworks, etc.)
├── app/                     # Next.js application routes
├── components/              # React components (navigation, search, layout)
├── content/                 # Build pipeline — markdown → JSON → pages
├── styles/                  # Global CSS and design tokens
├── tests/                   # Vitest + Playwright test suites
├── .github/workflows/       # CI/CD — build, test, deploy to GitHub Pages
├── next.config.ts           # Next.js configuration (static export)
├── package.json
└── LICENSE                  # MIT
```

---

## Design Principles

1. **Formal where it counts.** Every pattern includes the mathematical framing that makes it precise — routing as scored decision functions, memory as weighted retrieval, planning as constrained search. But formalism serves clarity, never obscures it.

2. **Architecture over hype.** The focus is on durable structural patterns (circuit breakers, saga workflows, fan-out/fan-in, state machines) rather than framework-specific APIs that change quarterly.

3. **Failure modes first.** Each chapter devotes equal attention to *what goes wrong* — sycophantic self-evaluation in reflection, infinite loops in planning, token budget explosion in multi-agent conversations — because production systems are defined by how they handle the unhappy path.

4. **Cross-referencing.** Chapters are heavily interlinked. Routing connects to multi-agent dispatch; memory management connects to RAG and context-window optimisation; guardrails connect to exception handling and HITL. A cross-reference map is provided in the master index.

---

## Contributing

Contributions that improve technical accuracy, add missing patterns, or fix errors are welcome.

1. **Fork** the repository and create a feature branch.
2. **Write or update** content in the `docs/` directory using standard Markdown.
3. **Test locally** — run `npm run dev` and verify your changes render correctly.
4. **Submit a pull request** with a clear description of the change and its rationale.

Please follow these guidelines:
- Use formal definitions where appropriate; avoid unsupported claims.
- Include references to papers, frameworks, or implementations where relevant.
- Keep mathematical notation consistent with existing chapters (LaTeX-style).
- Ensure new content integrates with the existing chapter structure and cross-reference map.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for engineers who ship autonomous systems, not slide decks.</sub>
</p>
