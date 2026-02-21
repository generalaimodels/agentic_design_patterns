

# Chapter 14: Agent-to-Agent (A2A) Communication Protocol

---

## 14.1 Definition and Formal Framework

### 14.1.1 What is Agent-to-Agent (A2A) Communication

Agent-to-Agent (A2A) communication is a **standardized interaction protocol** that enables autonomous AI agents—potentially built on heterogeneous frameworks, running on distributed infrastructure, and maintained by independent organizations—to discover each other's capabilities, negotiate task delegation, exchange structured messages, and collaboratively complete complex workflows **without requiring shared internal state or implementation knowledge**.

The fundamental premise: in a mature AI ecosystem, no single agent possesses all capabilities required for complex real-world tasks. A travel-planning agent must coordinate with a flight-booking agent, a hotel-reservation agent, a weather-forecasting agent, and a currency-conversion agent. Each of these agents may be built by different teams, use different LLM backbones, and expose different interfaces. A2A provides the **lingua franca** that enables this heterogeneous coordination.

**Formal definition:** An A2A system is a tuple:

$$
\mathcal{S}_{A2A} = (\mathcal{A}, \mathcal{M}, \mathcal{T}, \mathcal{P}, \mathcal{R})
$$

where:
- $\mathcal{A} = \{A_1, A_2, \ldots, A_n\}$ is the set of participating agents
- $\mathcal{M}$ is the structured message space (the set of all valid messages)
- $\mathcal{T}$ is the task lifecycle state machine governing task progression
- $\mathcal{P}$ is the communication protocol specification (transport, serialization, authentication)
- $\mathcal{R}$ is the registry/discovery service enabling capability advertisement and lookup

**Key distinctions from traditional API communication:**

| Property | Traditional API | A2A Communication |
|----------|----------------|-------------------|
| **Participant model** | Client-server (asymmetric) | Peer-to-peer (symmetric) |
| **Capability knowledge** | Client knows server's API a priori | Dynamic discovery via Agent Cards |
| **Interaction pattern** | Single request-response | Multi-turn, stateful task conversations |
| **Content type** | Fixed schema per endpoint | Multimodal (text, files, structured data, streams) |
| **Autonomy** | Client drives all decisions | Either agent can drive, pause, or redirect |
| **Error handling** | HTTP status codes | Rich task state machine with negotiation |
| **Composability** | Manual orchestration | Agents delegate to sub-agents autonomously |

**Architectural positioning:**

```
┌─────────────────────────────────────────────────────────┐
│                    User / Orchestrator                    │
└──────────────────────────┬──────────────────────────────┘
                           │ A2A Protocol
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Agent A  │◄──►│ Agent B  │◄──►│ Agent C  │
    │(Planning)│A2A │(Booking) │A2A │(Payment) │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │ MCP           │ MCP           │ MCP
    ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐
    │  Tools   │    │  Tools   │    │  Tools   │
    │(Calendar,│    │(Airline  │    │(Stripe,  │
    │ Maps)    │    │  APIs)   │    │ Banks)   │
    └──────────┘    └──────────┘    └──────────┘
```

A2A operates at the **inter-agent coordination layer**, while protocols like MCP (Model Context Protocol) operate at the **agent-to-tool layer**. This layered separation is fundamental to the architecture.

---

### 14.1.2 A2A as a Protocol: Structured Message Schemas

A2A formalizes agent communication as a **protocol**—a complete specification of message formats, transport mechanisms, state transitions, and behavioral contracts.

**Core protocol abstraction:**

$$
\text{A2A}: A_i \xrightarrow{\text{message}} A_j \text{ with structured message schemas}
$$

This notation captures three essential properties:

1. **Directionality**: Messages flow from sender $A_i$ to receiver $A_j$ (though the protocol supports bidirectional exchanges within a task)
2. **Structure**: Messages conform to predefined schemas, not free-form text
3. **Protocol-level semantics**: The protocol defines what each message type means and how the receiver must respond

**Protocol stack:**

The A2A protocol is layered atop standard web infrastructure:

```
┌─────────────────────────────────────┐
│     A2A Application Layer           │  Task lifecycle, Agent Cards,
│     (Semantic Protocol)             │  capability negotiation
├─────────────────────────────────────┤
│     A2A Message Layer               │  Message Parts, Artifacts,
│     (Structured Messaging)          │  multi-turn conversations
├─────────────────────────────────────┤
│     A2A Transport Layer             │  HTTP/HTTPS, SSE, WebSocket,
│     (Communication Transport)       │  push notifications
├─────────────────────────────────────┤
│     Security Layer                  │  OAuth 2.0, mTLS, API keys,
│     (Authentication & Encryption)   │  JWT tokens
├─────────────────────────────────────┤
│     Network Layer                   │  TCP/IP, DNS, load balancing
│     (Standard Internet)             │
└─────────────────────────────────────┘
```

**Message schema formalization:**

Every A2A message $m \in \mathcal{M}$ is a structured object:

$$
m = (\text{id}, \text{task\_id}, \text{role}, \text{parts}, \text{metadata}, \text{timestamp})
$$

where:
- $\text{id} \in \text{UUID}$: Unique message identifier
- $\text{task\_id} \in \text{UUID}$: The task this message belongs to
- $\text{role} \in \{\text{user}, \text{agent}\}$: Sender role relative to the task
- $\text{parts} \in \mathcal{P}^*$: Ordered sequence of message parts (text, files, data)
- $\text{metadata} \in \text{Map}\langle\text{String}, \text{Any}\rangle$: Extensible key-value metadata
- $\text{timestamp} \in \mathbb{R}$: Unix epoch timestamp

**JSON-RPC foundation:**

A2A uses **JSON-RPC 2.0** as its wire protocol, providing a standardized request-response envelope:

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": {
    "id": "task-uuid-123",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "Book a flight from SFO to JFK on December 15"
        }
      ]
    }
  }
}
```

**Protocol methods (core RPC surface):**

| Method | Description | Direction |
|--------|-------------|-----------|
| `tasks/send` | Send a message within a task (create or continue) | Client → Remote Agent |
| `tasks/sendSubscribe` | Send message and subscribe to streaming updates | Client → Remote Agent |
| `tasks/get` | Poll for current task state | Client → Remote Agent |
| `tasks/cancel` | Request task cancellation | Client → Remote Agent |
| `tasks/pushNotification/set` | Register webhook for push notifications | Client → Remote Agent |
| `tasks/pushNotification/get` | Retrieve current notification configuration | Client → Remote Agent |
| `tasks/resubscribe` | Reconnect to a streaming task | Client → Remote Agent |

---

### 14.1.3 A2A vs. MCP: Complementary Protocols

A2A and MCP (Model Context Protocol) address fundamentally different communication boundaries in the agentic AI stack. Understanding their complementarity is essential for correct system design.

**MCP (Model Context Protocol):**

MCP governs the interface between an **agent and its tools/resources**. It defines how an LLM-based agent invokes external functions, accesses databases, reads files, and interacts with APIs. The key characteristic: MCP operates **within** an agent's trust boundary.

$$
\text{MCP}: A_i \xrightarrow{\text{tool\_call}} \text{Tool}_k
$$

MCP treats the tool as a **transparent, deterministic function**: the agent knows exactly what the tool does (via its schema), controls when to invoke it, and directly receives the result.

**A2A (Agent-to-Agent Protocol):**

A2A governs the interface between **two autonomous agents**. The key characteristic: A2A operates **across** trust and capability boundaries. The calling agent does **not** know the internal implementation of the remote agent—it interacts with an **opaque** entity that may itself use LLMs, tools, sub-agents, and complex reasoning.

$$
\text{A2A}: A_i \xrightarrow{\text{task}} A_j \text{ (opaque)}
$$

**Comparison matrix:**

| Dimension | MCP | A2A |
|-----------|-----|-----|
| **Boundary** | Agent ↔ Tool (intra-agent) | Agent ↔ Agent (inter-agent) |
| **Opacity** | Transparent (tool schema known) | Opaque (internal implementation hidden) |
| **Statefulness** | Stateless function calls | Stateful task lifecycle |
| **Autonomy** | Tool is passive; agent drives | Both parties are autonomous |
| **Multi-turn** | Single call-response | Multi-turn conversations |
| **Delegation** | Agent delegates computation | Agent delegates entire sub-goals |
| **Discovery** | Static tool registration | Dynamic capability discovery |
| **Trust model** | Same trust domain | Cross-domain trust negotiation |
| **Streaming** | Not native | Native SSE/WebSocket support |
| **Error model** | Exception/error codes | Task state machine (retry, renegotiate) |

**Complementary usage pattern:**

```
User: "Plan my trip to Tokyo"
    │
    ▼
┌──────────────────────────────────────────┐
│  Travel Planning Agent (Orchestrator)     │
│                                          │
│  Uses MCP to access:                     │
│  ├── Calendar tool (check availability)  │
│  ├── Weather API tool (forecast)         │
│  └── Currency converter tool             │
│                                          │
│  Uses A2A to delegate to:               │
│  ├── Flight Booking Agent ──► [A2A]     │
│  │   └── Uses MCP: airline APIs          │
│  ├── Hotel Booking Agent ──► [A2A]      │
│  │   └── Uses MCP: hotel APIs            │
│  └── Local Guide Agent ──► [A2A]        │
│      └── Uses MCP: maps, review APIs     │
└──────────────────────────────────────────┘
```

**When to use which:**

- **Use MCP** when: The operation is a deterministic function call, the agent fully understands the tool's behavior, no negotiation is needed, and the tool has no autonomy.
- **Use A2A** when: The operation requires another agent's reasoning capabilities, the implementation is opaque, multi-turn interaction may be needed, or the remote entity is maintained by a different organization.

**Bridging MCP and A2A:**

An agent can expose its MCP tools as A2A capabilities through an **A2A wrapper**. Conversely, an A2A-accessible remote agent can be wrapped as an MCP tool for simpler integration:

```python
# Wrapping an A2A agent as an MCP tool
class A2AToolWrapper:
    """Makes a remote A2A agent appear as a local MCP tool."""
    
    def __init__(self, agent_card_url: str):
        self.remote_agent = A2AClient(agent_card_url)
    
    async def invoke(self, input_text: str) -> str:
        task = await self.remote_agent.send_task(
            message=Message(role="user", 
                          parts=[TextPart(text=input_text)])
        )
        # Wait for completion (blocking for tool semantics)
        result = await self.remote_agent.poll_until_complete(task.id)
        return result.artifacts[0].parts[0].text
```

---

### 14.1.4 Design Principles: Opaque Agents, Capability Discovery, Task Delegation

A2A is built upon four foundational design principles that constrain the protocol's design space and ensure scalability across heterogeneous ecosystems.

**Principle 1: Agent Opacity**

Each agent is treated as a **black box**. The protocol makes zero assumptions about:
- The LLM model powering the agent (GPT-4, Claude, Gemini, open-source)
- The agent framework (LangChain, CrewAI, AutoGen, custom)
- The internal reasoning strategy (chain-of-thought, ReAct, tree-of-thought)
- The tools or sub-agents used internally (via MCP or otherwise)
- The programming language or runtime environment

**Formal opacity property:**

$$
\forall A_i, A_j \in \mathcal{A}: \text{behavior}(A_j) = f(\text{AgentCard}(A_j), \text{messages exchanged})
$$

An agent $A_i$ can only reason about $A_j$ through: (1) $A_j$'s publicly advertised Agent Card, and (2) the messages exchanged during task execution. No introspection into $A_j$'s internals is possible or permitted.

**Why this matters:** Opacity enables agents built by different organizations, using different technology stacks, and evolving independently to interoperate without tight coupling. It mirrors the encapsulation principle from object-oriented design but at the system level.

**Principle 2: Capability Discovery**

Agents advertise their capabilities through **machine-readable Agent Cards** (Section 14.2.1). Other agents or orchestrators discover these capabilities through a **registry** or **well-known endpoint**:

$$
\text{discover}: q \to \{A_j \in \mathcal{A} : \text{capabilities}(A_j) \cap \text{requirements}(q) \neq \emptyset\}
$$

This replaces hardcoded integrations with dynamic, composable agent ecosystems. A new agent can join the ecosystem by publishing its Agent Card—no modification of existing agents is required.

**Principle 3: Task Delegation**

A2A frames all inter-agent interaction as **task delegation**. Agent $A_i$ (the client) delegates a task to agent $A_j$ (the remote agent), specifying the desired outcome but not the method:

$$
\text{delegate}: (A_i, \text{task\_description}) \to A_j
$$

The remote agent $A_j$ has full autonomy over:
- How to accomplish the task (reasoning strategy)
- Whether to involve sub-agents (recursive delegation)
- When to request additional input from $A_i$
- How to report progress and results

**Principle 4: Collaborative Multi-Turn Interaction**

Tasks are not single-shot request-response exchanges. The protocol supports **multi-turn conversations** within a task, where either agent can:
- Request clarification or additional input
- Report partial results
- Renegotiate scope or constraints
- Stream intermediate outputs

This mirrors human collaborative work patterns: a manager delegates a task to a specialist, who may ask clarifying questions, provide progress updates, and deliver results incrementally.

---

## 14.2 A2A Protocol Architecture

### 14.2.1 Agent Cards (Capability Advertisement)

An **Agent Card** is a machine-readable JSON document that serves as the **public identity and capability declaration** of an agent. It is analogous to an API specification (OpenAPI/Swagger) but designed for autonomous agent consumption.

Agent Cards are hosted at a **well-known URL** by convention:

$$
\text{AgentCard URL} = \texttt{https://\{agent-host\}/.well-known/agent.json}
$$

#### Metadata: Name, Description, Skills, Endpoints

**Complete Agent Card schema:**

```json
{
  "name": "FlightBookingAgent",
  "description": "Autonomous agent for searching, comparing, and booking 
                  commercial flights. Supports multi-leg itineraries, 
                  fare comparison, seat selection, and booking confirmation.",
  "url": "https://flights.example.com/a2a",
  "version": "2.1.0",
  "provider": {
    "organization": "TravelAI Corp",
    "url": "https://travelai.example.com",
    "contact": "agents@travelai.example.com"
  },
  "documentationUrl": "https://docs.travelai.example.com/flight-agent",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "authentication": {
    "schemes": ["OAuth2", "Bearer"],
    "credentials": "https://auth.travelai.example.com/.well-known/openid-configuration"
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json", "text/html"],
  "skills": [
    {
      "id": "flight-search",
      "name": "Flight Search",
      "description": "Search available flights between any two airports 
                      with flexible date ranges, class preferences, and 
                      airline filters.",
      "tags": ["travel", "flights", "search", "booking"],
      "examples": [
        "Find flights from SFO to NRT on December 15",
        "Search for business class flights to London next week",
        "Compare prices for round-trip flights NYC to Paris"
      ],
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["application/json", "text/plain"]
    },
    {
      "id": "flight-booking",
      "name": "Flight Booking",
      "description": "Book a specific flight with passenger details, 
                      seat preferences, and payment processing.",
      "tags": ["travel", "booking", "payment"],
      "examples": [
        "Book flight UA123 on Dec 15 for John Doe",
        "Reserve 2 seats on the cheapest SFO-JFK flight tomorrow"
      ],
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "itinerary-management",
      "name": "Itinerary Management",
      "description": "Modify, cancel, or check status of existing bookings.",
      "tags": ["travel", "management", "cancellation"],
      "examples": [
        "Cancel booking #BK-2024-1234",
        "Change my flight to a day later"
      ],
      "inputModes": ["text/plain"],
      "outputModes": ["text/plain", "application/json"]
    }
  ]
}
```

**Skill specification formalization:**

Each skill $s \in \text{Skills}(A_j)$ is a tuple:

$$
s = (\text{id}, \text{name}, \text{description}, \text{tags}, \text{examples}, \text{input\_modes}, \text{output\_modes})
$$

The $\text{description}$ field is designed for **LLM consumption**: it must be sufficiently detailed and unambiguous for another agent's LLM to determine whether this skill matches a given task requirement.

The $\text{examples}$ field provides **few-shot demonstrations** of valid task descriptions, enabling capability matching through semantic similarity.

#### Authentication Requirements

The Agent Card declares the authentication mechanisms the remote agent requires:

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `Bearer` | API key or JWT in Authorization header | Simple inter-agent auth |
| `OAuth2` | OAuth 2.0 flow with token exchange | Enterprise, user-delegated auth |
| `mTLS` | Mutual TLS with client certificates | High-security environments |
| `APIKey` | Static API key in header or query parameter | Development, simple deployments |
| `OpenIDConnect` | OIDC-based identity verification | Federated identity scenarios |

The authentication specification enables the **client agent** to programmatically determine how to authenticate before sending the first message, without manual configuration.

#### Supported Input/Output Modalities

Each skill declares its supported **MIME types** for input and output:

$$
\text{input\_modes}(s) \subseteq \{\texttt{text/plain}, \texttt{application/json}, \texttt{image/png}, \texttt{audio/wav}, \texttt{application/pdf}, \ldots\}
$$

$$
\text{output\_modes}(s) \subseteq \{\texttt{text/plain}, \texttt{application/json}, \texttt{text/html}, \texttt{image/png}, \texttt{video/mp4}, \ldots\}
$$

This enables **multimodal agent communication**: a document analysis agent can accept PDF inputs and return structured JSON outputs, while a chart generation agent accepts JSON data and returns PNG images.

**Content negotiation** follows HTTP conventions: the client specifies preferred output modes, and the remote agent selects the most appropriate one:

$$
\text{output\_mode} = \arg\max_{m \in \text{output\_modes}(s) \cap \text{client\_preferences}} \text{priority}(m)
$$

---

### 14.2.2 Task Lifecycle

The **task** is the fundamental unit of work in A2A. Every inter-agent interaction occurs within the context of a task, which progresses through a well-defined state machine.

#### Task Creation and Submission

A task is created when a client agent sends its first message via `tasks/send` or `tasks/sendSubscribe`:

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": {
    "id": "task-abc-123",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "Find the cheapest round-trip flight from SFO to NRT, 
                   departing Dec 15, returning Dec 29, economy class."
        }
      ]
    },
    "metadata": {
      "priority": "normal",
      "deadline": "2024-12-10T00:00:00Z",
      "budget_constraint": "max $1500",
      "preferred_airlines": ["ANA", "JAL", "United"]
    }
  }
}
```

**Task data structure:**

$$
\text{Task} = (\text{id}, \text{sessionId}, \text{status}, \text{history}, \text{artifacts}, \text{metadata})
$$

```json
{
  "id": "task-abc-123",
  "sessionId": "session-xyz-789",
  "status": {
    "state": "working",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "Searching flights..."}]
    },
    "timestamp": "2024-12-01T10:23:45Z"
  },
  "history": [
    {
      "role": "user",
      "parts": [{"type": "text", "text": "Find cheapest..."}],
      "timestamp": "2024-12-01T10:23:40Z"
    }
  ],
  "artifacts": [],
  "metadata": {}
}
```

#### Task States

The A2A protocol defines a finite set of task states with well-defined semantics:

| State | Semantics | Who Sets It |
|-------|-----------|-------------|
| `submitted` | Task received but not yet being processed | System (automatic) |
| `working` | Agent is actively processing the task | Remote agent |
| `input-required` | Agent needs additional information from client | Remote agent |
| `completed` | Task finished successfully; artifacts available | Remote agent |
| `failed` | Task failed; error details in status message | Remote agent |
| `canceled` | Task was canceled by client or agent | Either party |

**Formal state set:**

$$
\mathcal{S}_{\text{task}} = \{\text{submitted}, \text{working}, \text{input-required}, \text{completed}, \text{failed}, \text{canceled}\}
$$

#### Task State Transition Diagram

**Valid state transitions:**

$$
\text{submitted} \to \text{working}
$$
$$
\text{working} \to \text{completed} \mid \text{failed} \mid \text{input-required} \mid \text{canceled}
$$
$$
\text{input-required} \to \text{working} \mid \text{canceled}
$$

**State transition diagram:**

```
                              ┌──────────────┐
                    ┌────────►│   canceled    │◄──────────┐
                    │         └──────────────┘            │
                    │                                     │
              ┌─────┴──────┐                       ┌──────┴───────┐
   ──────────►│ submitted  │──────────────────────►│   working    │
              └────────────┘                       └──┬───┬───┬───┘
                                                      │   │   │
                              ┌────────────────────────┘   │   │
                              │                            │   │
                              ▼                            │   │
                    ┌──────────────────┐                   │   │
                    │ input-required   │───────────────────┘   │
                    └──────────────────┘    (client responds)  │
                                                               │
                              ┌─────────────────────────────────┘
                              │
                    ┌─────────┴───┐         ┌──────────────┐
                    │ completed   │         │   failed     │
                    │ (artifacts) │         │ (error info) │
                    └─────────────┘         └──────────────┘
```

**Transition semantics:**

1. **submitted → working**: The remote agent acknowledges receipt and begins processing. This transition is typically immediate.

2. **working → completed**: The agent has successfully produced results. Artifacts (output documents, structured data) are attached to the task.

3. **working → failed**: The agent encountered an unrecoverable error. The status message contains error details:
```json
{
  "state": "failed",
  "message": {
    "role": "agent",
    "parts": [{
      "type": "text",
      "text": "Failed to search flights: Airline API rate limit exceeded. 
               Retry after 60 seconds."
    }]
  }
}
```

4. **working → input-required**: The agent needs clarification or additional information from the client:
```json
{
  "state": "input-required",
  "message": {
    "role": "agent",
    "parts": [{
      "type": "text",
      "text": "I found 3 options. Do you prefer a direct flight for $1200 
               or a one-stop flight for $850 with a 3-hour layover in LAX?"
    }]
  }
}
```

5. **input-required → working**: The client provides the requested input, and the agent resumes processing.

6. **Any non-terminal → canceled**: Either party can cancel a task. The canceling party should provide a reason:
```json
{
  "jsonrpc": "2.0",
  "id": "req-005",
  "method": "tasks/cancel",
  "params": {
    "id": "task-abc-123",
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "Plans changed; no longer need this booking."}]
    }
  }
}
```

**Task lifecycle invariants:**

$$
\text{Terminal states} = \{\text{completed}, \text{failed}, \text{canceled}\}
$$

$$
\forall t \in \text{Terminal states}: \nexists s \in \mathcal{S}_{\text{task}} \text{ such that } t \to s
$$

Once a task reaches a terminal state, no further transitions are possible. The task and its artifacts are immutable.

---

### 14.2.3 Messaging Format

#### Message Parts: Text, Files, Structured Data

Each message in A2A is composed of an ordered sequence of **parts**, where each part carries a specific content type:

$$
\text{Message} = (\text{role}, [\text{Part}_1, \text{Part}_2, \ldots, \text{Part}_n])
$$

**Part types:**

**1. TextPart:**
```json
{
  "type": "text",
  "text": "Find flights from SFO to NRT departing December 15"
}
```
Used for natural language instructions, questions, and responses.

**2. FilePart:**
```json
{
  "type": "file",
  "file": {
    "name": "passenger_details.pdf",
    "mimeType": "application/pdf",
    "bytes": "base64-encoded-content..."
  }
}
```
Or with URI reference (for large files):
```json
{
  "type": "file",
  "file": {
    "name": "large_dataset.csv",
    "mimeType": "text/csv",
    "uri": "https://storage.example.com/files/dataset.csv"
  }
}
```

**3. DataPart (Structured Data):**
```json
{
  "type": "data",
  "data": {
    "departure": {"airport": "SFO", "date": "2024-12-15"},
    "arrival": {"airport": "NRT"},
    "passengers": 2,
    "class": "economy",
    "maxPrice": 1500,
    "currency": "USD"
  }
}
```

Structured data parts enable **machine-readable communication** between agents, avoiding the ambiguity of natural language when precise parameters must be conveyed.

**Multimodal message example:**

```json
{
  "role": "user",
  "parts": [
    {
      "type": "text",
      "text": "Analyze this medical image and compare with the patient 
               history in the attached PDF."
    },
    {
      "type": "file",
      "file": {
        "name": "xray_scan.png",
        "mimeType": "image/png",
        "bytes": "iVBORw0KGgo..."
      }
    },
    {
      "type": "file",
      "file": {
        "name": "patient_history.pdf",
        "mimeType": "application/pdf",
        "uri": "https://ehr.hospital.com/patients/12345/history.pdf"
      }
    },
    {
      "type": "data",
      "data": {
        "patient_id": "P-12345",
        "scan_type": "chest_xray",
        "priority": "urgent"
      }
    }
  ]
}
```

#### Artifact Exchange

**Artifacts** are the structured outputs produced by a task—the deliverables. They are distinct from conversational messages: messages are the dialogue about the work, while artifacts are the work product itself.

$$
\text{Artifact} = (\text{name}, \text{description}, [\text{Part}_1, \ldots, \text{Part}_n], \text{index}, \text{append}, \text{lastChunk})
$$

```json
{
  "artifacts": [
    {
      "name": "flight_options",
      "description": "Top 3 flight options matching search criteria",
      "index": 0,
      "append": false,
      "lastChunk": true,
      "parts": [
        {
          "type": "data",
          "data": {
            "flights": [
              {
                "airline": "ANA",
                "flight": "NH7",
                "departure": "2024-12-15T11:00:00-08:00",
                "arrival": "2024-12-16T16:30:00+09:00",
                "price": 1150,
                "stops": 0,
                "duration_hours": 11.5
              },
              {
                "airline": "United",
                "flight": "UA837",
                "departure": "2024-12-15T13:30:00-08:00",
                "arrival": "2024-12-16T18:00:00+09:00",
                "price": 980,
                "stops": 0,
                "duration_hours": 11.5
              }
            ]
          }
        }
      ]
    }
  ]
}
```

**Streaming artifacts:** For large outputs, artifacts can be delivered in chunks:
- `index`: Identifies which artifact this chunk belongs to
- `append`: If `true`, append this chunk to the existing artifact at this index
- `lastChunk`: If `true`, this is the final chunk for this artifact

#### Multi-Turn Task Conversations

A2A tasks naturally support **multi-turn conversations**, where the client and remote agent exchange multiple messages:

```
Turn 1 [Client → Agent]:
  "Find flights SFO to NRT, Dec 15, economy"
  
Turn 2 [Agent → Client] (state: input-required):
  "Found 5 options. Do you want direct flights only, or should I 
   include connections? Budget limit?"

Turn 3 [Client → Agent]:
  "Direct flights only, under $1200"

Turn 4 [Agent → Client] (state: working):
  "Narrowed to 2 options. Checking seat availability..."

Turn 5 [Agent → Client] (state: input-required):
  "ANA NH7 at $1150 has window seats available. United UA837 at $980 
   has only middle seats. Which do you prefer?"

Turn 6 [Client → Agent]:
  "Book the ANA flight, window seat, for John Doe"

Turn 7 [Agent → Client] (state: completed):
  [Artifact: booking confirmation PDF + structured booking data]
```

Each turn is stored in the task's `history` array, providing a complete audit trail of the interaction.

**Session management:** Multiple tasks can share a `sessionId`, enabling conversational context to persist across tasks:

$$
\text{Session} = \{t_1, t_2, \ldots, t_n\} \text{ where } \forall t_i: t_i.\text{sessionId} = \text{sid}
$$

This allows follow-up tasks to reference context from previous ones: "Change the flight I booked earlier to a day later."

---

## 14.3 Communication Patterns

### 14.3.1 Request-Response (Synchronous)

The simplest communication pattern: the client sends a message and blocks until the remote agent returns a complete response.

$$
A_i \xrightarrow{\text{request}} A_j \xrightarrow{\text{response}} A_i
$$

**Implementation via `tasks/send`:**

```
Client                         Remote Agent
  │                                  │
  │──── POST /a2a (tasks/send) ─────►│
  │                                  │
  │        [Agent processes task]    │
  │                                  │
  │◄─── HTTP 200 (Task object) ─────│
  │     state: "completed"           │
  │     artifacts: [...]             │
  │                                  │
```

**Characteristics:**
- **Blocking**: The client waits for the full response before proceeding
- **Simple**: No subscription management, no streaming infrastructure
- **Timeout-sensitive**: Long-running tasks risk HTTP timeout (typically 30–300 seconds)
- **Appropriate for**: Quick tasks (< 30 seconds), simple lookups, deterministic operations

**HTTP semantics:**

```http
POST /a2a HTTP/1.1
Host: flights.example.com
Content-Type: application/json
Authorization: Bearer <token>

{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": {
    "id": "task-quick-001",
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "What is the current exchange rate USD to JPY?"}]
    }
  }
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "id": "task-quick-001",
    "status": {
      "state": "completed",
      "message": {
        "role": "agent",
        "parts": [{"type": "text", "text": "1 USD = 149.23 JPY as of 2024-12-01 10:30 UTC"}]
      }
    },
    "artifacts": [{
      "name": "exchange_rate",
      "parts": [{"type": "data", "data": {"from": "USD", "to": "JPY", "rate": 149.23}}]
    }]
  }
}
```

---

### 14.3.2 Publish-Subscribe (Event-Driven)

In the publish-subscribe pattern, agents **subscribe to event topics** and receive notifications when relevant events occur. This decouples producers from consumers.

$$
A_i \xrightarrow{\text{subscribe}(\text{topic})} \text{EventBus} \xleftarrow{\text{publish}(\text{event})} A_j
$$

**Architecture:**

```
                    ┌──────────────────┐
                    │   Event Bus /    │
        publish ──► │   Message Broker │ ◄── publish
                    │  (Kafka, Redis   │
                    │   Pub/Sub, NATS) │
                    └──┬───┬───┬───┬──┘
                       │   │   │   │
            subscribe  │   │   │   │  subscribe
                       ▼   ▼   ▼   ▼
                     A₁   A₂   A₃   A₄
```

**Use cases in A2A:**
- **Monitoring agents** subscribe to task completion events from worker agents
- **Orchestrator agents** subscribe to status updates from multiple sub-agents
- **Logging agents** subscribe to all task events for audit trails
- **Alerting agents** subscribe to failure events for incident response

**Event message structure:**

```json
{
  "topic": "task.status.changed",
  "event": {
    "task_id": "task-abc-123",
    "agent_id": "flight-booking-agent",
    "previous_state": "working",
    "new_state": "completed",
    "timestamp": "2024-12-01T10:30:00Z",
    "artifacts_available": true
  }
}
```

**Subscription filtering:**

$$
\text{filter}(\text{event}) = \bigwedge_{i} \text{predicate}_i(\text{event}.\text{field}_i)
$$

Example: Subscribe to all task failures from agents in the "travel" domain:
```json
{
  "topic": "task.status.changed",
  "filter": {
    "new_state": "failed",
    "agent_tags": {"$contains": "travel"}
  }
}
```

---

### 14.3.3 Streaming (Long-Running Tasks via SSE)

For tasks that take significant time (seconds to minutes), **streaming** provides real-time progress updates and incremental results using **Server-Sent Events (SSE)**.

$$
A_i \xrightarrow{\text{tasks/sendSubscribe}} A_j \xrightarrow[\text{SSE stream}]{\text{event}_1, \text{event}_2, \ldots, \text{event}_n} A_i
$$

**Implementation via `tasks/sendSubscribe`:**

```
Client                         Remote Agent
  │                                  │
  │── POST /a2a (sendSubscribe) ────►│
  │                                  │
  │◄─── SSE: status(working) ───────│
  │                                  │
  │◄─── SSE: status(working,        │
  │      "Searching 3 airlines...") ─│
  │                                  │
  │◄─── SSE: artifact(chunk 1) ─────│
  │                                  │
  │◄─── SSE: artifact(chunk 2) ─────│
  │                                  │
  │◄─── SSE: status(completed) ─────│
  │                                  │
```

**SSE event format:**

```
event: task/status
data: {"id":"task-123","status":{"state":"working","message":{"role":"agent","parts":[{"type":"text","text":"Searching ANA flights..."}]}}}

event: task/status
data: {"id":"task-123","status":{"state":"working","message":{"role":"agent","parts":[{"type":"text","text":"Found 3 ANA options. Searching United..."}]}}}

event: task/artifact
data: {"id":"task-123","artifact":{"name":"partial_results","index":0,"append":false,"lastChunk":false,"parts":[{"type":"data","data":{"flights":[{"airline":"ANA","price":1150}]}}]}}

event: task/artifact
data: {"id":"task-123","artifact":{"name":"partial_results","index":0,"append":true,"lastChunk":true,"parts":[{"type":"data","data":{"flights":[{"airline":"United","price":980}]}}]}}

event: task/status
data: {"id":"task-123","status":{"state":"completed"}}
```

**Reconnection via `tasks/resubscribe`:**

If the SSE connection drops (network interruption, client restart), the client can reconnect to an in-progress task:

```json
{
  "jsonrpc": "2.0",
  "id": "req-010",
  "method": "tasks/resubscribe",
  "params": {
    "id": "task-abc-123"
  }
}
```

The remote agent resumes streaming from the current state, avoiding duplicate processing.

---

### 14.3.4 Push Notifications

For tasks where maintaining a persistent connection is impractical (mobile clients, serverless architectures, cross-network boundaries), A2A supports **push notifications** via webhooks.

$$
A_j \xrightarrow{\text{HTTP POST}} \text{webhook\_url}(A_i) \text{ when task state changes}
$$

**Registration:**

```json
{
  "jsonrpc": "2.0",
  "id": "req-020",
  "method": "tasks/pushNotification/set",
  "params": {
    "id": "task-abc-123",
    "pushNotificationConfig": {
      "url": "https://client-agent.example.com/webhooks/a2a",
      "token": "webhook-secret-token-xyz",
      "authentication": {
        "schemes": ["Bearer"],
        "credentials": "shared-secret-abc"
      }
    }
  }
}
```

**Notification delivery:**

When the task state changes, the remote agent sends an HTTP POST to the registered webhook:

```http
POST /webhooks/a2a HTTP/1.1
Host: client-agent.example.com
Content-Type: application/json
Authorization: Bearer shared-secret-abc

{
  "task_id": "task-abc-123",
  "state": "completed",
  "timestamp": "2024-12-01T10:35:00Z",
  "message": "Flight search completed. 3 options found.",
  "artifacts_available": true
}
```

The client agent then retrieves the full task details via `tasks/get`.

**Delivery guarantees:**
- **Retry with exponential backoff**: If the webhook endpoint returns an error (5xx, timeout), retry with increasing delays: $\Delta t_n = \min(2^n \cdot t_{\text{base}}, t_{\text{max}})$
- **Idempotency**: Notifications include unique event IDs; the receiver deduplicates based on these IDs
- **Dead letter queue**: After $n$ failed delivery attempts, the notification is stored in a dead letter queue for manual inspection

---

### 14.3.5 Broadcast Communication

Broadcast enables an agent to send a message to **all agents** in a group or matching a capability filter, without enumerating specific recipients.

$$
A_i \xrightarrow{\text{broadcast}(m, \text{filter})} \{A_j : \text{filter}(A_j) = \text{true}\}
$$

**Use cases:**
- **Capability auction**: "I need an agent that can translate Japanese legal documents. Who can do this?"
- **Status broadcast**: "The airline API is down. All travel agents should switch to cached data."
- **Coordination**: "All agents working on project X: the deadline has moved to Friday."

**Implementation approaches:**

1. **Registry-mediated broadcast**: The agent registry fans out the message to all matching agents:
```python
async def broadcast(self, message: Message, 
                    filter: AgentFilter) -> list[TaskResponse]:
    matching_agents = self.registry.find_agents(filter)
    tasks = []
    for agent in matching_agents:
        task = await agent.send_task(message)
        tasks.append(task)
    return tasks
```

2. **Topic-based broadcast**: Using the pub/sub infrastructure, publish to a topic that matching agents subscribe to.

3. **Gossip protocol**: Each agent forwards the message to $k$ known peers, ensuring eventual delivery to all agents with $O(\log n)$ hop overhead.

---

### 14.3.6 Conversational Turn-Taking

A2A's multi-turn task model naturally supports **conversational turn-taking**, where the client and remote agent alternate messages within a task context.

**Turn-taking protocol:**

$$
\text{Turn}_t: \begin{cases} A_i \to A_j & \text{if } t \text{ is odd (client turn)} \\ A_j \to A_i & \text{if } t \text{ is even (agent turn)} \end{cases}
$$

**State-based turn control:**

The task state machine enforces turn-taking discipline:
- When state is `input-required`: It is the **client's turn** to send a message
- When state is `working`: It is the **agent's turn** to process and respond
- When state is `completed`/`failed`: No more turns are expected

**Implicit turn signals:**

| Signal | Meaning |
|--------|---------|
| Agent sets state to `input-required` | "Your turn—I need information from you" |
| Client sends message to `input-required` task | "Here's what you asked for—your turn now" |
| Agent sets state to `working` | "Processing—wait for my response" |
| Agent sets state to `completed` | "I'm done; conversation is over" |

**Timeout handling:**

If the expected party does not respond within a configured timeout $T_{\text{turn}}$:

$$
\text{If } t_{\text{now}} - t_{\text{last\_message}} > T_{\text{turn}}, \text{ then escalate or cancel}
$$

The remote agent may:
- Send a reminder message
- Proceed with default assumptions (documented in the Agent Card)
- Transition to `failed` with a timeout reason

---

## 14.4 Message Passing Semantics

### 14.4.1 Structured vs. Unstructured Messages

A2A messages can carry content on a spectrum from fully unstructured (natural language text) to fully structured (typed JSON schemas):

**Unstructured (Natural Language):**

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Find me a good Italian restaurant near downtown Seattle for dinner tonight, party of 4"}
  ]
}
```

*Advantages*: Flexible, human-readable, leverages LLM's natural language understanding.

*Disadvantages*: Ambiguous ("good" is subjective; "near" is undefined; "tonight" depends on timezone), requires LLM parsing on the receiver side, error-prone.

**Structured (Typed Schema):**

```json
{
  "role": "user",
  "parts": [
    {
      "type": "data",
      "data": {
        "action": "restaurant_search",
        "cuisine": "Italian",
        "location": {"lat": 47.6062, "lng": -122.3321, "radius_km": 2},
        "date": "2024-12-01",
        "time": "19:00",
        "timezone": "America/Los_Angeles",
        "party_size": 4,
        "min_rating": 4.0,
        "price_range": "$$-$$$"
      }
    }
  ]
}
```

*Advantages*: Unambiguous, machine-parseable, validatable against schema, deterministic behavior.

*Disadvantages*: Rigid, requires both agents to agree on schema, less flexible for novel requests.

**Hybrid approach (recommended):**

Combine both: include a natural language description for LLM reasoning and structured data for precise parameters:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Find a good Italian restaurant for tonight's team dinner"},
    {
      "type": "data",
      "data": {
        "cuisine": "Italian",
        "location": {"lat": 47.6062, "lng": -122.3321},
        "date": "2024-12-01",
        "time": "19:00",
        "party_size": 4
      }
    }
  ]
}
```

The remote agent uses the structured data for precise parameter extraction and the text for understanding intent and preferences not captured in the schema.

---

### 14.4.2 Message Serialization (JSON, Protobuf)

Serialization governs how A2A messages are encoded for transmission over the wire.

**JSON (JavaScript Object Notation) — Default:**

A2A uses JSON as its primary serialization format, consistent with the JSON-RPC 2.0 foundation.

$$
\text{JSON}: \text{Object} \to \text{UTF-8 string}
$$

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": {"id": "task-123", "message": {"role": "user", "parts": [...]}}
}
```

| Property | Value |
|----------|-------|
| Human readability | High |
| Schema validation | JSON Schema |
| Binary data | Base64 encoding (33% overhead) |
| Compression | gzip/brotli over HTTP |
| Parsing speed | Moderate ($O(n)$, but string-heavy) |
| Ecosystem support | Universal |

**Protocol Buffers (Protobuf) — High-Performance Variant:**

For high-throughput A2A deployments (thousands of messages per second), Protobuf provides significant performance advantages:

$$
\text{Protobuf}: \text{Object} \to \text{binary bytes}
$$

```protobuf
syntax = "proto3";

message A2ARequest {
  string jsonrpc = 1;
  string id = 2;
  string method = 3;
  TaskSendParams params = 4;
}

message TaskSendParams {
  string task_id = 1;
  Message message = 2;
  map<string, string> metadata = 3;
}

message Message {
  string role = 1;
  repeated Part parts = 2;
  int64 timestamp = 3;
}

message Part {
  oneof content {
    TextPart text_part = 1;
    FilePart file_part = 2;
    DataPart data_part = 3;
  }
}
```

| Property | JSON | Protobuf |
|----------|------|----------|
| Message size | $n$ bytes | $0.3n$–$0.5n$ bytes |
| Serialization speed | $1\times$ | $3$–$10\times$ faster |
| Deserialization speed | $1\times$ | $3$–$10\times$ faster |
| Schema enforcement | Runtime (JSON Schema) | Compile-time (proto definition) |
| Human readability | High | Low (binary) |
| Binary data support | Base64 (overhead) | Native `bytes` field |

**Content negotiation:**

The client specifies preferred serialization via HTTP `Content-Type` and `Accept` headers:

```http
POST /a2a HTTP/1.1
Content-Type: application/json           # Request in JSON
Accept: application/protobuf, application/json  # Prefer Protobuf response
```

---

### 14.4.3 Message Routing and Addressing

Message routing determines how a message from agent $A_i$ reaches the correct recipient agent $A_j$.

**Direct addressing:**

The simplest model: the client knows the remote agent's URL from its Agent Card:

$$
\text{route}(m) = \text{AgentCard}(A_j).\text{url}
$$

```http
POST https://flights.example.com/a2a HTTP/1.1
```

**Registry-based routing:**

The client queries the registry to resolve the best agent for a task:

$$
\text{route}(m) = \text{Registry}.\text{resolve}(\text{required\_skill}, \text{constraints})
$$

```python
# Client queries registry
agent_url = registry.resolve(
    skill="flight-booking",
    constraints={
        "region": "asia-pacific",
        "min_uptime_sla": 0.999,
        "max_latency_p99_ms": 500
    }
)
# Route message to resolved agent
response = a2a_client.send(agent_url, message)
```

**Load-balanced routing:**

For agents deployed as scalable services, a load balancer distributes incoming A2A requests across instances:

$$
\text{route}(m) = \text{LoadBalancer}.\text{select}(\{A_j^{(1)}, A_j^{(2)}, \ldots, A_j^{(k)}\})
$$

Selection strategies:
- **Round-robin**: $\text{instance} = A_j^{(i \mod k)}$
- **Least-connections**: Route to the instance with fewest active tasks
- **Consistent hashing**: Route based on $\text{hash}(\text{session\_id}) \mod k$ for session affinity
- **Weighted routing**: Route proportionally to instance capacity

**Hierarchical routing (multi-hop):**

In complex deployments, a message may traverse multiple intermediary agents:

$$
A_i \to A_{\text{gateway}} \to A_{\text{regional}} \to A_j
$$

Each intermediary performs routing logic (access control, protocol translation, load balancing) before forwarding.

---

### 14.4.4 Message Ordering and Delivery Guarantees

Distributed message passing introduces fundamental challenges around ordering and delivery reliability.

#### At-Most-Once, At-Least-Once, Exactly-Once Semantics

**At-Most-Once Delivery:**

$$
P(\text{message delivered}) \leq 1 \quad \text{(may be lost, never duplicated)}
$$

- Send and forget: no acknowledgment, no retry
- Fastest but least reliable
- Appropriate for: status updates, metrics, non-critical notifications
- Implementation: Fire-and-forget HTTP request without retry logic

**At-Least-Once Delivery:**

$$
P(\text{message delivered}) = 1 \quad \text{(never lost, may be duplicated)}
$$

- Client retries until acknowledgment received
- Requires **idempotency** on the receiver side to handle duplicates
- Implementation: Client retries with exponential backoff; receiver deduplicates by message ID

```python
async def send_with_retry(client, message, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = await client.send(message)
            if response.status == 200:
                return response
        except (TimeoutError, ConnectionError):
            pass
        # Exponential backoff with jitter
        delay = min(2 ** attempt + random.uniform(0, 1), 60)
        await asyncio.sleep(delay)
    raise DeliveryError(f"Failed after {max_retries} attempts")
```

**Idempotency on receiver:**

$$
f(m) = f(m) \circ f(m) \quad \text{(applying the same message twice has the same effect as once)}
$$

Implemented by tracking processed message IDs:

```python
class IdempotentReceiver:
    def __init__(self):
        self.processed_ids = set()  # Use Redis/DB in production
    
    async def handle(self, message):
        if message.id in self.processed_ids:
            return self.cached_response(message.id)  # Return cached result
        
        result = await self.process(message)
        self.processed_ids.add(message.id)
        self.cache_response(message.id, result)
        return result
```

**Exactly-Once Delivery:**

$$
P(\text{message delivered exactly once}) = 1 \quad \text{(never lost, never duplicated)}
$$

True exactly-once delivery is **impossible** in asynchronous distributed systems (proven by the Two Generals' Problem and FLP impossibility result). In practice, it is approximated by combining at-least-once delivery with idempotent processing:

$$
\text{Exactly-Once} \approx \text{At-Least-Once Delivery} + \text{Idempotent Processing}
$$

For A2A, the task ID serves as the idempotency key: re-sending a message with the same task ID and message content results in no additional processing.

**Message ordering guarantees:**

Within a single task, A2A guarantees **causal ordering**: messages from the same sender within the same task are delivered in the order they were sent.

$$
\text{If } A_i \text{ sends } m_1 \text{ before } m_2 \text{ in task } t, \text{ then } A_j \text{ receives } m_1 \text{ before } m_2
$$

Across tasks, no ordering guarantees are made—tasks are independent and may be processed concurrently.

---

### 14.4.5 Message Queuing and Buffering

For asynchronous communication and workload management, A2A deployments use message queues as intermediaries.

**Queue architecture:**

```
Client Agent ──► [Message Queue] ──► Remote Agent
                 (RabbitMQ, SQS,
                  Kafka, Redis)
```

**Benefits of queue-based buffering:**

1. **Temporal decoupling**: Client and remote agent need not be online simultaneously
2. **Load leveling**: Queue absorbs traffic spikes, delivering messages at the agent's processing rate
3. **Reliability**: Persistent queues survive agent restarts
4. **Priority scheduling**: High-priority tasks are dequeued before low-priority ones

**Queue configuration per task priority:**

$$
\text{dequeue\_order}(m) = \text{priority}(m) \cdot w_p + \text{age}(m) \cdot w_a
$$

where $w_p$ and $w_a$ weight priority versus fairness (preventing starvation of low-priority tasks).

**Dead letter queues (DLQ):**

Messages that fail processing after $n$ attempts are moved to a dead letter queue for inspection:

$$
m \to \text{DLQ} \quad \text{if } \text{attempts}(m) > n_{\text{max}}
$$

```python
# Queue configuration example (AWS SQS-style)
queue_config = {
    "main_queue": "a2a-tasks",
    "dead_letter_queue": "a2a-tasks-dlq",
    "max_receive_count": 3,           # Move to DLQ after 3 failures
    "visibility_timeout_seconds": 300, # Lock message for 5 min during processing
    "message_retention_days": 14       # Keep messages for 14 days
}
```

**Backpressure mechanisms:**

When the remote agent is overwhelmed, it signals the sender to slow down:

1. **Queue depth monitoring**: If queue depth exceeds threshold, new tasks are rejected with HTTP 429 (Too Many Requests)
2. **Rate limiting**: Enforce maximum tasks/second per client agent
3. **Circuit breaker**: If failure rate exceeds threshold, temporarily reject all new tasks:

$$
\text{circuit\_state} = \begin{cases} \text{closed (normal)} & \text{if failure\_rate} < \theta_{\text{open}} \\ \text{open (rejecting)} & \text{if failure\_rate} \geq \theta_{\text{open}} \\ \text{half-open (testing)} & \text{after cooldown period } T_{\text{cool}} \end{cases}
$$

---

## 14.5 Discovery and Negotiation

### 14.5.1 Agent Registry and Discovery Services

An **Agent Registry** is a centralized or federated service that maintains a catalog of available agents, their capabilities, health status, and access endpoints.

**Registry data model:**

$$
\text{Registry} = \{(\text{AgentCard}_i, \text{HealthStatus}_i, \text{Metadata}_i)\}_{i=1}^{N}
$$

**Core registry operations:**

| Operation | Description | API |
|-----------|-------------|-----|
| `register(agent_card)` | Add or update an agent's registration | `POST /registry/agents` |
| `deregister(agent_id)` | Remove an agent from the registry | `DELETE /registry/agents/{id}` |
| `discover(query, filters)` | Find agents matching capability requirements | `GET /registry/agents?q=...` |
| `health_check(agent_id)` | Check if an agent is currently available | `GET /registry/agents/{id}/health` |
| `list(filters, pagination)` | Enumerate registered agents | `GET /registry/agents` |

**Discovery mechanisms:**

**1. Well-Known URL Convention:**

Any web server can expose an Agent Card at a standardized path:

$$
\texttt{https://\{domain\}/.well-known/agent.json}
$$

A crawler or orchestrator can discover agents by probing known domains:

```python
async def discover_agent(domain: str) -> AgentCard | None:
    url = f"https://{domain}/.well-known/agent.json"
    try:
        response = await httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            return AgentCard.parse(response.json())
    except Exception:
        return None
```

**2. Centralized Registry (Hub Model):**

```
           ┌─────────────────┐
           │  Agent Registry  │
           │   (Centralized)  │
           └──┬──┬──┬──┬──┬──┘
              │  │  │  │  │
    register  │  │  │  │  │  discover
              ▼  ▼  ▼  ▼  ▼
            A₁ A₂ A₃ A₄ A₅
```

- Single point of truth for all available agents
- Supports rich querying (by skill, region, SLA, price)
- Risk: single point of failure (mitigated by replication)

**3. Federated Registry (Mesh Model):**

```
    ┌──────────┐     sync     ┌──────────┐
    │Registry A│◄────────────►│Registry B│
    └────┬─────┘              └────┬─────┘
         │                         │
    A₁ A₂ A₃                  A₄ A₅ A₆
```

- Multiple registries synchronize their catalogs
- No single point of failure
- Supports cross-organizational discovery
- Eventual consistency: agent updates propagate asynchronously

**4. DNS-Based Discovery:**

Agents register DNS SRV records for their A2A endpoints:

```
_a2a._https.flights.example.com. SRV 10 60 443 a2a.flights.example.com
```

**Agent health monitoring:**

The registry periodically probes registered agents and updates their status:

```python
class AgentHealthMonitor:
    async def check_health(self, agent: RegisteredAgent) -> HealthStatus:
        try:
            start = time.monotonic()
            response = await httpx.get(
                f"{agent.url}/.well-known/agent.json",
                timeout=5.0
            )
            latency = time.monotonic() - start
            
            if response.status_code == 200:
                return HealthStatus(
                    state="healthy",
                    latency_ms=latency * 1000,
                    last_checked=datetime.utcnow()
                )
            else:
                return HealthStatus(state="degraded", ...)
        except Exception as e:
            return HealthStatus(state="unhealthy", error=str(e))
    
    async def monitor_loop(self, interval_seconds=30):
        while True:
            for agent in self.registry.all_agents():
                health = await self.check_health(agent)
                self.registry.update_health(agent.id, health)
            await asyncio.sleep(interval_seconds)
```

---

### 14.5.2 Capability Matching

Capability matching determines which agent is best suited to handle a given task query. This is the core intelligence of the discovery system.

**Formal definition:**

$$
\text{match}(q, A_j) = \text{sim}\bigl(\text{skill\_description}(A_j),\, q\bigr) \geq \tau
$$

where $\text{sim}$ is a semantic similarity function and $\tau$ is the matching threshold.

**Multi-factor matching:**

In practice, capability matching considers multiple dimensions beyond semantic similarity:

$$
\text{score}(q, A_j) = \sum_{i} w_i \cdot f_i(q, A_j)
$$

| Factor $f_i$ | Description | Weight Range |
|--------------|-------------|-------------|
| Semantic skill match | $\cos(\text{Embed}(q), \text{Embed}(\text{skills}(A_j)))$ | High (0.4–0.6) |
| Modality compatibility | $\text{input\_modes}(q) \subseteq \text{input\_modes}(A_j)$ | Binary gate |
| Historical performance | Average task success rate for similar queries | Medium (0.1–0.2) |
| Latency SLA | $\text{p99\_latency}(A_j) \leq \text{required\_latency}(q)$ | Binary gate |
| Cost | $\text{cost\_per\_task}(A_j) \leq \text{budget}(q)$ | Medium (0.1–0.2) |
| Availability | $\text{health}(A_j) = \text{healthy}$ | Binary gate |
| Trust score | Reputation based on past interactions | Low–Medium (0.05–0.15) |

**Implementation with LLM-based matching:**

For complex queries where simple semantic similarity is insufficient, an LLM can perform capability matching:

```python
matching_prompt = """Given the following task and list of available agents 
with their capabilities, select the best agent for the task. Consider 
skill relevance, modality support, and any constraints mentioned.

Task: {query}

Available Agents:
{agent_cards_summary}

Selected Agent (with reasoning):"""

async def match_agent(query: str, 
                       agents: list[AgentCard]) -> AgentCard:
    agents_summary = "\n".join(
        f"- {a.name}: {a.description}. Skills: {[s.name for s in a.skills]}"
        for a in agents
    )
    response = await llm.generate(
        matching_prompt.format(query=query, 
                               agent_cards_summary=agents_summary)
    )
    selected_agent_name = parse_selection(response)
    return next(a for a in agents if a.name == selected_agent_name)
```

**Embedding-based skill matching pipeline:**

```python
class CapabilityMatcher:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.skill_embeddings = {}  # agent_id -> skill_embedding_matrix
    
    def index_agent(self, agent_card: AgentCard):
        """Pre-compute embeddings for all skills of an agent."""
        texts = []
        for skill in agent_card.skills:
            # Combine skill description with examples
            skill_text = f"{skill.name}: {skill.description}. " + \
                        f"Examples: {'; '.join(skill.examples)}"
            texts.append(skill_text)
        embeddings = self.model.encode(texts)
        self.skill_embeddings[agent_card.name] = embeddings
    
    def match(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Find best matching agents for a query."""
        q_emb = self.model.encode([query])[0]
        scores = []
        for agent_name, skill_embs in self.skill_embeddings.items():
            # Max similarity across all skills of this agent
            max_sim = max(
                cosine_similarity(q_emb, s_emb) 
                for s_emb in skill_embs
            )
            scores.append((agent_name, max_sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

---

### 14.5.3 Protocol Negotiation

Before substantive task communication begins, agents may negotiate **communication parameters** to ensure compatibility.

**Negotiation dimensions:**

| Dimension | Options | Resolution Strategy |
|-----------|---------|-------------------|
| Serialization format | JSON, Protobuf, MessagePack | Client preference + server support |
| Communication pattern | Sync, streaming, push | Agent Card `capabilities` field |
| Authentication scheme | OAuth2, Bearer, mTLS | Agent Card `authentication` field |
| Content modality | Text, JSON, image, audio | Skill-level `inputModes`/`outputModes` |
| Language | English, Japanese, multilingual | Metadata negotiation |
| Compression | None, gzip, brotli | HTTP `Accept-Encoding` |
| API version | v1, v2, v2.1 | Agent Card `version` field |

**Version negotiation:**

The client specifies its supported A2A protocol version, and the remote agent responds with the highest mutually supported version:

$$
\text{negotiated\_version} = \max(\text{client\_versions} \cap \text{server\_versions})
$$

```http
POST /a2a HTTP/1.1
A2A-Version: 2.0, 1.5, 1.0     # Client supports these versions
Accept: application/json        # Preferred serialization

# Server response includes negotiated version
A2A-Version: 2.0                # Server selected highest mutual version
```

**Capability-based negotiation:**

If the remote agent's capabilities don't fully match the client's requirements, the agents negotiate a feasible subset:

```json
// Client request
{
  "desired_capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "multimodal_input": ["text", "image", "audio"]
  }
}

// Server response
{
  "negotiated_capabilities": {
    "streaming": true,
    "pushNotifications": false,  // Not supported
    "multimodal_input": ["text", "image"]  // Audio not supported
  },
  "alternatives": {
    "pushNotifications": "Use polling via tasks/get instead",
    "audio_input": "Convert audio to text before sending"
  }
}
```

---

### 14.5.4 Service Level Agreements (SLAs) Between Agents

SLAs define the **contractual performance expectations** between communicating agents. They are essential in production multi-agent systems where reliability, latency, and cost must be guaranteed.

**SLA specification:**

$$
\text{SLA}(A_i, A_j) = (\text{availability}, \text{latency}, \text{throughput}, \text{cost}, \text{quality})
$$

```json
{
  "sla": {
    "availability": {
      "uptime_percentage": 99.9,
      "measurement_window": "monthly",
      "planned_maintenance_notice_hours": 24
    },
    "latency": {
      "p50_ms": 200,
      "p95_ms": 500,
      "p99_ms": 2000,
      "timeout_ms": 30000
    },
    "throughput": {
      "max_concurrent_tasks": 100,
      "max_tasks_per_minute": 500,
      "burst_limit": 50
    },
    "cost": {
      "per_task_usd": 0.05,
      "per_token_input_usd": 0.00001,
      "per_token_output_usd": 0.00003,
      "monthly_cap_usd": 10000
    },
    "quality": {
      "min_task_success_rate": 0.95,
      "max_hallucination_rate": 0.05,
      "response_format_compliance": 0.99
    }
  }
}
```

**SLA monitoring:**

$$
\text{compliance}(A_j, \text{SLA}, \Delta t) = \frac{|\{t \in \Delta t : \text{met\_sla}(t)\}|}{|\{t \in \Delta t\}|}
$$

```python
class SLAMonitor:
    def record_task(self, task_id: str, metrics: TaskMetrics):
        self.metrics_store.append({
            "task_id": task_id,
            "latency_ms": metrics.latency_ms,
            "success": metrics.status == "completed",
            "cost_usd": metrics.cost_usd,
            "timestamp": datetime.utcnow()
        })
    
    def check_compliance(self, window_hours: int = 24) -> SLAReport:
        recent = self.metrics_store.query(
            since=datetime.utcnow() - timedelta(hours=window_hours)
        )
        return SLAReport(
            availability=sum(1 for m in recent if m["success"]) / len(recent),
            p50_latency=np.percentile([m["latency_ms"] for m in recent], 50),
            p99_latency=np.percentile([m["latency_ms"] for m in recent], 99),
            total_cost=sum(m["cost_usd"] for m in recent),
            task_count=len(recent)
        )
```

**SLA violation handling:**

| Violation | Response |
|-----------|----------|
| Latency exceeds SLA for >5% of requests | Alert; consider alternative agent |
| Availability drops below 99.9% | Failover to backup agent |
| Cost exceeds monthly cap | Throttle or pause delegation |
| Quality drops below threshold | Route to higher-quality (possibly more expensive) agent |

**Automatic failover:**

$$
\text{route}(q) = \begin{cases} A_j^{\text{primary}} & \text{if SLA}(A_j^{\text{primary}}) \text{ is met} \\ A_j^{\text{secondary}} & \text{if SLA}(A_j^{\text{primary}}) \text{ is violated} \\ A_j^{\text{fallback}} & \text{if both primary and secondary violate SLA} \end{cases}
$$

---

## 14.6 Security in A2A Communication

### 14.6.1 Mutual Authentication

In A2A, **both parties** must authenticate each other—unlike traditional client-server where only the server authenticates. This is because both agents are autonomous entities that may execute consequential actions.

$$
\text{Auth}(A_i, A_j): A_i \text{ verifies identity of } A_j \land A_j \text{ verifies identity of } A_i
$$

**Authentication mechanisms:**

**1. Mutual TLS (mTLS):**

Both client and server present X.509 certificates during the TLS handshake:

```
Client Agent                           Remote Agent
     │                                       │
     │─── ClientHello ─────────────────────►│
     │◄── ServerHello + ServerCert ─────────│
     │─── ClientCert + CertVerify ─────────►│
     │◄── Finished ─────────────────────────│
     │                                       │
     │  [Encrypted A2A communication]        │
```

Both certificates are issued by a trusted Certificate Authority (CA), and both parties verify the certificate chain.

**2. OAuth 2.0 with Client Credentials Grant:**

For agent-to-agent authentication (no human user involved), the OAuth 2.0 Client Credentials flow is appropriate:

```
Client Agent                   Auth Server               Remote Agent
     │                              │                          │
     │── client_id + secret ──────►│                          │
     │◄── access_token ────────────│                          │
     │                              │                          │
     │── A2A request + token ──────────────────────────────►│
     │                              │◄── validate token ──────│
     │                              │─── token valid ────────►│
     │◄── A2A response ────────────────────────────────────│
```

```python
async def authenticate_to_agent(target_agent_card: AgentCard) -> str:
    """Obtain access token for communicating with target agent."""
    auth_config = target_agent_card.authentication
    
    if "OAuth2" in auth_config.schemes:
        token_response = await httpx.post(
            auth_config.credentials,  # Token endpoint
            data={
                "grant_type": "client_credentials",
                "client_id": MY_CLIENT_ID,
                "client_secret": MY_CLIENT_SECRET,
                "scope": "a2a:tasks:send a2a:tasks:read"
            }
        )
        return token_response.json()["access_token"]
```

**3. JWT-Based Authentication:**

Each agent issues signed JWTs containing its identity and claims:

$$
\text{JWT} = \text{Base64}(\text{header}) \cdot \text{Base64}(\text{payload}) \cdot \text{Signature}
$$

```json
{
  "header": {"alg": "RS256", "typ": "JWT"},
  "payload": {
    "iss": "planning-agent.example.com",
    "sub": "agent:planning-agent-v2",
    "aud": "flights.example.com",
    "exp": 1701500000,
    "iat": 1701496400,
    "scopes": ["tasks:send", "tasks:read"],
    "agent_card_url": "https://planning-agent.example.com/.well-known/agent.json"
  }
}
```

The receiving agent verifies the JWT signature using the issuer's public key (obtained from a JWKS endpoint).

---

### 14.6.2 Message Encryption and Integrity

**Transport-Level Encryption:**

All A2A communication MUST use TLS 1.3 (minimum TLS 1.2), providing:
- **Confidentiality**: Messages are encrypted in transit
- **Integrity**: Message tampering is detected via MAC
- **Forward secrecy**: Ephemeral key exchange ensures past messages remain secure even if long-term keys are compromised

$$
\text{TLS 1.3}: \text{ECDHE key exchange} + \text{AES-256-GCM encryption} + \text{SHA-384 MAC}
$$

**Message-Level Encryption (End-to-End):**

For scenarios where intermediaries (load balancers, API gateways, message queues) should not have access to message content:

$$
\text{Encrypt}: m \xrightarrow{K_{\text{pub}}(A_j)} \text{encrypted}(m)
$$

$$
\text{Decrypt}: \text{encrypted}(m) \xrightarrow{K_{\text{priv}}(A_j)} m
$$

Implementation using hybrid encryption (asymmetric for key exchange, symmetric for data):

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_message(message: bytes, recipient_public_key) -> EncryptedMessage:
    # Generate ephemeral key pair
    ephemeral_private = ec.generate_private_key(ec.SECP384R1())
    
    # ECDH shared secret
    shared_key = ephemeral_private.exchange(ec.ECDH(), recipient_public_key)
    
    # Derive AES key from shared secret
    aes_key = HKDF(shared_key, length=32, info=b"a2a-message-encryption")
    
    # Encrypt message
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, message, None)
    
    return EncryptedMessage(
        ephemeral_public_key=ephemeral_private.public_key(),
        nonce=nonce,
        ciphertext=ciphertext
    )
```

**Message Integrity (Signing):**

The sender signs each message with its private key, enabling the receiver to verify authenticity and integrity:

$$
\text{sig} = \text{Sign}(K_{\text{priv}}(A_i),\, \text{Hash}(m))
$$

$$
\text{Verify}: \text{Verify}(K_{\text{pub}}(A_i),\, m,\, \text{sig}) \in \{\text{valid}, \text{invalid}\}
$$

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": { ... },
  "signature": {
    "algorithm": "Ed25519",
    "value": "base64-encoded-signature...",
    "key_id": "agent-key-2024-01",
    "signed_fields": ["params", "id", "method"]
  }
}
```

---

### 14.6.3 Authorization and Permission Delegation

**Authorization** determines what actions an authenticated agent is allowed to perform.

**Scope-based authorization:**

Each agent's access token contains scopes defining permitted operations:

$$
\text{scopes}(A_i \to A_j) \subseteq \{\texttt{tasks:send}, \texttt{tasks:read}, \texttt{tasks:cancel}, \texttt{artifacts:read}, \texttt{admin}\}
$$

The remote agent enforces scope checks before processing each request:

```python
def authorize_request(token: JWT, method: str) -> bool:
    required_scope = {
        "tasks/send": "tasks:send",
        "tasks/get": "tasks:read",
        "tasks/cancel": "tasks:cancel",
        "tasks/sendSubscribe": "tasks:send",
    }
    return required_scope[method] in token.scopes
```

**Permission Delegation (Delegation Chains):**

When agent $A_i$ delegates a task to $A_j$, and $A_j$ needs to further delegate to $A_k$, the original user's permissions must be propagated securely.

$$
\text{User} \xrightarrow{\text{delegate}} A_i \xrightarrow{\text{delegate}} A_j \xrightarrow{\text{delegate}} A_k
$$

**On-Behalf-Of (OBO) token exchange:**

```
User token (scope: full) 
  → A_i exchanges for delegated token (scope: travel-booking)
    → A_j exchanges for further-delegated token (scope: flight-search-only)
```

Each delegation step **narrows** the scope, following the principle of least privilege:

$$
\text{scope}(A_k) \subseteq \text{scope}(A_j) \subseteq \text{scope}(A_i) \subseteq \text{scope}(\text{User})
$$

```python
async def delegate_with_reduced_scope(
    parent_token: str,
    target_agent: AgentCard,
    required_scopes: list[str]
) -> str:
    """Exchange parent token for a delegated token with reduced scope."""
    response = await httpx.post(
        AUTH_SERVER_TOKEN_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": parent_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": target_agent.url,
            "scope": " ".join(required_scopes)
        }
    )
    return response.json()["access_token"]
```

**Resource-level authorization:**

Beyond method-level scopes, agents may enforce resource-level access control:

$$
\text{authorize}(A_i, \text{method}, \text{resource}) = \text{policy}(A_i.\text{role}, \text{method}, \text{resource}.\text{owner})
$$

Example: Agent $A_i$ can read tasks it created but not tasks created by other agents:
```python
def authorize_task_access(agent_id: str, task: Task) -> bool:
    if task.creator_id == agent_id:
        return True  # Creator can always access
    if agent_id in task.authorized_readers:
        return True  # Explicitly authorized
    return False
```

---

### 14.6.4 Trust Hierarchies Among Agents

In a multi-agent ecosystem, not all agents are equally trustworthy. A **trust hierarchy** formalizes the degree to which one agent trusts another.

**Trust model formalization:**

$$
\text{trust}: \mathcal{A} \times \mathcal{A} \to [0, 1]
$$

where $\text{trust}(A_i, A_j) = 1$ represents complete trust and $\text{trust}(A_i, A_j) = 0$ represents zero trust.

**Trust hierarchy levels:**

```
Level 4: Fully Trusted (Internal, same organization)
    │     - Can execute any delegated action
    │     - Full data access
    │
Level 3: Trusted Partner (Verified external organization)
    │     - Can execute pre-approved action types
    │     - Filtered data access
    │
Level 2: Semi-Trusted (Verified but limited history)
    │     - Can execute read-only actions
    │     - Anonymized/redacted data
    │
Level 1: Untrusted (Unknown agent)
    │     - Sandboxed execution only
    │     - No sensitive data exposure
    │
Level 0: Blocked (Known-bad actor)
          - All communication rejected
```

**Trust computation:**

Trust scores can be computed dynamically based on multiple signals:

$$
\text{trust}(A_i, A_j) = \alpha \cdot \text{reputation}(A_j) + \beta \cdot \text{cert\_level}(A_j) + \gamma \cdot \text{history}(A_i, A_j)
$$

where:
- $\text{reputation}(A_j)$: Global reputation score from the registry (based on aggregate performance)
- $\text{cert\_level}(A_j)$: Certification/verification level of the agent's organization
- $\text{history}(A_i, A_j)$: Direct interaction history between these two agents

**Trust-based behavior modification:**

| Trust Level | Behavior |
|-------------|----------|
| High ($> 0.8$) | Delegate complex tasks; share detailed data; minimal monitoring |
| Medium ($0.4$–$0.8$) | Delegate simple tasks; redact sensitive fields; periodic verification |
| Low ($0.1$–$0.4$) | Delegate only read-only tasks; full output validation; human approval for actions |
| Minimal ($< 0.1$) | Refuse delegation; or delegate only with complete sandboxing and output filtering |

**Zero-Trust Architecture for A2A:**

Following zero-trust principles, every A2A interaction is authenticated and authorized regardless of network location or prior trust:

$$
\forall (A_i, A_j, m): \text{verify}(\text{identity}(A_i)) \land \text{authorize}(\text{action}(m)) \land \text{encrypt}(m)
$$

Key principles:
1. **Never trust, always verify**: Even agents within the same organization must authenticate
2. **Least privilege**: Each agent receives only the permissions it needs for the specific task
3. **Assume breach**: Design systems to limit blast radius if any single agent is compromised
4. **Continuous verification**: Re-validate authentication and authorization periodically, not just at connection establishment

---

## 14.7 Interoperability and Standards

### 14.7.1 Cross-Framework Agent Communication

The AI agent ecosystem is fragmented across numerous frameworks, each with its own agent abstractions, tool interfaces, and communication patterns:

| Framework | Agent Model | Communication Model |
|-----------|------------|-------------------|
| **LangChain/LangGraph** | `AgentExecutor`, graph-based workflows | Function calling, tool invocation |
| **CrewAI** | Role-based agents in crews | Inter-agent delegation |
| **AutoGen** (Microsoft) | Conversational agents | Multi-agent chat |
| **Semantic Kernel** (Microsoft) | Plugin-based agents | Kernel functions |
| **Haystack** | Pipeline-based agents | Component chaining |
| **Google ADK** | Agent Development Kit | A2A-native |
| **Custom** | Proprietary implementations | Proprietary protocols |

**A2A as a universal bridge:**

A2A enables cross-framework communication by providing a **framework-agnostic protocol layer**:

```
┌────────────────┐    A2A    ┌────────────────┐
│  LangChain     │◄────────►│  CrewAI Agent  │
│  Agent         │           │                │
└───────┬────────┘           └───────┬────────┘
        │ (internal)                  │ (internal)
   LangChain                     CrewAI
   runtime                      runtime
```

**A2A adapter pattern:**

Each framework implements an **A2A adapter** that translates between the framework's native agent interface and the A2A protocol:

```python
# LangChain → A2A Adapter
class LangChainA2AAdapter:
    """Exposes a LangChain agent as an A2A-compatible remote agent."""
    
    def __init__(self, langchain_agent, agent_card: AgentCard):
        self.agent = langchain_agent
        self.card = agent_card
        self.tasks = {}
    
    async def handle_task_send(self, request: TaskSendRequest) -> Task:
        task_id = request.params.id
        message_text = extract_text(request.params.message)
        
        # Translate A2A message to LangChain invocation
        self.tasks[task_id] = Task(
            id=task_id, 
            status=TaskStatus(state="working")
        )
        
        try:
            # Run LangChain agent
            result = await self.agent.ainvoke({"input": message_text})
            
            # Translate LangChain output to A2A artifact
            self.tasks[task_id].status = TaskStatus(state="completed")
            self.tasks[task_id].artifacts = [
                Artifact(
                    name="result",
                    parts=[TextPart(text=result["output"])]
                )
            ]
        except Exception as e:
            self.tasks[task_id].status = TaskStatus(
                state="failed",
                message=Message(
                    role="agent",
                    parts=[TextPart(text=str(e))]
                )
            )
        
        return self.tasks[task_id]
    
    async def serve(self, host: str, port: int):
        """Start A2A server for this LangChain agent."""
        app = FastAPI()
        
        @app.get("/.well-known/agent.json")
        async def agent_card():
            return self.card.dict()
        
        @app.post("/a2a")
        async def handle_a2a(request: dict):
            method = request["method"]
            if method == "tasks/send":
                return await self.handle_task_send(
                    TaskSendRequest(**request)
                )
            elif method == "tasks/get":
                return self.tasks.get(request["params"]["id"])
            # ... other methods
        
        uvicorn.run(app, host=host, port=port)
```

**Cross-framework communication flow:**

```
AutoGen Agent (Python)                      CrewAI Agent (Python)
      │                                           ▲
      │ AutoGen runtime                            │ CrewAI runtime
      ▼                                           │
┌──────────────────┐                    ┌──────────────────┐
│ AutoGen A2A      │                    │ CrewAI A2A       │
│ Adapter          │                    │ Adapter          │
└────────┬─────────┘                    └────────▲─────────┘
         │                                       │
         │        A2A Protocol (JSON-RPC)         │
         └───────────────────────────────────────┘
                 (HTTP/HTTPS transport)
```

---

### 14.7.2 Protocol Bridges and Adapters

When agents use incompatible protocols (not just frameworks), **protocol bridges** provide translation between protocol families.

**Bridge architecture:**

$$
\text{Bridge}: \text{Protocol}_A \leftrightarrow \text{Protocol}_B
$$

```
Agent (Protocol A)          Bridge              Agent (Protocol B)
      │                       │                        │
      │── Protocol A msg ───►│                        │
      │                       │── translate ──►        │
      │                       │── Protocol B msg ────►│
      │                       │                        │
      │                       │◄── Protocol B resp ──│
      │                       │── translate ──►        │
      │◄── Protocol A resp ──│                        │
```

**Common bridge types:**

**1. A2A ↔ MCP Bridge:**

Enables an A2A agent to invoke an MCP tool server, or an MCP client to interact with an A2A remote agent:

```python
class A2AtoMCPBridge:
    """Translates A2A task requests into MCP tool invocations."""
    
    def __init__(self, mcp_client):
        self.mcp = mcp_client
    
    async def handle_a2a_task(self, task: TaskSendRequest) -> Task:
        # Parse A2A message to determine MCP tool and arguments
        message_text = extract_text(task.params.message)
        
        # Use LLM to map natural language to MCP tool call
        tool_call = await self.plan_tool_call(message_text)
        
        # Execute MCP tool
        mcp_result = await self.mcp.call_tool(
            name=tool_call.name,
            arguments=tool_call.arguments
        )
        
        # Wrap MCP result as A2A artifact
        return Task(
            id=task.params.id,
            status=TaskStatus(state="completed"),
            artifacts=[Artifact(
                name="tool_result",
                parts=[DataPart(data=mcp_result)]
            )]
        )
```

**2. A2A ↔ REST API Bridge:**

Wraps a traditional REST API as an A2A agent:

```python
class RESTtoA2ABridge:
    """Makes a REST API accessible as an A2A agent."""
    
    def __init__(self, api_base_url: str, openapi_spec: dict):
        self.api_url = api_base_url
        self.spec = openapi_spec
    
    async def handle_task(self, message: str) -> Task:
        # Map natural language request to REST endpoint
        endpoint, method, params = await self.map_to_endpoint(message)
        
        # Call REST API
        response = await httpx.request(
            method=method,
            url=f"{self.api_url}{endpoint}",
            json=params
        )
        
        return Task(
            status=TaskStatus(state="completed"),
            artifacts=[Artifact(
                parts=[DataPart(data=response.json())]
            )]
        )
```

**3. A2A ↔ gRPC Bridge:**

For high-performance microservices using gRPC:

$$
\text{A2A (JSON-RPC/HTTP)} \xleftrightarrow{\text{Bridge}} \text{gRPC (Protobuf/HTTP2)}
$$

The bridge translates:
- JSON messages ↔ Protobuf messages
- HTTP/1.1 request-response ↔ HTTP/2 streaming
- A2A task lifecycle ↔ gRPC unary/streaming calls

**Bridge deployment patterns:**

```
Pattern 1: Sidecar Bridge
┌──────────────────────────────┐
│ Pod/Container                │
│ ┌──────────┐  ┌───────────┐ │
│ │ Legacy   │  │ A2A       │ │
│ │ Agent    │◄►│ Bridge    │◄──── A2A traffic
│ │ (gRPC)   │  │ (Sidecar) │ │
│ └──────────┘  └───────────┘ │
└──────────────────────────────┘

Pattern 2: Gateway Bridge
┌───────────────┐     ┌──────────┐     ┌──────────┐
│ A2A Agent     │────►│ Protocol │────►│ Legacy   │
│               │ A2A │ Gateway  │ REST│ Service  │
└───────────────┘     └──────────┘     └──────────┘
```

---

### 14.7.3 Versioning and Backward Compatibility

As the A2A protocol evolves, maintaining backward compatibility across versions is critical for ecosystem stability.

**Semantic versioning:**

A2A follows semantic versioning $\text{MAJOR}.\text{MINOR}.\text{PATCH}$:

$$
\text{Version} = M.m.p
$$

| Component | When Incremented | Compatibility |
|-----------|-----------------|---------------|
| $M$ (Major) | Breaking changes to wire protocol | **Not** backward compatible |
| $m$ (Minor) | New features, new optional fields | Backward compatible |
| $p$ (Patch) | Bug fixes, clarifications | Fully compatible |

**Backward compatibility rules:**

**1. Additive changes are safe (minor version):**
- Adding new optional fields to existing message schemas
- Adding new RPC methods
- Adding new task states (if existing agents can ignore them)
- Adding new capability flags to Agent Cards

**2. Breaking changes require major version bump:**
- Removing or renaming existing fields
- Changing field types
- Changing the semantics of existing methods
- Modifying the task state machine transitions

**Version negotiation mechanism:**

The Agent Card declares supported protocol versions:

```json
{
  "name": "FlightBookingAgent",
  "protocolVersions": ["2.0", "1.5", "1.0"],
  "preferredVersion": "2.0"
}
```

The client selects the highest mutually supported version:

```python
def negotiate_version(
    client_versions: list[str],
    server_versions: list[str]
) -> str:
    # Parse and sort versions in descending order
    client_set = {parse_version(v) for v in client_versions}
    server_set = {parse_version(v) for v in server_versions}
    
    # Find highest compatible version
    # Major versions must match; select highest minor
    compatible = client_set & server_set
    if not compatible:
        # Fallback: find compatible major versions
        client_majors = {v.major for v in client_set}
        server_majors = {v.major for v in server_set}
        common_majors = client_majors & server_majors
        if not common_majors:
            raise IncompatibleVersionError(
                f"No compatible versions: client={client_versions}, "
                f"server={server_versions}"
            )
        best_major = max(common_majors)
        compatible = {v for v in client_set | server_set 
                     if v.major == best_major}
    
    return str(max(compatible))
```

**Graceful degradation:**

When agents with different versions communicate, the newer agent should **degrade gracefully** by:
1. Omitting features not supported by the older version
2. Using only message fields defined in the older version's schema
3. Translating new task states to the closest equivalent in the older version

```python
def adapt_task_for_version(task: Task, target_version: str) -> Task:
    if parse_version(target_version) < parse_version("2.0"):
        # v1.x doesn't support "input-required" state
        if task.status.state == "input-required":
            task.status.state = "working"  # Map to closest v1.x state
            task.status.message.parts.insert(0, TextPart(
                text="[ACTION REQUIRED] "
            ))
        
        # v1.x doesn't support DataPart
        for artifact in task.artifacts:
            artifact.parts = [
                TextPart(text=json.dumps(p.data)) 
                if isinstance(p, DataPart) else p
                for p in artifact.parts
            ]
    
    return task
```

**Deprecation policy:**

$$
\text{Deprecation timeline}: \text{Announce} \xrightarrow{6 \text{ months}} \text{Warn} \xrightarrow{6 \text{ months}} \text{Remove}
$$

1. **Announce**: Document the deprecation in release notes; add deprecation warnings to Agent Card
2. **Warn**: Return deprecation headers in responses:
```http
A2A-Deprecated: method=tasks/legacySend; sunset=2025-06-01
A2A-Successor: method=tasks/send
```
3. **Remove**: In the next major version, remove the deprecated feature

**Migration tooling:**

```python
class A2AVersionMigrator:
    """Helps agents migrate between protocol versions."""
    
    def migrate_agent_card(self, card: dict, 
                           from_version: str, 
                           to_version: str) -> dict:
        if from_version == "1.0" and to_version == "2.0":
            # v1.0 → v2.0 migration
            card["capabilities"] = {
                "streaming": False,  # Default: not supported
                "pushNotifications": False,
                "stateTransitionHistory": False
            }
            # v2.0 requires skills array (v1.0 had flat description)
            if "skills" not in card:
                card["skills"] = [{
                    "id": "default",
                    "name": card.get("name", "Unknown"),
                    "description": card.get("description", ""),
                    "tags": [],
                    "examples": []
                }]
            # v2.0 requires explicit input/output modes
            for skill in card["skills"]:
                skill.setdefault("inputModes", ["text/plain"])
                skill.setdefault("outputModes", ["text/plain"])
        
        card["protocolVersions"] = [to_version]
        return card
```

---

**End-to-End A2A System Architecture Summary:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DISCOVERY LAYER                                  │
│                                                                          │
│   Agent Registry ◄──── Agent Cards (.well-known/agent.json)             │
│        │                    │                                            │
│   Capability Matching  ◄──── Skill descriptions + embeddings            │
│        │                                                                 │
│   Protocol Negotiation  ──── Version + capability negotiation           │
└────────┬─────────────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYER                                     │
│                                                                          │
│   Mutual Authentication (mTLS / OAuth2 / JWT)                           │
│   Message Encryption (TLS 1.3 + optional E2E)                           │
│   Authorization (Scopes + Permission Delegation)                        │
│   Trust Hierarchies (Dynamic trust scoring)                             │
└────────┬─────────────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────────────────────┐
│                      COMMUNICATION LAYER                                  │
│                                                                          │
│   Patterns: Request-Response │ Streaming (SSE) │ Push │ Pub/Sub         │
│   Transport: HTTP/HTTPS │ WebSocket │ Message Queues                    │
│   Serialization: JSON (default) │ Protobuf (high-perf)                  │
│   Delivery: At-least-once + Idempotent processing                       │
└────────┬─────────────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────────────────────┐
│                        TASK LAYER                                         │
│                                                                          │
│   Task Lifecycle: submitted → working → input-required → completed      │
│   Multi-Turn Conversations: Conversational turn-taking                  │
│   Artifacts: Structured deliverables with multimodal parts              │
│   Session Management: Cross-task context via sessionId                  │
└────────┬─────────────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────────────────────┐
│                    INTEROPERABILITY LAYER                                  │
│                                                                          │
│   Framework Adapters: LangChain │ CrewAI │ AutoGen │ Custom             │
│   Protocol Bridges: A2A ↔ MCP │ A2A ↔ REST │ A2A ↔ gRPC               │
│   Versioning: Semantic versioning + backward compatibility              │
│   Graceful Degradation: Feature detection + adaptive behavior           │
└──────────────────────────────────────────────────────────────────────────┘
```