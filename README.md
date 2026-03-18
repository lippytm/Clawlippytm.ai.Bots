# Clawlippytm.ai.Bots

> **AI Full Stack · Generative AI · DevOps · Synthetic Intelligence Engines · Swarms · Agents · Bots**

A Python framework that wires together three core sub-systems — **Diagnostics**,
**Cognitive Reasoning**, and **Feedback Loops** — into a single, orchestrated
`SyntheticIntelligenceEngine`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   SyntheticIntelligenceEngine                        │
│                                                                      │
│  ┌─────────────────┐  ┌───────────────────────┐  ┌───────────────┐  │
│  │   Diagnostics   │  │  Cognitive Reasoning  │  │ Feedback Loops│  │
│  │                 │  │                       │  │               │  │
│  │ HealthMonitor   │  │  ReasoningEngine      │  │ FeedbackMgr   │  │
│  │ MetricsCollector│  │  KnowledgeBase        │  │ Evaluator     │  │
│  │ DiagnosticsLogger│ │  DecisionMaker        │  │ Adapter       │  │
│  └─────────────────┘  └───────────────────────┘  └───────────────┘  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Agents / Swarm                              │ │
│  │   BaseAgent  <->  SwarmCoordinator                              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Modules

| Package | Module | Responsibility |
|---------|--------|----------------|
| `diagnostics` | `HealthMonitor` | Register / run named health checks; produce `SystemHealthReport` |
| `diagnostics` | `MetricsCollector` | Thread-safe counters, gauges, histograms & percentiles |
| `diagnostics` | `DiagnosticsLogger` | JSON-structured logging with per-call context dictionaries |
| `cognitive_reasoning` | `KnowledgeBase` | Thread-safe key-value store with confidence scores |
| `cognitive_reasoning` | `ReasoningEngine` | Forward-chaining inference over registered rules |
| `cognitive_reasoning` | `DecisionMaker` | Scores candidate actions via pluggable scoring functions |
| `feedback_loops` | `FeedbackManager` | Collects `FeedbackRecord`s and dispatches to handlers |
| `feedback_loops` | `Evaluator` | Scores agent outputs against registered criteria |
| `feedback_loops` | `Adapter` | Adjusts agent params based on accumulated feedback |
| `agents` | `BaseAgent` | Abstract agent wired with all sub-systems; lifecycle management |
| `agents` | `SwarmCoordinator` | Manages a pool of agents; broadcast & targeted dispatch |

---

## Quick Start

```python
from engine import SyntheticIntelligenceEngine
from agents.base_agent import AgentTask, BaseAgent

# 1. Define a custom agent
class MyAgent(BaseAgent):
    def execute(self, task: AgentTask):
        return f"processed: {task.payload}"

# 2. Boot the engine
engine = SyntheticIntelligenceEngine(engine_id="prod-1")

# 3. Register an agent
engine.register_agent(MyAgent(agent_id="worker-1"))

# 4. Add a reasoning rule
engine.reasoning_engine.register_rule(
    "high_load",
    lambda ctx: ("alert", "scale_up") if ctx.get("cpu_pct", 0) > 80 else None,
)
engine.knowledge_base.store("cpu_pct", 95)

# 5. Reason about the situation
chain = engine.reason("Is the system overloaded?")
print(chain.final_answer["alert"])  # "scale_up"

# 6. Run a task
result = engine.run_task(AgentTask(description="work", payload={"data": [1, 2, 3]}))
print(result.output)

# 7. Check system health
status = engine.status()
print(status.health.overall_status)  # "healthy"

# 8. Shut down gracefully
engine.shutdown()
```

---

## Closed-Loop Execution Flow

```
Diagnose --> Reason --> Decide --> Execute --> Evaluate --> Adapt
   ^                                                          |
   +----------------------- Feedback ------------------------+
```

Every task execution automatically:
1. Logs start/stop with structured JSON via `DiagnosticsLogger`
2. Records timing in `MetricsCollector`
3. Evaluates the result via `Evaluator`
4. Submits a `FeedbackRecord` to `FeedbackManager`
5. Triggers `Adapter.adapt()` to adjust parameters for the next run

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 84 tests pass.
