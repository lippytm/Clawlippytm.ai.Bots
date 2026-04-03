# Clawlippytm.ai.Bots

AI Toolkits for all of my Repositories

---

## Overview

`clawlippytm` is the core Python AI-toolkit package for Clawlippytm repositories.  It
provides a fully-featured bot framework built around six tightly integrated sub-systems:

| Module | Purpose |
|---|---|
| `BotAttributes` | Canonical set of all bot configuration attributes |
| `DiagnosticsSystem` | Multi-pass diagnostics with iterative **feedback loops** |
| `CognitiveReasoner` | Chain-of-thought reasoning with configurable depth, **self-critique**, and **confidence scoring** |
| `CreativityEngine` | Temperature-driven response enrichment (analogies, metaphors, narrative framing) |
| `AgentOrchestrator` | Multi-agent coordination for AI Full-Stack, DevOps, Synthetic Intelligence pipelines |
| `DevOpsEngine` | AI-powered CI/CD pipeline engine (Build → Test → Lint → Deploy → Monitor) |

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│         ClawBot.respond()           │
│                                     │
│  1. DiagnosticsSystem.analyse()     │  ← pre-pass (feedback loops)
│         │                           │
│  2. CognitiveReasoner.reason()      │  ← chain-of-thought + self-critique + confidence
│         │                           │
│  3. CreativityEngine.enrich()       │  ← analogy / metaphor / narrative
│         │                           │
│  4. DiagnosticsSystem.analyse()     │  ← post-pass (feedback loops)
│         │                           │
│  5. Safety correction (if needed)   │
└─────────────────────────────────────┘
    │
    ▼
Bot Response

Optional extensions:
  ClawBot.dispatch_task()  →  AgentOrchestrator  →  SyntheticAgent (DevOps/FullStack/Synthetic/…)
  ClawBot.run_pipeline()   →  DevOpsEngine       →  PipelineRun (Build/Test/Lint/Deploy/Monitor)
```

---

## Quick Start

```python
from clawlippytm import ClawBot, BotAttributes

# Create a bot with default attributes
bot = ClawBot()
response = bot.respond("What is machine learning?")
print(response)

# Customise attributes
attrs = BotAttributes(
    name="MyBot",
    reasoning_depth=4,         # deeper chain-of-thought (1–5)
    creativity_temperature=0.9, # more creative responses (0.0–1.0)
    feedback_loops=3,           # more diagnostic iterations
    self_critique=True,         # enable self-critique loop
)
custom_bot = ClawBot(attributes=attrs)
print(custom_bot.respond("Explain quantum entanglement."))
print(custom_bot.status())
```

### Multi-Agent Mode

```python
from clawlippytm import ClawBot, BotAttributes

# Enable the multi-agent orchestrator
bot = ClawBot(attributes=BotAttributes(agent_mode=True))

# Dispatch tasks to specialised agents
result = bot.dispatch_task("Optimise database queries", role="fullstack")
print(result["output"])

result = bot.dispatch_task("Run smoke tests on the staging deployment", role="devops")
print(result["output"])

print(bot.status()["orchestrator"])
```

### DevOps Pipeline Mode

```python
from clawlippytm import ClawBot, BotAttributes

# Create a DevOps bot
bot = ClawBot(attributes=BotAttributes(
    role="devops",
    devops_environment="production",
))

# Run the full CI/CD pipeline
run = bot.run_pipeline(branch="main")
print(run.passed)           # True
print(run.to_dict())        # Full pipeline run record

print(bot.devops.health_check())
```

### Agents and DevOps standalone

```python
from clawlippytm import AgentOrchestrator, AgentTask, AgentRole
from clawlippytm import DevOpsEngine, PipelineStage

# Multi-agent orchestration
orch = AgentOrchestrator(max_agents=5)
task = AgentTask(description="Scaffold a new microservice", role=AgentRole.FULLSTACK)
result = orch.dispatch(task)
print(result.output)

# DevOps pipeline
engine = DevOpsEngine(
    environment="staging",
    stages=[PipelineStage.BUILD, PipelineStage.TEST, PipelineStage.DEPLOY],
)
run = engine.run_pipeline(branch="feature/new-api")
print(run.passed, run.total_duration_ms)
print(engine.health_check())
```

---

## BotAttributes — all fields

| Attribute | Default | Description |
|---|---|---|
| `name` | `"ClawBot"` | Bot display name |
| `version` | `"1.0.0"` | Semantic version |
| `description` | `"A Clawlippytm AI bot"` | Short description |
| `tone` | `"friendly"` | Tone: friendly / formal / humorous / neutral |
| `verbosity` | `"balanced"` | concise / balanced / verbose |
| `empathy_level` | `0.75` | 0.0 – 1.0 |
| `multi_turn` | `True` | Support multi-turn conversations |
| `memory_enabled` | `True` | Retain context across turns |
| `tool_use` | `True` | Allow external tool / API calls |
| `streaming` | `False` | Streaming response support |
| `reasoning_depth` | `3` | Chain-of-thought depth (1–5) |
| `creativity_temperature` | `0.7` | Creativity 0.0 (deterministic) – 1.0 (creative) |
| `self_critique` | `True` | Enable self-critique refinement loop |
| `safety_filter` | `True` | Apply safety corrections on flagged content |
| `ethical_guidelines` | *(list)* | Ethical rules the bot follows |
| `diagnostics_enabled` | `True` | Enable the diagnostics system |
| `feedback_loops` | `2` | Number of diagnostic feedback loop iterations |
| `role` | `"general"` | Bot role: general / devops / fullstack / synthetic / coordinator |
| `agent_mode` | `False` | Enable multi-agent orchestration |
| `max_agents` | `5` | Maximum simultaneous agents in the pool |
| `devops_environment` | `"staging"` | Default deployment environment for the DevOps engine |

---

## Sub-systems

### DiagnosticsSystem — feedback loops

Runs configurable multi-pass analysis on both incoming messages and outgoing
responses.  Each pass checks for safety issues, clarity, tone, and coherence.
High-severity findings inject a synthetic context marker that influences
subsequent iterations — this is the **feedback loop** mechanism.

```python
from clawlippytm import DiagnosticsSystem

diag = DiagnosticsSystem(enabled=True, feedback_loops=3)
result = diag.analyse("You are an idiot.")
print(result.has_issues)         # True
print(result.highest_severity)  # "medium"
for issue in result.issues:
    print(issue.category, issue.severity, issue.description)
```

### CognitiveReasoner — chain-of-thought + self-critique + confidence

Decomposes prompts into sub-questions, explores them recursively to the
configured depth, synthesises a response, and (when enabled) runs a
self-critique pass to identify and correct weaknesses.  Each reasoning step
now carries a **confidence score** (1.0 at depth 1, decreasing at deeper
levels) and the output exposes an `average_confidence` property.

```python
from clawlippytm import CognitiveReasoner

reasoner = CognitiveReasoner(depth=4, self_critique=True)
output = reasoner.reason("Why is diversity important in teams?")
print(output.response)
print(output.average_confidence)  # e.g. 0.85
for step in output.trace:
    print(step.depth, step.confidence, step.question)
```

### CreativityEngine — enhanced creativity

Enriches responses using temperature-scaled transforms:

- **Lexical diversification** — replaces common words with richer alternatives
- **Analogy injection** (temperature ≥ 0.4)
- **Metaphor injection** (temperature ≥ 0.65)
- **Narrative framing** (temperature ≥ 0.8)

```python
from clawlippytm import CreativityEngine

engine = CreativityEngine(temperature=0.85, seed=42)
enriched = engine.enrich("Machine learning allows computers to learn from data.")
print(enriched)
print(engine.summary())
```

### AgentOrchestrator — multi-agent coordination

Coordinates a pool of `SyntheticAgent` instances, each specialised for a
role.  Tasks are routed to the matching agent (with GENERAL as fallback) and
results accumulated for auditing.

```python
from clawlippytm import AgentOrchestrator, AgentTask, AgentRole, SyntheticAgent

orch = AgentOrchestrator(max_agents=5)

# Dispatch individual tasks
results = orch.dispatch_all([
    AgentTask("Provision Kubernetes cluster", role=AgentRole.DEVOPS, priority=1),
    AgentTask("Build React frontend",         role=AgentRole.FULLSTACK, priority=3),
    AgentTask("Generate synthetic test data", role=AgentRole.SYNTHETIC, priority=5),
])
for r in results:
    print(r.agent_role, r.output)

print(orch.summary())
```

### DevOpsEngine — CI/CD pipeline

Simulates a full Build → Test → Lint → Deploy → Monitor pipeline.

```python
from clawlippytm import DevOpsEngine, PipelineStage

engine = DevOpsEngine(environment="production")
run = engine.run_pipeline(branch="release/v2.0")
print(run.passed, run.total_duration_ms)
for stage in run.stages:
    print(stage.stage.value, stage.passed, stage.message)

print(engine.health_check())
print(engine.summary())
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

---

## License

See [LICENSE](LICENSE).
