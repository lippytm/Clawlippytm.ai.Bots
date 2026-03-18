# Clawlippytm.ai.Bots

AI Toolkits for all of my Repositories

---

## Overview

`clawlippytm` is the core Python AI-toolkit package for Clawlippytm repositories.  It
provides a fully-featured bot framework built around four tightly integrated sub-systems:

| Module | Purpose |
|---|---|
| `BotAttributes` | Canonical set of all bot configuration attributes |
| `DiagnosticsSystem` | Multi-pass diagnostics with iterative **feedback loops** |
| `CognitiveReasoner` | Chain-of-thought reasoning with configurable depth and **self-critique** |
| `CreativityEngine` | Temperature-driven response enrichment (analogies, metaphors, narrative framing) |

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
│  2. CognitiveReasoner.reason()      │  ← chain-of-thought + self-critique
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

### CognitiveReasoner — chain-of-thought + self-critique

Decomposes prompts into sub-questions, explores them recursively to the
configured depth, synthesises a response, and (when enabled) runs a
self-critique pass to identify and correct weaknesses.

```python
from clawlippytm import CognitiveReasoner

reasoner = CognitiveReasoner(depth=4, self_critique=True)
output = reasoner.reason("Why is diversity important in teams?")
print(output.response)
print(output.self_critique)   # critique text, or None
for step in output.trace:
    print(step.depth, step.question)
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
