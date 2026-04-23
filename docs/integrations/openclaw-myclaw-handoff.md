# Bot → OpenClaw → MyClaw Handoff

This document defines how `Clawlippytm.ai.Bots` should hand off more complex interactions into `OpenClaw-lippytm.AI-` and `MyClaw.lippytm.AI-`.

## Goal

Bots should remain simple and useful on the surface while more complex tasks move into assistant and swarm layers with preserved context.

---

## Handoff Pattern

1. bot detects need for deeper handling
2. bot packages structured context
3. OpenClaw receives assistant-facing handoff
4. OpenClaw builds task envelope
5. MyClaw routes the task to the correct specialist flow
6. result or next step is returned to the assistant or operator path

---

## Context To Preserve

- original user intent
- bot class and conversation state
- lead or customer reference
- recommended next step
- confidence or escalation reason

---

## Best Practices

- do not drop context during handoff
- keep bot language simple and user-friendly
- keep swarm details mostly behind the scenes
- escalate premium or sensitive cases intentionally

---

## Rule of thumb

The best handoff feels continuous to the user even though multiple systems are cooperating behind the scenes.
