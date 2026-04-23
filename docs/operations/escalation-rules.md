# Escalation Rules

This document defines escalation rules for the `Clawlippytm.ai.Bots` product family.

## Goal

Escalation should help bots stay useful and trustworthy by moving the user to the right next system when confidence, authority, or fit is too low.

---

## Escalation Levels

### Level 1 — Bot Handoff
Use when another bot class is a better fit.

Examples:
- concierge bot routes to intake bot
- support bot routes to research bot
- sales bot routes to offer or strategy path

### Level 2 — Workflow Handoff
Use when a form or structured process is the best next step.

Examples:
- strategy intake form
- funding intake form
- automation discovery sequence

### Level 3 — Assistant or Swarm Handoff
Use when a more dynamic system is needed.

Examples:
- OpenClaw assistant session
- MyClaw task routing
- specialist or supervisor path

### Level 4 — Human or Premium Handoff
Use when the case is high-value, sensitive, or too complex.

Examples:
- premium buildout inquiry
- custom planning
- repeated unresolved issue

---

## Escalation Triggers

Escalate when:

- intent stays unclear after reasonable clarification
- confidence is low
- the request is complex or premium in nature
- the task needs structured intake or scheduling
- the bot starts looping or stops being useful

---

## Best Practices

- explain the next step clearly
- preserve context during handoff
- avoid making escalation feel like failure
- match escalation path to bot role

---

## Rule of thumb

A strong bot does not try to do everything. It guides the user to the right next layer when needed.
