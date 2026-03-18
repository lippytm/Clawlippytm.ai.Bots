"""
clawlippytm.cognitive_reasoning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cognitive reasoning engine for Clawlippytm.Bots.

The :class:`CognitiveReasoner` implements a chain-of-thought inspired
pipeline that:

1. Decomposes the user prompt into sub-questions (breadth).
2. Explores each sub-question recursively up to a configurable *depth*.
3. Synthesises the findings into a coherent candidate response.
4. Optionally runs a self-critique pass to refine the response.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in the chain-of-thought trace."""

    depth: int
    question: str
    answer: str
    sub_steps: List["ReasoningStep"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "question": self.question,
            "answer": self.answer,
            "sub_steps": [s.to_dict() for s in self.sub_steps],
        }


@dataclass
class ReasoningOutput:
    """Complete output from one :meth:`CognitiveReasoner.reason` call."""

    prompt: str
    response: str
    trace: List[ReasoningStep]
    self_critique: Optional[str] = None
    refined: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "refined": self.refined,
            "self_critique": self.self_critique,
            "trace": [s.to_dict() for s in self.trace],
        }


# ---------------------------------------------------------------------------
# Cognitive reasoning engine
# ---------------------------------------------------------------------------

class CognitiveReasoner:
    """
    Chain-of-thought cognitive reasoning engine.

    The reasoner decomposes problems into sub-questions, builds a
    reasoning trace, and synthesises a response.  When *self_critique*
    is enabled an additional reflective pass evaluates the candidate
    response and proposes improvements.

    Parameters
    ----------
    depth:
        Maximum recursion depth for sub-question exploration (1–5).
    self_critique:
        Whether to run the self-critique feedback loop.
    """

    def __init__(self, depth: int = 3, self_critique: bool = True) -> None:
        self.depth: int = max(1, min(depth, 5))
        self.self_critique_enabled: bool = self_critique
        self._call_count: int = 0
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def reason(
        self,
        prompt: str,
        diagnostic_context: Optional[Any] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ReasoningOutput:
        """
        Reason about *prompt* using chain-of-thought decomposition.

        Parameters
        ----------
        prompt:
            The user's input text.
        diagnostic_context:
            Optional :class:`~clawlippytm.diagnostics.DiagnosticResult`
            injected from the diagnostics pre-pass.
        history:
            Conversation history for contextual awareness.

        Returns
        -------
        ReasoningOutput
            Full reasoning output including response and trace.
        """
        self._call_count += 1
        history = history or []

        # --- Step 1: decompose the prompt into guiding sub-questions ---
        sub_questions = self._decompose(prompt)

        # --- Step 2: explore each sub-question ---
        trace: List[ReasoningStep] = []
        for sq in sub_questions:
            step = self._explore(sq, current_depth=1)
            trace.append(step)
            self._total_steps += 1 + len(step.sub_steps)

        # --- Step 3: synthesise a response from the trace ---
        response = self._synthesise(prompt, trace, history)

        # --- Step 4: optional self-critique pass ---
        critique: Optional[str] = None
        refined = False
        if self.self_critique_enabled:
            critique = self._critique(response, trace)
            if critique:
                response = self._refine(response, critique)
                refined = True

        return ReasoningOutput(
            prompt=prompt,
            response=response,
            trace=trace,
            self_critique=critique,
            refined=refined,
        )

    def reset(self) -> None:
        """Reset internal counters."""
        self._call_count = 0
        self._total_steps = 0

    def summary(self) -> Dict[str, Any]:
        """Return a summary of reasoning activity."""
        return {
            "depth": self.depth,
            "self_critique_enabled": self.self_critique_enabled,
            "calls": self._call_count,
            "total_steps": self._total_steps,
        }

    # ------------------------------------------------------------------
    # Internal reasoning pipeline
    # ------------------------------------------------------------------

    def _decompose(self, prompt: str) -> List[str]:
        """
        Break the prompt into sub-questions to guide exploration.

        This is a heuristic implementation that covers common question
        structures.  A real system would use an LLM here.
        """
        prompt_lower = prompt.lower().strip()
        sub_questions: List[str] = []

        # What / definition questions
        if any(w in prompt_lower for w in ("what", "define", "explain")):
            sub_questions.append(f"What is the core concept behind: '{prompt}'?")
            sub_questions.append(f"What are the key components or aspects?")

        # How questions
        if any(w in prompt_lower for w in ("how", "steps", "process", "implement")):
            sub_questions.append(f"How does this work at a high level?")
            sub_questions.append(f"What are the concrete steps or implementation details?")

        # Why questions
        if any(w in prompt_lower for w in ("why", "reason", "cause", "purpose")):
            sub_questions.append(f"What is the primary reason or motivation?")
            sub_questions.append(f"Are there secondary or contextual factors?")

        # Comparison / choice questions
        if any(w in prompt_lower for w in ("compare", "versus", "vs", "difference", "better")):
            sub_questions.append(f"What are the key distinguishing characteristics?")
            sub_questions.append(f"What are the trade-offs?")

        # Fallback: always include at least two guiding questions
        if not sub_questions:
            sub_questions.append(f"What is the main point of: '{prompt}'?")
            sub_questions.append(f"What context or background is relevant?")

        return sub_questions[: self.depth + 2]   # allow up to depth+2 sub-questions

    def _explore(self, question: str, current_depth: int) -> ReasoningStep:
        """
        Recursively explore a sub-question up to *self.depth*.

        In a production system this would call an LLM.  Here we generate
        a structured placeholder answer that conveys the chain-of-thought
        intent clearly.
        """
        answer = self._generate_answer(question, current_depth)
        sub_steps: List[ReasoningStep] = []

        if current_depth < self.depth:
            follow_ups = self._follow_up_questions(question, answer)
            for fq in follow_ups:
                sub_step = self._explore(fq, current_depth + 1)
                sub_steps.append(sub_step)

        return ReasoningStep(
            depth=current_depth,
            question=question,
            answer=answer,
            sub_steps=sub_steps,
        )

    def _generate_answer(self, question: str, depth: int) -> str:
        """
        Generate a heuristic answer for *question* at the given *depth*.

        Deeper levels produce more detailed / specific answers, simulating
        the progressive refinement of chain-of-thought reasoning.
        """
        detail_levels = [
            "At a high level: ",
            "Going deeper: ",
            "In more detail: ",
            "Specifically: ",
            "At the most granular level: ",
        ]
        prefix = detail_levels[min(depth - 1, len(detail_levels) - 1)]
        # Simple heuristic: extract key nouns / verbs for the answer body
        keywords = [
            w for w in question.split()
            if len(w) > 4 and w.lower() not in
            {"what", "does", "this", "that", "with", "from", "have", "will",
             "would", "could", "should", "their", "there", "about", "which"}
        ]
        keyword_str = ", ".join(keywords[:5]) if keywords else question.strip("?")
        return (
            f"{prefix}considering '{keyword_str}', the reasoning at depth {depth} "
            f"indicates a nuanced understanding that spans multiple perspectives "
            f"and integrates prior context."
        )

    def _follow_up_questions(self, question: str, answer: str) -> List[str]:
        """Generate follow-up sub-questions from a question/answer pair."""
        return [
            f"What are the implications of: '{answer[:60].rstrip()}…'?",
            f"Are there exceptions or edge cases related to '{question[:50].rstrip()}…'?",
        ]

    def _synthesise(
        self,
        prompt: str,
        trace: List[ReasoningStep],
        history: List[Dict[str, str]],
    ) -> str:
        """
        Synthesise a final response from the full reasoning trace.
        """
        summary_parts: List[str] = []
        for step in trace:
            summary_parts.append(step.answer)
            for sub in step.sub_steps:
                summary_parts.append(sub.answer)

        body = "  ".join(summary_parts)
        context_note = ""
        if history:
            context_note = (
                f" (drawing on {len(history)} prior conversation turn(s))"
            )
        response = (
            f"Based on a {self.depth}-level chain-of-thought analysis{context_note}: "
            + textwrap.fill(body, width=200)
        )
        return response

    def _critique(self, response: str, trace: List[ReasoningStep]) -> Optional[str]:
        """
        Self-critique: identify weaknesses in the generated response.
        """
        issues: List[str] = []
        if len(response) < 80:
            issues.append("Response is very short; consider adding more detail.")
        if len(response) > 2000:
            issues.append("Response is very long; consider summarising key points.")
        if not any(c in response for c in ".!?"):
            issues.append("Response lacks sentence-ending punctuation.")
        if not trace:
            issues.append("No reasoning trace available; response may lack grounding.")
        return "  ".join(issues) if issues else None

    def _refine(self, response: str, critique: str) -> str:
        """
        Apply self-critique feedback to produce a refined response.
        """
        return (
            response
            + f"\n\n[Self-critique applied: {critique}]"
        )
