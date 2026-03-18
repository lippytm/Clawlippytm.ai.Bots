"""
clawlippytm.bot
~~~~~~~~~~~~~~~

Core bot definition for Clawlippytm.Bots.

Defines BotAttributes (the canonical set of personality / capability
attributes) and the ClawBot class that combines all sub-systems into a
single, fully-featured AI-toolkit bot.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

from .cognitive_reasoning import CognitiveReasoner
from .creativity import CreativityEngine
from .diagnostics import DiagnosticsSystem


# ---------------------------------------------------------------------------
# Bot attribute descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BotAttributes:
    """
    Canonical set of attributes for every Clawlippytm.Bot.

    All fields are optional so that partial attribute sets can be constructed
    for sub-bots or specialised roles, then merged with the defaults provided
    by :meth:`BotAttributes.defaults`.
    """

    # --- Identity ---
    name: str = "ClawBot"
    version: str = "1.0.0"
    description: str = "A Clawlippytm AI bot"

    # --- Personality / tone ---
    tone: str = "friendly"            # friendly | formal | humorous | neutral
    verbosity: str = "balanced"       # concise | balanced | verbose
    empathy_level: float = 0.75       # 0.0 – 1.0

    # --- Capability flags ---
    multi_turn: bool = True           # supports multi-turn conversation
    memory_enabled: bool = True       # retains context across turns
    tool_use: bool = True             # can call external tools / APIs
    streaming: bool = False           # streaming response support

    # --- Reasoning / creativity knobs ---
    reasoning_depth: int = 3          # 1 (shallow) – 5 (deep)
    creativity_temperature: float = 0.7  # 0.0 (deterministic) – 1.0 (creative)
    self_critique: bool = True        # enables self-critique loop

    # --- Safety / ethics ---
    safety_filter: bool = True
    ethical_guidelines: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Do no harm",
            "Respect privacy",
            "Be transparent about being an AI",
            "Avoid misinformation",
        ]
    )

    # --- Diagnostics ---
    diagnostics_enabled: bool = True
    feedback_loops: int = 2           # how many diagnostic feedback iterations

    @classmethod
    def defaults(cls) -> "BotAttributes":
        """Return a fully-populated default attribute set."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise attributes to a plain dictionary."""
        return dataclasses.asdict(self)

    def update(self, **kwargs: Any) -> "BotAttributes":
        """Return a *new* BotAttributes with the given fields overridden."""
        current = dataclasses.asdict(self)
        current.update(kwargs)
        return BotAttributes(**current)


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class ClawBot:
    """
    The core Clawlippytm.Bot.

    Combines :class:`~clawlippytm.diagnostics.DiagnosticsSystem`,
    :class:`~clawlippytm.cognitive_reasoning.CognitiveReasoner`, and
    :class:`~clawlippytm.creativity.CreativityEngine` into a single,
    unified agent.

    Parameters
    ----------
    attributes:
        Optional :class:`BotAttributes` instance.  Defaults are used when
        *None* is supplied.
    """

    def __init__(self, attributes: Optional[BotAttributes] = None) -> None:
        self.attributes: BotAttributes = attributes or BotAttributes.defaults()
        self.diagnostics: DiagnosticsSystem = DiagnosticsSystem(
            enabled=self.attributes.diagnostics_enabled,
            feedback_loops=self.attributes.feedback_loops,
        )
        self.reasoner: CognitiveReasoner = CognitiveReasoner(
            depth=self.attributes.reasoning_depth,
            self_critique=self.attributes.self_critique,
        )
        self.creativity: CreativityEngine = CreativityEngine(
            temperature=self.attributes.creativity_temperature,
        )
        self._conversation_history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> str:
        """
        Process *user_input* and return a bot response.

        The pipeline is:

        1. Record the user turn in conversation history.
        2. Run diagnostics on the incoming message (with feedback loops).
        3. Apply cognitive reasoning to generate a candidate response.
        4. Optionally enrich the response with creative elaboration.
        5. Run a post-generation diagnostic pass.
        6. Return the final response.

        Parameters
        ----------
        user_input:
            Raw text from the user.

        Returns
        -------
        str
            The bot's response.
        """
        self._conversation_history.append({"role": "user", "content": user_input})

        # --- Diagnostic pre-pass ---
        diag_result = self.diagnostics.analyse(
            text=user_input,
            context=self._conversation_history[:-1],
        )

        # --- Cognitive reasoning ---
        reasoning_output = self.reasoner.reason(
            prompt=user_input,
            diagnostic_context=diag_result,
            history=self._conversation_history,
        )

        # --- Creativity enrichment ---
        candidate = self.creativity.enrich(
            base_response=reasoning_output.response,
            reasoning_trace=reasoning_output.trace,
        )

        # --- Diagnostic post-pass (feedback loop on output) ---
        final_diag = self.diagnostics.analyse(
            text=candidate,
            context=self._conversation_history,
        )
        # Apply safety correction when the input OR output triggered a
        # high-severity diagnostic issue.
        input_is_high = diag_result.highest_severity == "high"
        output_is_high = final_diag.highest_severity == "high"
        if self.attributes.safety_filter and (input_is_high or output_is_high):
            candidate = self._apply_safety_correction(candidate, final_diag)

        self._conversation_history.append({"role": "bot", "content": candidate})
        return candidate

    def reset(self) -> None:
        """Clear the conversation history and all diagnostic state."""
        self._conversation_history.clear()
        self.diagnostics.reset()
        self.reasoner.reset()

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of the bot's current operational status."""
        return {
            "name": self.attributes.name,
            "version": self.attributes.version,
            "conversation_turns": len(self._conversation_history),
            "diagnostics": self.diagnostics.summary(),
            "reasoning": self.reasoner.summary(),
            "creativity": self.creativity.summary(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_safety_correction(
        self, text: str, diag_result: Any
    ) -> str:
        """
        Apply a lightweight safety correction when the diagnostic pass
        flags issues in the generated output.
        """
        prefix = (
            "[Safety note: parts of this response have been reviewed. "
            "Please interpret with care.] "
        )
        return prefix + text
