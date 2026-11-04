"""Idea council: LLM Council (after karpathy/llm-council) with a fact-check gate."""

from .claude_code import ClaudeCodeClient
from .confidence import Evidence, Verdict, assess
from .config import CouncilConfig
from .council import Council, CouncilResult
from .verify import ClaimCheck, Verifier

__all__ = [
    "ClaimCheck",
    "ClaudeCodeClient",
    "Council",
    "CouncilConfig",
    "CouncilResult",
    "Evidence",
    "Verdict",
    "Verifier",
    "assess",
]
