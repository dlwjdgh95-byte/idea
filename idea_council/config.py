"""Council configuration.

The council runs inside Claude by default (the ``claude-code`` backend, see
:mod:`idea_council.claude_code`): one Claude model of each size.

The optional ``openrouter`` backend keeps karpathy/llm-council's line-up (one
model each from OpenAI, Google, Anthropic and xAI, Gemini as chairman) with
ids that OpenRouter still lists (checked 2026-11-04; the README's
gemini-3-pro-preview and grok-4 are gone). Those ids go stale, so the CLI
checks them against OpenRouter's model list before a run. Override any
model with environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

CLAUDE_MEMBERS: tuple[str, ...] = ("opus", "sonnet", "haiku", "fable")
CLAUDE_CHAIRMAN = "opus"
CLAUDE_VERIFIER = "sonnet"

OPENROUTER_MEMBERS: tuple[str, ...] = (
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-5.5",
    "x-ai/grok-4.7",
)
OPENROUTER_CHAIRMAN = "google/gemini-3.1-pro-preview"
OPENROUTER_VERIFIER = "anthropic/claude-sonnet-5.5"

BACKENDS: tuple[str, ...] = ("claude-code", "openrouter")
DEFAULT_BACKEND = "claude-code"
DEFAULT_MEMBERS, DEFAULT_CHAIRMAN, DEFAULT_VERIFIER = CLAUDE_MEMBERS, CLAUDE_CHAIRMAN, CLAUDE_VERIFIER
DEFAULT_ROLES: tuple[str, ...] = ("expander", "critic", "builder", "connector")


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class CouncilConfig:
    members: tuple[str, ...] = DEFAULT_MEMBERS
    chairman: str = DEFAULT_CHAIRMAN
    verifier: str = DEFAULT_VERIFIER
    roles: tuple[str, ...] = DEFAULT_ROLES
    # Shuffle which model gets which role on every run, so a model's
    # ranking is not confounded with the role it always plays.
    rotate_roles: bool = True
    max_claims_to_verify: int = 8
    web_search_max_results: int = 5
    check_source_urls: bool = True
    seed: int | None = None
    timeout_s: float = 180.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("a council needs at least two members")
        if not self.roles:
            raise ValueError("at least one role is required")
        if self.max_claims_to_verify < 1:
            raise ValueError("COUNCIL_MAX_CLAIMS must be 1 or more (use --no-verify to skip the gate)")
        if len(set(self.members)) != len(self.members):
            raise ValueError("council members must be distinct model ids")

    def all_models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.members, self.chairman, self.verifier)))

    @classmethod
    def from_env(cls, backend: str = DEFAULT_BACKEND) -> CouncilConfig:
        """Read overrides from the environment; the default line-up depends on the backend."""
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; use one of {', '.join(BACKENDS)}")
        if backend == "openrouter":
            members, chairman, verifier = OPENROUTER_MEMBERS, OPENROUTER_CHAIRMAN, OPENROUTER_VERIFIER
        else:
            members, chairman, verifier = CLAUDE_MEMBERS, CLAUDE_CHAIRMAN, CLAUDE_VERIFIER
        seed = os.environ.get("COUNCIL_SEED")
        return cls(
            members=_split(os.environ.get("COUNCIL_MODELS")) or members,
            chairman=os.environ.get("CHAIRMAN_MODEL") or chairman,
            verifier=os.environ.get("VERIFIER_MODEL") or verifier,
            roles=_split(os.environ.get("COUNCIL_ROLES")) or DEFAULT_ROLES,
            rotate_roles=os.environ.get("COUNCIL_ROTATE_ROLES", "1") != "0",
            max_claims_to_verify=int(os.environ.get("COUNCIL_MAX_CLAIMS", "8")),
            seed=int(seed) if seed else None,
        )
