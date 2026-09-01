"""Turn Home Assistant names into a budgeted biasing prompt.

Whisper accepts an ``initial_prompt`` that conditions the decoder, and biasing it
toward the names in the user's home is the cheapest accuracy win available: it
takes proper nouns the model has never heard ("Ecobee", "Nanit") from garbage to
correct without changing model, audio, or decode settings. Qwen3-ASR takes the
same string as its context prompt.

The catch is size. Whisper's decoder context is 448 tokens and the prompt is
hard-capped at ``max_length // 2 - 1`` == 223, past which it is silently
truncated -- and quality degrades well before the cap, because a long prompt
makes the model hallucinate and echo words that were never spoken. A house with
200 exposed entities cannot be sent wholesale, so the names are placed in
priority order and cut off at a budget:

  1. areas and floors that hold an exposed entity -- few in number, spoken in
     nearly every command ("turn on the *office* lamp"), and demonstrably real
     targets, because something in them can actually be commanded
  2. entity names in the domains a speaker names out loud -- lights, switches,
     media players and the rest (see ``hass_api.PRIORITY_DOMAINS``). These are
     the high-value proper nouns: the thing being turned on usually *is* the
     name that gets misheard
  3. the remaining areas and floors -- sayable, but with nothing exposed in them
     there is no command they can complete
  4. the remaining entity names -- in a big home mostly sensors, which are
     hundreds in number and usually asked about by area ("the temperature in the
     office") rather than by their own name

Aliases are not a tier. An alias exists precisely because it is what the speaker
says *instead of* the name the integration gave the thing, so it is at least as
likely to be spoken as the name it replaces -- ranking every alias below every
name drops the very words biasing exists to catch. Each one sits with the name
it belongs to, so truncation cuts whole things rather than stranding a name
whose spoken form was thrown away.

Truncation is deliberate and deterministic: fill in that order, stop at the
first name that would not fit, and log what was dropped. No scoring, no
sampling -- the same home produces the same prompt every time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# Stay well under Whisper's hard cap of 223.
DEFAULT_PROMPT_MAX_TOKENS = 200

# Counts tokens in a candidate prompt using the model's own tokenizer.
CountTokens = Callable[[str], int]

# Punctuation that already separates a prefix from the name list.
_PREFIX_ENDINGS = (".", ":", ",", ";", "!", "?")


def estimate_tokens(text: str) -> int:
    """Approximate a token count when the model's tokenizer is unavailable.

    Deliberately an over-estimate. Going under means Whisper silently truncates
    the prompt, which is worse than fitting fewer names than we could have.

    Measured against Whisper's own BPE, a comma-joined list of realistic entity
    names runs about 2.95 characters per token, so 2 leaves comfortable headroom.
    This is only ever called on a whole candidate prompt, which is long; a very
    short string on its own can still come in under (rare names are dense --
    "Nanit" is 5 characters but 3 tokens), and that does not matter here.
    """
    return max(1, len(text) // 2)


def clean_names(*groups: Iterable[Any]) -> List[str]:
    """Merge name groups in order: strip, collapse whitespace, de-dupe.

    De-duplication is case-insensitive and keeps the first spelling seen, so a
    name that appears as both an area and an alias is only paid for once.
    """
    names: List[str] = []
    seen = set()
    for group in groups:
        for value in group or ():
            if not isinstance(value, str):
                continue

            name = " ".join(value.split()).strip()
            key = name.lower()
            if name and (key not in seen):
                seen.add(key)
                names.append(name)

    return names


@dataclass
class RecognitionContext:
    """The names from one snapshot of Home Assistant, in priority order.

    Each field is one tier of the order documented at the top of this module,
    highest first; ``hass_api`` decides which name lands in which, and puts each
    alias directly behind the name it belongs to. Floors are grouped with areas
    -- they are the same kind of name to a speaker, and there are only ever a
    handful of them.

    A new snapshot is a new instance, so the prompt cache below dies with the
    names it was built from.
    """

    # Areas and floors with at least one exposed entity in them.
    used_areas: List[str] = field(default_factory=list)

    # Entities in the domains a speaker names out loud.
    priority_entities: List[str] = field(default_factory=list)

    # Areas and floors with nothing exposed in them.
    empty_areas: List[str] = field(default_factory=list)

    # Every other exposed entity.
    other_entities: List[str] = field(default_factory=list)

    # Cached by (prefix, budget, tokenizer). Building the prompt costs one
    # tokenizer call per name, and the same prompt is reused for every utterance
    # until the names change.
    _prompts: Dict[Tuple[str, int, str], Optional[str]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def all_names(self) -> List[str]:
        """Every name in priority order, de-duped across the tiers."""
        return clean_names(
            self.used_areas,
            self.priority_entities,
            self.empty_areas,
            self.other_entities,
        )

    def __bool__(self) -> bool:
        return bool(
            self.used_areas
            or self.priority_entities
            or self.empty_areas
            or self.other_entities
        )

    def whisper_prompt(
        self,
        count_tokens: CountTokens,
        max_tokens: int = DEFAULT_PROMPT_MAX_TOKENS,
        prefix: Optional[str] = None,
        tokenizer_key: str = "",
    ) -> Optional[str]:
        """A comma-joined prompt of as many names as fit within ``max_tokens``.

        ``count_tokens`` should use the model's own tokenizer so the budget is
        exact for that model; ``tokenizer_key`` identifies it for caching.
        ``prefix`` is the server's configured ``--initial-prompt``, kept at the
        front and always included -- an explicit setting outranks anything
        discovered from Home Assistant.

        Returns None when there is nothing to send at all.
        """
        key = (prefix or "", max_tokens, tokenizer_key)
        if key not in self._prompts:
            self._prompts[key] = self._build_prompt(count_tokens, max_tokens, prefix)

        return self._prompts[key]

    def _build_prompt(
        self,
        count_tokens: CountTokens,
        max_tokens: int,
        prefix: Optional[str],
    ) -> Optional[str]:
        """Fill the budget in priority order, stopping at the first name that
        does not fit."""
        head = (prefix or "").strip()
        if head and not head.endswith(_PREFIX_ENDINGS):
            # So the prefix reads as its own sentence rather than running into
            # the first name.
            head += "."

        names = self.all_names()
        if not names:
            return head or None

        chosen: List[str] = []
        for name in names:
            candidate = _join(head, chosen + [name])
            if count_tokens(candidate) > max_tokens:
                break

            chosen.append(name)

        if len(chosen) < len(names):
            _LOGGER.debug(
                "Prompt budget of %s tokens fit %s/%s names; %s dropped",
                max_tokens,
                len(chosen),
                len(names),
                len(names) - len(chosen),
            )

        if not chosen:
            # Not even one name fits, which means the budget is smaller than the
            # prefix. Send the prefix alone rather than nothing.
            return head or None

        return _join(head, chosen)


def _join(head: str, names: List[str]) -> str:
    """Render a prefix and a name list as the final prompt string."""
    body = ", ".join(names) + "."
    return f"{head} {body}" if head else body
