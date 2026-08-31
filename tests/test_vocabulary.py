"""Tests for the biasing prompt built from Home Assistant's names.

Dependency-free: the token counter is injected, so no model or tokenizer is
needed and the budget is exact and predictable.
"""

import pytest

from wyoming_faster_whisper.vocabulary import (
    RecognitionContext,
    clean_names,
    estimate_tokens,
)


def _words(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def _context(**kwargs) -> RecognitionContext:
    return RecognitionContext(**kwargs)


# --- clean_names ----------------------------------------------------------


def test_clean_names_strips_and_collapses_whitespace():
    assert clean_names(["  Living   Room  "]) == ["Living Room"]


def test_clean_names_dedupes_case_insensitively_keeping_first_spelling():
    assert clean_names(["Kitchen"], ["kitchen", "Office"]) == ["Kitchen", "Office"]


def test_clean_names_drops_empty_and_non_strings():
    assert clean_names(["", "   ", None, 5, "Office"]) == ["Office"]


def test_clean_names_tolerates_none_groups():
    assert clean_names(None, ["Office"]) == ["Office"]


# --- priority order ------------------------------------------------------


def test_all_names_follows_the_tier_order():
    context = _context(
        used_areas=["Office"],
        # hass_api folds each alias in behind its own name.
        priority_entities=["Office Lamp", "Desk Lamp"],
        empty_areas=["Garage"],
        other_entities=["Ecobee", "Thermostat"],
    )
    assert context.all_names() == [
        "Office",
        "Office Lamp",
        "Desk Lamp",
        "Garage",
        "Ecobee",
        "Thermostat",
    ]


def test_names_repeated_across_tiers_are_only_paid_for_once():
    context = _context(used_areas=["Office"], other_entities=["office", "Ecobee"])
    assert context.all_names() == ["Office", "Ecobee"]


def test_budget_keeps_higher_priority_names_and_drops_the_rest():
    context = _context(used_areas=["Office"], other_entities=["Ecobee", "Nanit"])

    # Three words fits "Office, Ecobee," plus nothing more.
    prompt = context.whisper_prompt(_words, max_tokens=2)
    assert prompt == "Office, Ecobee."


def test_a_priority_entity_outranks_an_area_with_nothing_in_it():
    context = _context(priority_entities=["Office Lamp"], empty_areas=["Garage"])
    assert context.whisper_prompt(_words, max_tokens=2) == "Office Lamp."


def test_truncation_is_deterministic():
    context = _context(other_entities=[f"Name{i}" for i in range(50)])
    first = context.whisper_prompt(_words, max_tokens=5)
    second = _context(other_entities=[f"Name{i}" for i in range(50)]).whisper_prompt(
        _words, max_tokens=5
    )
    assert first == second


# --- prompt rendering ----------------------------------------------------


def test_prompt_is_comma_joined_and_ends_with_a_period():
    context = _context(used_areas=["Office", "Kitchen"])
    assert context.whisper_prompt(_words, max_tokens=100) == "Office, Kitchen."


def test_prefix_is_kept_verbatim_when_it_ends_in_punctuation():
    context = _context(other_entities=["Ecobee"])
    prompt = context.whisper_prompt(_words, max_tokens=100, prefix="Vocabulary:")
    assert prompt == "Vocabulary: Ecobee."


def test_prefix_without_punctuation_becomes_its_own_sentence():
    context = _context(other_entities=["Ecobee"])
    prompt = context.whisper_prompt(_words, max_tokens=100, prefix="  Smart home  ")
    assert prompt == "Smart home. Ecobee."


def test_prefix_counts_against_the_budget():
    context = _context(other_entities=["Ecobee", "Nanit"])

    # "Vocabulary: Ecobee." is two words; adding Nanit would make three.
    prompt = context.whisper_prompt(_words, max_tokens=2, prefix="Vocabulary:")
    assert prompt == "Vocabulary: Ecobee."


def test_prefix_survives_a_budget_too_small_for_any_name():
    context = _context(other_entities=["Ecobee"])
    prompt = context.whisper_prompt(_words, max_tokens=1, prefix="Vocabulary:")
    assert prompt == "Vocabulary:"


def test_no_names_and_no_prefix_gives_no_prompt():
    assert _context().whisper_prompt(_words, max_tokens=100) is None


def test_no_names_falls_back_to_the_prefix_alone():
    prompt = _context().whisper_prompt(_words, max_tokens=100, prefix="Vocabulary:")
    assert prompt == "Vocabulary:"


# --- caching -------------------------------------------------------------


def test_prompt_is_built_once_per_budget_and_tokenizer():
    calls = []

    def counting(text: str) -> int:
        calls.append(text)
        return _words(text)

    context = _context(other_entities=["Ecobee", "Nanit"])
    first = context.whisper_prompt(counting, max_tokens=100)
    used = len(calls)
    second = context.whisper_prompt(counting, max_tokens=100)

    assert first == second
    assert len(calls) == used, "cached prompt should not re-tokenize"


def test_a_different_tokenizer_rebuilds_the_prompt():
    context = _context(other_entities=["Ecobee"])
    context.whisper_prompt(_words, max_tokens=100, tokenizer_key="a")

    calls = []

    def counting(text: str) -> int:
        calls.append(text)
        return _words(text)

    context.whisper_prompt(counting, max_tokens=100, tokenizer_key="b")
    assert calls, "a new tokenizer must not reuse another's budget"


def test_equality_ignores_the_prompt_cache():
    left = _context(other_entities=["Ecobee"])
    right = _context(other_entities=["Ecobee"])
    left.whisper_prompt(_words, max_tokens=100)

    assert left == right, "a built prompt must not make snapshots unequal"


# --- misc ----------------------------------------------------------------


def test_context_is_falsey_when_empty():
    assert not _context()
    assert _context(other_entities=["Thermostat"])


@pytest.mark.parametrize("text", ["", "a", "Ecobee", "Living Room Lamp" * 20])
def test_estimate_tokens_is_positive(text):
    assert estimate_tokens(text) >= 1


def test_estimate_tokens_over_counts_a_realistic_prompt():
    """The fallback must never come in under the real count.

    Under-counting overflows Whisper's 223-token cap and the prompt is silently
    truncated. The token figure below was measured with Whisper's own BPE
    (faster-whisper 1.2.1, rhasspy/faster-whisper-tiny-int8).
    """
    names = [
        "Office",
        "Kitchen",
        "Living Room",
        "Upstairs",
        "Main Floor",
        "Ecobee",
        "Nanit",
        "Office Lamp",
        "Desk Lamp",
        "Thermostat",
        "Bedroom Blinds",
        "Aqara Switch",
        "Sonos Arc",
        "Nest Hub Max",
        "Roborock S7",
        "LIFX Strip",
        "Wyze Cam",
        "Shelly Plug",
        "Zooz ZEN32",
    ]
    prompt = ", ".join(names) + "."
    real_tokens = 76  # 217 characters, so 2.85 per token

    assert estimate_tokens(prompt) >= real_tokens
