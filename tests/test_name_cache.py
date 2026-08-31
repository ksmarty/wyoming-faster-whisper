"""Tests for the background refresh of Home Assistant's names.

The Home Assistant client is faked, so nothing here touches the network. What is
being pinned down is the degradation behavior: a refresh that fails, or that is
still running when the audio stops, must cost freshness and never a transcript.
"""

import asyncio

import pytest

pytest.importorskip("aiohttp")

# pylint: disable=wrong-import-position
from wyoming.info import Info  # noqa: E402

from wyoming_faster_whisper.dispatch_handler import DispatchEventHandler  # noqa: E402
from wyoming_faster_whisper.hass_api import HomeAssistantError  # noqa: E402
from wyoming_faster_whisper.name_cache import HassNameCache  # noqa: E402
from wyoming_faster_whisper.vocabulary import RecognitionContext  # noqa: E402


class FakeHass:
    """Stands in for HomeAssistant, with scriptable results and delay."""

    def __init__(self, context=None, delay=0.0):
        self.context = context if context is not None else RecognitionContext()
        self.delay = delay
        self.fail = False
        self.calls = 0

    async def get_context(self) -> RecognitionContext:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise HomeAssistantError("unreachable")

        return self.context


def _cache(hass, **kwargs) -> HassNameCache:
    kwargs.setdefault("max_tokens", 1000)
    return HassNameCache(hass, **kwargs)


def _names(*entities) -> RecognitionContext:
    return RecognitionContext(priority_entities=list(entities))


# --- prompt ---------------------------------------------------------------


async def test_prompt_is_the_prefix_alone_before_any_fetch():
    cache = _cache(FakeHass(), prefix="Vocabulary:")
    assert await cache.initial_prompt() == "Vocabulary:"


async def test_prompt_is_none_before_any_fetch_without_a_prefix():
    assert await _cache(FakeHass()).initial_prompt() is None


async def test_fetched_names_reach_the_prompt():
    cache = _cache(FakeHass(_names("Ecobee", "Nanit")), prefix="Vocabulary:")
    assert await cache.refresh()
    assert await cache.initial_prompt() == "Vocabulary: Ecobee, Nanit."


# --- refresh scheduling ---------------------------------------------------


async def test_scheduled_refresh_lands_before_the_prompt_is_needed():
    cache = _cache(FakeHass(_names("Ecobee")))
    cache.schedule_refresh()

    # initial_prompt waits for the in-flight fetch, as it does at AudioStop.
    assert await cache.initial_prompt() == "Ecobee."


async def test_concurrent_schedules_collapse_onto_one_fetch():
    hass = FakeHass(_names("Ecobee"), delay=0.05)
    cache = _cache(hass)

    cache.schedule_refresh()
    cache.schedule_refresh()
    cache.schedule_refresh()
    await cache.initial_prompt()

    assert hass.calls == 1


async def test_refresh_seconds_throttles_repeat_fetches():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass, refresh_seconds=60.0)

    cache.schedule_refresh()
    await cache.initial_prompt()
    cache.schedule_refresh()
    await cache.initial_prompt()

    assert hass.calls == 1


async def test_zero_refresh_seconds_fetches_every_utterance():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass, refresh_seconds=0.0)

    for _ in range(3):
        cache.schedule_refresh()
        await cache.initial_prompt()

    assert hass.calls == 3


# --- degradation ----------------------------------------------------------


async def test_a_failed_fetch_leaves_no_names_but_keeps_the_prefix():
    hass = FakeHass()
    hass.fail = True
    cache = _cache(hass, prefix="Vocabulary:")

    assert not await cache.refresh()
    assert await cache.initial_prompt() == "Vocabulary:"


async def test_a_failed_refresh_keeps_the_previous_names():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass)
    await cache.refresh()

    hass.fail = True
    cache.schedule_refresh()

    assert await cache.initial_prompt() == "Ecobee."


async def test_a_slow_refresh_does_not_hold_up_transcription():
    hass = FakeHass(_names("Ecobee"), delay=0.02)
    cache = _cache(hass)
    await cache.refresh()

    hass.context = _names("Ecobee", "Nanit")
    hass.delay = 5.0
    cache.schedule_refresh()

    # The slow fetch is abandoned for this utterance; the old names are used.
    assert await cache.initial_prompt() == "Ecobee."


async def test_a_refresh_that_timed_out_is_not_cancelled():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass, wait_timeout=0.01)
    await cache.refresh()

    hass.context = _names("Ecobee", "Nanit")
    hass.delay = 0.1
    cache.schedule_refresh()

    # Times out and falls back...
    assert await cache.initial_prompt() == "Ecobee."

    # ...but the fetch kept running, so the next utterance has the new names.
    await asyncio.sleep(0.2)
    assert await cache.initial_prompt(wait=False) == "Ecobee, Nanit."


async def test_streaming_path_does_not_wait():
    hass = FakeHass(_names("Ecobee"), delay=5.0)
    cache = _cache(hass, prefix="Vocabulary:")
    cache.schedule_refresh()

    # wait=False returns immediately with whatever is on hand (nothing yet).
    assert await cache.initial_prompt(wait=False) == "Vocabulary:"


# --- snapshot reuse -------------------------------------------------------


async def test_unchanged_names_keep_the_cached_prompt():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass)
    await cache.refresh()
    before = cache._context  # pylint: disable=protected-access

    # A fresh but equal context arrives.
    hass.context = _names("Ecobee")
    await cache.refresh()

    assert cache._context is before  # pylint: disable=protected-access


async def test_renamed_entity_replaces_the_snapshot():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass)
    await cache.refresh()

    hass.context = _names("Ecobee", "Nanit")
    await cache.refresh()

    assert await cache.initial_prompt(wait=False) == "Ecobee, Nanit."


# --- once per utterance ---------------------------------------------------


def _handler(cache) -> DispatchEventHandler:
    """A handler with no I/O: only the name-refresh latch is exercised."""
    return DispatchEventHandler(Info(), None, cache, None, None)


async def test_an_utterance_asks_for_exactly_one_refresh():
    # A finished fetch is immediately due again at refresh_seconds=0, so without
    # a per-utterance latch every audio chunk would start another one.
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(_cache(hass))

    for _ in range(50):
        # Stands in for AudioStart followed by a stream of AudioChunks.
        handler._refresh_names()  # pylint: disable=protected-access
        await asyncio.sleep(0)

    await asyncio.sleep(0.05)
    assert hass.calls == 1


async def test_the_next_utterance_refreshes_again():
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(_cache(hass))

    for _ in range(2):
        handler._refresh_names()  # pylint: disable=protected-access
        await asyncio.sleep(0.01)
        handler._reset()  # pylint: disable=protected-access

    assert hass.calls == 2


async def test_no_refresh_without_a_cache():
    # Nothing to latch, and nothing to crash on.
    _handler(None)._refresh_names()  # pylint: disable=protected-access


async def test_has_names_reflects_whether_a_fetch_succeeded():
    hass = FakeHass(_names("Ecobee"))
    hass.fail = True
    cache = _cache(hass)

    await cache.refresh()
    assert not cache.has_names

    hass.fail = False
    await cache.refresh()
    assert cache.has_names
