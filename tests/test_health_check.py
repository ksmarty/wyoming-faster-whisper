"""Tests for the Docker health check.

Runs the check against a real socket rather than a mocked client, since what it
is meant to prove is that a Describe gets an Info back over the wire.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

import pytest
from wyoming.asr import Transcript
from wyoming.event import Event, async_read_event, async_write_event
from wyoming.info import AsrProgram, Attribution, Describe, Info

from wyoming_faster_whisper.health_check import DEFAULT_URI, check, default_uri, main

Responder = Callable[[Event, asyncio.StreamWriter], Awaitable[bool]]

INFO = Info(
    asr=[
        AsrProgram(
            name="faster-whisper",
            description="Faster Whisper transcription with CTranslate2",
            attribution=Attribution(name="Test", url="http://example.com"),
            installed=True,
            version="1.0.0",
            models=[],
        )
    ]
)

# -----------------------------------------------------------------------------


@asynccontextmanager
async def fake_server(respond: Responder) -> AsyncIterator[str]:
    """Serve on a random port with a handler of the test's choosing."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                event = await async_read_event(reader)
                if (event is None) or (not await respond(event, writer)):
                    break
        finally:
            writer.close()

    server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"tcp://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


async def _describe_responder(
    event: Event, writer: asyncio.StreamWriter, info: Optional[Info] = INFO
) -> bool:
    """Answer Describe with info, like the real server does."""
    if Describe.is_type(event.type):
        await async_write_event((info or INFO).event(), writer)

    return True


@asynccontextmanager
async def hanging_server() -> AsyncIterator[str]:
    """Serve a socket that accepts a connection and then never answers."""
    release = asyncio.Event()

    async def respond(event: Event, writer: asyncio.StreamWriter) -> bool:
        await release.wait()
        return False

    async with fake_server(respond) as uri:
        try:
            yield uri
        finally:
            # Let the handler return: wait_closed() blocks until it does.
            release.set()


async def unused_uri() -> str:
    """Return a URI for a port that nothing is listening on."""
    async with fake_server(_describe_responder) as uri:
        pass

    return uri


# -----------------------------------------------------------------------------


async def test_check_succeeds() -> None:
    """A server that answers Describe with ASR programs is healthy."""
    async with fake_server(_describe_responder) as uri:
        await check(uri)


async def test_check_skips_unexpected_events() -> None:
    """Anything that isn't Info is ignored rather than failing the check."""

    async def respond(event: Event, writer: asyncio.StreamWriter) -> bool:
        if Describe.is_type(event.type):
            await async_write_event(Transcript(text="surprise").event(), writer)
            await async_write_event(INFO.event(), writer)

        return True

    async with fake_server(respond) as uri:
        await check(uri)


async def test_check_fails_without_asr_programs() -> None:
    """Info with no ASR programs means the server came up without a backend."""

    async def respond(event: Event, writer: asyncio.StreamWriter) -> bool:
        return await _describe_responder(event, writer, info=Info(asr=[]))

    async with fake_server(respond) as uri:
        with pytest.raises(RuntimeError, match="No ASR programs"):
            await check(uri)


async def test_check_fails_when_connection_closes() -> None:
    """A socket that accepts and then hangs up is not healthy."""

    async def respond(event: Event, writer: asyncio.StreamWriter) -> bool:
        return False  # disconnect without answering

    async with fake_server(respond) as uri:
        with pytest.raises(RuntimeError, match="Connection closed"):
            await check(uri)


async def test_check_fails_when_nothing_is_listening() -> None:
    """Nothing bound to the port at all."""
    with pytest.raises(OSError):
        await check(await unused_uri())


async def test_check_hangs_until_the_timeout() -> None:
    """A server that accepts but never answers is caught by --timeout, not hung on."""
    async with hanging_server() as uri:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(check(uri), timeout=0.1)


# -----------------------------------------------------------------------------


async def test_main_exits_zero_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module Docker runs succeeds quietly."""
    async with fake_server(_describe_responder) as uri:
        monkeypatch.setattr("sys.argv", ["health_check", "--uri", uri])
        await main()


async def test_main_exits_one_when_unhealthy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Docker only sees the exit status, so the reason has to be printed."""
    monkeypatch.setattr(
        "sys.argv",
        ["health_check", "--uri", await unused_uri(), "--timeout", "5"],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 1
    assert "unhealthy:" in capsys.readouterr().err


async def test_main_reports_a_timeout_by_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """TimeoutError has an empty message, which would print nothing useful."""
    async with hanging_server() as uri:
        monkeypatch.setattr(
            "sys.argv", ["health_check", "--uri", uri, "--timeout", "0.1"]
        )

        with pytest.raises(SystemExit):
            await main()

    assert "unhealthy: TimeoutError" in capsys.readouterr().err


# -----------------------------------------------------------------------------


def test_default_uri_is_loopback() -> None:
    """Nothing in the environment: the image's own port."""
    assert default_uri({}) == DEFAULT_URI


def test_default_uri_follows_the_environment() -> None:
    """A container that moved the port moved it with WYO_WHISPER_URI."""
    assert (
        default_uri({"WYO_WHISPER_URI": "tcp://127.0.0.1:10400"})
        == "tcp://127.0.0.1:10400"
    )


@pytest.mark.parametrize(
    ("listen_uri", "expected"),
    [
        ("tcp://0.0.0.0:10400", "tcp://127.0.0.1:10400"),
        ("tcp://[::]:10400", "tcp://[::1]:10400"),
        ("unix:///tmp/whisper.socket", "unix:///tmp/whisper.socket"),
    ],
)
def test_default_uri_rewrites_wildcard_hosts(listen_uri: str, expected: str) -> None:
    """The server's listen-everywhere host is not an address to connect to.

    0.0.0.0 reaches the local host on Linux, but "::" does not, and neither is
    guaranteed to elsewhere. unix:// is passed through untouched.
    """
    assert default_uri({"WYO_WHISPER_URI": listen_uri}) == expected


def test_default_uri_ignores_an_empty_variable() -> None:
    """Empty means unset, the same as it does for the server's options."""
    assert default_uri({"WYO_WHISPER_URI": "  "}) == DEFAULT_URI


def test_default_uri_reads_the_file_form(tmp_path: Path) -> None:
    """Every option has a _FILE form; the trailing newline is not part of it."""
    uri_path = tmp_path / "uri"
    uri_path.write_text("tcp://0.0.0.0:10400\n", encoding="utf-8")

    assert (
        default_uri({"WYO_WHISPER_URI_FILE": str(uri_path)}) == "tcp://127.0.0.1:10400"
    )


def test_default_uri_survives_an_unreadable_file(tmp_path: Path) -> None:
    """A bad _FILE is the server's error to report, not a stat traceback here."""
    assert default_uri({"WYO_WHISPER_URI_FILE": str(tmp_path / "missing")}) == (
        DEFAULT_URI
    )
