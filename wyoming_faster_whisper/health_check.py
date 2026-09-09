"""Health check for the Wyoming server.

Used by the Dockerfile's HEALTHCHECK. Asking for info rather than just opening a
socket is deliberate: the port is bound by the OS, so a connect-only check stays
green even if the event loop is wedged. A Describe/Info round trip proves the
accept loop and the event handler are both still running.

Only imports wyoming and this package's environment-variable names, so it stays
fast and does not depend on the optional extras or on a backend being
importable.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse, urlunparse

from wyoming.client import AsyncClient
from wyoming.info import Describe, Info

from .env_args import ENV_PREFIX, FILE_SUFFIX

DEFAULT_URI = "tcp://127.0.0.1:10300"

# What a server binds to when it means "every interface". These are addresses to
# listen on, not addresses to connect to: 0.0.0.0 happens to reach the local
# host on Linux, but "::" does not, and an empty host is not a host at all.
_WILDCARD_HOSTS = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}


async def check(uri: str) -> None:
    """Ask the server for its info, or raise."""
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Describe().event())

        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("Connection closed without info")

            if Info.is_type(event.type):
                info = Info.from_event(event)
                if not info.asr:
                    raise RuntimeError("No ASR programs in info")

                return

            # The server sends nothing else in response to Describe, but skip
            # anything unexpected instead of failing on it.


def default_uri(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return the URI to check, following the server's own configuration.

    The image's entrypoint only supplies its baked-in ``--uri`` when
    ``WYO_WHISPER_URI`` is unset, so that variable is where a container's port
    differs from the default. A ``--uri`` passed to ``docker run`` instead is
    invisible here; pass the same one to this check.
    """
    if environ is None:
        environ = os.environ

    return _to_loopback(_configured_uri(environ) or DEFAULT_URI)


def _configured_uri(environ: Mapping[str, str]) -> Optional[str]:
    """Read WYO_WHISPER_URI, preferring the file it may point at."""
    file_path = environ.get(f"{ENV_PREFIX}URI{FILE_SUFFIX}")
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8").strip() or None
        except OSError:
            # The server reports this one at startup; here it would only turn a
            # readable "unhealthy: connection refused" into a stat error.
            return None

    return (environ.get(f"{ENV_PREFIX}URI") or "").strip() or None


def _to_loopback(uri: str) -> str:
    """Rewrite a listen-everywhere host into one that can be connected to."""
    parsed = urlparse(uri)
    if parsed.scheme != "tcp":
        return uri

    host = _WILDCARD_HOSTS.get(parsed.hostname or "")
    if host is None:
        return uri

    if ":" in host:
        host = f"[{host}]"  # IPv6 literals are bracketed in a URI

    port = f":{parsed.port}" if parsed.port else ""

    return urlunparse(parsed._replace(netloc=f"{host}{port}"))


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        default=default_uri(),
        help="unix:// or tcp:// (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for info (default: 15)",
    )
    args = parser.parse_args()

    try:
        await asyncio.wait_for(check(args.uri), timeout=args.timeout)
    except Exception as err:  # pylint: disable=broad-except
        # Docker only reports the exit status, so the reason has to be printed.
        # It shows up in "docker inspect" as the health check's output.
        # TimeoutError, among others, has an empty message, so fall back to the
        # class name.
        print(f"unhealthy: {str(err) or type(err).__name__}", file=sys.stderr)
        sys.exit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
