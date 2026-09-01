"""Environment variable defaults for the command-line options.

Every ``--option`` of the server also reads ``WYO_WHISPER_OPTION``: the flag name
uppercased, dashes as underscores, with a project-specific prefix so a compose
stack shared with other Wyoming services cannot cross-configure this one.

Values may also be placed in a file named by ``WYO_WHISPER_OPTION_FILE``, the
convention Docker Compose and Swarm secrets use: a secret is mounted as a file
under ``/run/secrets`` rather than delivered as a variable, so the variable
points at the file instead. This keeps a long-lived Home Assistant token out of
both the command line and the environment.

Precedence is command line, then ``_FILE``, then the plain variable. An explicit
flag always wins, so an argument can never be silently shadowed by a stale
variable left over in a container.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ENV_PREFIX = "WYO_WHISPER_"
FILE_SUFFIX = "_FILE"

# Accepted spellings for flags like --debug, which take no value on the command
# line and so need one invented for the environment.
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", ""}

_LOGGER = logging.getLogger(__name__)

# argparse offers no public way to walk a parser's options or to tell one kind of
# action from another, and these names have been stable since argparse was added.
# pylint: disable=protected-access
_VALUELESS_ACTIONS = (argparse._HelpAction, argparse._VersionAction)
_AppendAction = argparse._AppendAction
_StoreConstAction = argparse._StoreConstAction
# pylint: enable=protected-access

# -----------------------------------------------------------------------------


def env_var_name(dest: str) -> str:
    """Return the environment variable that fills in an argument."""
    return ENV_PREFIX + dest.upper()


def parse_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> argparse.Namespace:
    """Parse arguments, falling back to the environment for missing options."""
    if environ is None:
        environ = os.environ

    known_vars = set()

    # Filled in after parsing so a command-line value replaces the environment's
    # instead of being appended to it (see below).
    append_defaults: Dict[str, List[Any]] = {}

    for action in parser._actions:  # pylint: disable=protected-access
        if isinstance(action, _VALUELESS_ACTIONS):
            # --help/--version take no value
            continue

        if (not action.option_strings) or (action.dest == argparse.SUPPRESS):
            continue

        var_name = env_var_name(action.dest)
        known_vars.add(var_name)

        value = _read_env(parser, environ, var_name)
        if value is None:
            continue

        default = _convert(parser, action, var_name, value)

        if isinstance(action, _AppendAction):
            append_defaults[action.dest] = default
            action.default = None
        else:
            action.default = default

        # The environment supplied it, so the command line does not have to.
        action.required = False

    args = parser.parse_args(argv)

    for dest, default in append_defaults.items():
        # argparse appends command-line values onto an action's default, which
        # would merge the environment's list with the command line's. Only use
        # the environment when the command line gave nothing.
        if getattr(args, dest, None) is None:
            setattr(args, dest, default)

    _warn_unknown_vars(environ, known_vars)

    return args


# -----------------------------------------------------------------------------


def _read_env(
    parser: argparse.ArgumentParser, environ: Mapping[str, str], var_name: str
) -> Optional[str]:
    """Read a variable's value, preferring the file it may point at."""
    file_path = environ.get(var_name + FILE_SUFFIX)
    if file_path:
        try:
            # Trailing newlines are an artifact of writing the file, never part
            # of a token or a URL.
            return Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"cannot read {var_name}{FILE_SUFFIX} ({file_path}): {exc}")

    return environ.get(var_name)


def _convert(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    var_name: str,
    value: str,
) -> Any:
    """Turn a variable's string into the value the action would have produced."""
    if isinstance(action, _StoreConstAction):
        # --debug and friends: the flag's presence is the value on the command
        # line, so the variable has to spell out true or false. action.default
        # is still the original one here.
        return action.const if _to_bool(parser, var_name, value) else action.default

    if isinstance(action, _AppendAction):
        # Paths, which may contain spaces or commas, so only the platform's path
        # separator splits them.
        return [
            _value(parser, action, var_name, item)
            for item in value.split(os.pathsep)
            if item
        ]

    if action.nargs in ("*", "+"):
        items = [
            _value(parser, action, var_name, item)
            for item in re.split(r"[,\s]+", value.strip())
            if item
        ]
        if (action.nargs == "+") and (not items):
            parser.error(f"{var_name} needs at least one value")

        return items

    if (action.nargs == "?") and (action.const is not None) and (not value):
        # An empty value means the flag with no value, e.g. --zeroconf
        return action.const

    return _value(parser, action, var_name, value)


def _value(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    var_name: str,
    value: str,
) -> Any:
    """Apply the action's type conversion and check it against its choices."""
    typed_value: Any = value
    if callable(action.type):
        try:
            typed_value = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f"{var_name}: invalid value {value!r} ({exc})")

    if (action.choices is not None) and (typed_value not in action.choices):
        choices = ", ".join(str(choice) for choice in action.choices)
        parser.error(f"{var_name}: {value!r} is not one of: {choices}")

    return typed_value


def _to_bool(parser: argparse.ArgumentParser, var_name: str, value: str) -> bool:
    """Interpret a variable that stands in for a flag."""
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True

    if normalized in _FALSE_VALUES:
        return False

    parser.error(f"{var_name}: {value!r} is not a true/false value")
    raise AssertionError  # unreachable: parser.error exits


def _warn_unknown_vars(environ: Mapping[str, str], known_vars: set) -> None:
    """Warn about variables that look like options but match none of them.

    A misspelled variable is otherwise silent: the server starts with a default
    the user believed they had changed.
    """
    for name in sorted(environ):
        if not name.startswith(ENV_PREFIX):
            continue

        base_name = name[: -len(FILE_SUFFIX)] if name.endswith(FILE_SUFFIX) else name
        if base_name not in known_vars:
            _LOGGER.warning("Ignoring unknown environment variable: %s", name)
