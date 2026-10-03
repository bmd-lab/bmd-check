"""Observation transports for Agent's fixed read-only commands.

Every observational read is expressed as one fixed argument vector of the form
``["ssh", *options, <host>, <command>]``, where ``<command>`` is built by Agent
itself from fixed templates and already-authorized, shell-quoted paths. The
transport decides only *how* that vector runs:

* SSH (any configured host alias): runs ``ssh`` with ``BatchMode=yes`` added, so
  a missing key or unknown host fails immediately instead of waiting for
  interactive authentication.
* Local (``ssh_host = "local"``): for when Agent runs on the machine it
  observes. The identical ``<command>`` text runs through ``/bin/sh -c`` and
  ``ssh`` is never executed.

Neither transport accepts commands from users: they only execute vectors whose
shape Agent's own acquisition code produced, and callers never pass
``shell=True``. ``"local"`` is a reserved transport selector, not a host name.
"""

from __future__ import annotations

from collections.abc import Sequence
import subprocess
from typing import Any

LOCAL_TRANSPORT = "local"
LOCAL_SHELL = "/bin/sh"
SSH_BATCH_MODE_OPTION = "BatchMode=yes"

# The only ``-o`` options Agent's own code adds to an observational SSH vector.
_PERMITTED_SSH_OPTIONS = ("ConnectTimeout=", "BatchMode=")


class TransportError(ValueError):
    """An observational command did not have the fixed shape Agent produces."""


def is_local_transport(ssh_host: str | None) -> bool:
    return ssh_host == LOCAL_TRANSPORT


def run_ssh_command(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run an observational SSH vector non-interactively (adds ``BatchMode=yes``)."""

    parts = _argument_list(command, kwargs)
    if parts[0] != "ssh":
        raise TransportError("observational SSH runner received a non-SSH command")
    if LOCAL_TRANSPORT in parts[-2:]:
        # Local mode must never fall through to a real ``ssh local ...``.
        raise TransportError(
            'the "local" transport was selected but the SSH runner was used'
        )
    if SSH_BATCH_MODE_OPTION not in parts:
        parts = ["ssh", "-o", SSH_BATCH_MODE_OPTION, *parts[1:]]
    return subprocess.run(parts, **kwargs)


def local_command(command: Sequence[str], kwargs: dict[str, Any] | None = None) -> list[str]:
    """Translate one Agent-built observational vector into its local form."""

    parts = _argument_list(command, kwargs or {})
    if len(parts) < 3 or parts[0] != "ssh" or parts[-2] != LOCAL_TRANSPORT:
        raise TransportError("local transport received an unexpected command shape")
    options = parts[1:-2]
    if len(options) % 2 != 0:
        raise TransportError("local transport received unexpected SSH options")
    for flag, value in zip(options[::2], options[1::2]):
        if flag != "-o" or not value.startswith(_PERMITTED_SSH_OPTIONS):
            raise TransportError("local transport received unexpected SSH options")
    remote_command = parts[-1]
    if not remote_command.strip():
        raise TransportError("local transport received an empty command")
    return [LOCAL_SHELL, "-c", remote_command]


def run_local_command(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run an observational vector on this machine; never executes ``ssh``."""

    return subprocess.run(local_command(command, kwargs), **kwargs)


def observation_runner(ssh_host: str):
    """Return the runner for a configured cluster's transport."""

    return run_local_command if is_local_transport(ssh_host) else run_ssh_command


class LocalObservationSession:
    """Drop-in for ReusableSshSession when observing this machine.

    Same ``runner(role)`` and context-manager interface; nothing is opened or
    cleaned up because no connection exists.
    """

    def __init__(self, *, runner=subprocess.run) -> None:
        self._runner = runner
        self.ssh_host = LOCAL_TRANSPORT

    def __enter__(self) -> "LocalObservationSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def close(self) -> None:
        return None

    def runner(self, role: str):
        from bmd_agent.profiling import profiled_runner

        def run(command: object, **kwargs: Any):
            translated = local_command(command, kwargs)  # type: ignore[arg-type]
            return profiled_runner(self._runner, role=role)(translated, **kwargs)

        return run


def _argument_list(command: Sequence[str], kwargs: dict[str, Any]) -> list[str]:
    if kwargs.get("shell") is True:
        raise TransportError("observational transports do not permit shell=True")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise TransportError("observational transports require an argument sequence")
    parts = [str(item) for item in command]
    if not parts:
        raise TransportError("observational transports received an empty command")
    return parts
