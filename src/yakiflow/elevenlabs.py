"""ElevenLabs speech-to-text support.

This module hosts everything specific to the ElevenLabs backends. Only the
API-key resolution lives here so far; the transcribers themselves follow.
"""

from __future__ import annotations

import os
import subprocess

from .config import Settings


KEYRING_SERVICE = "yakiflow"
KEYRING_ENTRY = "elevenlabs"
API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"

_NO_KEY_ERROR = (
    "no ElevenLabs API key is configured; provide one via "
    "-c elevenlabs.api-key=…, the ELEVENLABS_API_KEY environment variable, "
    "elevenlabs.api-key / api-key-file / api-key-command in a configuration "
    "file, or 'yakiflow secret set elevenlabs' (needs yakiflow[keyring])"
)


def _configured_slot(settings: Settings) -> str | None:
    """Name which spelling of the config API-key slot is set, if any."""
    elevenlabs = settings.elevenlabs
    if elevenlabs.api_key is not None:
        return "api_key"
    if elevenlabs.api_key_file is not None:
        return "api_key_file"
    if elevenlabs.api_key_command is not None:
        return "api_key_command"
    return None


def _slot_key(settings: Settings) -> str | None:
    """Resolve the configured slot to its key.

    A configured source that fails to produce a key is an error, never a
    silent fall-through to a lower-priority source: the key that would be used
    instead is not the one the user pointed at.
    """
    elevenlabs = settings.elevenlabs
    slot = _configured_slot(settings)
    if slot is None:
        return None
    if slot == "api_key":
        key = (elevenlabs.api_key or "").strip()
        if not key:
            raise ValueError("elevenlabs.api-key is empty")
        return key
    if slot == "api_key_file":
        assert elevenlabs.api_key_file is not None
        try:
            text = elevenlabs.api_key_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"cannot read elevenlabs.api-key-file {elevenlabs.api_key_file}: {exc}"
            ) from exc
        key = text.rstrip()
        if not key:
            raise ValueError(
                f"elevenlabs.api-key-file {elevenlabs.api_key_file} is empty"
            )
        return key
    assert elevenlabs.api_key_command is not None
    result = subprocess.run(
        elevenlabs.api_key_command,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            "elevenlabs.api-key-command exited with status "
            f"{result.returncode}{suffix}"
        )
    key = result.stdout.strip()
    if not key:
        raise ValueError("elevenlabs.api-key-command printed no API key")
    return key


def _keyring_key() -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_ENTRY) or None
    except Exception:
        # An unavailable Secret Service backend means this layer has no key,
        # the same as when the optional dependency is not installed at all.
        return None


def elevenlabs_api_key(settings: Settings) -> str:
    """Resolve the API key at time of use; never stored in ``Settings``.

    Priority, highest first: the config slot when the command line set it, the
    environment variable, the config slot from configuration files, the OS
    keyring.
    """
    from_cli = settings.elevenlabs.api_key_from_cli
    if from_cli:
        key = _slot_key(settings)
        if key:
            return key
    env = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if env:
        return env
    if not from_cli:
        key = _slot_key(settings)
        if key:
            return key
    key = _keyring_key()
    if key:
        return key
    raise ValueError(_NO_KEY_ERROR)


def elevenlabs_api_key_source(settings: Settings) -> str:
    """Name where the key would come from, without touching the key itself."""
    slot_labels = {
        "api_key": "config",
        "api_key_file": "file",
        "api_key_command": "command",
    }
    slot = _configured_slot(settings)
    if slot is not None and settings.elevenlabs.api_key_from_cli:
        return slot_labels[slot]
    if os.environ.get(API_KEY_ENV_VAR, "").strip():
        return "env"
    if slot is not None:
        return slot_labels[slot]
    if _keyring_key():
        return "keyring"
    return "not set"
