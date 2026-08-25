from __future__ import annotations

from typing import Any, Mapping

from yakiflow.config import Settings, build_settings


def make_settings(values: Mapping[str, Any] | None = None, /, **groups: Any) -> Settings:
    """Build unresolved ``Settings`` from one nested mapping.

    Accepts either a single nested dict or keyword arguments per top-level
    field/group, e.g. ``make_settings(target_language="zh-CN",
    agent={"backend": "codex"})``.
    """
    merged: dict[str, Any] = dict(values or {})
    merged.update(groups)
    return build_settings(merged)
