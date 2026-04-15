from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _lookup_key(payload, dotted_key: str):
    value = payload
    for part in str(dotted_key).split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _resolve_default(repo_root: Path, field: dict):
    default_cfg = field.get("default", {})
    kind = default_cfg.get("kind", "static")
    if kind == "static":
        return default_cfg.get("value", "")
    if kind == "json":
        path = repo_root / default_cfg["path"]
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        value = _lookup_key(payload, default_cfg["key"])
        transform = default_cfg.get("transform")
        if transform == "len":
            return len(value or [])
        if transform == "csv":
            return ",".join(str(item) for item in (value or []))
        return value
    raise ValueError(f"Unsupported default kind: {kind}")


def _normalize_value(raw_value: str, field: dict):
    field_type = field.get("type", "str")
    if field_type == "int":
        value = int(str(raw_value).strip())
        minimum = field.get("min")
        maximum = field.get("max")
        if minimum is not None and value < int(minimum):
            raise ValueError(f"值不能小于 {minimum}")
        if maximum is not None and value > int(maximum):
            raise ValueError(f"值不能大于 {maximum}")
        return str(value)
    if field_type == "csv":
        parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
        if not parts:
            raise ValueError("请输入至少一个值")
        return ",".join(parts)
    if field_type == "bool":
        text = str(raw_value).strip().lower()
        truthy = {"1", "true", "yes", "y", "on", "enable", "enabled", "是", "开", "开启"}
        falsy = {"0", "false", "no", "n", "off", "disable", "disabled", "否", "关", "关闭"}
        if text in truthy:
            return "1"
        if text in falsy:
            return "0"
        raise ValueError("请输入开启/关闭、是/否、1/0 或 yes/no")
    return str(raw_value).strip()


def _display_value(value: str, field: dict, lang: str) -> str:
    field_type = field.get("type", "str")
    if field_type == "bool":
        enabled = str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enable", "enabled"}
        if lang == "en":
            return "Enabled" if enabled else "Disabled"
        return "开启" if enabled else "关闭"
    return str(value)


def _estimate_suffix(field: dict, profile: str, value: str, lang: str) -> str:
    estimate = field.get("estimate")
    if not estimate:
        return ""
    profiles = estimate.get("profiles", [])
    if profiles and profile not in profiles:
        return ""
    if estimate.get("kind") == "sim_hours":
        try:
            steps = int(str(value).strip())
            seconds_per_step = float(estimate["seconds_per_step"])
        except (KeyError, TypeError, ValueError):
            return ""
        hours = steps * seconds_per_step / 3600.0
        note = estimate.get("note", "") if lang == "zh" else "estimated using the default simulation step length"
        if lang == "en":
            if note:
                return f", about {hours:.1f} simulation hours ({seconds_per_step:.1f} sec/step, {note})"
            return f", about {hours:.1f} simulation hours ({seconds_per_step:.1f} sec/step)"
        if note:
            return f"，约 {hours:.1f} 小时仿真时间（{seconds_per_step:.1f} 秒/步，{note}）"
        return f"，约 {hours:.1f} 小时仿真时间（{seconds_per_step:.1f} 秒/步）"
    return ""


def _prompt_field(label: str, default_value: str, field: dict, profile: str, lang: str):
    default_prefix = "Default" if lang == "en" else "默认"
    display_default = _display_value(default_value, field, lang)
    print(f"{label} [{default_prefix}: {display_default}{_estimate_suffix(field, profile, default_value, lang)}]", file=sys.stderr)
    description = field.get("zh_description")
    if description:
        prefix = "Description" if lang == "en" else "说明"
        print(f"{prefix}: {description}", file=sys.stderr)
    if lang == "en":
        print("Press Enter to keep the default; type q to cancel.", file=sys.stderr)
    else:
        print("直接回车使用默认值；输入 q 取消本次配置。", file=sys.stderr)
    user_input = input("> ").strip()
    if user_input.lower() == "q":
        raise KeyboardInterrupt
    return default_value if user_input == "" else user_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Start quick config helper")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="zh")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    schema = _load_schema(Path(args.schema))
    profile_entry = schema.get("profiles", {}).get(args.profile, [])
    if isinstance(profile_entry, dict):
        profile_fields = profile_entry.get("fields", [])
        profile_title = (
            profile_entry.get(f"{args.lang}_title")
            or profile_entry.get("zh_title")
            or args.profile
        )
        profile_intro = (
            profile_entry.get(f"{args.lang}_intro")
            or profile_entry.get("zh_intro")
            or ""
        )
    else:
        profile_fields = profile_entry
        profile_title = "快速配置"
        profile_intro = ""
    fields = schema.get("fields", {})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_lines: list[str] = []
    resolved_summary: list[tuple[str, str]] = []
    if profile_fields:
        print("============================================================", file=sys.stderr)
        print(profile_title, file=sys.stderr)
        if profile_intro:
            print(profile_intro, file=sys.stderr)
        else:
            if args.lang == "en":
                print("Press Enter to keep the default; type q to cancel.", file=sys.stderr)
            else:
                print("直接回车使用默认值；输入 q 取消本次配置。", file=sys.stderr)
        print("============================================================", file=sys.stderr)
    try:
        for field_id in profile_fields:
            field = fields[field_id]
            default_value = _resolve_default(repo_root, field)
            label = field.get(f"{args.lang}_label") or field.get("zh_label") or field_id
            if args.lang == "en":
                if "en_description" in field:
                    field = dict(field)
                    field["zh_description"] = field["en_description"]
            while True:
                try:
                    raw_value = _prompt_field(label, str(default_value), field, args.profile, args.lang)
                    value = _normalize_value(raw_value, field)
                    estimate_suffix = _estimate_suffix(field, args.profile, value, args.lang)
                    if estimate_suffix:
                        prefix = "[Tip]" if args.lang == "en" else "[提示]"
                        print(f"{prefix} {label}{estimate_suffix}", file=sys.stderr)
                    resolved_lines.append(f'{field["env"]}={value}')
                    resolved_summary.append((label, value))
                    break
                except ValueError as exc:
                    prefix = "[Error]" if args.lang == "en" else "[错误]"
                    print(f"{prefix} {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        return 2

    if resolved_summary:
        print("============================================================", file=sys.stderr)
        print("Execution Summary" if args.lang == "en" else "本次执行配置摘要", file=sys.stderr)
        for label, value in resolved_summary:
            field = next(
                (
                    field_cfg
                    for field_cfg in fields.values()
                    if field_cfg.get("zh_label") == label or field_cfg.get("en_label") == label
                ),
                None,
            )
            print(f"- {label}: {_display_value(value, field or {}, args.lang)}", file=sys.stderr)
        print("============================================================", file=sys.stderr)

    output_path.write_text("\n".join(resolved_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
