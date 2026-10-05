#!/usr/bin/env python3
"""Generate report.md from validated research JSONs.

Reads outline.yaml (item order), fields.yaml (field structure, categories),
and every *.json in results/. Skips uncertain/empty values. Output: report.md
"""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report.md"

TOC_FIELDS = ["literature_verdict", "scale_regime"]
SKIP_KEYS = {"_source_file", "uncertain"}
LONG_TEXT = 100


def load_yaml(path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def field_structure(fields_yaml):
    """Return (ordered [(category, [field names])], field -> category)."""
    ordered = []
    f2c = {}
    for cat in fields_yaml.get("field_categories", []):
        names = [f["name"] for f in cat.get("fields", [])]
        ordered.append((cat["category"], names))
        for n in names:
            f2c[n] = cat["category"]
    return ordered, f2c


def item_order(outline):
    """Item ids in outline order; fall back handled by caller."""
    return [it["id"] for it in outline.get("items", []) if "id" in it]


def find_field(data, name, category_keys):
    """Lookup order: top level -> category sub-dict -> traverse nested dicts."""
    if name in data:
        return data[name]
    for ck in category_keys:
        sub = data.get(ck)
        if isinstance(sub, dict) and name in sub:
            return sub[name]
    stack = [v for v in data.values() if isinstance(v, dict)]
    while stack:
        obj = stack.pop()
        if name in obj:
            return obj[name]
        stack.extend(v for v in obj.values() if isinstance(v, dict))
    return None


def is_uncertain(name, value, uncertain_names):
    if name in uncertain_names:
        return True
    if value is None:
        return True
    if isinstance(value, str) and ("[uncertain]" in value or not value.strip()):
        return True
    return False


def fmt_value(value, indent=0):
    """Format a field value as markdown lines."""
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}**:")
                lines.extend(fmt_value(v, indent + 1))
            else:
                lines.append(f"{pad}- **{k}**: {v}")
    elif isinstance(value, list):
        if all(not isinstance(x, (dict, list)) for x in value):
            joined = ", ".join(str(x) for x in value)
            if len(value) <= 4 and len(joined) <= LONG_TEXT:
                lines.append(f"{pad}{joined}")
            else:
                lines.extend(f"{pad}- {x}" for x in value)
        else:
            for x in value:
                if isinstance(x, dict):
                    row = " | ".join(f"{k}: {v}" for k, v in x.items()
                                     if not isinstance(v, (dict, list)))
                    lines.append(f"{pad}- {row}")
                    for k, v in x.items():
                        if isinstance(v, (dict, list)):
                            lines.extend(fmt_value({k: v}, indent + 1))
                else:
                    lines.append(f"{pad}- {x}")
    else:
        text = str(value)
        if len(text) > LONG_TEXT:
            lines.extend(f"{pad}> {ln}" for ln in text.split("\n"))
        else:
            lines.append(f"{pad}{text}")
    return lines


def anchor(title):
    a = title.lower()
    a = re.sub(r"[^\w\s-]", "", a)
    return re.sub(r"\s+", "-", a.strip())


def main():
    outline = load_yaml(ROOT / "outline.yaml")
    fields_yaml = load_yaml(ROOT / "fields.yaml")
    categories, f2c = field_structure(fields_yaml)
    defined = set(f2c)
    cat_keys = {c: [c, c.replace("_", " ")] for c, _ in categories}

    json_by_id = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                  for p in RESULTS.glob("*.json")}
    order = [i for i in item_order(outline) if i in json_by_id]
    order += sorted(set(json_by_id) - set(order))

    toc, body = [], []
    toc.append("## Table of Contents\n")
    for idx, iid in enumerate(order, 1):
        data = json_by_id[iid]
        uncertain_names = set(data.get("uncertain") or [])
        summary = []
        for tf in TOC_FIELDS:
            v = find_field(data, tf, cat_keys.get(f2c.get(tf, ""), []))
            if not is_uncertain(tf, v, uncertain_names):
                summary.append(str(v).split("\n")[0])
        suffix = f" — {' | '.join(summary)}" if summary else ""
        toc.append(f"{idx}. [{iid}](#{anchor(iid)}){suffix}")

        body.append(f"\n## {iid}\n")
        shown = set()
        for cat, names in categories:
            block = []
            for name in names:
                v = find_field(data, name, cat_keys[cat])
                if is_uncertain(name, v, uncertain_names):
                    continue
                shown.add(name)
                block.append(f"**{name}**")
                block.extend(fmt_value(v, 1))
                block.append("")
            if block:
                body.append(f"### {cat.replace('_', ' ').title()}")
                body.extend(block)

        # Extra fields not defined in fields.yaml
        extras = []
        def collect_extras(obj, prefix=""):
            for k, v in obj.items():
                if k in SKIP_KEYS or k in shown or k in defined:
                    continue
                if k in {c for c, _ in categories} or k in {c.replace("_", " ") for c, _ in categories}:
                    if isinstance(v, dict):
                        collect_extras(v, prefix)
                    continue
                if is_uncertain(k, v, uncertain_names):
                    continue
                extras.append((k, v))
        collect_extras(data)
        if extras:
            body.append("### Other Info")
            for k, v in extras:
                body.append(f"**{k}**")
                body.extend(fmt_value(v, 1))
                body.append("")

        if uncertain_names:
            body.append("### Uncertain (skipped above)")
            body.extend(f"- {n}" for n in sorted(uncertain_names))
            body.append("")

    header = [
        f"# {outline.get('topic', 'Research Report')}",
        "",
        f"Compiled from {len(order)} validated result files "
        f"(`results/*.json`, 100% field coverage). Fields marked uncertain "
        f"by the research agents are omitted from the body and listed per item.",
        "",
    ]
    OUT.write_text("\n".join(header + toc + body) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(order)} items)")


if __name__ == "__main__":
    main()
