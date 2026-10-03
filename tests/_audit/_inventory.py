"""One-shot inventory of the test suite and the source public API.

Read-only analysis: parses tests/ and src/chessrl/ with the ast module and emits
a structured JSON inventory so the gap analysis can be done offline.
"""
import ast
import json
import pathlib

ROOT = pathlib.Path("C:/Users/Trinity/Programmes/Chess-player")
TESTS = ROOT / "tests"
SRC = ROOT / "src" / "chessrl"


def docstring_first_lines(node, n=3):
    ds = ast.get_docstring(node)
    if not ds:
        return ""
    lines = [ln.strip() for ln in ds.strip().splitlines() if ln.strip()]
    return " ".join(lines[:n])


def decorators_of(node):
    out = []
    for d in node.decorator_list:
        if isinstance(d, ast.Call):
            out.append(ast.unparse(d.func))
        elif isinstance(d, ast.Attribute):
            out.append(ast.unparse(d))
        elif isinstance(d, ast.Name):
            out.append(d.id)
    return out


def imports_of(tree):
    mods = set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
            for a in n.names:
                names.add(f"{n.module}.{a.asname or a.name}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name)
    return mods, names


def test_functions(tree):
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            funcs.append({
                "name": node.name,
                "doc": docstring_first_lines(node),
                "decos": decorators_of(node),
                "lineno": node.lineno,
            })
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                    funcs.append({
                        "name": f"{node.name}.{sub.name}",
                        "doc": docstring_first_lines(sub),
                        "decos": decorators_of(sub),
                        "lineno": sub.lineno,
                    })
    return funcs


def public_api(tree, filename):
    # honour __all__ if present
    all_names = None
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    try:
                        all_names = sorted({ast.literal_eval(e) for e in n.value.elts})
                    except Exception:
                        all_names = None
    pub = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if n.name.startswith("_"):
                continue
            pub.append(n.name)
    if all_names is not None:
        pub = all_names
    return sorted(set(pub))


def main():
    # --- source public API ---
    src_api = {}
    for f in sorted(SRC.glob("*.py")):
        if f.name == "__init__.py":
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception as e:
            src_api[f.name] = {"error": str(e)}
            continue
        src_api[f.name] = {"public": public_api(tree, f.name)}

    # --- tests ---
    test_files = {}
    for f in sorted(TESTS.glob("*.py")):
        try:
            src_text = f.read_text(encoding="utf-8")
            tree = ast.parse(src_text)
        except Exception as e:
            test_files[f.name] = {"error": str(e)}
            continue
        mods, names = imports_of(tree)
        chesrl_imports = sorted(m for m in names if m.startswith("chessrl"))
        test_files[f.name] = {
            "functions": test_functions(tree),
            "n_lines": len(src_text.splitlines()),
            "imports_chessrl": chesrl_imports,
            "imports_modules": sorted(m for m in mods if m.startswith("chessrl")),
        }

    out = {
        "src_api": src_api,
        "tests": test_files,
    }
    pathlib.Path("C:/Users/Trinity/Programmes/Chess-player/_inventory.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    # quick console summary
    print(f"source modules: {len(src_api)}")
    print(f"test files: {len(test_files)}")
    total = sum(len(t.get('functions', [])) for t in test_files.values())
    print(f"total test functions: {total}")
    for fn, t in test_files.items():
        print(f"  {fn:28s} {len(t.get('functions', [])):4d} tests  {t.get('n_lines',0):5d} lines")


if __name__ == "__main__":
    main()
