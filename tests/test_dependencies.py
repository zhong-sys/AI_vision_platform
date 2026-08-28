import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXCLUDED_DIRS = {".git", ".venv", ".local_archive", "build", "dist", "tests"}


def _is_excluded(path):
    return any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts)


def _module_name(path):
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_modules():
    result = {}
    for path in ROOT.rglob("*.py"):
        if _is_excluded(path):
            continue
        result[_module_name(path)] = path
    return result


def _resolve_absolute(name, modules):
    candidates = [name]
    parts = name.split(".")
    candidates.extend(".".join(parts[:index]) for index in range(len(parts) - 1, 0, -1))
    return next((candidate for candidate in candidates if candidate in modules), None)


def _resolve_import(current, node, imported_name, modules):
    if node.level:
        package_parts = current.split(".")[:-1]
        if node.level > len(package_parts) + 1:
            return None
        base = package_parts[: len(package_parts) - node.level + 1]
        if node.module:
            base.extend(node.module.split("."))
        if imported_name:
            base.extend(imported_name.split("."))
        return _resolve_absolute(".".join(base), modules)

    module = node.module or ""
    if module and imported_name:
        target = _resolve_absolute(module + "." + imported_name, modules)
        if target:
            return target
    name = node.module or imported_name
    return _resolve_absolute(name, modules)


def _build_graph(modules):
    graph = {name: set() for name in modules}
    duplicate_imports = []
    for current, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    key = (alias.name, alias.asname)
                    if key in seen:
                        duplicate_imports.append((path, node.lineno, alias.name))
                    seen.add(key)
                    target = _resolve_absolute(alias.name, modules)
                    if target and target != current:
                        graph[current].add(target)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    key = ("." * node.level + (node.module or ""), alias.name, alias.asname)
                    if key in seen:
                        duplicate_imports.append((path, node.lineno, key[0]))
                    seen.add(key)
                    target = _resolve_import(current, node, alias.name, modules)
                    if target and target != current:
                        graph[current].add(target)
    return graph, duplicate_imports


def _find_cycles(graph):
    state = {}
    stack = []
    cycles = []

    def visit(node):
        state[node] = 1
        stack.append(node)
        for child in graph[node]:
            if state.get(child) == 1:
                start = stack.index(child)
                cycles.append(stack[start:] + [child])
            elif state.get(child) != 2:
                visit(child)
        stack.pop()
        state[node] = 2

    for node in graph:
        if state.get(node) is None:
            visit(node)
    return cycles


class DependencyGraphTests(unittest.TestCase):
    def test_no_obvious_local_import_cycles(self):
        graph, duplicate_imports = _build_graph(_local_modules())
        self.assertFalse(_find_cycles(graph))
        self.assertFalse(duplicate_imports)


if __name__ == "__main__":
    unittest.main()
