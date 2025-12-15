# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Self-contained packaging for nemo-run.

This module provides :class:`~nemotron.kit.packaging.self_contained_packager.SelfContainedPackager`,
which builds a tarball containing only:

- `main.py`: a single-file script produced by inlining `nemotron.*` imports
- `config.yaml`: the training config (paths should already be rewritten to /nemo_run/code
  by ConfigBuilder.save() before this packager runs)
- Any config files referenced in config.yaml (e.g., blend JSON files)

The AST inliner is intentionally small and conservative:

- Only `package_prefix` imports are inlined (default: `nemotron`).
- External imports are preserved.
- `from nemotron.x import *` is rejected.
"""

from __future__ import annotations

import ast
import os
import tarfile
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_run.core.packaging import Packager


@dataclass(kw_only=True)
class SelfContainedPackager(Packager):
    """Packager that produces a self-contained `main.py` by inlining `nemotron.*` imports.

    Expects config.yaml to already have paths rewritten to /nemo_run/code/... by
    ConfigBuilder.save(). Scans for those paths and includes the corresponding
    local files in the tarball.
    """

    script_path: str
    train_path: str
    inline_package: str = "nemotron"
    remote_code_dir: str = "/nemo_run/code"

    def package(self, path: Path, job_dir: str, name: str) -> str:
        """Create a tarball containing `main.py`, `config.yaml`, and referenced files.

        The train_path config should already have paths rewritten to /nemo_run/code
        by ConfigBuilder.save(). This method scans for those paths and includes
        the corresponding local files in the tarball.

        Args:
            path: Repo root (nemo-run passes cwd for generic packagers).
            job_dir: Local staging directory for nemo-run.
            name: Package name (used for the tarball filename).

        Returns:
            Path to the created tarball.
        """
        from io import BytesIO

        from omegaconf import OmegaConf

        repo_root = Path(path)
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        staging_dir = Path(job_dir) / "code"
        staging_dir.mkdir(parents=True, exist_ok=True)

        script_file = Path(self.script_path)
        if not script_file.is_absolute():
            script_file = repo_root / self.script_path

        # Inline nemotron imports into main.py
        inlined = inline_imports(
            script_file,
            repo_root=repo_root,
            package_prefix=self.inline_package,
        )

        # Load config (already has paths rewritten to /nemo_run/code)
        # Resolve most interpolations, but preserve ${art:...} which requires runtime resolution
        config = OmegaConf.load(self.train_path)
        config_dict = _to_container_preserve_art(config)

        # Scan config for /nemo_run/code paths and collect corresponding local files
        extra_files: list[tuple[str, str]] = []  # (local_path, archive_path)
        self._collect_referenced_files(config_dict, repo_root, extra_files)

        # Build tarball manually to include extra files
        # On macOS, bsdtar can include AppleDouble `._*` entries unless disabled.
        prev = os.environ.get("COPYFILE_DISABLE")
        os.environ["COPYFILE_DISABLE"] = "1"
        try:
            with tarfile.open(output_file, "w:gz") as tf:
                # Add main.py
                main_info = tarfile.TarInfo(name="main.py")
                main_data = inlined.encode("utf-8")
                main_info.size = len(main_data)
                main_info.mode = 0o644
                tf.addfile(main_info, BytesIO(main_data))

                # Add config.yaml (copy from train_path which already has rewritten paths)
                tf.add(self.train_path, arcname="config.yaml")

                # Add extra referenced files with their relative paths
                for local_path, archive_path in extra_files:
                    if Path(local_path).exists():
                        tf.add(local_path, arcname=archive_path)

            return output_file
        finally:
            if prev is None:
                os.environ.pop("COPYFILE_DISABLE", None)
            else:
                os.environ["COPYFILE_DISABLE"] = prev

    def _collect_referenced_files(
        self, obj: Any, repo_root: Path, extra_files: list[tuple[str, str]]
    ) -> None:
        """Recursively scan config for /nemo_run/code paths and collect local files.

        When paths like /nemo_run/code/src/... are found, the corresponding local
        files (repo_root/src/...) are added to extra_files for tarball inclusion.

        Args:
            obj: Config object (dict, list, or scalar)
            repo_root: Local repository root
            extra_files: List to collect (local_path, archive_path) tuples
        """
        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_referenced_files(v, repo_root, extra_files)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_referenced_files(v, repo_root, extra_files)
        elif isinstance(obj, str):
            # Check if it's a /nemo_run/code path
            if obj.startswith(self.remote_code_dir + "/"):
                # Extract relative path after /nemo_run/code/
                rel_path = obj[len(self.remote_code_dir) + 1:]
                local_path = repo_root / rel_path
                # If the local file exists, add to extra_files
                if local_path.exists() and local_path.is_file():
                    extra_files.append((str(local_path), rel_path))


def _to_container_preserve_art(config: Any) -> Any:
    """Convert OmegaConf config to container, resolving all interpolations except ${art:...}.

    The ${art:...} interpolations require runtime artifact resolution and must be preserved.
    All other interpolations (like ${run.wandb.project}) are resolved at packaging time.

    Args:
        config: OmegaConf DictConfig or ListConfig

    Returns:
        Plain dict/list with interpolations resolved except ${art:...}
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    def _convert(obj: Any) -> Any:
        if isinstance(obj, DictConfig):
            result = {}
            for key in obj.keys():
                # Get the raw value to check if it's an ${art:...} interpolation
                raw_node = OmegaConf.to_container(obj, resolve=False).get(key)
                if isinstance(raw_node, str) and "${art:" in raw_node:
                    # Preserve ${art:...} interpolations as-is
                    result[key] = raw_node
                else:
                    # Resolve and convert recursively
                    try:
                        result[key] = _convert(obj[key])
                    except Exception:
                        # If resolution fails, keep raw value
                        result[key] = raw_node
            return result
        elif isinstance(obj, ListConfig):
            return [_convert(item) for item in obj]
        else:
            return obj

    return _convert(config)


def _read_text(path: Path) -> str:
    """Read Python source text using `tokenize.open` to honor PEP-263 encodings."""
    with tokenize.open(path) as f:
        return f.read()


def _node_source(lines: list[str], node: ast.AST) -> str:
    """Extract the exact source segment for an AST node (requires lineno metadata)."""
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    start = int(getattr(node, "lineno")) - 1
    end = int(getattr(node, "end_lineno"))
    return "".join(lines[start:end])


def _is_nemotron_import(node: ast.AST, *, package_prefix: str) -> bool:
    """Return True if the node is an import from `package_prefix` (import/from-import)."""
    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return False
        return node.module == package_prefix or node.module.startswith(package_prefix + ".")
    if isinstance(node, ast.Import):
        return any(
            a.name == package_prefix or a.name.startswith(package_prefix + ".") for a in node.names
        )
    return False


def _resolve_module_path(repo_root: Path, module: str) -> Path:
    """Resolve a module name to a file path under a `src/` layout.

    Supports both:
    - `src/<module>.py`
    - `src/<module>/__init__.py`
    """
    base = repo_root / "src" / Path(*module.split("."))
    py = base.with_suffix(".py")
    if py.exists():
        return py
    init = base / "__init__.py"
    if init.exists():
        return init
    raise FileNotFoundError(f"Could not resolve module '{module}' under {repo_root / 'src'}")


@dataclass
class _ModuleInline:
    """Extracted content of a module to inline into the entry script."""

    module: str
    external_imports: list[str]
    prelude_assignments: list[str]
    body: str
    exports: set[str]


def _module_exports(mod_ast: ast.Module) -> set[str]:
    """Compute names exported by a module (functions/classes/top-level assigns).

    Dunder names are excluded.
    """
    exports: set[str] = set()
    for node in mod_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    exports.add(t.id)
    return {n for n in exports if n and not (n.startswith("__") and n.endswith("__"))}


def _parse_module_for_inlining(
    module: str,
    *,
    repo_root: Path,
    package_prefix: str,
) -> tuple[_ModuleInline, list[str]]:
    """Parse a module and split it into (inline block, dependency modules)."""
    path = _resolve_module_path(repo_root, module)
    text = _read_text(path)
    lines = text.splitlines(keepends=True)
    mod_ast = ast.parse(text, filename=str(path))

    external_imports: list[str] = []
    body_parts: list[str] = []
    prelude_assignments: list[str] = []
    dependencies: list[str] = []

    for node in mod_ast.body:
        # Never inline/emit __future__ imports from library modules.
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _is_nemotron_import(node, package_prefix=package_prefix):
                # Track dependency modules and synthesize alias assignments where needed.
                if isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.append(node.module)
                    for alias in node.names:
                        if alias.name == "*":
                            raise ValueError(
                                f"Star import not supported in SelfContainedPackager: {module}"
                            )
                        if alias.asname and alias.asname != alias.name:
                            prelude_assignments.append(f"{alias.asname} = {alias.name}\n")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                        asname = alias.asname
                        if asname:
                            prelude_assignments.append(
                                f"{asname} = __nemotron_namespaces['{alias.name}']\n"
                            )
                continue

            external_imports.append(_node_source(lines, node))
            continue

        body_parts.append(_node_source(lines, node))

    exports = _module_exports(mod_ast)
    return (
        _ModuleInline(
            module=module,
            external_imports=external_imports,
            prelude_assignments=prelude_assignments,
            body="".join(body_parts),
            exports=exports,
        ),
        dependencies,
    )


def inline_imports(
    entry_path: Path,
    *,
    repo_root: Path,
    package_prefix: str = "nemotron",
) -> str:
    """Inline `package_prefix` imports into a single self-contained script.

    The resulting script:
    - removes `import {package_prefix}...` and `from {package_prefix}...` statements
    - appends source for referenced modules (under `repo_root/src/`)
    - synthesizes `types.SimpleNamespace` objects for `import nemotron.x as x` patterns
      so that attribute access continues to work.

    Args:
        entry_path: The script whose `package_prefix` imports should be inlined.
        repo_root: Repo root containing `src/`.
        package_prefix: Package prefix to inline (default: `nemotron`).

    Returns:
        A Python source string suitable for writing to `main.py`.
    """

    entry_text = _read_text(entry_path)
    entry_lines = entry_text.splitlines(keepends=True)
    entry_ast = ast.parse(entry_text, filename=str(entry_path))

    shebang = entry_lines[0] if entry_lines and entry_lines[0].startswith("#!") else ""

    # Preserve module docstring (exact source) if present.
    docstring_src = ""
    if (
        entry_ast.body
        and isinstance(entry_ast.body[0], ast.Expr)
        and isinstance(getattr(entry_ast.body[0], "value", None), ast.Constant)
        and isinstance(getattr(entry_ast.body[0].value, "value", None), str)
    ):
        docstring_src = _node_source(entry_lines, entry_ast.body[0])

    future_imports: list[str] = []
    entry_external_imports: list[str] = []
    entry_body_parts: list[str] = []

    # These are used to generate alias bindings for removed nemotron imports.
    entry_alias_assignments: list[str] = []
    entry_module_aliases: list[tuple[str, str]] = []  # (module, asname)
    entry_dependencies: list[str] = []

    for node in entry_ast.body:
        # Skip shebang/docstring handled above.
        if node is entry_ast.body[0] and docstring_src:
            continue

        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future_imports.append(_node_source(entry_lines, node))
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _is_nemotron_import(node, package_prefix=package_prefix):
                if isinstance(node, ast.ImportFrom) and node.module:
                    entry_dependencies.append(node.module)
                    for alias in node.names:
                        if alias.name == "*":
                            raise ValueError(
                                "Star import not supported in SelfContainedPackager: "
                                f"{entry_path}"
                            )
                        if alias.asname and alias.asname != alias.name:
                            entry_alias_assignments.append(f"{alias.asname} = {alias.name}\n")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            entry_module_aliases.append((alias.name, alias.asname))
                        entry_dependencies.append(alias.name)
                continue

            entry_external_imports.append(_node_source(entry_lines, node))
            continue

        entry_body_parts.append(_node_source(entry_lines, node))

    # DFS inline modules with dependencies-first ordering.
    module_blocks: dict[str, _ModuleInline] = {}
    ordered_modules: list[str] = []

    def visit(module: str, *, stack: set[str]) -> None:
        if module in module_blocks:
            return
        if module in stack:
            return
        stack.add(module)
        mod_inline, deps = _parse_module_for_inlining(
            module,
            repo_root=repo_root,
            package_prefix=package_prefix,
        )
        for dep in deps:
            if dep == package_prefix or dep.startswith(package_prefix + "."):
                visit(dep, stack=stack)
        module_blocks[module] = mod_inline
        ordered_modules.append(module)
        stack.remove(module)

    for dep in entry_dependencies:
        if dep == package_prefix or dep.startswith(package_prefix + "."):
            visit(dep, stack=set())

    need_namespace = bool(ordered_modules)
    namespace_prelude = (
        "import types\n__nemotron_namespaces: dict[str, types.SimpleNamespace] = {}\n"
        if need_namespace
        else ""
    )

    # Generate entry module-alias bindings (these will run after namespaces are built).
    for mod, asname in entry_module_aliases:
        entry_alias_assignments.append(f"{asname} = __nemotron_namespaces['{mod}']\n")

    # Collect and dedupe external imports from entry + modules.
    external_imports: list[str] = []
    seen_imports: set[str] = set()

    def add_imports(import_lines: list[str]) -> None:
        for s in import_lines:
            key = s.strip()
            if not key:
                continue
            if key not in seen_imports:
                external_imports.append(s)
                seen_imports.add(key)

    add_imports(entry_external_imports)
    for mod in ordered_modules:
        add_imports(module_blocks[mod].external_imports)

    # Assemble output.
    out: list[str] = []
    if shebang:
        out.append(shebang)
    if docstring_src:
        out.append(docstring_src)
    out.extend(future_imports)
    out.extend(external_imports)
    if namespace_prelude:
        out.append(namespace_prelude)

    for mod in ordered_modules:
        block = module_blocks[mod]
        out.append(f"# --- begin inlined: {block.module} ---\n")
        out.extend(block.prelude_assignments)
        out.append(block.body)
        exports = sorted(block.exports)
        if exports:
            args = ", ".join(f"{n}={n}" for n in exports)
            out.append(
                f"__nemotron_namespaces['{block.module}'] = types.SimpleNamespace({args})\n"
            )
        else:
            out.append(f"__nemotron_namespaces['{block.module}'] = types.SimpleNamespace()\n")
        out.append(f"# --- end inlined: {block.module} ---\n\n")

    out.extend(entry_alias_assignments)
    out.append("".join(entry_body_parts))

    return "".join(out)


__all__ = ["SelfContainedPackager"]
