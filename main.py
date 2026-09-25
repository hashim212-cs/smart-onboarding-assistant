"""
Smart Onboarding Assistant — FastAPI Backend
Exposes an intelligent agent that scans a project directory and returns
a structured overview (file tree, tech stack, architecture) for new developers.

Also exposes a /generate-docs endpoint that runs a multi-phase document-
understanding subagent and writes a comprehensive ONBOARDING.md file.
"""

from __future__ import annotations

import os
import re
import json
import textwrap
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Onboarding Assistant",
    description=(
        "An intelligent agent that scans a project directory and generates "
        "a developer-friendly onboarding overview: file structure, tech stack, "
        "and architecture summary. Also generates a full ONBOARDING.md via the "
        "document-understanding subagent."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKIP_DIRS: set[str] = {
    ".git", ".svn", ".hg",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt", "dist", "build", "out",
    "venv", ".venv", "env", ".env", "virtualenv",
    ".idea", ".vscode", ".vs",
    "coverage", ".coverage",
    "target", "vendor", ".terraform",
    "eggs", ".eggs",
}

_SKIP_FILES: set[str] = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
    ".env", ".env.local", ".env.production",
}

# Files whose full text is valuable to read for deep understanding
_READABLE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".go", ".rs", ".rb", ".php", ".cs",
    ".html", ".css", ".scss",
    ".sql", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json", ".env.example",
    ".md", ".rst", ".txt",
    ".tf", ".hcl",
    ".graphql", ".gql",
    ".Dockerfile",
}

_STACK_SIGNALS: dict[str, tuple[str, str]] = {
    "requirements.txt":   ("Python (pip)",        "language/runtime"),
    "pyproject.toml":     ("Python (pyproject)",  "language/runtime"),
    "setup.py":           ("Python (setup.py)",   "language/runtime"),
    "Pipfile":            ("Python (pipenv)",     "language/runtime"),
    "poetry.lock":        ("Poetry",              "package-manager"),
    "package.json":       ("Node.js",             "language/runtime"),
    "package-lock.json":  ("npm",                 "package-manager"),
    "yarn.lock":          ("Yarn",                "package-manager"),
    "pnpm-lock.yaml":     ("pnpm",                "package-manager"),
    "tsconfig.json":      ("TypeScript",          "language/runtime"),
    "next.config.js":     ("Next.js",             "framework"),
    "next.config.ts":     ("Next.js",             "framework"),
    "nuxt.config.js":     ("Nuxt.js",             "framework"),
    "vite.config.ts":     ("Vite",                "build-tool"),
    "vite.config.js":     ("Vite",                "build-tool"),
    "webpack.config.js":  ("Webpack",             "build-tool"),
    "angular.json":       ("Angular",             "framework"),
    "svelte.config.js":   ("Svelte",              "framework"),
    "pom.xml":            ("Maven (Java)",        "build-tool"),
    "build.gradle":       ("Gradle",              "build-tool"),
    "build.gradle.kts":   ("Gradle (Kotlin)",     "build-tool"),
    "go.mod":             ("Go",                  "language/runtime"),
    "Cargo.toml":         ("Rust",                "language/runtime"),
    "Gemfile":            ("Ruby (Bundler)",      "language/runtime"),
    "composer.json":      ("PHP (Composer)",      "language/runtime"),
    "Dockerfile":         ("Docker",              "infrastructure"),
    "docker-compose.yml": ("Docker Compose",      "infrastructure"),
    "docker-compose.yaml":("Docker Compose",      "infrastructure"),
    ".github":            ("GitHub Actions",      "ci-cd"),
    "Jenkinsfile":        ("Jenkins",             "ci-cd"),
    ".circleci":          ("CircleCI",            "ci-cd"),
    ".travis.yml":        ("Travis CI",           "ci-cd"),
    "serverless.yml":     ("Serverless Framework","infrastructure"),
    "serverless.yaml":    ("Serverless Framework","infrastructure"),
    "schema.prisma":      ("Prisma ORM",          "database"),
    "alembic.ini":        ("Alembic (DB)",        "database"),
    "Makefile":           ("Make",                "build-tool"),
    ".eslintrc.json":     ("ESLint",              "linter"),
    "eslint.config.js":   ("ESLint",              "linter"),
    ".prettierrc":        ("Prettier",            "formatter"),
    "jest.config.js":     ("Jest",                "testing"),
    "jest.config.ts":     ("Jest",                "testing"),
    "vitest.config.ts":   ("Vitest",              "testing"),
    "pytest.ini":         ("pytest",              "testing"),
    "conftest.py":        ("pytest",              "testing"),
}

_EXT_TO_LANG: dict[str, str] = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JavaScript", ".tsx": "TypeScript",
    ".java": "Java", ".kt": "Kotlin", ".go": "Go",
    ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".cs": "C#", ".fs": "F#", ".cpp": "C++", ".c": "C",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".sql": "SQL", ".sh": "Shell", ".bash": "Shell",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON",
    ".toml": "TOML", ".md": "Markdown", ".rst": "reStructuredText",
    ".tf": "Terraform", ".hcl": "HCL",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FileNode(BaseModel):
    name: str
    path: str
    kind: str = Field(..., description="'file' or 'directory'")
    children: list["FileNode"] = Field(default_factory=list)
    size_bytes: int | None = None
    extension: str | None = None

FileNode.model_rebuild()


class TechStackItem(BaseModel):
    name: str
    category: str
    detected_via: str


class LanguageStat(BaseModel):
    language: str
    file_count: int
    line_count: int


class ArchitectureOverview(BaseModel):
    entry_points: list[str]
    config_files: list[str]
    test_directories: list[str]
    api_indicators: list[str]
    notable_patterns: list[str]


class OnboardingReport(BaseModel):
    project_root: str
    total_files: int
    total_directories: int
    file_tree: FileNode
    tech_stack: list[TechStackItem]
    language_stats: list[LanguageStat]
    architecture: ArchitectureOverview
    summary: str


class DocsGenerationResult(BaseModel):
    output_path: str
    project_root: str
    files_read: int
    sections_generated: list[str]
    message: str

# ---------------------------------------------------------------------------
# Phase 1 — File tree
# ---------------------------------------------------------------------------

def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".")


def _build_file_tree(root: Path, rel_root: Path) -> tuple[FileNode, int, int]:
    total_files = 0
    total_dirs = 0
    children: list[FileNode] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        entries = []

    for entry in entries:
        rel_path = str(entry.relative_to(rel_root)).replace("\\", "/")
        if entry.is_dir():
            if _should_skip_dir(entry.name):
                continue
            child_node, cf, cd = _build_file_tree(entry, rel_root)
            child_node.name = entry.name
            child_node.path = rel_path
            child_node.kind = "directory"
            children.append(child_node)
            total_files += cf
            total_dirs += 1 + cd
        elif entry.is_file():
            if entry.name in _SKIP_FILES:
                continue
            ext = entry.suffix.lower() or None
            size = None
            try:
                size = entry.stat().st_size
            except OSError:
                pass
            children.append(FileNode(
                name=entry.name, path=rel_path, kind="file",
                size_bytes=size, extension=ext,
            ))
            total_files += 1

    node = FileNode(
        name=root.name,
        path=str(root.relative_to(rel_root)).replace("\\", "/") if root != rel_root else ".",
        kind="directory",
        children=children,
    )
    return node, total_files, total_dirs

# ---------------------------------------------------------------------------
# Phase 2 — Tech stack detection
# ---------------------------------------------------------------------------

def _detect_tech_stack(root: Path) -> list[TechStackItem]:
    detected: dict[str, TechStackItem] = {}

    def _check(path: Path, depth: int = 0) -> None:
        if depth > 2:
            return
        try:
            entries = list(path.iterdir())
        except PermissionError:
            return
        for entry in entries:
            name = entry.name
            if entry.is_dir() and _should_skip_dir(name):
                continue
            if name in _STACK_SIGNALS:
                display, category = _STACK_SIGNALS[name]
                if display not in detected:
                    detected[display] = TechStackItem(
                        name=display, category=category, detected_via=name
                    )
            for pattern, (display, category) in _STACK_SIGNALS.items():
                if "*" in pattern:
                    ext = pattern.lstrip("*")
                    if name.endswith(ext) and display not in detected:
                        detected[display] = TechStackItem(
                            name=display, category=category, detected_via=name
                        )
            if entry.is_dir() and depth < 2 and not _should_skip_dir(name):
                _check(entry, depth + 1)

    _check(root)

    req_file = root / "requirements.txt"
    if req_file.exists():
        try:
            reqs = req_file.read_text(encoding="utf-8", errors="ignore").lower()
            for pkg, label, cat in [
                ("fastapi",     "FastAPI",      "framework"),
                ("flask",       "Flask",        "framework"),
                ("django",      "Django",       "framework"),
                ("sqlalchemy",  "SQLAlchemy",   "database"),
                ("celery",      "Celery",       "task-queue"),
                ("redis",       "Redis",        "database"),
                ("httpx",       "HTTPX",        "http-client"),
                ("boto3",       "AWS SDK",      "cloud"),
                ("openai",      "OpenAI",       "ai/ml"),
                ("langchain",   "LangChain",    "ai/ml"),
                ("torch",       "PyTorch",      "ai/ml"),
                ("tensorflow",  "TensorFlow",   "ai/ml"),
                ("pandas",      "Pandas",       "data"),
                ("numpy",       "NumPy",        "data"),
                ("uvicorn",     "Uvicorn",      "server"),
                ("pydantic",    "Pydantic",     "validation"),
            ]:
                if pkg in reqs and label not in detected:
                    detected[label] = TechStackItem(
                        name=label, category=cat, detected_via="requirements.txt"
                    )
        except OSError:
            pass

    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            for pkg, label, cat in [
                ("react",       "React",        "framework"),
                ("vue",         "Vue.js",       "framework"),
                ("svelte",      "Svelte",       "framework"),
                ("express",     "Express",      "framework"),
                ("@nestjs/core","NestJS",       "framework"),
                ("graphql",     "GraphQL",      "api"),
                ("prisma",      "Prisma ORM",   "database"),
                ("mongoose",    "Mongoose",     "database"),
                ("typeorm",     "TypeORM",      "database"),
                ("socket.io",   "Socket.IO",    "realtime"),
                ("axios",       "Axios",        "http-client"),
            ]:
                if pkg in all_deps and label not in detected:
                    detected[label] = TechStackItem(
                        name=label, category=cat, detected_via="package.json"
                    )
        except (OSError, json.JSONDecodeError):
            pass

    return sorted(detected.values(), key=lambda x: (x.category, x.name))

# ---------------------------------------------------------------------------
# Phase 3 — Language stats
# ---------------------------------------------------------------------------

def _count_language_stats(root: Path) -> list[LanguageStat]:
    stats: dict[str, dict[str, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fname in filenames:
            if fname in _SKIP_FILES:
                continue
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            lang = _EXT_TO_LANG.get(ext)
            if lang is None:
                if fname == "Dockerfile":
                    lang = "Dockerfile"
                else:
                    continue
            if lang not in stats:
                stats[lang] = {"files": 0, "lines": 0}
            stats[lang]["files"] += 1
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                stats[lang]["lines"] += text.count("\n") + 1
            except OSError:
                pass
    return sorted(
        [LanguageStat(language=l, file_count=v["files"], line_count=v["lines"])
         for l, v in stats.items()],
        key=lambda s: s.line_count, reverse=True,
    )

# ---------------------------------------------------------------------------
# Phase 4 — Architecture detection
# ---------------------------------------------------------------------------

def _entry_point_score(name: str) -> int:
    priorities = [
        "main.py", "app.py", "server.py", "run.py", "manage.py",
        "index.ts", "index.js", "server.ts", "server.js",
        "main.go", "main.rs", "main.java", "app.go",
    ]
    try:
        return len(priorities) - priorities.index(name)
    except ValueError:
        return 0


def _detect_architecture(root: Path) -> ArchitectureOverview:
    entry_names = {
        "main.py", "app.py", "server.py", "run.py", "manage.py",
        "index.ts", "index.js", "server.ts", "server.js",
        "main.go", "main.rs", "main.java", "app.go",
    }
    config_names = {
        "config.py", "config.ts", "config.js", "settings.py",
        "appsettings.json", "application.yml", "application.yaml",
        ".env.example", "pyproject.toml", "package.json",
        "Makefile", "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
    }
    test_dir_names = {"tests", "test", "__tests__", "spec", "specs", "e2e"}
    api_pattern_names = {"routes", "routers", "controllers", "api", "endpoints", "views", "handlers"}

    entry_points: list[str] = []
    config_files: list[str] = []
    test_dirs: list[str] = []
    api_dirs: list[str] = []
    patterns: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        rel = Path(dirpath).relative_to(root)
        for dname in dirnames:
            dname_lower = dname.lower()
            rel_dir = str(rel / dname).replace("\\", "/")
            if dname_lower in test_dir_names:
                test_dirs.append(rel_dir)
            if dname_lower in api_pattern_names:
                api_dirs.append(rel_dir)
            if dname_lower in {"models", "model"}:
                patterns.add("MVC-style models layer")
            if dname_lower in {"views", "templates"}:
                patterns.add("MVC-style views/templates layer")
            if dname_lower in {"controllers", "routes", "routers"}:
                patterns.add("Controller/Router pattern")
            if dname_lower in {"services", "service"}:
                patterns.add("Service layer pattern")
            if dname_lower in {"repositories", "repo", "repos"}:
                patterns.add("Repository pattern")
            if dname_lower in {"middleware"}:
                patterns.add("Middleware layer")
            if dname_lower in {"schemas", "schema"}:
                patterns.add("Schema/validation layer")
            if dname_lower in {"migrations", "migration"}:
                patterns.add("Database migrations")
            if dname_lower in {"utils", "helpers", "lib", "common", "shared"}:
                patterns.add("Shared utilities/helpers layer")
            if dname_lower in {"components", "pages", "layouts"}:
                patterns.add("Component-based frontend structure")
        for fname in filenames:
            if fname in entry_names:
                entry_points.append(str(rel / fname).replace("\\", "/"))
            if fname in config_names:
                config_files.append(str(rel / fname).replace("\\", "/"))
        for fname in filenames:
            if re.search(r"\.(graphql|gql)$", fname):
                patterns.add("GraphQL schema")
                break

    entry_points.sort(key=lambda p: -_entry_point_score(Path(p).name))
    return ArchitectureOverview(
        entry_points=entry_points,
        config_files=config_files,
        test_directories=test_dirs,
        api_indicators=api_dirs,
        notable_patterns=sorted(patterns),
    )

# ---------------------------------------------------------------------------
# Scanning agent orchestrator
# ---------------------------------------------------------------------------

def _build_summary(raw: dict[str, Any]) -> str:
    stack_names = [t["name"] for t in raw["tech_stack"]]
    top_langs = [s["language"] for s in raw["language_stats"][:3]]
    entries = raw["architecture"]["entry_points"]
    patterns = raw["architecture"]["notable_patterns"]
    parts: list[str] = []
    if top_langs:
        parts.append(f"Primary languages: {', '.join(top_langs)}.")
    if stack_names:
        parts.append(f"Tech stack includes: {', '.join(stack_names[:8])}.")
    if entries:
        parts.append(f"Suggested entry point(s): {', '.join(entries[:3])}.")
    if patterns:
        parts.append(f"Detected architectural patterns: {', '.join(patterns[:5])}.")
    parts.append(
        f"Project contains {raw['total_files']} files across "
        f"{raw['total_directories']} directories."
    )
    return " ".join(parts)


def run_onboarding_agent(target: str) -> OnboardingReport:
    """
    Intelligent agent that scans `target` directory and returns a full
    onboarding report covering file structure, tech stack, and architecture.
    """
    root = Path(target).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {target!r}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {target!r}")

    tree, total_files, total_dirs = _build_file_tree(root, root)
    tech_stack = _detect_tech_stack(root)
    lang_stats = _count_language_stats(root)
    architecture = _detect_architecture(root)

    raw: dict[str, Any] = {
        "total_files": total_files,
        "total_directories": total_dirs,
        "tech_stack": [t.model_dump() for t in tech_stack],
        "language_stats": [s.model_dump() for s in lang_stats],
        "architecture": architecture.model_dump(),
    }
    summary = _build_summary(raw)

    return OnboardingReport(
        project_root=str(root),
        total_files=total_files,
        total_directories=total_dirs,
        file_tree=tree,
        tech_stack=tech_stack,
        language_stats=lang_stats,
        architecture=architecture,
        summary=summary,
    )

# ---------------------------------------------------------------------------
# Document-understanding subagent
# ---------------------------------------------------------------------------

# Max bytes to read per file — keeps the doc concise without truncating small files
_MAX_FILE_BYTES = 8_000

# Files that are always worth reading in full for deep understanding
_PRIORITY_FILES = {
    "main.py", "app.py", "server.py", "run.py",
    "requirements.txt", "package.json", "pyproject.toml",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "README.md", "CONTRIBUTING.md",
    "alembic.ini", "conftest.py", "pytest.ini",
    "tsconfig.json", "vite.config.ts", "vite.config.js",
    "next.config.js", "next.config.ts",
    ".env.example",
}


def _collect_source_files(root: Path) -> list[Path]:
    """
    Return all readable source files, sorted so priority files come first.
    """
    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fname in filenames:
            if fname in _SKIP_FILES:
                continue
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            if ext in _READABLE_EXTENSIONS or fname in _PRIORITY_FILES:
                collected.append(fpath)

    def _sort_key(p: Path) -> tuple[int, str]:
        is_priority = 0 if p.name in _PRIORITY_FILES else 1
        return (is_priority, str(p))

    return sorted(collected, key=_sort_key)


def _read_file_excerpt(path: Path) -> str:
    """Read up to _MAX_FILE_BYTES of a file, appending a truncation notice if needed."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    truncated = len(raw) > _MAX_FILE_BYTES
    text = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... [truncated — file is {len(raw):,} bytes total]"
    return text


def _render_file_tree_text(node: FileNode, prefix: str = "", is_last: bool = True) -> str:
    """Render a FileNode tree as an ASCII tree string."""
    connector = "└── " if is_last else "├── "
    icon = "📁 " if node.kind == "directory" else ""
    lines = [f"{prefix}{connector}{icon}{node.name}"]
    if node.kind == "directory" and node.children:
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension
        for i, child in enumerate(node.children):
            lines.append(
                _render_file_tree_text(child, child_prefix, i == len(node.children) - 1)
            )
    return "\n".join(lines)


def _infer_file_purpose(path: Path, content: str) -> str:
    """
    Apply heuristic rules to produce a one-line purpose description for a file.
    Falls back to extension-based descriptions.
    """
    name = path.name
    stem = path.stem.lower()
    ext = path.suffix.lower()
    first_lines = content[:400].lower()

    # FastAPI / Starlette patterns
    if ext == ".py":
        if "fastapi()" in first_lines or "from fastapi import" in first_lines:
            return "FastAPI application — defines routes and app factory"
        if "@app.route" in first_lines or "from flask import" in first_lines:
            return "Flask application — defines routes and app factory"
        if "django.conf" in first_lines or "urlpatterns" in first_lines:
            return "Django configuration or URL routing"
        if "sqlalchemy" in first_lines and ("base" in stem or "model" in stem):
            return "SQLAlchemy ORM model definitions"
        if "pytest" in first_lines or stem.startswith("test_") or stem.endswith("_test"):
            return "Test module (pytest)"
        if stem == "conftest":
            return "pytest shared fixtures and configuration"
        if stem in {"settings", "config", "configuration"}:
            return "Application configuration / settings"
        if stem in {"schemas", "schema"}:
            return "Pydantic / serialization schema definitions"
        if stem in {"utils", "helpers", "common"}:
            return "Shared utility functions"
        if stem in {"main", "app", "server", "run"}:
            return "Application entry point"
        if "celery" in first_lines:
            return "Celery task definitions"
        if "alembic" in first_lines:
            return "Alembic database migration"
        if "argparse" in first_lines or "click" in first_lines:
            return "CLI script"
        return "Python module"

    if ext in {".ts", ".tsx"}:
        if "react" in first_lines or "jsx" in first_lines:
            return "React TypeScript component"
        if "express" in first_lines or "router" in first_lines:
            return "Express/Node router (TypeScript)"
        if "nestjs" in first_lines or "@module" in first_lines:
            return "NestJS module"
        return "TypeScript module"

    if ext in {".js", ".jsx"}:
        if "react" in first_lines:
            return "React JavaScript component"
        if "express" in first_lines:
            return "Express/Node router"
        return "JavaScript module"

    # Config / infra files
    if name == "Dockerfile":
        return "Docker image build instructions"
    if name in {"docker-compose.yml", "docker-compose.yaml"}:
        return "Docker Compose multi-service orchestration"
    if name == "requirements.txt":
        return "Python package dependencies"
    if name == "package.json":
        return "Node.js project manifest and dependencies"
    if name in {"pyproject.toml", "setup.py", "setup.cfg"}:
        return "Python project build configuration"
    if ext in {".yaml", ".yml"}:
        if "github" in str(path).replace("\\", "/"):
            return "GitHub Actions CI/CD workflow"
        return "YAML configuration file"
    if ext == ".json":
        return "JSON configuration or data file"
    if ext == ".toml":
        return "TOML configuration file"
    if ext == ".md":
        return "Markdown documentation"
    if ext in {".graphql", ".gql"}:
        return "GraphQL schema definition"
    if ext == ".sql":
        return "SQL script or migration"
    if ext in {".sh", ".bash"}:
        return "Shell script"
    if ext in {".tf", ".hcl"}:
        return "Terraform infrastructure definition"
    if ext == ".env.example" or name == ".env.example":
        return "Environment variable template — copy to .env and fill in values"
    return f"{ext.lstrip('.').upper() or name} file"


def _extract_api_endpoints(content: str, file_ext: str) -> list[str]:
    """Extract HTTP route definitions from Python or JS/TS source."""
    endpoints: list[str] = []
    if file_ext == ".py":
        # FastAPI / Flask style: @app.get("/path") or @router.post("/path")
        for m in re.finditer(
            r'@\w+\.(get|post|put|patch|delete|head|options)\s*\(\s*["\']([^"\']+)["\']',
            content, re.IGNORECASE
        ):
            endpoints.append(f"{m.group(1).upper()} {m.group(2)}")
    elif file_ext in {".ts", ".js"}:
        # Express style: router.get('/path', ...) or app.post('/path', ...)
        for m in re.finditer(
            r'\b(?:router|app)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']',
            content, re.IGNORECASE
        ):
            endpoints.append(f"{m.group(1).upper()} {m.group(2)}")
    return endpoints


def _extract_env_vars(content: str) -> list[str]:
    """Pull environment variable names referenced in source code."""
    found: set[str] = set()
    # os.getenv("VAR") / os.environ["VAR"] / process.env.VAR
    for m in re.finditer(
        r'(?:os\.getenv|os\.environ(?:\[|\.))\s*["\']([A-Z_][A-Z0-9_]+)["\']'
        r'|process\.env\.([A-Z_][A-Z0-9_]+)',
        content
    ):
        name = m.group(1) or m.group(2)
        if name:
            found.add(name)
    # .env.example lines: KEY=value
    for m in re.finditer(r'^([A-Z_][A-Z0-9_]+)\s*=', content, re.MULTILINE):
        found.add(m.group(1))
    return sorted(found)


# ---- Subagent phases -------------------------------------------------------

def _phase_read_files(root: Path) -> dict[str, Any]:
    """
    Subagent Phase A: Read every source file and build a rich content index.
    Returns a dict keyed by relative path with content + metadata.
    """
    source_files = _collect_source_files(root)
    index: dict[str, Any] = {}

    for fpath in source_files:
        rel = str(fpath.relative_to(root)).replace("\\", "/")
        content = _read_file_excerpt(fpath)
        ext = fpath.suffix.lower()
        index[rel] = {
            "content": content,
            "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
            "purpose": _infer_file_purpose(fpath, content),
            "endpoints": _extract_api_endpoints(content, ext),
            "env_vars": _extract_env_vars(content),
        }

    return index


def _phase_understand(
    report: OnboardingReport,
    file_index: dict[str, Any],
) -> dict[str, Any]:
    """
    Subagent Phase B: Cross-reference the scan report and file index to
    produce structured understanding objects for each doc section.
    """
    root = Path(report.project_root)

    # Aggregate all API endpoints across files
    all_endpoints: list[tuple[str, str]] = []
    for rel_path, meta in file_index.items():
        for ep in meta["endpoints"]:
            all_endpoints.append((rel_path, ep))

    # Aggregate all env vars
    all_env_vars: set[str] = set()
    for meta in file_index.values():
        all_env_vars.update(meta["env_vars"])

    # Build a flat file→purpose map (skip tiny/generated files)
    file_purposes: dict[str, str] = {
        rel: meta["purpose"]
        for rel, meta in file_index.items()
        if meta["size_bytes"] > 10
    }

    # Infer setup steps from detected tech stack + files
    setup_steps: list[str] = []
    stack_names = {t.name for t in report.tech_stack}
    lang_names = {s.language for s in report.language_stats}

    if "Python (pip)" in stack_names or "Python" in lang_names:
        setup_steps.append("Create a virtual environment: `python -m venv venv`")
        setup_steps.append(
            "Activate it: `source venv/bin/activate`  (Windows: `venv\\Scripts\\activate`)"
        )
        setup_steps.append("Install dependencies: `pip install -r requirements.txt`")
    if "Poetry" in stack_names:
        setup_steps = ["Install dependencies: `poetry install`"]  # replace pip steps
    if "Node.js" in stack_names:
        setup_steps.append("Install Node dependencies: `npm install`")
    if "Yarn" in stack_names:
        setup_steps = [s for s in setup_steps if "npm install" not in s]
        setup_steps.append("Install Node dependencies: `yarn install`")

    if all_env_vars or (root / ".env.example").exists():
        setup_steps.append("Copy `.env.example` to `.env` and fill in credentials")

    if "Docker" in stack_names:
        setup_steps.append("Build Docker image: `docker build -t <image-name> .`")
    if "Docker Compose" in stack_names:
        setup_steps.append("Start all services: `docker compose up`")

    # Add run step from entry points
    if report.architecture.entry_points:
        ep = report.architecture.entry_points[0]
        if ep.endswith(".py"):
            if "FastAPI" in stack_names or "Uvicorn" in stack_names:
                mod = ep.replace("/", ".").removesuffix(".py")
                setup_steps.append(f"Start the server: `uvicorn {mod}:app --reload`")
            else:
                setup_steps.append(f"Run the application: `python {ep}`")
        elif ep.endswith((".js", ".ts")):
            setup_steps.append("Start the application: `npm start` or `npm run dev`")
        elif ep.endswith(".go"):
            setup_steps.append("Run the application: `go run ./...`")

    if "pytest" in stack_names:
        setup_steps.append("Run tests: `pytest`")

    # Compute dependency list from requirements.txt
    deps: list[str] = []
    req_file = root / "requirements.txt"
    if req_file.exists():
        try:
            deps = [
                line.strip() for line in
                req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except OSError:
            pass

    return {
        "all_endpoints": all_endpoints,
        "all_env_vars": sorted(all_env_vars),
        "file_purposes": file_purposes,
        "setup_steps": setup_steps,
        "dependencies": deps,
    }


def _phase_render_markdown(
    report: OnboardingReport,
    file_index: dict[str, Any],
    understanding: dict[str, Any],
) -> str:
    """
    Subagent Phase C: Render the complete ONBOARDING.md document as a string.
    """
    root = Path(report.project_root)
    project_name = root.name
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    sections.append(textwrap.dedent(f"""\
        # 🚀 {project_name} — Onboarding Guide

        > Auto-generated by **Smart Onboarding Assistant** on {now}.
        > Run `GET /generate-docs` to refresh this file.

        ---
    """))

    # ── Overview ────────────────────────────────────────────────────────────
    sections.append("## 📋 Overview\n")
    sections.append(f"{report.summary}\n")

    # ── Tech Stack ──────────────────────────────────────────────────────────
    if report.tech_stack:
        sections.append("## 🛠️ Tech Stack\n")
        categories: dict[str, list[TechStackItem]] = {}
        for item in report.tech_stack:
            categories.setdefault(item.category, []).append(item)
        for cat in sorted(categories):
            items_str = ", ".join(f"**{i.name}**" for i in categories[cat])
            sections.append(f"- **{cat.title()}**: {items_str}")
        sections.append("")

    # ── Language Breakdown ──────────────────────────────────────────────────
    if report.language_stats:
        sections.append("## 📊 Language Breakdown\n")
        sections.append("| Language | Files | Lines |")
        sections.append("|----------|------:|------:|")
        for s in report.language_stats:
            sections.append(f"| {s.language} | {s.file_count} | {s.line_count:,} |")
        sections.append("")

    # ── Project Structure ───────────────────────────────────────────────────
    sections.append("## 📁 Project Structure\n")
    sections.append("```")
    sections.append(f"{project_name}/")
    for i, child in enumerate(report.file_tree.children):
        sections.append(
            _render_file_tree_text(child, prefix="", is_last=(i == len(report.file_tree.children) - 1))
        )
    sections.append("```\n")

    # ── File Glossary ───────────────────────────────────────────────────────
    file_purposes = understanding["file_purposes"]
    if file_purposes:
        sections.append("## 📖 File Glossary\n")
        sections.append("| File | Purpose |")
        sections.append("|------|---------|")
        for rel_path in sorted(file_purposes):
            purpose = file_purposes[rel_path]
            sections.append(f"| `{rel_path}` | {purpose} |")
        sections.append("")

    # ── Entry Points ────────────────────────────────────────────────────────
    if report.architecture.entry_points:
        sections.append("## 🎯 Entry Points\n")
        for ep in report.architecture.entry_points:
            meta = file_index.get(ep, {})
            purpose = meta.get("purpose", "")
            desc = f" — {purpose}" if purpose else ""
            sections.append(f"- **`{ep}`**{desc}")
        sections.append("")

    # ── API Endpoints ────────────────────────────────────────────────────────
    all_endpoints = understanding["all_endpoints"]
    if all_endpoints:
        sections.append("## 🌐 API Endpoints\n")
        sections.append("| Method & Path | Defined In |")
        sections.append("|---------------|------------|")
        for (src_file, ep) in all_endpoints:
            sections.append(f"| `{ep}` | `{src_file}` |")
        sections.append("")

    # ── Architecture Patterns ────────────────────────────────────────────────
    if report.architecture.notable_patterns or report.architecture.api_indicators:
        sections.append("## 🏗️ Architecture\n")
        if report.architecture.notable_patterns:
            sections.append("**Detected patterns:**\n")
            for p in report.architecture.notable_patterns:
                sections.append(f"- {p}")
            sections.append("")
        if report.architecture.test_directories:
            td = ", ".join(f"`{d}`" for d in report.architecture.test_directories)
            sections.append(f"**Test directories:** {td}\n")
        if report.architecture.api_indicators:
            ai = ", ".join(f"`{d}`" for d in report.architecture.api_indicators)
            sections.append(f"**API/routing directories:** {ai}\n")

    # ── Setup Guide ─────────────────────────────────────────────────────────
    setup_steps = understanding["setup_steps"]
    if setup_steps:
        sections.append("## ⚙️ Setup & Running\n")
        for i, step in enumerate(setup_steps, 1):
            sections.append(f"{i}. {step}")
        sections.append("")

    # ── Dependencies ────────────────────────────────────────────────────────
    deps = understanding["dependencies"]
    if deps:
        sections.append("## 📦 Dependencies\n")
        sections.append("```")
        sections.append("\n".join(deps))
        sections.append("```\n")

    # ── Environment Variables ────────────────────────────────────────────────
    env_vars = understanding["all_env_vars"]
    if env_vars:
        sections.append("## 🔐 Environment Variables\n")
        sections.append(
            "The following environment variables are referenced in the codebase. "
            "Copy `.env.example` → `.env` and populate them before running.\n"
        )
        for var in env_vars:
            sections.append(f"- `{var}`")
        sections.append("")

    # ── Configuration Files ──────────────────────────────────────────────────
    if report.architecture.config_files:
        sections.append("## 🗂️ Configuration Files\n")
        for cf in report.architecture.config_files:
            meta = file_index.get(cf, {})
            purpose = meta.get("purpose", "")
            desc = f" — {purpose}" if purpose else ""
            sections.append(f"- **`{cf}`**{desc}")
        sections.append("")

    # ── Security Notes ───────────────────────────────────────────────────────
    security_file = root / "SECURITY.MD"
    if not security_file.exists():
        security_file = root / "SECURITY.md"
    if security_file.exists():
        sections.append("## 🔒 Security Notes\n")
        sections.append(
            "This project includes security guidelines. "
            f"See [`{security_file.name}`]({security_file.name}) for the full policy.\n"
        )
        sections.append("**Quick reminders:**\n")
        sections.append("- Never commit `.env` files or hardcode credentials")
        sections.append("- Use environment variables for all secrets")
        sections.append("- Run `git diff` before every commit\n")

    # ── Footer ───────────────────────────────────────────────────────────────
    sections.append("---\n")
    sections.append(
        "_This file was auto-generated. Re-run `GET /generate-docs` after "
        "significant code changes to keep it up to date._"
    )

    return "\n".join(sections) + "\n"


def run_docs_agent(target: str, output_filename: str = "ONBOARDING.md") -> DocsGenerationResult:
    """
    Multi-phase document-understanding subagent:

      Phase A — read all source files into a content index
      Phase B — cross-reference with scan report for deep understanding
      Phase C — render the full ONBOARDING.md markdown document
      Phase D — write the file to disk

    Returns metadata about what was generated.
    """
    root = Path(target).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {target!r}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {target!r}")

    # Run the base scanning agent first
    report = run_onboarding_agent(target)

    # Phase A
    file_index = _phase_read_files(root)

    # Phase B
    understanding = _phase_understand(report, file_index)

    # Phase C
    markdown = _phase_render_markdown(report, file_index, understanding)

    # Phase D — write to disk
    output_path = root / output_filename
    output_path.write_text(markdown, encoding="utf-8")

    sections_present = [
        s for s in [
            "Overview", "Tech Stack", "Language Breakdown", "Project Structure",
            "File Glossary", "Entry Points", "API Endpoints", "Architecture",
            "Setup & Running", "Dependencies", "Environment Variables",
            "Configuration Files", "Security Notes",
        ]
        if s.lower().replace(" ", "") in markdown.lower().replace(" ", "")
    ]

    return DocsGenerationResult(
        output_path=str(output_path),
        project_root=str(root),
        files_read=len(file_index),
        sections_generated=sections_present,
        message=(
            f"ONBOARDING.md written to {output_path} "
            f"({len(markdown):,} characters, {len(sections_present)} sections)."
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Smart Onboarding Assistant"}


@app.get(
    "/analyze",
    response_model=OnboardingReport,
    summary="Analyze a project directory",
    description=(
        "Runs the scanning agent against the given directory and returns a "
        "structured report: file tree, tech stack, language stats, and "
        "architecture overview."
    ),
)
def analyze_project(
    path: str = Query(
        default=".",
        description="Absolute or relative path to the project directory to analyze.",
        examples={"default": {"value": "."}},
    ),
) -> OnboardingReport:
    try:
        return run_onboarding_agent(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc


@app.get(
    "/analyze/summary",
    summary="Short onboarding summary",
    description="Returns only the plain-text summary paragraph from the analysis.",
)
def analyze_summary(
    path: str = Query(default=".", description="Path to the project directory."),
) -> dict[str, str]:
    try:
        report = run_onboarding_agent(path)
        return {"summary": report.summary, "project_root": report.project_root}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc


@app.post(
    "/generate-docs",
    response_model=DocsGenerationResult,
    summary="Generate ONBOARDING.md",
    description=(
        "Runs the multi-phase document-understanding subagent against the target "
        "directory. Reads all source files, extracts API endpoints, environment "
        "variables, setup steps, and architectural patterns, then writes a "
        "comprehensive ONBOARDING.md to the project root."
    ),
)
def generate_docs(
    path: str = Query(
        default=".",
        description="Absolute or relative path to the project directory.",
        examples={"default": {"value": "."}},
    ),
    output: str = Query(
        default="ONBOARDING.md",
        description="Name of the output file (written to the project root).",
    ),
) -> DocsGenerationResult:
    try:
        return run_docs_agent(path, output)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Docs agent error: {exc}") from exc


@app.get(
    "/generate-docs/preview",
    response_class=PlainTextResponse,
    summary="Preview ONBOARDING.md (plain text)",
    description="Same as POST /generate-docs but returns the markdown as plain text without writing to disk.",
)
def preview_docs(
    path: str = Query(default=".", description="Path to the project directory."),
) -> str:
    try:
        root = Path(path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {path!r}")
        report = run_onboarding_agent(path)
        file_index = _phase_read_files(root)
        understanding = _phase_understand(report, file_index)
        return _phase_render_markdown(report, file_index, understanding)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview error: {exc}") from exc


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
