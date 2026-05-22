## `parse/` — Dependency Analysis & Import Tracking 🕵️📦

Automatically discover what your script *actually* uses vs. what's installed or declared. Perfect for cleaning up bloated environments, auditing dependencies, and generating minimal `requirements.txt` files.

---

### 🎯 Quick Start

```bash
# Analyze a single script
python -c "from yumbox.parse import analyze_dependencies; analyze_dependencies('train.py')"

# Analyze with arguments and requirement files
python -c "
from yumbox.parse import analyze_dependencies
analyze_dependencies(
    script_path='main.py',
    script_args='--config configs/exp.yaml --gpu 0',
    requirements_files=['requirements.txt', 'requirements-dev.txt'],
    pyproject_files=['pyproject.toml']
)
"
```

#### Sample Output
```
🔍 Finding imports from script execution...
📦 Getting installed packages...
📄 Parsing requirements.txt...
📄 Parsing pyproject.toml...

📊 ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 Found 42 unique imports from script
💾 Found 187 installed packages (pip freeze)
📋 Found 23 packages in requirements files
⚙️  Found 18 packages in pyproject files

✅ THIRD-PARTY PACKAGES ACTUALLY USED (15):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • numpy
  • pandas
  • torch
  • transformers
  • ...

❓ IMPORTS NOT IN PIP FREEZE (27):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   (Probably standard library or built-in modules)
  • os
  • sys
  • re
  • json
  • ...

💾 INSTALLED BUT NOT USED (172):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • aiohttp
  • boto3
  • flask
  • ...

📋 REQUIREMENTS FILE ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 In requirements but not used (8):
  • black
  • flake8
  • pytest

🔍 Used but not in requirements (3):
  • omegaconf
  • safetensors
  • faiss-cpu

🎯 DEPENDENCY EFFICIENCY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Using 8.0% of installed packages (15/187)
📋 Using 65.2% of requirements packages
```

---

### 🧰 Function Reference

#### `analyze_dependencies()` — The Main Event

```python
def analyze_dependencies(
    script_path: str,
    script_args: str | list[str] = None,
    requirements_files: str | list[str] = None,
    pyproject_files: str | list[str] = None,
):
    """
    Analyze a Python script's runtime dependencies and compare against installed/declared packages.

    Args:
        script_path: Path to the Python script to analyze
        script_args: Command-line arguments to pass to the script (string or list)
        requirements_files: Path(s) to requirements.txt file(s) to parse
        pyproject_files: Path(s) to pyproject.toml file(s) to parse

    Returns:
        None (prints report to stdout)
    """
```

##### Parameters Deep Dive

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `script_path` | `str` | Entry point script to execute and trace | `"train.py"`, `"src/main.py"` |
| `script_args` | `str \| list[str]` | CLI args passed to script during analysis | `"--batch 32 --lr 1e-4"` or `["--batch", "32"]` |
| `requirements_files` | `str \| list[str]` | requirements.txt files to compare against | `"requirements.txt"` or `["base.txt", "dev.txt"]` |
| `pyproject_files` | `str \| list[str]` | pyproject.toml files (Poetry/PEP 621) to parse | `"pyproject.toml"` |

##### How It Works
1. **Runtime Import Tracing**: Executes your script in a subprocess with a custom `__import__` hook that logs every top-level module imported
2. **Package Discovery**: Runs `pip freeze` to get installed packages, parses `requirements.txt` and `pyproject.toml` for declared dependencies
3. **Name Normalization**: Maps import names to package names (e.g., `cv2` → `opencv-python`, `PIL` → `pillow`)
4. **Set Analysis**: Computes intersections/differences to produce the report

---

#### Helper Functions (For Advanced Use)

| Function | Purpose | Returns |
|----------|---------|---------|
| `get_all_imports(script_path, script_args)` | Execute script and capture all imported module names | `list[str]` of top-level imports |
| `get_installed_packages()` | Run `pip freeze` and parse output | `set[str]` of normalized package names |
| `parse_requirements_txt(filepath)` | Parse a requirements.txt file | `set[str]` of package names |
| `parse_pyproject_toml(filepath)` | Parse pyproject.toml (Poetry/PEP 621) | `set[str]` of package names |
| `normalize_package_name(name)` | Map import name → PyPI package name | `str` normalized name |

```python
# Example: Manual import tracing
from yumbox.parse import get_all_imports, normalize_package_name

imports = get_all_imports("inference.py", ["--model", "bert-base"])
packages = [normalize_package_name(imp) for imp in imports]
print(f"Runtime dependencies: {packages}")
```

---

### 🔁 Common Patterns

#### Generate Minimal requirements.txt
```python
from yumbox.parse import analyze_dependencies, get_all_imports, normalize_package_name
import subprocess

# 1. Find actually used packages
imports = get_all_imports("train.py", [])
used_packages = {normalize_package_name(imp) for imp in imports if imp}

# 2. Get installed versions for used packages
freeze_output = subprocess.run(
    ["pip", "freeze"], capture_output=True, text=True
).stdout

installed = {}
for line in freeze_output.strip().split("\n"):
    if "==" in line:
        name, version = line.split("==")
        installed[name.lower().replace("_", "-")] = version

# 3. Write minimal requirements
with open("requirements.minimal.txt", "w") as f:
    for pkg in sorted(used_packages):
        if pkg in installed:
            f.write(f"{pkg}=={installed[pkg]}\n")
        else:
            f.write(f"{pkg}\n")  # No version if not found in freeze

print("✅ Created requirements.minimal.txt")
```

#### CI/CD Dependency Audit
```yaml
# .github/workflows/audit.yml
name: Dependency Audit
on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run dependency analysis
        run: |
          python -c "
          from yumbox.parse import analyze_dependencies
          analyze_dependencies(
              script_path='tests/test_imports.py',
              requirements_files=['requirements.txt']
          )
          "
      - name: Fail if unused dependencies > threshold
        run: |
          # Custom logic: parse output and fail if too many unused
          python scripts/check_efficiency.py --threshold 80
```

#### Docker Image Optimization
```dockerfile
# Stage 1: Analyze dependencies
FROM python:3.10-slim as analyzer
WORKDIR /app
COPY . .
RUN pip install yumbox
RUN python -c "
from yumbox.parse import analyze_dependencies
analyze_dependencies('main.py', requirements_files=['requirements.txt'])
" > /tmp/audit.txt

# Stage 2: Build minimal image
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# Use audit.txt to install only what's needed
# Extract package names using awk for robust section boundary detection
RUN PACKAGES=$(awk '
  # Start capturing when we find the target section header
  /✅ THIRD-PARTY PACKAGES ACTUALLY USED/ { capturing=1; next }
  
  # Extract package names from bullet points (format: "  • package_name")
  capturing && /^  • / { 
    sub(/^  • /, "")      # Remove bullet prefix
    if (NF > 0 && $0 !~ /^\.\.\.$/) print  # Skip empty/"..." lines
    next 
  }
  
  # Stop at section boundaries: dashed separator, new emoji header, or blank line
  capturing && (/^━+$/ || /^[❓💾📋⚙️🎯🔍]/ || /^$/) { exit }
' /tmp/audit.txt | xargs) && \
  if [ -n "$PACKAGES" ]; then \
    pip install --no-cache-dir $PACKAGES; \
  else \
    echo "⚠️  No packages extracted from audit.txt" && exit 1; \
  fi
COPY . .
CMD ["python", "main.py"]
```

#### Pre-commit Hook for Dependency Hygiene
```python
# .pre-commit-hooks/check_deps.py
#!/usr/bin/env python3
import sys
from yumbox.parse import analyze_dependencies

def main():
    script = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    
    # Run analysis silently, capture output
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        analyze_dependencies(script, requirements_files=["requirements.txt"])
    
    output = f.getvalue()
    
    # Check efficiency threshold
    for line in output.split("\n"):
        if "Using" in line and "%" in line:
            # Parse: "📈 Using 45.2% of installed packages"
            try:
                pct = float(line.split("%")[0].split()[-1])
                if pct < 50:  # Fail if using <50% of installed
                    print(f"❌ Dependency efficiency too low: {pct:.1f}%")
                    return 1
            except:
                pass
    
    print("✅ Dependency check passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-dependencies
        name: Check dependency efficiency
        entry: python .pre-commit-hooks/check_deps.py
        language: system
        files: ^(main\.py|train\.py|requirements\.txt)$
        pass_filenames: false
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| Script has side effects (writes files, calls APIs) | Wrap analysis in a test environment or mock external calls in the script |
| Dynamic imports (`importlib.import_module`) not traced | The `__import__` hook catches most, but `__import__(var)` with runtime values may slip through |
| Namespace packages (e.g., `google.cloud.*`) | Only top-level module (`google`) is captured; this is intentional for package-level analysis |
| Editable installs (`-e .`) not in pip freeze output | `parse_requirements_txt` skips `-e` lines; consider adding manual mapping if needed |
| Script requires GPU/external resources | Use `script_args` to pass flags that disable heavy initialization during analysis |

#### Pro Tips
```python
# Tip 1: Analyze multiple entry points and merge results
def analyze_project(entry_points: list[str], **kwargs):
    all_imports = set()
    for script in entry_points:
        imports = get_all_imports(script, kwargs.get("script_args", []))
        all_imports.update(imports)
    return [normalize_package_name(imp) for imp in all_imports if imp]

# Tip 2: Export analysis results as JSON for tooling integration
import json

def analyze_to_json(script_path, **kwargs):
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        analyze_dependencies(script_path, **kwargs)
    
    # Parse the text output into structured data
    # (Implement parser based on your output format)
    return json.loads(parse_report(f.getvalue()))

# Tip 3: Compare two environments (dev vs prod)
def diff_environments(script_path, env1_reqs, env2_reqs):
    imports = set(get_all_imports(script_path, []))
    
    used_in_1 = imports & set(env1_reqs)
    used_in_2 = imports & set(env2_reqs)
    
    return {
        "only_in_env1": used_in_1 - used_in_2,
        "only_in_env2": used_in_2 - used_in_1,
        "in_both": used_in_1 & used_in_2,
    }
```

---

### 🔐 Security & Best Practices

```python
# Never run untrusted scripts with analyze_dependencies!
# The function executes the script, which could run arbitrary code.

def safe_analyze(script_path, allowed_imports=None, timeout=30):
    """
    Safer wrapper with import allowlist and execution timeout.
    """
    import signal
    
    # Validate script path
    if not os.path.abspath(script_path).startswith(os.getcwd()):
        raise ValueError("Script must be in current working directory")
    
    # Set timeout
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Script execution exceeded {timeout}s")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        imports = get_all_imports(script_path, [])
        
        # Filter by allowlist if provided
        if allowed_imports:
            imports = [imp for imp in imports if imp in allowed_imports]
        
        return imports
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
```

```python
# Log analysis for audit trails
from yumbox.config import setup_logger

# One-time setup: colorful console + file logging + library capture
logger = setup_logger(
    name="yumbox.parse.audit",
    level="INFO",
    path="./logs",                    # Auto-saves timestamped log files
    capture_libs=["subprocess"],      # Also log subprocess calls for debugging
    suppress_libs=["urllib3", "certifi"],  # Silence noisy HTTP libs
)

def audited_analyze(script_path, **kwargs):
    logger.info(f"🔍 Starting dependency analysis: {script_path}")
    
    try:
        result = analyze_dependencies(script_path, **kwargs)
        logger.info(f"✅ Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise
```

---

### 📦 Module Organization

```
parse/
├── __init__.py      # Exports analyze_dependencies
└── dep_parser.py    # Core dependency analysis logic
```

> 💡 **Why runtime tracing?** Static analysis (AST parsing) misses dynamic imports, conditional imports, and imports inside functions. By actually *running* the script with a custom `__import__` hook, we capture what's truly loaded — almost no false negatives.

---

> 🍱 **Yumbox Philosophy**: Dependency management in a large project, where refactors have been postponed can quickly become hell when you need to ship for production ASAP before hitting deadline. A simple manual grep, is burdensome, wastes valuable time, and still fails to capture the dependencies you actually need. If your Docker image grows to 2GB because of unused dependencies, that's catastrophic.

Happy auditing! 🚀 If your use case needs a new output format or integration point, the code is intentionally modular — extend away.
