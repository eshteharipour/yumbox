## `config/` — Logging & Configuration, done right 🎛️

Centralized config management and reusable logger setup — whether you're debugging locally, running batch jobs, or orchestrating distributed training. The colored logger setup provides functionalities to *fix* the logging behaviour of libraries you don't control.

---

### 🎯 Quick Start

```python
from yumbox.config import BFG, setup_logger, execution_wrapper

# 1. Set cache directory (auto-creates if needed)
BFG["cache_dir"] = "/tmp/my_project_cache"

# 2. Setup logger with console + file output + library capture
logger = setup_logger(
    name="my_project",
    level="INFO",                    # Console: INFO and above
    path="/logs/my_project",         # File: save to timestamped log
    file_level="DEBUG",              # File: capture DEBUG+ for forensics
    redirect_prints=True,            # Redirect print() → log file
    capture_libs=["transformers", "torch"],  # Also log these libraries
    suppress_libs=["rustboard_core", "httpcore"],  # Silence noisy deps
    suppress_libs_std=["debug info", "internal:"]  # Filter stdout patterns
)

# 3. Wrap your main logic for auto-timing + error logging
@execution_wrapper
def train():
    logger.info("Starting training...")
    # your code here

# Or use main_run for script entrypoints
if __name__ == "__main__":
    from yumbox.config import main_run
    main_run(train)
```

---

### ⚙️ `BFG` — Global Config Store (Big Fat Global)

A simple config wrapper around a config dict. Use it to store your config globally that dictate the global behaviour of your project. Currently Yumbox uses this to enable and disable cache behaviour:

| Key | Type | Behavior |
|-----|------|----------|
| `cache_dir` | `str \| None` | Auto-creates directory on assignment; used by all `@*cache` decorators |

```python
from yumbox.config import BFG

# Set (directory created automatically)
BFG["cache_dir"] = "~/.cache/yumbox"  # ✅ expands ~, creates dirs

# Get
cache_path = BFG["cache_dir"]  # Returns str or None

# No cache? Decorators gracefully become no-ops
BFG["cache_dir"] = None  # ✅ valid, caching disabled
```

> 💡 **Why `BFG`?** It's a single source of truth. Change it once, and every caching decorator, data loader, and utility respects it.

---

### 🪵 `setup_logger()` — The Logger That Does Everything

A single call to get production-grade logging with zero boilerplate.

#### Parameters

| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `name` | `str` | — | Logger name (e.g., `"my_project"`) |
| `level` | `str \| int` | `logging.INFO` | Console log level |
| `path` | `str` | `""` | Directory for log files (empty = no file logging) |
| `stream` | `str \| io.TextIO` | `"stdout"` | Console output stream (`"stdout"` or `sys.stderr`) |
| `file_level` | `str \| int` | `logging.INFO` | File log level (often `DEBUG` for debugging) |
| `redirect_prints` | `bool` | `False` | Redirect `print()` statements to log file |
| `capture_libs` | `list[str]` | `[]` | Libraries whose logs to capture (e.g., `["torch", "transformers"]`) |
| `capture_all_libs` | `bool` | `False` | Capture *all* library logs via root logger |
| `suppress_libs` | `list[str]` | `[]` | Libraries to silence completely |
| `suppress_libs_std` | `list[str]` | `[]` | Text patterns to filter from stdout/stderr (e.g., `["debug info"]`) |

#### 🎨 Console Output: Color-Coded & Readable

```
2024-06-01 12:34:56 - my_project - INFO - Starting training... (train.py:42)
2024-06-01 12:34:57 - my_project - WARNING - Learning rate too high (train.py:88)
2024-06-01 12:35:01 - my_project - ERROR - NaN detected in loss (train.py:102)
```

| Level | Color | Style |
|-------|-------|-------|
| `DEBUG` | 🔵 Blue | Normal |
| `INFO` | 🔷 Bold Cyan | Normal |
| `WARNING` | 🟡 Yellow | Normal |
| `ERROR` | 🔴 Red | Normal |
| `CRITICAL` | ⚪ White on 🔴 Red BG | Bold |

#### 📁 File Output: Plain & Parseable

Log files use a clean, machine-friendly format:
```
2024-06-01 12:34:56 my_project INFO: Starting training...
2024-06-01 12:34:57 my_project WARNING: Learning rate too high
```

Perfect for `grep`, `jq`, or shipping to ELK/Splunk.

---

### 🔇 Suppressing Noisy Libraries

Some dependencies log *too much*. Silence them cleanly:

```python
# Silence specific libraries entirely
setup_logger(
    name="my_app",
    suppress_libs=["rustboard_core", "httpcore", "PIL"]
)

# Filter stdout/stderr by pattern (great for C++ lib noise)
setup_logger(
    name="my_app",
    redirect_prints=True,
    suppress_libs_std=["debug info", "internal:", "[DEBUG]"]
)

# Capture only what you care about
setup_logger(
    name="my_app",
    capture_libs=["transformers"],  # Log HF libs
    suppress_libs=["urllib3", "botocore"]  # Silence AWS/HTTP noise
)
```

> ⚠️ **Order matters**: `suppress_libs` is applied *before* `capture_libs`, so suppressed libs stay silent even if also listed in `capture_libs`.

---

### 🧰 Utility Functions

#### `execution_wrapper` — Auto-Timing + Error Logging
```python
from yumbox.config import execution_wrapper

@execution_wrapper
def long_running_task():
    # If this raises, traceback is logged + execution time reported
    ...

# Output:
# ERROR: Traceback (most recent call last): ...
# INFO: Execution time: 0h 2m 34.12s
```

#### `main_run` — Script Entry Point Helper
```python
from yumbox.config import main_run

def main():
    # Your script logic
    ...

if __name__ == "__main__":
    main_run(main)  # Same as @execution_wrapper, but without decorator
```

#### `redir_print` — Capture `print()` Output
```python
from yumbox.config import redir_print

def legacy_func():
    print("Hello from old code")

output = redir_print(legacy_func)
assert output == "Hello from old code\n"
```

#### `log_df_info` — Pretty-Print Pandas Stats
```python
import pandas as pd
from yumbox.config import log_df_info

df = pd.DataFrame({"a": [1,2], "b": [3,4]})
log_df_info(df)
# Logs:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2 entries, 0 to 1
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   a       2 non-null      int64
#  1   b       2 non-null      int64
```

#### `inspect_loggers` — Debug Logging Setup
```python
from yumbox.config import inspect_loggers
inspect_loggers()
# Prints: Logger: yumbox.config, Level: 20
#         Logger: torch.distributed, Level: 30
#         ...
```

---

### 🔁 Common Patterns

#### Local Dev vs. Production
```python
# dev.py
setup_logger("my_project", level="DEBUG", path=None)  # Console only, verbose

# prod.py
setup_logger(
    "my_project",
    level="INFO",
    path="/var/log/my_project",
    file_level="DEBUG",  # Keep debug in file for post-mortem
    redirect_prints=True,
    capture_libs=["transformers"]
)
```

#### Jupyter-Friendly Logging
```python
# In notebook cells:
from yumbox.config import setup_logger
logger = setup_logger("notebook", level="INFO", path=None)  # Console only

logger.info("Cell executed")  # Shows in notebook output
```

#### Multi-Module Logging Consistency
```python
# utils/logging.py
from yumbox.config import setup_logger
def get_logger(name: str):
    return setup_logger(name, level="INFO", path="/logs/app")

# In any module:
from utils.logging import get_logger
logger = get_logger(__name__)  # Consistent formatting across project
```

---

> 💡 **Pro tip**: Combine `BFG["cache_dir"]` + `setup_logger(path=...)` at the very start of your script. Now *everything* — caching, logging, error handling — is configured in two lines. That's the Yumbox way. 🍱

Happy configuring! If you need a new config key or log formatter tweak, the code is intentionally simple — extend away.
