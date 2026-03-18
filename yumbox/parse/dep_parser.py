#!/usr/bin/env python3
"""
Comprehensive dependency analyzer - find what you actually use vs what you have
"""

import os
import re
import subprocess
import sys


def get_all_imports(script_path, script_args):
    """Get all imports by running script in subprocess with import tracking

    Args:
        script_path: Path to the Python script to analyze
        script_args: List of arguments to pass to the script
    Returns:
        List of imported module names
    """
    tracker_code = f"""
import sys
import os
import builtins

# Set up argv
sys.argv = {[script_path] + script_args}

# Change to script directory
script_dir = os.path.dirname(os.path.abspath("{script_path}"))
if script_dir:
    os.chdir(script_dir)

# Track imports
imported = set()
original_import = builtins.__import__

def track_import(name, globals=None, locals=None, fromlist=(), level=0):
    imported.add(name.split(".")[0])
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = track_import

# Run the script
try:
    with open("{script_path}", "r") as f:
        code = f.read()
    exec(compile(code, "{script_path}", "exec"))
except SystemExit:
    pass  # Script called sys.exit(), that's fine
except Exception as e:
    print(f"Script execution error: {{e}}", file=sys.stderr)

# Output results
for imp in sorted(imported):
    print(imp)
"""

    # Run in subprocess
    result = subprocess.run(
        [sys.executable, "-c", tracker_code],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

    if result.stderr:
        print("⚠️  Errors during execution:")
        print(result.stderr)

    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def get_installed_packages():
    """Get all installed packages from pip freeze

    Returns:
        Set of normalized package names
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )

    if result.returncode != 0:
        print("❌ Error running pip freeze:")
        print(result.stderr)
        return set()

    # Parse pip freeze output (package==version)
    packages = set()
    for line in result.stdout.strip().split("\n"):
        if line and "==" in line:
            package_name = line.split("==")[0].lower().replace("_", "-")
            packages.add(package_name)
        elif line and not line.startswith("#") and not line.startswith("-e"):
            # Handle other formats
            clean_name = line.split("==")[0].split("@")[0].lower().replace("_", "-")
            if clean_name:
                packages.add(clean_name)

    return packages


def parse_requirements_txt(filepath):
    """Parse requirements.txt file

    Args:
        filepath: Path to requirements.txt file

    Returns:
        Set of normalized package names
    """
    packages = set()
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Extract package name (before ==, >=, etc.)
                    package = (
                        re.split(r"[><=!]", line)[0].strip().lower().replace("_", "-")
                    )
                    if package:
                        packages.add(package)
    except FileNotFoundError:
        print(f"❌ Requirements file not found: {filepath}")
    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")

    return packages


def parse_pyproject_toml(filepath):
    """Parse pyproject.toml file

    Args:
        filepath: Path to pyproject.toml file

    Returns:
        Set of normalized package names
    """
    packages = set()
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("❌ Need tomllib or tomli to parse pyproject.toml files")
            return packages

    try:
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        # Check different sections where dependencies might be
        deps_sections = [
            data.get("project", {}).get("dependencies", []),
            data.get("tool", {}).get("poetry", {}).get("dependencies", {}),
            data.get("build-system", {}).get("requires", []),
        ]

        for deps in deps_sections:
            if isinstance(deps, list):
                for dep in deps:
                    if isinstance(dep, str):
                        package = (
                            re.split(r"[><=!]", dep)[0]
                            .strip()
                            .lower()
                            .replace("_", "-")
                        )
                        if package:
                            packages.add(package)
            elif isinstance(deps, dict):
                for package in deps.keys():
                    if package != "python":  # Skip python version spec
                        packages.add(package.lower().replace("_", "-"))

    except FileNotFoundError:
        print(f"❌ pyproject.toml not found: {filepath}")
    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")

    return packages


def normalize_package_name(name):
    """Normalize package names for comparison

    Args:
        name: Import or package name

    Returns:
        Normalized package name
    """
    mappings = {
        "cv2": "opencv-python",
        "PIL": "pillow",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "bs4": "beautifulsoup4",
        "dateutil": "python-dateutil",
        "magic": "python-magic",
        "psutil": "psutil",
        "requests": "requests",
        "urllib3": "urllib3",
        "certifi": "certifi",
        "charset_normalizer": "charset-normalizer",
        "idna": "idna",
    }

    normalized = name.lower().replace("_", "-")
    return mappings.get(normalized, normalized)


def analyze_dependencies(
    script_path: str,
    script_args: str | list[str] = None,
    requirements_files: str | list[str] = None,
    pyproject_files: str | list[str] = None,
):
    """Analyze script dependencies and print detailed report

    Args:
        script_path: Path to Python script to analyze
        script_args: Arguments to pass to script (string or list)
        requirements_files: Path(s) to requirements.txt file(s) (string or list)
        pyproject_files: Path(s) to pyproject.toml file(s) (string or list)
    """
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    # Normalize inputs to lists
    if script_args is None:
        script_args = []
    elif isinstance(script_args, str):
        import shlex

        script_args = shlex.split(script_args)

    if requirements_files is None:
        requirements_files = []
    elif isinstance(requirements_files, str):
        requirements_files = [requirements_files]

    if pyproject_files is None:
        pyproject_files = []
    elif isinstance(pyproject_files, str):
        pyproject_files = [pyproject_files]

    print("🔍 Finding imports from script execution...")
    imports = get_all_imports(script_path, script_args)
    imports = [imp for imp in imports if imp]  # Remove empty lines

    print("📦 Getting installed packages...")
    installed_packages = get_installed_packages()

    # Parse requirement files
    requirements_packages = set()
    if requirements_files:
        for req_file in requirements_files:
            print(f"📄 Parsing {req_file}...")
            requirements_packages.update(parse_requirements_txt(req_file))

    # Parse pyproject.toml files
    pyproject_packages = set()
    if pyproject_files:
        for proj_file in pyproject_files:
            print(f"📄 Parsing {proj_file}...")
            pyproject_packages.update(parse_pyproject_toml(proj_file))

    # Normalize import names
    import_packages = set()
    for imp in imports:
        normalized = normalize_package_name(imp)
        import_packages.add(normalized)

    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔬 Found {len(imports)} unique imports from script")
    print(f"💾 Found {len(installed_packages)} installed packages (pip freeze)")
    if requirements_files:
        print(f"📋 Found {len(requirements_packages)} packages in requirements files")
    if pyproject_files:
        print(f"⚙️  Found {len(pyproject_packages)} packages in pyproject files")

    # Find intersections
    actually_used = import_packages.intersection(installed_packages)

    print(f"\n✅ THIRD-PARTY PACKAGES ACTUALLY USED ({len(actually_used)}):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for pkg in sorted(actually_used):
        print(f"  • {pkg}")

    # Imports not found in pip freeze (probably stdlib)
    not_in_pip = import_packages - installed_packages
    print(f"\n❓ IMPORTS NOT IN PIP FREEZE ({len(not_in_pip)}):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   (Probably standard library or built-in modules)")
    for pkg in sorted(not_in_pip):
        print(f"  • {pkg}")

    # Installed but not used - SHOW ALL
    unused_installed = installed_packages - import_packages
    print(f"\n💾 INSTALLED BUT NOT USED ({len(unused_installed)}):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for pkg in sorted(unused_installed):
        print(f"  • {pkg}")

    # Requirements file analysis
    if requirements_files:
        in_requirements_not_used = requirements_packages - import_packages
        used_not_in_requirements = actually_used - requirements_packages

        print(f"\n📋 REQUIREMENTS FILE ANALYSIS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📝 In requirements but not used ({len(in_requirements_not_used)}):")
        for pkg in sorted(in_requirements_not_used):
            print(f"  • {pkg}")

        print(f"\n🔍 Used but not in requirements ({len(used_not_in_requirements)}):")
        for pkg in sorted(used_not_in_requirements):
            print(f"  • {pkg}")

    # Pyproject file analysis
    if pyproject_files:
        in_pyproject_not_used = pyproject_packages - import_packages
        used_not_in_pyproject = actually_used - pyproject_packages

        print(f"\n⚙️  PYPROJECT FILE ANALYSIS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📝 In pyproject but not used ({len(in_pyproject_not_used)}):")
        for pkg in sorted(in_pyproject_not_used):
            print(f"  • {pkg}")

        print(f"\n🔍 Used but not in pyproject ({len(used_not_in_pyproject)}):")
        for pkg in sorted(used_not_in_pyproject):
            print(f"  • {pkg}")

    # Efficiency metrics
    print(f"\n🎯 DEPENDENCY EFFICIENCY:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if installed_packages:
        efficiency = len(actually_used) / len(installed_packages) * 100
        print(
            f"📈 Using {efficiency:.1f}% of installed packages ({len(actually_used)}/{len(installed_packages)})"
        )

    if requirements_packages:
        req_efficiency = (
            len(actually_used.intersection(requirements_packages))
            / len(requirements_packages)
            * 100
        )
        print(f"📋 Using {req_efficiency:.1f}% of requirements packages")

    if pyproject_packages:
        proj_efficiency = (
            len(actually_used.intersection(pyproject_packages))
            / len(pyproject_packages)
            * 100
        )
        print(f"⚙️  Using {proj_efficiency:.1f}% of pyproject packages")
