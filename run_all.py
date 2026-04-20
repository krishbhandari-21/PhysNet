"""
run_all.py — Convenience runner for all PhysNet examples.

Usage (from PhysNet/ directory):
    python run_all.py
"""
import subprocess
import sys
import os

EXAMPLES = [
    "examples/run_heat.py",
    "examples/run_burgers.py",
    "examples/run_poisson.py",
]

if __name__ == "__main__":
    python = sys.executable
    base = os.path.dirname(os.path.abspath(__file__))

    for script in EXAMPLES:
        path = os.path.join(base, script)
        print(f"\n{'='*60}")
        print(f"  Running: {script}")
        print(f"{'='*60}\n")
        result = subprocess.run([python, path], cwd=base)
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n\nAll examples completed successfully!")
