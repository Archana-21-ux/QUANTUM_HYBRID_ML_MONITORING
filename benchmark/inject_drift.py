"""CLI wrapper: generate every drift scenario declared in configs/default.yaml.

Usage: uv run python benchmark/inject_drift.py
"""

from quantumguard.drift.injection import generate_all

if __name__ == "__main__":
    for path in generate_all():
        print(f"generated {path}")
