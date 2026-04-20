"""
Root conftest.py — ensures PhysNet/ is on sys.path so all test imports resolve.
"""
import sys
import os

# Add PhysNet root to path so `core`, `physics`, `utils` are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
