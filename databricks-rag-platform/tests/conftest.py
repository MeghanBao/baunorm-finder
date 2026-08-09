"""Make the platform package importable as `src.*` when running pytest from the
databricks-rag-platform/ directory."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
