import sys
from pathlib import Path

# Allow tests to import from src/ without a PYTHONPATH prefix
sys.path.insert(0, str(Path(__file__).parent / "src"))
