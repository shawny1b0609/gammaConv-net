"""Execute the walkthrough from the repository root."""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "examples/walkthrough.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="python3").execute(cwd=str(root))
out = root / "runs/walkthrough.ipynb"
out.parent.mkdir(exist_ok=True)
nbformat.write(notebook, out)
print("Walkthrough executed successfully.")
