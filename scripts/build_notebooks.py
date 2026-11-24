"""Execute compact notebook views; never fit or select models."""
from pathlib import Path
import sys
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parents[1]
views=[('EDA','Data provenance and research history',"from pathlib import Path\nimport json,pandas as pd\nr=Path.cwd()\nprint(json.dumps(json.loads((r/'data/source.json').read_text()),indent=2))\ndisplay(pd.read_csv(r/'results/invalid_raw_rows.csv'))\nprint('The Kaggle audit is historical. The unified comparison uses the frozen Yahoo snapshots described in comparison_2026/results.json.')"),('Modeling','Unified 2026 comparison — retrospective',"from pathlib import Path\nimport pandas as pd,json\nfrom IPython.display import display, Image\nr=Path.cwd()\ndisplay(pd.read_csv(r/'comparison_2026/metrics.csv'))\ndisplay(pd.read_csv(r/'comparison_2026/excess_return_intervals.csv'))\ndisplay(Image(filename=str(r/'comparison_2026/figures/equity_drawdown.png')))\nprint(json.dumps(json.loads((r/'comparison_2026/verification.json').read_text()),indent=2))")]
for name,title,code in views:
    book=n.v4.new_notebook(cells=[n.v4.new_markdown_cell('# '+title+'\n\nA view of frozen experimental outputs. No training, model selection or test tuning occurs here. See README and docs/RESEARCH_HISTORY.md.'),n.v4.new_code_cell(code)])
    book.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
    client=NotebookClient(book,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}})
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'];client.km=km
    client.execute();n.write(book,root/'notebooks'/f'{name}.ipynb');print(name,'executed')
