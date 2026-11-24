from pathlib import Path
import hashlib,json,platform,subprocess
import torch
R=Path(__file__).resolve().parent
files=list(R.glob('*.py'))+[R/'protocol.json',R/'trial_plan.json']
out={'host':'Apple M5','platform':platform.platform(),'device_neural':'mps' if torch.backends.mps.is_available() else 'cpu','boosting_threads':1,'publication':'local only, no commits or pushes','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R.parent,text=True).strip()}
(R/'source_manifest.json').write_text(json.dumps(out,indent=2))
