"""Download the original public Kaggle archive; verify member fingerprint."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,io
URL='https://www.kaggle.com/api/v1/datasets/download/camnugent/sandp500'
root=Path(__file__).resolve().parents[1]
archive=urllib.request.urlopen(URL).read()
with zipfile.ZipFile(io.BytesIO(archive)) as z:data=z.read('all_stocks_5yr.csv')
digest=hashlib.sha256(data).hexdigest()
manifest=root/'data/source.json'
if manifest.exists():
    expected=json.loads(manifest.read_text())['csv_sha256']
    if digest!=expected:raise RuntimeError(f'Dataset version changed: expected {expected}, got {digest}')
p=root/'data/raw/all_stocks_5yr.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(p,digest)
