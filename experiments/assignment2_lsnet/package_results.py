"""Private lightweight evidence package; leaves all checkpoints in the project."""
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    report=json.loads((ROOT/'reports/VERIFICATION.json').read_text());assert report['status']=='PASS'
    included=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in {'cache','__pycache__','.git','deliverables'} for part in rel.parts):continue
        if p.suffix in {'.pt','.zip','.tar','.gz'}:continue
        if p.name.endswith('.tmp') or p.name.startswith('.'):continue
        included.append(p)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(included)}
    target=ROOT/'deliverables';target.mkdir(exist_ok=True)
    path=target/'Assignment2_evidence.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in included:archive.write(p,str(p.relative_to(ROOT)))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    (target/'README.txt').write_text('Private evidence package: source, reports, raw predictions, learning histories and provenance.\nFull model and optimizer checkpoints remain under outputs/fits on UAEUHPC.\nDataset and existing dependencies remain in sibling project folders; see README.md.\nFinal Word/PowerPoint files require local rendering and visual QA after result retrieval.\n')
    print(f'Private evidence package created: {path}')

if __name__=='__main__':main()
