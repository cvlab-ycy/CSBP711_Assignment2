"""Create private source/results and full evidence archives without publishing."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCUMENTS=['README.md','START_HERE.md','SUBMISSION_CHECKLIST.md','PRESENTATION_GUIDE.md','CORRECTION_HISTORY.md','requirements.txt','environment.yml','.gitignore']
SKIP={'__pycache__','.pytest_cache','matplotlib-cache','.DS_Store'}
def keep(p):
    return p.is_file() and not any(part in SKIP or part.startswith('._') for part in p.parts) and p.suffix not in ['.pyc','.zip','.gz'] and p.name!='.experiment.lock'
def files_in(folder):return [p for p in (ROOT/folder).rglob('*') if keep(p)]
def main():
    target=ROOT/'deliverables';target.mkdir(exist_ok=True)
    artifacts=[target/'CSBP711_Report.docx',target/'CSBP711_Slides_Final.pptx']
    assert all(p.exists() for p in artifacts)
    shared=[ROOT/f for f in DOCUMENTS]
    for folder in ['src','config','scripts','tests','reports']:shared+=files_in(folder)
    shared+=artifacts
    source=sorted(set(shared))
    full=source+files_in('outputs')+files_in('data/prepared')+files_in('archive')+files_in('logs')+list((ROOT/'data/raw').glob('*.gz'))
    manifest={}
    for title,files in [('CSBP711_Source_and_Results.zip',source),('CSBP711_Full_Project.zip',sorted(set(full)))]:
        dest=target/title
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            contents=[]
            for p in files:
                rel=p.relative_to(ROOT).as_posix();z.write(p,'advanced_ai/'+rel)
                contents.append({'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            z.writestr('advanced_ai/PACKAGE_MANIFEST.json',json.dumps({'files':contents},indent=2)+'\n')
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        manifest[title]={'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'files':len(files),'zip_integrity':'passed'}
    for p in artifacts:manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (target/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
