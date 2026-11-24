from pathlib import Path
import ast
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parent

def main():
    allowed = [p for p in ROOT.rglob('*') if p.is_file() and not any(part in {'.venv','__pycache__','.pytest_cache','data','models','.git'} for part in p.relative_to(ROOT).parts) and p.suffix not in {'.zip','.pyc','.db','.sqlite','.onnx','.bin'} and p.name != '.env']
    for p in allowed:
        if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
    destination=ROOT/'cyberlife_os_v2.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in allowed: z.write(p,Path('cyberlife_os')/p.relative_to(ROOT))
    print(destination)
if __name__ == '__main__': main()
