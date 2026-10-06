"""Finish this one pilot run, then package portable results (not dense caches)."""

from research_paths import research_path
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

root=Path(str(research_path('')))
out=root/'runs/moin_multiscale_pilot_v1'
deadline=time.monotonic()+2400
last=-1
while True:
    if (out/'failure.json').exists():raise RuntimeError((out/'failure.json').read_text())
    count=len(list(out.glob('*/metrics.json')))
    if count!=last:
        print('PILOT',count,'/20 complete',flush=True);last=count
    if count==20:break
    if time.monotonic()>deadline:raise TimeoutError('Pilot did not finish within40 minutes')
    time.sleep(5)
subprocess.run([sys.executable,str(root/'work/inspect_multiscale_pilot.py'),'--all'],check=True)
subprocess.run([sys.executable,str(root/'work/build_multiscale_report.py'),str(out)],check=True)
archive=root/'work/multiscale-pilot-report.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix!='.npy' and p.name!='labels.npz':tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
print('REPORT_READY',archive,archive.stat().st_size,flush=True)
