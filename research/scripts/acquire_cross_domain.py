
from research_paths import research_path
import io,json,zipfile,hashlib,sys,concurrent.futures
from pathlib import Path
import requests

ROOT=Path(str(research_path('runs/eco_cross_domain_v1')))
OUT=ROOT/'sources';OUT.mkdir(parents=True,exist_ok=True)
class RemoteFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.session=requests.Session()
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=120,stream=True)
        if r.status_code!=206:raise RuntimeError(f'Range unsupported {r.status_code}')
        b=r.content
        assert len(b)==n,(len(b),n,r.headers)
        self.pos+=n;return b
def get(url,path):
    if path.exists():return
    with requests.get(url,stream=True,timeout=180) as r:
        r.raise_for_status()
        with path.with_suffix('.part').open('wb') as f:
            for b in r.iter_content(2**20):f.write(b)
    path.with_suffix('.part').replace(path)
def task(name):
    if name=='fungi':
        r=requests.get('https://zenodo.org/api/records/18612453',timeout=45);r.raise_for_status();d=r.json()
        (OUT/'fungi_metadata.json').write_text(json.dumps(d,indent=2))
        print('FUNGI',json.dumps({'description':d['metadata']['description'],'license':d['metadata'].get('license'),'files':[(x['key'],x['size'],x['links']['self']) for x in d['files']]}),flush=True)
    if name=='diatoms':
        d=json.loads((OUT/'diatoms_metadata.json').read_text());f=d['files'][0]
        z=zipfile.ZipFile(RemoteFile(f['links']['self'],f['size']))
        entries=[{'name':i.filename,'size':i.file_size,'compressed':i.compress_size} for i in z.infolist()]
        (OUT/'diatoms_zip_index.json').write_text(json.dumps(entries,indent=2))
        print('DIATOMS_INDEX',len(entries),json.dumps(entries[:15]),flush=True)
        for i in z.infolist():
            if i.file_size<2**20 and ('readme' in i.filename.lower() or i.filename.endswith('.txt')):
                p=OUT/('diatoms_'+Path(i.filename).name);p.write_bytes(z.read(i));print('SMALL',str(p),flush=True)
    if name=='uav':
        d=json.loads((OUT/'uav_metadata.json').read_text());f=next(x for x in d['files'] if x['key']=='test.zip')
        p=OUT/'uav_test.zip';get(f['links']['self'],p)
        assert hashlib.md5(p.read_bytes()).hexdigest()==f['checksum'].split(':')[-1]
        z=zipfile.ZipFile(p);entries=[{'name':i.filename,'size':i.file_size} for i in z.infolist()]
        (OUT/'uav_zip_index.json').write_text(json.dumps(entries,indent=2));print('UAV_INDEX',entries,flush=True)
    if name=='rellis':
        p=OUT/'rellis_examples.zip';get('https://drive.usercontent.google.com/download?id=1wIig-LCie571DnK72p2zNAYYWeclEz1D&export=download',p)
        z=zipfile.ZipFile(p);print('RELLIS_INDEX',z.namelist(),flush=True)
        z.extractall(OUT/'rellis_examples')
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(task,n):n for n in ['diatoms','uav','fungi','rellis']}
        for job in concurrent.futures.as_completed(jobs):
            try:job.result()
            except Exception as e:print('FAILED',jobs[job],repr(e),flush=True)
