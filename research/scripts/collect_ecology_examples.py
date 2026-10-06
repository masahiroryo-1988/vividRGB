"""Deterministic source sampling before inference; preserve original bytes and native windows."""

from research_paths import research_path
import argparse,json,zipfile,io,hashlib,re,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import requests
import pyarrow.parquet as pq
import rasterio
from pycocotools import mask as maskutil
from acquire_cross_domain import RemoteFile as OriginalRemoteFile,get
import time
class RemoteFile(OriginalRemoteFile):
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        for attempt in range(8):
            r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=180)
            if r.status_code==206:
                b=r.content;assert len(b)==n;self.pos+=n;return b
            if r.status_code not in [429,500,502,503,504]:r.raise_for_status()
            delay=max(5,min(60,int(r.headers.get('Retry-After','0')) or 5*2**attempt))
            print('HTTP_RETRY',r.status_code,delay,flush=True);time.sleep(delay)
        raise RuntimeError(f'Range download failed: {r.status_code}')
ROOT=Path(str(research_path('runs/eco_collection75_v1')));SRC=ROOT/'sources';SEED=20261002
Image.MAX_IMAGE_PIXELS=None
def sha(p):
    q=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):q.update(b)
    return q.hexdigest()
def archive(domain,key):
    d=json.loads((SRC/f'{domain}_metadata.json').read_text());f=next(x for x in d['files'] if x['key']==key)
    return zipfile.ZipFile(RemoteFile(f['links']['self'],f['size']))
def extract(z,name,p):
    if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    return p
def rng(domain):return np.random.default_rng(SEED+int(hashlib.sha256(domain.encode()).hexdigest()[:6],16))
def prepare(domain,index,original,meta,mask=None,center=None):
    dest=ROOT/f'{domain}_{index:02d}';dest.mkdir(exist_ok=True)
    im=Image.open(original).convert('RGB');w,h=im.size;assert min(w,h)>=1024,(original,im.size)
    sw,sh=min(2048,w)//16*16,min(2048,h)//16*16
    cx,cy=center if center is not None else (w//2,h//2)
    x=max(0,min(w-sw,int(cx-sw//2)));y=max(0,min(h-sh,int(cy-sh//2)));box=(x,y,x+sw,y+sh)
    im.crop(box).save(dest/'input.png');small=im.copy();small.thumbnail((720,720));draw=ImageDraw.Draw(small)
    ratio=small.width/w;draw.rectangle(tuple(int(v*ratio) for v in box),outline='#ffac32',width=3);small.save(dest/'source_overview.jpg',quality=90)
    thumb=im.crop(box);thumb.thumbnail((640,640));thumb.save(dest/'original.jpg',quality=93)
    row={'id':dest.name,'dataset':domain,'original_path':str(original),'original_dimensions':[w,h],'working_dimensions':[sw,sh],'native_window':list(box),'resize_ratio':1,
         'source_sha256':sha(original),'input_sha256':sha(dest/'input.png'),'seed':SEED,'reference_available':mask is not None,
         'biological_occupancy':None,'occupancy_basis':'Not measured: no eligible pixel-level reference.','smallest_resolved_structure':'Not measured; image dimensions do not establish optical resolution.',**meta}
    if mask is not None:
        assert mask.shape==(h,w),(mask.shape,(h,w))
        m=mask[y:y+sh,x:x+sw];Image.fromarray(m.astype(np.uint16)).save(dest/'reference.png');row['reference_sha256']=sha(dest/'reference.png')
        if domain in ['mycelium','bam']:row['biological_occupancy']=float((m>0).mean());row['occupancy_basis']='Fraction of working pixels inside annotated mycelium/crowns.'
        row['reference_ids']=np.unique(m).tolist()
    (dest/'source.json').write_text(json.dumps(row,indent=2,ensure_ascii=False));print('PREPARED',dest.name,[w,h],list(box),flush=True);return row
def finish(domain,rows):
    assert len(rows)==10 and len({r['source_sha256'] for r in rows})==10
    (ROOT/f'{domain}_selection.json').write_text(json.dumps({'dataset':domain,'images':rows},indent=2,ensure_ascii=False))
    can=Image.new('RGB',(2000,860),'#f0f4ed');draw=ImageDraw.Draw(can)
    for i,r in enumerate(rows):
        im=Image.open(ROOT/r['id']/'original.jpg');im.thumbnail((390,385));x=i%5*400;y=i//5*430;can.paste(im,(x,y+28));draw.text((x+5,y+5),r['id']+' '+r.get('stratum',''),fill='black')
    can.save(ROOT/f'{domain}_selection.jpg',quality=94)
def mycelium():
    rows=[];g=rng('mycelium')
    for key,strata in [('labeled-GL.zip',[('GL','testset',4)]),('labeled-GS_PO_TS.zip',[('GS','GS',2),('PO','PO',2),('TS','TS',2)])]:
        z=archive('mycelium',key)
        for species,part,count in strata:
            names=sorted(n for n in z.namelist() if '/'+part+'/image/' in n and n.endswith('.jpg') and '__MACOSX' not in n)
            chosen=[names[i] for i in g.choice(len(names),count,replace=False)]
            for name in chosen:
                orig=extract(z,name,ROOT/'originals/mycelium'/species/Path(name).name)
                prefix=name.replace('/image/','/mask/').rsplit('.',1)[0];mn=next(n for n in z.namelist() if n.rsplit('.',1)[0]==prefix)
                mp=extract(z,mn,ROOT/'originals/mycelium'/species/(Path(mn).stem+'_mask'+Path(mn).suffix))
                m=np.asarray(Image.open(mp));m=m[...,0] if m.ndim==3 else m;m=(m>0).astype(np.uint8)
                boundary=(m>0)&~binary_erosion(m>0);yy,xx=np.nonzero(boundary);j=int(g.integers(len(xx)))
                rows.append(prepare('mycelium',len(rows)+1,orig,{'stratum':species,'source':'https://zenodo.org/records/15224240','license':'CC BY 4.0','source_member':name,'reference_kind':'binary_colony','reference_note':'Outer colony boundary; internal hyphae/voids are not separately annotated.','selection':'Four GL test images and two each GS,PO,TS; uniform fixed-seed sample. Native window centered on a uniformly sampled annotated boundary point.','physical_resolution':None},m,(xx[j],yy[j])))
    finish('mycelium',rows)
def fungal_network():
    rows=[];g=rng('fungal_network')
    for key,count in [('ascomycota.zip',4),('basidiomycota.zip',3),('Zygomycetous3.zip',3)]:
        z=archive('fungal_network',key);names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.tif','.png')))
        for j in g.choice(len(names),count,replace=False):
            name=names[j];orig=extract(z,name,ROOT/'originals/fungal_network'/name)
            rows.append(prepare('fungal_network',len(rows)+1,orig,{'stratum':Path(key).stem,'source':'https://zenodo.org/records/5725751','license':'CC BY 4.0','source_member':name,'selection':'Fixed-seed sample:4 ascomycota,3 basidiomycota,3 zygomycetous images; native center window.','reference_kind':None,'physical_resolution':None}))
    finish('fungal_network',rows)
def bam():
    g=rng('bam');rows=[];a=json.loads((SRC/'bam_archive.json').read_text());z=zipfile.ZipFile(RemoteFile(a['url'],a['size']));used_bounds=[]
    for split,count in [(1,5),(2,5)]:
        q=json.loads((SRC/f'instances_tree_TestSet{split}2023.json').read_text());ims=q['images'];ann={x['id']:[] for x in ims}
        for v in q['annotations']:ann[v['image_id']].append(v)
        selected=0
        for ix in g.permutation(len(ims)):
            r=ims[ix];name=f'coco2048/test2023/Test-Set-{split}/'+r['file_name'];orig=extract(z,name,ROOT/'originals/bam'/r['file_name'])
            with rasterio.open(orig) as ds:bounds=list(ds.bounds);crs=str(ds.crs);res=list(map(abs,ds.res))
            # Reject overlapping georeferenced tiles from the same acquisition area.
            area=r['file_name'].split('_')[0]
            if any(area==a and min(bounds[2],b[2])>max(bounds[0],b[0]) and min(bounds[3],b[3])>max(bounds[1],b[1]) for a,b in used_bounds):continue
            used_bounds.append((area,bounds));mask=np.zeros((2048,2048),np.uint16)
            for k,v in enumerate(sorted(ann[r['id']],key=lambda a:a['area'],reverse=True),1):
                rr=maskutil.frPyObjects(v['segmentation'],2048,2048);binary=maskutil.decode(maskutil.merge(rr));mask[binary>0]=k
            rows.append(prepare('bam',len(rows)+1,orig,{'stratum':f'test{split} / {area}','source':'https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/bamforests','license':'CC BY-NC-SA 4.0','source_member':name,'selection':'Five tiles from each official test split, fixed-seed order, rejecting geospatial overlap within an acquisition area.','reference_kind':'tree_instances','reference_note':'COCO crown polygons; smaller annotation wins overlap; background is0.','annotated_objects':len(ann[r['id']]),'physical_resolution':{'x':res[0],'y':res[1],'units':'CRS map units / pixel','crs':crs},'bounds':bounds},mask))
            selected+=1
            if selected==count:break
        assert selected==count
    finish('bam',rows)
def coralscapes():
    rows=[];g=rng('coralscapes');files=sorted(SRC.glob('coralscapes_test*.parquet'));allrows=[]
    for f in files:allrows.extend(pq.read_table(f).to_pylist())
    groups={}
    for i,v in enumerate(allrows):groups.setdefault(v['image']['path'].split('_')[0],[]).append(i)
    chosen=[]
    for site in g.permutation(sorted(groups)):
        chosen.append(int(g.choice(groups[site])))
        if len(chosen)==10:break
    if len(chosen)<10:chosen+=list(map(int,g.choice([i for i in range(len(allrows)) if i not in chosen],10-len(chosen),replace=False)))
    for i in chosen:
        v=allrows[i];name=v['image']['path'];orig=ROOT/'originals/coralscapes'/name;orig.parent.mkdir(exist_ok=True,parents=True);orig.write_bytes(v['image']['bytes'])
        mp=ROOT/'originals/coralscapes'/v['label']['path'];mp.write_bytes(v['label']['bytes']);m=np.asarray(Image.open(mp))
        rows.append(prepare('coralscapes',len(rows)+1,orig,{'stratum':name.split('_')[0],'source':'https://huggingface.co/datasets/EPFL-ECEO/coralscapes','license':'Apache-2.0','source_member':name,'selection':'One fixed-seed random test frame per available site before filling remaining slots uniformly, no model-outcome selection.','reference_kind':'semantic_benthic','reference_note':'Published semantic label IDs;0 is ignored for reference scoring.','physical_resolution':None},m))
    finish('coralscapes',rows)
def neon():
    rows=[];g=rng('neon');q=json.loads((SRC/'neon_public_metadata.json').read_text());f=next(x for x in q['files'] if x['key']=='training.zip');z=zipfile.ZipFile(RemoteFile(f['links']['self'],f['size']));sites=set()
    names=sorted(n for n in z.namelist() if n.startswith('RGB/') and n.lower().endswith('.tif'))
    for i in g.permutation(len(names)):
        name=names[i];site=Path(name).name.split('_')[1]
        if site in sites:continue
        orig=extract(z,name,ROOT/'originals/neon'/Path(name).name)
        with rasterio.open(orig) as ds:w,h=ds.width,ds.height;res=list(map(abs,ds.res));crs=str(ds.crs)
        if min(w,h)<1024:continue
        row=prepare('neon',len(rows)+1,orig,{'stratum':site,'source':'https://zenodo.org/records/5914554','license':q['metadata'].get('license',{}).get('id','See source'),'source_member':name,'selection':'One fixed-seed sampled archived training RGB tile per site with short side>=1024; native center window. Models are not trained on these data.','reference_kind':None,'reference_note':'No pixel-level crown contours in this release; box annotations are not used as silhouette truth.','physical_resolution':{'x':res[0],'y':res[1],'units':'CRS map units / pixel','crs':crs}})
        rows.append(row);sites.add(site)
        if len(rows)==10:break
    finish('neon',rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('datasets',nargs='+');a=ap.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(globals()[name]):name for name in a.datasets}
        for j in concurrent.futures.as_completed(jobs):
            try:j.result()
            except Exception as e:print('FAILED',jobs[j],repr(e),flush=True);raise
