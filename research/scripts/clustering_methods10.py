"""Frozen-feature clustering inspection; does not modify the original benchmark."""

from research_paths import research_path
import argparse, json, time, shutil, warnings
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
import ecology75_benchmark as old
import multiscale_benchmark as mb
import phase1_benchmark as p1
from phase1_fourflip_benchmark import fourflip_encode

ROOT=Path(str(research_path('runs/eco_clustering10_v1')))
METHODS=['whole','s05','s10','s20']
DATASETS=['moin','fungal_network','neon','coralscapes','pmid']

def selection():
    f=ROOT/'selection.json'
    if f.exists(): return json.loads(f.read_text())['images']
    rng=np.random.default_rng(20261003); chosen=[]
    for ds in DATASETS:
        rows=json.loads((old.ROOT/f'{ds}_selection.json').read_text())['images']
        ids=[0,9] if ds=='moin' else sorted(rng.choice(len(rows),2,replace=False).tolist())
        chosen.extend(rows[i] for i in ids)
    mb.save(f,{'seed':20261003,'selection':'MOIN first and last frozen rank-stratified examples; two fixed-seed random examples per other dataset; chosen before new clustering.','images':chosen})
    return chosen

def extract(rows):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    proc=AutoImageProcessor.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True)
    model=AutoModel.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    mean=torch.tensor(proc.image_mean,device='cuda')[None,:,None,None]
    std=torch.tensor(proc.image_std,device='cuda')[None,:,None,None]
    def encode(ims):
        arr=np.stack([np.asarray(i) for i in ims]);h,w=arr.shape[1:3]
        pix=torch.from_numpy(arr).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=model(pixel_values=(pix-mean)/std).last_hidden_state[:,1+model.config.num_register_tokens:]
        return z.reshape(len(ims),h//16,w//16,768).half().cpu().numpy()
    for row in rows:
        source=old.ROOT/row['id'];dest=ROOT/row['id'];dest.mkdir(exist_ok=True)
        assert p1.sha(source/'input.png')==row['input_sha256']
        im=Image.open(source/'input.png').convert('RGB');w,h=im.size
        preview=im.copy();preview.thumbnail((1024,1024));preview.save(dest/'input.jpg',quality=94)
        shutil.copy2(source/'source.json',dest/'source.json')
        tr=np.load(source/'transform.npz');basis,mu=tr['basis'],tr['mean']
        shutil.copy2(source/'transform.npz',dest/'transform.npz')
        gh,gw=round(h*256/max(w,h)),round(w*256/max(w,h))
        # Sample cell centers, not cell averages: same image locations for every treatment.
        ys=np.minimum(((np.arange(gh)+.5)*h/gh).astype(int),h-1)
        xs=np.minimum(((np.arange(gw)+.5)*w/gw).astype(int),w-1)
        for method in METHODS:
            path=dest/f'{method}_features.npz'
            if path.exists():continue
            start=time.perf_counter()
            if method=='whole':
                pr=mb.project(np.load(source/'tokens_whole.npz')['z'],basis,mu)[0]
                field=F.interpolate(torch.from_numpy(pr).permute(2,0,1)[None],size=(h,w),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
                meta={'tiles':1,'cached_whole_tokens':True}
            else:
                c,s,boxes=old.geometry(w,h,int(method[1:]));field=np.zeros((h,w,16),np.float32);weight=np.zeros((h,w,1),np.float32)
                blend=(np.maximum(np.hanning(c),.05)[:,None]*np.maximum(np.hanning(c),.05)[None,:])[...,None].astype(np.float32)
                for i in range(0,len(boxes),12):
                    chunk=boxes[i:i+12]
                    z=fourflip_encode([im.crop(b).resize((384,384),Image.Resampling.BICUBIC) for b in chunk],encode)
                    mb.add_tiles(field,weight,mb.project(z,basis,mu),chunk,blend,torch,F)
                    if i%1200==0:print('TILES',row['id'],method,i,len(boxes),flush=True)
                assert weight.min()>0;field/=weight;del weight
                meta={'tiles':len(boxes),'crop_pixels':c,'stride_pixels':s,'four_flips':True,'overlap':1-s/c}
            grid=field[ys[:,None],xs[None,:]].copy();assert grid.shape==(gh,gw,16) and np.isfinite(grid).all()
            np.savez_compressed(path,features=grid,ys=ys,xs=xs)
            meta.update(seconds=time.perf_counter()-start,grid_shape=[gh,gw],feature_sha256=p1.sha(path),transform_sha256=p1.sha(source/'transform.npz'))
            mb.save(dest/f'{method}_extraction.json',meta)
            # Exactly the same sampled grid is visualized and clustered.
            arr=np.uint8(np.clip((grid[:,:,:3]-tr['low'])/np.maximum(tr['high']-tr['low'],1e-6),0,1)*255)
            Image.fromarray(arr).resize(preview.size,Image.Resampling.BILINEAR).save(dest/f'{method}_pca.jpg',quality=95)
            shutil.copy2(source/f'{method}_pca.png',dest/f'{method}_original_pca.png')
            del field;print('EXTRACTED',row['id'],method,flush=True)
        torch.cuda.empty_cache()

def cluster(rows):
    for row in rows:
        dest=ROOT/row['id'];rgb=np.asarray(Image.open(dest/'input.jpg'))
        h,w=rgb.shape[:2]
        for method in METHODS:
            file=dest/f'{method}_features.npz'
            if not file.exists():continue
            grid=np.load(file)['features'];gh,gw=grid.shape[:2];x=np.ascontiguousarray(grid.reshape(-1,16));n=len(x)
            for alg in ['kmeans','gmm','hdbscan']:
                outf=dest/f'{method}_{alg}.json'
                if outf.exists():continue
                start=time.perf_counter();extra={}
                with warnings.catch_warnings(record=True) as caught:
                    if alg=='kmeans':
                        model=KMeans(n_clusters=6,n_init=10,random_state=42).fit(x);labels=model.labels_
                        extra={'inertia':float(model.inertia_),'iterations':int(model.n_iter_)}
                    elif alg=='gmm':
                        model=GaussianMixture(n_components=6,covariance_type='full',n_init=3,max_iter=300,tol=1e-3,reg_covar=1e-5,random_state=42).fit(x)
                        prob=model.predict_proba(x);labels=prob.argmax(1)
                        extra={'converged':bool(model.converged_),'iterations':int(model.n_iter_),'mean_max_posterior':float(prob.max(1).mean()),'bic':float(model.bic(x))}
                        np.savez_compressed(dest/f'{method}_{alg}_confidence.npz',confidence=prob.max(1).reshape(gh,gw).astype(np.float32))
                    else:
                        minimum=max(32,round(.005*n))
                        model=HDBSCAN(min_cluster_size=minimum,min_samples=15,cluster_selection_method='eom',metric='euclidean',n_jobs=2).fit(x)
                        labels=model.labels_
                        extra={'min_cluster_size':minimum,'min_samples':15,'mean_membership_strength':float(model.probabilities_.mean())}
                    extra['warnings']=[str(v.message) for v in caught]
                elapsed=time.perf_counter()-start
                labels=labels.reshape(gh,gw).astype(np.int32);values,counts=np.unique(labels,return_counts=True)
                groups=[int(v) for v in values if v>=0]
                # A deterministic palette; numeric IDs/colors are arbitrary across methods.
                colors=np.random.default_rng(33).integers(25,240,(max(groups,default=0)+1,3),dtype=np.uint8)
                colored=np.full((gh,gw,3),[180,180,180],np.uint8);named=labels>=0;colored[named]=colors[labels[named]]
                pic=Image.fromarray(colored).resize((w,h),Image.Resampling.NEAREST);pic.save(dest/f'{method}_{alg}_clusters.png')
                full=np.asarray(Image.fromarray(labels).resize((w,h),Image.Resampling.NEAREST))
                mask=full>=0;overlay=rgb.copy();overlay[mask]=np.uint8(.55*rgb[mask]+.45*np.asarray(pic)[mask])
                Image.fromarray(overlay).save(dest/f'{method}_{alg}_overlay.jpg',quality=94)
                boundary=np.zeros(full.shape,bool)
                boundary[:,1:]|=(full[:,1:]!=full[:,:-1])&(full[:,1:]>=0)&(full[:,:-1]>=0)
                boundary[1:]|=(full[1:]!=full[:-1])&(full[1:]>=0)&(full[:-1]>=0)
                edge=rgb.copy();edge[boundary]=[255,35,150];Image.fromarray(edge).save(dest/f'{method}_{alg}_boundary.jpg',quality=95)
                np.savez_compressed(dest/f'{method}_{alg}_labels.npz',labels=labels)
                out={'id':row['id'],'method':method,'algorithm':alg,'grid_shape':[gh,gw],'samples':n,'feature_sha256':p1.sha(file),'seconds':elapsed,'cluster_count':len(groups),'noise_fraction':float((labels<0).mean()),'area_fractions':{str(int(v)):float(c/n) for v,c in zip(values,counts)},**extra}
                mb.save(outf,out);print('CLUSTERED',row['id'],method,alg,len(groups),out['noise_fraction'],round(elapsed,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['extract','cluster','all']);args=ap.parse_args()
    ROOT.mkdir(exist_ok=True);rows=selection()
    mb.save(ROOT/'protocol.json',{'script_sha256':p1.sha(Path(__file__)),'model':p1.MODEL,'revision':p1.REV,'methods':METHODS,'grid':'Aspect-preserving grid, longest side 256; native-pixel cell centers sampled from original dense PCA16 reconstruction. Every algorithm fits ALL identical grid vectors. Nearest-neighbor labels; bilinear PCA display. Fine boundaries below the grid spacing are unresolved.','features':'Frozen original per-image PCA16 basis, mean and RGB limits reused; no whitening, extra L2 normalization, XY coordinates or size balancing. PCA calibration originally included six feature treatments, including whole-fourflip and Local40, though those are not tested here.','local':'Original 384px bicubic enlargement, inverse-aligned four-flip averaging, approximately75% overlap, Hann weighted blending; no AnyUp.','algorithms':{'kmeans':'K6,n_init10,seed42','gmm':'6 full-covariance components,n_init3,max_iter300,tol.001,reg_covar.00001,seed42; maximum-posterior hard labels','hdbscan':'Euclidean, min_cluster_size=max(32,round(.005*N)),min_samples15,EOM; -1 unassigned gray; no forced assignment'},'purpose':'Visual algorithm comparison, not leaf-instance accuracy validation or cluster-based species identification. No winner selected by appearance.'})
    with threadpool_limits(limits=2):
        if args.mode in ['extract','all']:extract(rows)
        if args.mode in ['cluster','all']:cluster(rows)
