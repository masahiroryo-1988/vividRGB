"""Native-resolution ecology comparison. No AnyUp, text prompts, or outcome selection."""

from research_paths import research_path
import argparse,json,time,subprocess,threading,atexit
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import phase1_benchmark as p1
import multiscale_benchmark as mb
import cross_domain_benchmark as cross
from phase1_fourflip_benchmark import fourflip_encode,_predict
ROOT=Path(str(research_path('runs/eco_collection75_v1')))
METHODS=['whole','whole_fourflip','s05','s10','s20','s40']

class Power:
    def __init__(self):
        self.samples=[]
        self.proc=subprocess.Popen(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits','-lms','200'],stdout=subprocess.PIPE,text=True)
        def read():
            for line in self.proc.stdout:
                try:self.samples.append((time.perf_counter(),float(line.strip())))
                except ValueError:pass
        threading.Thread(target=read,daemon=True).start();atexit.register(self.proc.terminate)
    def between(self,start,end):
        q=np.array(self.samples)
        if len(q)<2:return None
        ts=np.r_[start,q[(q[:,0]>start)&(q[:,0]<end),0],end]
        watts=np.interp(ts,q[:,0],q[:,1])
        return float(np.sum(np.diff(ts)*(watts[:-1]+watts[1:])*.5))

def geometry(w,h,scale):
    c=max(16,round(min(w,h)*scale/100));s=max(1,round(c/4))
    return c,s,[(x,y,x+c,y+c) for y in p1.starts(h,c,s) for x in p1.starts(w,c,s)]

def rows(a):
    out=[]
    for name in a.datasets.split(','):
        f=ROOT/f'{name}_selection.json'
        if not f.exists():raise FileNotFoundError(f)
        out+=json.loads(f.read_text())['images']
    return out[:a.limit] if a.limit else out

def inputs(r):
    d=ROOT/r['id'];im=Image.open(d/'input.png').convert('RGB');assert p1.sha(d/'input.png')==r['input_sha256']
    rgb=np.asarray(im);gt=np.asarray(Image.open(d/'reference.png')) if r['reference_available'] else None
    valid=binary_erosion(gt>0 if gt is not None and r['dataset']=='coralscapes' else np.ones(rgb.shape[:2],bool),iterations=2,border_value=0)
    rgbvalid=binary_erosion(np.ones(rgb.shape[:2],bool),iterations=2,border_value=0)
    edges,_=p1.rgb_edges(rgb,rgbvalid)
    return d,im,rgb,gt,valid,rgbvalid,edges

def score(labels,gt,valid,rgbvalid,edges):
    b=p1.boundaries(labels);out={'rgb_edge':p1.bf1(b,edges,rgbvalid),'segments':int(len(np.unique(labels[rgbvalid]))),'boundary_density':float(b[rgbvalid].mean()),'reference':None}
    if gt is not None:
        gb=p1.boundaries(gt);nb=int((gb&valid).sum());ix=np.flatnonzero(valid)
        ix=np.random.default_rng(77).choice(ix,min(len(ix),100000),replace=False)
        out['reference']={'boundary':p1.bf1(b,gb,valid) if nb else None,'boundary_5px':p1.bf1(b,gb,valid,tolerance=5) if nb else None,'ari':float(adjusted_rand_score(gt.flat[ix],labels.flat[ix])),'reference_boundary_pixels':nb,'valid_pixels':int(valid.sum())}
    return out

def previews(d,name,labels,rgb,valid,field=None,limits=None):
    cross.preview(d,name,labels,rgb,valid,field,limits)
    colors=np.random.default_rng(33).integers(25,240,(int(labels.max())+1,3),dtype=np.uint8)
    Image.fromarray(colors[labels]).save(d/f'{name}_clusters.png')
    if field is not None:
        lo,hi=limits;arr=np.uint8(np.clip((field[:,:,:3]-lo)/np.maximum(hi-lo,1e-6),0,1)*255)
        Image.fromarray(arr).save(d/f'{name}_pca.png')

def dino(a):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    proc=AutoImageProcessor.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True)
    model=AutoModel.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    mean=torch.tensor(proc.image_mean,device='cuda')[None,:,None,None];std=torch.tensor(proc.image_std,device='cuda')[None,:,None,None]
    power=Power()
    def encode(ims):
        arr=np.stack([np.asarray(i) for i in ims]);h,w=arr.shape[1:3];pix=torch.from_numpy(arr).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model(pixel_values=(pix-mean)/std).last_hidden_state[:,1+model.config.num_register_tokens:]
        return z.reshape(len(ims),h//16,w//16,768).half().cpu().numpy()
    def flip(ims):return fourflip_encode(ims,encode)
    flip([Image.new('RGB',(384,384),(100,130,90))])
    for r in rows(a):
        d,im,rgb,gt,valid,rgbvalid,edges=inputs(r);w,h=im.size;metafile=d/'dino_metrics.json'
        result=json.loads(metafile.read_text()) if metafile.exists() else {'id':r['id'],'methods':{}}
        if all(m in result['methods'] for m in a.methods.split(',')):continue
        with threadpool_limits(limits=2):
            whole={}
            for name,fn in [('whole',encode),('whole_fourflip',flip)]:
                f=d/f'tokens_{name}.npz'
                if not f.exists():
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();st=time.perf_counter();z=fn([im]);torch.cuda.synchronize();en=time.perf_counter()
                    mb.save(d/f'{name}_encoding.json',{'seconds':en-st,'gpu_board_joules':power.between(st,en),'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2})
                    np.savez_compressed(f,z=z)
                whole[name]=np.load(f)['z']
            if not (d/'transform.npz').exists():
                st=time.perf_counter();g=np.random.default_rng(44);samples=[]
                for z in whole.values():
                    zz=z.reshape(-1,768);samples.append(zz[g.choice(len(zz),min(2048,len(zz)),replace=False)])
                for scale in [5,10,20,40]:
                    _,_,boxes=geometry(w,h,scale);boxes=[boxes[i] for i in g.choice(len(boxes),min(12,len(boxes)),replace=False)]
                    zz=flip([im.crop(b).resize((384,384),Image.Resampling.BICUBIC) for b in boxes]).reshape(-1,768)
                    samples.append(zz[g.choice(len(zz),2048,replace=False)])
                sam=np.concatenate(samples).astype(np.float32);pc=PCA(n_components=16,svd_solver='randomized',random_state=42).fit(sam)
                lo,hi=np.quantile(pc.transform(sam)[:,:3],[.01,.99],axis=0);coords=np.random.default_rng(45).choice(w*h,min(8192,w*h),replace=False)
                np.savez(d/'transform.npz',basis=pc.components_.T,mean=pc.mean_,coordinates=coords,low=lo,high=hi)
                mb.save(d/'calibration.json',{'seconds':time.perf_counter()-st,'explained_variance_ratio':pc.explained_variance_ratio_.tolist()})
            tr=np.load(d/'transform.npz');basis,mu,coords,lo,hi=[tr[n] for n in ['basis','mean','coordinates','low','high']]
            for name in a.methods.split(','):
                if name in result['methods']:continue
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                if name.startswith('whole'):
                    pr=mb.project(whole[name],basis,mu)[0]
                    field=F.interpolate(torch.from_numpy(pr).permute(2,0,1)[None],size=(h,w),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
                    meta=json.loads((d/f'{name}_encoding.json').read_text());meta['tiles']=1;meta['encoding_seconds']=meta['seconds'];meta['seconds']+=time.perf_counter()-start
                    meta['encoding_gpu_board_joules']=meta.pop('gpu_board_joules');meta['gpu_board_joules']=None
                    meta['energy_note']='Cached encoding energy retained separately; full extraction energy was not measured continuously and is withheld from comparison.'
                else:
                    c,s,boxes=geometry(w,h,int(name[1:]));field=np.zeros((h,w,16),np.float32);weights=np.zeros((h,w,1),np.float32)
                    blend=(np.maximum(np.hanning(c),.05)[:,None]*np.maximum(np.hanning(c),.05)[None,:])[...,None].astype(np.float32)
                    for i in range(0,len(boxes),12):
                        chunk=boxes[i:i+12];z=flip([im.crop(b).resize((384,384),Image.Resampling.BICUBIC) for b in chunk]);pr=mb.project(z,basis,mu)
                        mb.add_tiles(field,weights,pr,chunk,blend,torch,F)
                        if i%600==0:print('TILES',r['id'],name,i+len(chunk),len(boxes),flush=True)
                    assert weights.min()>0;field/=weights;del weights
                    torch.cuda.synchronize();end=time.perf_counter();meta={'seconds':end-start,'gpu_board_joules':power.between(start,end),'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'tiles':len(boxes),'crop_pixels':c,'stride_pixels':s,'actual_overlap':1-s/c}
                scores={};clstart=time.perf_counter()
                for k in [6,12]:
                    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(field.reshape(-1,16)[coords]);lab=_predict(km,field,w,h).astype(np.uint16)
                    scores[str(k)]=score(lab,gt,valid,rgbvalid,edges);np.savez_compressed(d/f'{name}_k{k}_labels.npz',labels=lab)
                    if k==6:previews(d,name,lab,rgb,valid,field,(lo,hi))
                result['methods'][name]={'k':scores,'extraction':meta,'clustering_scoring_preview_seconds':time.perf_counter()-clstart};mb.save(metafile,result)
                print('DONE',r['id'],name,json.dumps(scores['6']),flush=True);del field
            if gt is not None and not (d/'reference_overlay.jpg').exists():
                shown=rgb.copy();shown[p1.boundaries(gt)&valid]=[255,40,160];pic=Image.fromarray(shown);pic.thumbnail((640,640));pic.save(d/'reference_overlay.jpg',quality=94)
        torch.cuda.empty_cache()

def sam3(a):
    import torch
    from torchvision.ops import nms
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(2);torch.manual_seed(42)
    ck=Path(str(research_path('work/sam3_checkpoint_path.txt'))).read_text().strip()
    model=build_sam3_image_model(device='cpu',checkpoint_path=ck,load_from_HF=False,enable_inst_interactivity=True)
    model.backbone._apply(lambda t:t.to(torch.bfloat16) if t.is_floating_point() else t);model.cuda().eval()
    processor=Sam3Processor(model,device='cuda');model.inst_interactive_predictor._transforms.max_hole_area=0;model.inst_interactive_predictor._transforms.max_sprinkle_area=0
    power=Power()
    for r in rows(a):
        d,im,rgb,gt,valid,rgbvalid,edges=inputs(r)
        if (d/'sam3_metrics.json').exists():continue
        w,h=im.size;axis=(np.arange(32)+.5)/32;pts=np.array([(x*w,y*h) for y in axis for x in axis],np.float32)
        masks=[];qs=[];boxes=[];stabilities=[];origpoints=[];areas=[]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();st=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(im)
            for i in range(0,len(pts),4):
                logits,ious,_=model.predict_inst(state,point_coords=pts[i:i+4,None,:],point_labels=np.ones((len(pts[i:i+4]),1),np.int32),multimask_output=True,return_logits=True,normalize_coords=True)
                for j in range(len(ious)):
                    for k in range(3):
                        q=float(ious[j,k])
                        if q<=.8:continue
                        m=logits[j,k];den=int((m>-1).sum());stab=float((m>1).sum()/den) if den else 0
                        if stab<.95:continue
                        m=m>0;yy,xx=np.nonzero(m)
                        if not len(xx):continue
                        masks.append(np.packbits(m));qs.append(q);stabilities.append(stab);boxes.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]);areas.append(int(len(xx)));origpoints.append(pts[i+j].tolist())
                if i%256==0:print('SAM3_POINTS',r['id'],i+4,len(pts),len(masks),flush=True)
            keep=nms(torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),torch.tensor(qs),.7).cpu().numpy() if qs else np.array([],int)
        torch.cuda.synchronize();end=time.perf_counter();meta={'seconds':end-st,'gpu_board_joules':power.between(st,end),'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'candidates_before_nms':len(masks),'mask_count':len(keep)}
        accepted=[masks[i] for i in keep];quality=np.array(qs)[keep];area=np.array(areas)[keep]
        np.savez_compressed(d/'sam3_proposals.npz',packed=np.array(accepted,dtype=np.uint8),quality=quality,area=area,shape=[h,w],stability=np.array(stabilities)[keep],boxes=np.array(boxes).reshape(-1,4)[keep],points=np.array(origpoints).reshape(-1,2)[keep])
        scores={}
        for variant,order in [('quality',np.argsort(quality,kind='stable')),('small_first',np.argsort(-area,kind='stable'))]:
            lab=np.zeros((h,w),np.uint16)
            for j in order:lab[np.unpackbits(accepted[j],count=h*w).reshape(h,w).astype(bool)]=int(j)+1
            scores[variant]=score(lab,gt,valid,rgbvalid,edges);np.savez_compressed(d/f'sam3_{variant}_labels.npz',labels=lab)
            previews(d,'sam3' if variant=='quality' else 'sam3_small_first',lab,rgb,valid)
        mb.save(d/'sam3_metrics.json',{'id':r['id'],'metrics':scores,'extraction':meta});print('SAM3_DONE',r['id'],json.dumps(meta),flush=True)
        del state;torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dino','sam3']);ap.add_argument('--datasets',default='mycelium,bam,coralscapes,fungal_network,neon,tara,pmid');ap.add_argument('--methods',default=','.join(METHODS));ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    mb.save(ROOT/f'{a.mode}_protocol.json',{'code_sha256':p1.sha(Path(__file__)),'dino':p1.MODEL,'dino_revision':p1.REV,'methods':METHODS,'local':'5/10/20/40% of working short side, rounded; bicubic384; four inverse-aligned flip vectors averaged; stride round(crop/4), approximately75% overlap; separable Hann floor.05 and normalize coverage; bilinear token upsampling. No AnyUp or multiscale merging. No explicit L2 vector normalization.','pca':'PCA16 per image; balanced2048tokens per six treatments; no labels; sharedbasis and1/99 percentile RGB displaylimits. Independent KMeans K6 primary/K12 sensitivity,8192 same coordinates,n_init10,seed42.','sam3':'Pinned3c879f39826c281e95690f02c7821c4de09afae7; SAM3 interactive instance head; automatic32x32uniform points,batch4,multimask3,predIoU>.8,stability>=.95 at logits+/-1,threshold0,boxNMS.7,crop_layers0,no text/manual points,no hole/sprinkle postprocessing. Primary highest predictedIoU wins; smaller-mask-first overlap sensitivity.','evaluation':'Reference boundaryF1 Manhattan2pixels primary/5sensitivity andARI shared100000pixels. Coral0ignored with2pixel erosion; otherreferencebackground included. RGBedgeF1 top15% grayscale gradients only appearanceproxy. No box-as-mask truth.','cost':'Single run, model-load/warmup excluded; separate extraction timings; GPUboardenergy estimated trapezoidal integration of nvidia-smi power.draw sampled200ms. SharedPCA calibration and clustering cost reported separately. Whole timing includes cached whole encoding. Not wall-plug or carbon measurement.'})
    with threadpool_limits(limits=2):globals()[a.mode](a)
