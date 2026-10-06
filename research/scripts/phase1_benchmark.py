"""Versioned, label-free Phase 1 benchmark. No metric here is leaf accuracy."""

from research_paths import research_path
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, binary_erosion
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

METHODS=['whole','crop384','crop_native']
MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'
REV='5931719e67bbdb9737e363e781fb0c67687896bc'
VERSION='phase1-v1.0'
PALETTE=np.array([[48,153,100],[217,91,59],[68,117,196],[215,179,48],[158,85,175],[38,176,184],[225,117,171],[129,106,78]],np.uint8)

def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def starts(n,c,s):return sorted(set(list(range(0,n-c+1,s))+[n-c]))
def boundaries(lab):
 b=np.zeros(lab.shape,bool);b[:,:-1]|=lab[:,:-1]!=lab[:,1:];b[:-1]|=lab[:-1]!=lab[1:];return b
def bf1(pred,ref,valid,tolerance=2):
 pred=pred&valid;ref=ref&valid
 precision=float((pred&binary_dilation(ref,iterations=tolerance)).sum()/pred.sum()) if pred.any() else 0.
 recall=float((ref&binary_dilation(pred,iterations=tolerance)).sum()/ref.sum()) if ref.any() else 0.
 return {'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
def rgb_edges(rgb,valid):
 gray=gaussian_filter(rgb.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)/255,1)
 gx=sobel(gray,axis=1)/8;gy=sobel(gray,axis=0)/8;g=np.hypot(gx,gy)
 threshold=max(1e-6,float(np.quantile(g[valid],.85)));return g>=threshold,g
def seam_score(field,rgb,valid,crop,stride):
 # Compare feature jumps along crop borders against off-border jumps matched by RGB contrast decile.
 numer=denom=weight=0.;h,w=valid.shape
 for axis,n in [(1,w),(0,h)]:
  fg=np.sqrt(np.mean(np.diff(field,axis=axis)**2,axis=2));rg=np.sqrt(np.mean(np.diff(rgb.astype(np.float32)/255,axis=axis)**2,axis=2))
  vv=(valid[:,:-1]&valid[:,1:]) if axis==1 else (valid[:-1]&valid[1:])
  coord=np.arange(n-1)+.5;positions=sorted({p for st in starts(n,crop,stride) for p in (st,st+crop) if 0<p<n})
  line=np.min(np.abs(coord[:,None]-np.array(positions)[None,:]),axis=1)<=1.5
  near=line[None,:] if axis==1 else line[:,None];near=np.broadcast_to(near,fg.shape)
  cuts=np.quantile(rg[vv],np.linspace(0,1,11));bins=np.searchsorted(cuts[1:-1],rg,side='right')
  for q in range(10):
   a=vv&near&(bins==q);b=vv&~near&(bins==q)
   if a.sum()<10 or b.sum()<10:continue
   count=int(a.sum());numer+=float(fg[a].mean())*count;denom+=float(fg[b].mean())*count;weight+=count
 return {'ratio':numer/denom if denom>1e-12 else None,'seam_mean':numer/weight if weight else None,'matched_control_mean':denom/weight if weight else None,'matched_edges':weight}

def protocol():
 return {'version':VERSION,'model':MODEL,'model_revision':REV,'seed':42,'methods':METHODS,'crop_fraction':.1,'crop_stride_fraction':2/3,'crop_encoder_size':384,'patch_size':16,'feature_normalization':'raw 768D tokens; no explicit L2 normalization','pca':'Shared centered unwhitened PCA16 per photo, equal eligible token counts per method from original inputs only; frozen for mirrored inputs and future methods','render':'Bilinear token interpolation; crop Hann blending with per-axis floor .05; no AnyUp or flip/shift ensemble','clustering':'Separate K-means per method, K=4/6/8, n_init10 seed42; 4096 matched eligible spatial samples; mirrored field predicted with the original fitted centers','robustness':'Independent horizontal reflection pass, flip outputs back, exclude 16px margins; no reflection averaging','edge_proxy':'Grayscale sigma1 Sobel, strongest 15% eligible RGB gradients, 2px boundary tolerance. An appearance proxy, not botanical truth.','seam_proxy':'RMS feature jumps near crop starts/ends versus off-seam jumps matched within RGB-contrast deciles and orientation. Ratio 1 means no excess, not perfect segmentation. Whole-image evaluated at same hypothetical crop lines.','runtime':'GPU-synchronized per-input encoding time and peak allocated GPU bytes. Shared PCA and CPU evaluation timings recorded separately. Model warmup excluded.','uncertainty':'2000 paired site/session-block bootstrap replicates; pooled-photo means and site means; not independent-pixel inference','references':'No ground truth currently. Leaf IoU and boundary accuracy remain unmeasured. Reference masks may be added through score_leaf_references.py.','primary_k':6,'no_composite_score':True}

def initialize(base,out):
 selection=json.loads((base/'selection.json').read_text());rows=selection['images'];assert len({r['source_sha256'] for r in rows})==len(rows)
 groups={}
 for r in rows:groups.setdefault(r['site'],set()).add(r['session'])
 # Reserve one entire session per site for future reference evaluation. All proxy results remain visible.
 prior_ids={r['id'] for r in json.loads((base.parent/'moin_dino_three_20260929/selection.json').read_text())['images']}
 reserved={}
 for site,sessions in groups.items():
  unseen=[s for s in sessions if not any(r['site']==site and r['session']==s and r['id'] in prior_ids for r in rows)]
  reserved[site]=sorted(unseen or list(sessions),key=lambda s:hashlib.sha256((site+s+'42').encode()).hexdigest())[0]
 for r in rows:r['reference_split']='reserved_session' if r['session']==reserved[r['site']] else 'development';r['previously_viewed_phase1']=r['id'] in prior_ids
 p=protocol();p['source_sha256']=sha(Path(__file__));p['base_selection_sha256']=sha(base/'selection.json');p['reserved_sessions']=reserved
 save(out/'protocol.json',p);save(out/'manifest.json',{'source':str(base),'entries':selection['jpeg_entries'],'unique_images':len(rows),'deduplication':selection['deduplication'],'images':rows})
 return rows,p

def run(args):
 import torch
 import torch.nn.functional as F
 from transformers import AutoImageProcessor,AutoModel
 base=Path(args.base);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if args.shards==1:rows,p=initialize(base,out)
 else:rows=json.loads((out/'manifest.json').read_text())['images'];p=json.loads((out/'protocol.json').read_text())
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
 proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,local_files_only=True);model=AutoModel.from_pretrained(MODEL,revision=REV,local_files_only=True,attn_implementation='sdpa').cuda().eval()
 mean=torch.tensor(proc.image_mean,device='cuda')[None,:,None,None];std=torch.tensor(proc.image_std,device='cuda')[None,:,None,None]
 def encode(images):
  arr=np.stack([np.array(im) for im in images]);h,w=arr.shape[1:3];arr=np.pad(arr,((0,0),(0,(-h)%16),(0,(-w)%16),(0,0)),mode='edge');pix=torch.from_numpy(arr).permute(0,3,1,2).cuda().float()/255
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model(pixel_values=(pix-mean)/std).last_hidden_state[:,1+model.config.num_register_tokens:]
  return z.reshape(len(images),math.ceil(h/16),math.ceil(w/16),768).half().cpu().numpy()
 encode([Image.new('RGB',(384,384))]);torch.cuda.synchronize()
 def extract(im,method,boxes):
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  if method=='whole':z=encode([im])
  else:
   blocks=[]
   for b in range(0,len(boxes),12):
    ims=[im.crop(box) for box in boxes[b:b+12]]
    if method=='crop384':ims=[i.resize((384,384),Image.Resampling.BICUBIC) for i in ims]
    blocks.append(encode(ims))
   z=np.concatenate(blocks)
  torch.cuda.synchronize();return z,{'encoding_seconds':time.perf_counter()-t,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'token_count':int(np.prod(z.shape[:3]))}
 def dense(z,basis,mu,method,boxes,w,h,crop):
  flat=z.reshape(-1,768);pr=np.empty((len(flat),16),np.float32)
  for a in range(0,len(flat),8192):pr[a:a+8192]=(flat[a:a+8192].astype(np.float32)-mu)@basis
  pr=pr.reshape(*z.shape[:3],16);field=np.zeros((h,w,16),np.float32);weight=np.zeros((h,w,1),np.float32);blend=(np.maximum(np.hanning(crop),.05)[:,None]*np.maximum(np.hanning(crop),.05)[None,:])[...,None].astype(np.float32)
  for i,b in enumerate([(0,0,w,h)] if method=='whole' else boxes):
   bh,bw=b[3]-b[1],b[2]-b[0];target=(bh,bw) if method=='crop384' else (math.ceil(bh/16)*16,math.ceil(bw/16)*16)
   d=F.interpolate(torch.from_numpy(pr[i]).permute(2,0,1)[None],size=target,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()[:bh,:bw];wt=1 if method=='whole' else blend
   field[b[1]:b[3],b[0]:b[2]]+=d*wt;weight[b[1]:b[3],b[0]:b[2]]+=wt
  assert weight.min()>0;return field/weight
 done=0
 for rec_index,rec in enumerate(rows):
  if rec_index%args.shards!=args.shard:continue
  dest=out/rec['id'];result=dest/'metrics.json'
  if result.exists():continue
  if args.limit and done>=args.limit:break
  start=time.perf_counter();dest.mkdir(exist_ok=True);key=rec['id']
  try:
   im=Image.open(base/'clean_inputs'/f'{key}.png').convert('RGB');assert sha(base/'clean_inputs'/f'{key}.png')==rec['clean_input_sha256'];rgb=np.array(im);w,h=im.size;crop=max(16,round(min(w,h)*.1));stride=max(1,round(crop*2/3));boxes=[(x,y,x+crop,y+crop) for y in starts(h,crop,stride) for x in starts(w,crop,stride)]
   excluded=np.load(base/key/'artifact_exclusion.npz')['excluded'];valid=binary_erosion(~excluded,iterations=16,border_value=0);assert valid.sum()>4096
   coords=np.random.default_rng(43).choice(np.flatnonzero(valid.ravel()),4096,replace=False);features={};timings={};pools={}
   for m in METHODS:
    z,timing=extract(im,m,boxes);features[m]=z;timings[m]=timing;masks=[]
    for b in ([(0,0,w,h)] if m=='whole' else boxes):
     mask=excluded[b[1]:b[3],b[0]:b[2]]
     if m=='crop384':mask=np.array(Image.fromarray(mask).resize((384,384),Image.Resampling.NEAREST))
     mh,mw=mask.shape;mask=np.pad(mask,((0,(-mh)%16),(0,(-mw)%16)),constant_values=True);masks.append(~mask.reshape(mask.shape[0]//16,16,mask.shape[1]//16,16).any((1,3)))
    pools[m]=np.flatnonzero(np.stack(masks).ravel())
   n=min(6000,*[len(v) for v in pools.values()]);rng=np.random.default_rng(42);sample=np.concatenate([features[m].reshape(-1,768)[rng.choice(pools[m],n,replace=False)].astype(np.float32) for m in METHODS]);t=time.perf_counter()
   with threadpool_limits(limits=2):pca=PCA(n_components=16,svd_solver='randomized',random_state=42).fit(sample)
   pca_seconds=time.perf_counter()-t;basis=pca.components_.T.copy();basis*=np.where(basis[np.argmax(abs(basis),axis=0),np.arange(16)]<0,-1,1);scale=float(np.sqrt(np.mean(pca.explained_variance_)));fields={}
   with threadpool_limits(limits=2):
    for m in METHODS:fields[m]=dense(features.pop(m),basis,pca.mean_,m,boxes,w,h,crop)
   del sample,features
   projected=np.concatenate([fields[m].reshape(-1,16)[coords] for m in METHODS]);lo,hi=np.percentile(projected[:,:3],[1,99],axis=0);edge,gradient=rgb_edges(rgb,valid);metrics={};saved={'valid':valid};transforms={'pca_basis':basis,'pca_mean':pca.mean_,'explained_variance_ratio':pca.explained_variance_ratio_,'rgb_low':lo,'rgb_high':hi,'training_flat_coordinates':coords}
   thumb=im.copy();thumb.thumbnail((480,480));thumb.save(dest/'original.jpg',quality=86)
   for m in METHODS:
    t=time.perf_counter();field=fields.pop(m);flip_z,flip_timing=extract(ImageOps.mirror(im),m,boxes)
    with threadpool_limits(limits=2):flipped=dense(flip_z,basis,pca.mean_,m,boxes,w,h,crop)[:,::-1].copy()
    del flip_z
    error=np.sqrt(np.mean((field[valid]-flipped[valid])**2));row={**timings[m],'flip_encoding_seconds':flip_timing['encoding_seconds'],'flip_feature_nrmse':float(error/scale),'feature_spatial_sd':float(np.sqrt(np.mean(np.var(field[valid],axis=0)))/scale),'seam':seam_score(field,rgb,valid,crop,stride),'k':{}}
    with threadpool_limits(limits=2):
     for k in [4,6,8]:
      km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(field.reshape(-1,16)[coords]);transforms[f'{m}_centers_k{k}']=km.cluster_centers_
      def predict(a):
       flat=a.reshape(-1,16);return np.concatenate([km.predict(flat[j:j+32768]) for j in range(0,len(flat),32768)]).reshape(h,w)
      labels=predict(field);lf=predict(flipped);areas=np.bincount(labels[valid],minlength=k)/valid.sum();effective=float(np.exp(-np.sum(areas[areas>0]*np.log(areas[areas>0]))));b=boundaries(labels);f1=bf1(b,edge,valid)
      row['k'][str(k)]={'rgb_edge':f1,'flip_ari':float(adjusted_rand_score(labels[valid],lf[valid])),'boundary_density':float(b[valid].mean()),'effective_clusters':effective,'occupied_clusters':int((areas>0).sum()),'dominant_fraction':float(areas.max())}
      saved[f'{m}_k{k}']=labels.astype(np.uint8)
      if k==6:
       saved[m+'_flip_k6']=lf.astype(np.uint8);overlay=np.uint8(.55*rgb+.45*PALETTE[labels]);overlay[~valid]=[145,145,145];img=Image.fromarray(overlay);img.thumbnail((480,480));img.save(dest/f'{m}_overlay.jpg',quality=86)
    color=np.uint8(np.clip((field[:,:,:3]-lo)/np.maximum(hi-lo,1e-6),0,1)*255);color[~valid]=[145,145,145];img=Image.fromarray(color);img.thumbnail((480,480));img.save(dest/f'{m}_pca.jpg',quality=86)
    row['evaluation_seconds_including_flip']=time.perf_counter()-t;metrics[m]=row;del field,flipped
   np.savez_compressed(dest/'labels.npz',**saved);np.savez_compressed(dest/'transform.npz',**transforms)
   record={k:rec[k] for k in ['id','site','session','source_sha256','reference_split','previously_viewed_phase1']};record.update(timing_mode='isolated' if args.shards==1 else 'concurrent',version=VERSION,size=[w,h],crop=crop,stride=stride,crop_count=len(boxes),eligible_pixels=int(valid.sum()),pca_tokens_per_method=n,pca_variance=float(pca.explained_variance_ratio_.sum()),shared_pca_seconds=pca_seconds,methods=metrics,total_seconds=time.perf_counter()-start)
   save(result,record);done+=1;save(out/f'progress-{args.shard}.json',{'completed':len(list(out.glob('*/metrics.json'))),'total':len(rows),'last_id':key,'last_seconds':record['total_seconds'],'state':'running'});print('DONE',key,round(record['total_seconds'],2),flush=True)
  except Exception:
   save(out/'failure.json',{'id':key,'traceback':traceback.format_exc()});raise
  torch.cuda.empty_cache()
 if args.shards==1:aggregate(out)

def aggregate(out):
 manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(p.read_text()) for p in sorted(out.glob('*/metrics.json'))];assert len({r['id'] for r in rows})==len(rows)
 names=['seam_ratio','flip_feature_nrmse','flip_ari','rgb_edge_f1','boundary_density','effective_clusters','encoding_seconds','peak_gpu_mib'];summary={};contrasts=[];blocks=sorted({r['site']+'/'+r['session'] for r in rows});indices={g:[i for i,r in enumerate(rows) if r['site']+'/'+r['session']==g] for g in blocks};rng=np.random.default_rng(20260929)
 # Shared block draws for paired comparisons; physical plot identity is not reliably available.
 draws=[np.concatenate([indices[blocks[j]] for j in rng.integers(0,len(blocks),len(blocks))]) for _ in range(2000)] if rows else []
 def val(r,m,n):
  q=r['methods'][m];k=q['k']['6']
  return q['seam']['ratio'] if n=='seam_ratio' else k['rgb_edge']['f1'] if n=='rgb_edge_f1' else k[n] if n in k else q[n]
 for n in names:
  summary[n]={}
  for m in METHODS:
   v=np.array([val(r,m,n) if val(r,m,n) is not None else np.nan for r in rows]);boot=[float(np.nanmean(v[idx])) for idx in draws]
   summary[n][m]={'mean':float(np.nanmean(v)),'median':float(np.nanmedian(v)),'n':int(np.isfinite(v).sum()),'block_bootstrap_mean_ci95':np.quantile(boot,[.025,.975]).tolist(),'site_means':{s:float(np.nanmean([val(r,m,n) for r in rows if r['site']==s])) for s in sorted({r['site'] for r in rows})}}
  for a,b in [('whole','crop384'),('whole','crop_native'),('crop384','crop_native')]:
   v=np.array([(val(r,b,n)-val(r,a,n)) if val(r,b,n) is not None and val(r,a,n) is not None else np.nan for r in rows]);boot=[float(np.nanmean(v[idx])) for idx in draws];contrasts.append({'metric':n,'a':a,'b':b,'definition':'B minus A','mean_difference':float(np.nanmean(v)),'ci95':np.quantile(boot,[.025,.975]).tolist()})
 save(out/'results.json',{'version':VERSION,'complete':len(rows)==manifest['unique_images'],'completed':len(rows),'total':manifest['unique_images'],'blocks':len(blocks),'block_unit':'site/session; spatial plot correlations across sessions may remain','summary':summary,'paired_contrasts':contrasts,'images':rows,'leaf_accuracy':{'status':'not measured','reviewed_reference_windows':0},'protocol':json.loads((out/'protocol.json').read_text())})
 save(out/'progress.json',{'completed':len(rows),'total':manifest['unique_images'],'state':'complete' if len(rows)==manifest['unique_images'] else 'partial'})
 print('AGGREGATE',len(rows),flush=True)

def selftest():
 v=np.ones((120,120),bool);a=np.zeros((120,120),np.uint8);a[:,60:]=1;b=boundaries(a);assert bf1(b,b,v)['f1']==1;assert bf1(np.zeros_like(b),b,v)['f1']==0
 rgb=np.zeros((120,120,3),np.uint8);f=np.random.default_rng(2).normal(scale=.01,size=(120,120,16)).astype(np.float32);baseline=seam_score(f,rgb,v,30,20)['ratio'];f[:,20:]+=10;spike=seam_score(f,rgb,v,30,20)['ratio'];assert .8<baseline<1.2 and spike>baseline*3,(baseline,spike)
 assert adjusted_rand_score(a.ravel(),(1-a).ravel())==1
 print('PASS: boundary identity/empty, label permutation, synthetic seam sensitivity',baseline,spike)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','aggregate','selftest','prepare']);ap.add_argument('--base',default=str(research_path('runs/moin_plot_all_20260928')));ap.add_argument('--out',default=str(research_path('runs/moin_phase1_benchmark_v1')));ap.add_argument('--limit',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard',type=int,default=0);args=ap.parse_args()
 if args.command=='run':run(args)
 elif args.command=='aggregate':aggregate(Path(args.out))
 elif args.command=='prepare':initialize(Path(args.base),Path(args.out))
 else:selftest()
