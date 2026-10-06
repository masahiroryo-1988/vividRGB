"""Frozen exploratory cross-domain segmentation comparison. No model tuning to labels."""

from research_paths import research_path
import argparse,json,hashlib,time,sys
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
from phase1_fourflip_benchmark import fourflip_encode,_predict
ROOT=Path(str(research_path('runs/eco_cross_domain_v1')))
METHODS=['whole','whole_fourflip','s05','s10','s20','s40']
PROMPTS={'diatoms':['diatom'],'fungi':['fungal colony'],'landscape':['grass','tree','bush','water','soil','sky','rock'],
         'uav':['grass','herbaceous plant','bare soil','stone','wood','straw','leaf litter']}
def save(p,d):mb.save(p,d)
def read_input(row):
    dest=ROOT/row['id'];im=Image.open(dest/'input.png').convert('RGB');rgb=np.asarray(im)
    assert p1.sha(dest/'input.png')==row['input_sha256']
    valid=np.ones(rgb.shape[:2],bool);gt=None
    if row['reference_available']:
        gt=np.asarray(Image.open(dest/'reference.png'))
        if row['domain'] in ['landscape','uav']:valid=gt>0
    valid=binary_erosion(valid,iterations=2,border_value=0)
    return dest,im,rgb,valid,gt
def score(labels,rgb,valid,gt):
    rgbvalid=binary_erosion(np.ones(valid.shape,bool),iterations=2,border_value=0)
    edges,_=p1.rgb_edges(rgb,rgbvalid);boundary=p1.boundaries(labels)
    out={'rgb_edge':p1.bf1(boundary,edges,rgbvalid),'segments':int(len(np.unique(labels[rgbvalid]))),
         'boundary_density':float(boundary[rgbvalid].mean()),'reference':None}
    if gt is not None:
        # Fixed same-pixel evaluation. Oracle map is diagnostic, not a deployable classifier.
        g,gi=np.unique(gt[valid],return_inverse=True);p,pi=np.unique(labels[valid],return_inverse=True)
        contingency=np.bincount(pi*len(g)+gi,minlength=len(p)*len(g)).reshape(len(p),len(g))
        mapping=np.argmax(contingency,axis=1);mapped=mapping[pi]
        ious=[]
        for j in range(len(g)):
            inter=np.sum((gi==j)&(mapped==j));union=np.sum((gi==j)|(mapped==j));ious.append(float(inter/union) if union else 0.)
        # Sample at most100000 valid pixels deterministically for ARI to bound memory/time.
        ix=np.random.default_rng(77).choice(len(gi),min(len(gi),100000),replace=False)
        gb=p1.boundaries(gt);nb=int((gb&valid).sum())
        out['reference']={'boundary':p1.bf1(boundary,gb,valid) if nb else None,'reference_boundary_pixels':nb,
            'ari':float(adjusted_rand_score(gi[ix],pi[ix])), 'oracle_majority_miou':float(np.mean(ious)),
            'class_ids':g.tolist(),'class_ious':ious,'valid_pixels':int(valid.sum())}
    return out
def preview(dest,name,labels,rgb,valid,field=None,limits=None):
    rng=np.random.default_rng(33);colors=rng.integers(25,240,(int(labels.max())+1,3),dtype=np.uint8)
    overlay=np.uint8(.55*rgb+.45*colors[labels]);bd=rgb.copy();bd[p1.boundaries(labels)]=[255,40,160]
    for kind,a in [('overlay',overlay),('boundary',bd)]:
        im=Image.fromarray(a);im.thumbnail((640,640));im.save(dest/f'{name}_{kind}.jpg',quality=92)
    if field is not None:
        lo,hi=limits;a=np.uint8(np.clip((field[:,:,:3]-lo)/np.maximum(hi-lo,1e-6),0,1)*255)
        im=Image.fromarray(a);im.thumbnail((640,640));im.save(dest/f'{name}_pca.jpg',quality=92)
def run_dino(limit):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    proc=AutoImageProcessor.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True)
    model=AutoModel.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    mean=torch.tensor(proc.image_mean,device='cuda')[None,:,None,None];std=torch.tensor(proc.image_std,device='cuda')[None,:,None,None]
    def encode(images):
        a=np.stack([np.asarray(im) for im in images]);h,w=a.shape[1:3]
        x=torch.from_numpy(a).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model(pixel_values=(x-mean)/std).last_hidden_state[:,1+model.config.num_register_tokens:]
        return z.reshape(len(images),h//16,w//16,768).half().cpu().numpy()
    rows=json.loads((ROOT/'manifest.json').read_text())['images'];done=0
    for row in rows:
        dest,im,rgb,valid,gt=read_input(row)
        if (dest/'dino_metrics.json').exists():continue
        if limit and done>=limit:break
        start=time.perf_counter();w,h=im.size;assert w%16==0 and h%16==0
        with threadpool_limits(limits=2):
            zwhole=encode([im]);zflip=fourflip_encode([im],encode)
            rng=np.random.default_rng(44);samples=[]
            for z in [zwhole,zflip]:
                a=z.reshape(-1,768);samples.append(a[rng.choice(len(a),min(2048,len(a)),replace=False)])
            for scale in [5,10,20,40]:
                _,_,boxes=mb.geometry(w,h,scale);boxes=[boxes[i] for i in rng.choice(len(boxes),min(12,len(boxes)),replace=False)]
                z=fourflip_encode([im.crop(b).resize((384,384),Image.Resampling.BICUBIC) for b in boxes],encode).reshape(-1,768)
                samples.append(z[rng.choice(len(z),2048,replace=False)])
            pca=PCA(n_components=16,svd_solver='randomized',random_state=42).fit(np.concatenate(samples).astype(np.float32))
            basis=pca.components_.T.astype(np.float32);mu=pca.mean_.astype(np.float32)
            # Shared colors derived from balanced token sample, without annotations.
            display=pca.transform(np.concatenate(samples).astype(np.float32))[:,:3];lo,hi=np.quantile(display,[.01,.99],axis=0)
            coords=np.random.default_rng(45).choice(w*h,min(8192,w*h),replace=False)
            np.savez(dest/'transform.npz',basis=basis,mean=mu,coordinates=coords,low=lo,high=hi)
            rowscores={};stored={}
            for name in METHODS:
                st=time.perf_counter()
                if name.startswith('whole'):
                    z=zwhole if name=='whole' else zflip
                    pr=mb.project(z,basis,mu)[0]
                    field=F.interpolate(torch.from_numpy(pr).permute(2,0,1)[None],size=(h,w),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
                    meta={'note':'Whole-image encoding included in shared PCA preparation; seconds exclude that cached encoding.'}
                else:field,meta=mb.extract(im,int(name[1:]),basis,mu,encode,torch,F)
                scores={}
                for k in [6,12]:
                    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(field.reshape(-1,16)[coords]);labels=_predict(km,field,w,h).astype(np.uint16)
                    scores[str(k)]=score(labels,rgb,valid,gt)
                    if k==6:stored[name]=labels;preview(dest,name,labels,rgb,valid,field,(lo,hi))
                rowscores[name]={'k':scores,'extraction':meta,'processing_seconds':time.perf_counter()-st}
                ref=scores['6']['reference']
                print('DINO',row['id'],name,'RGB F1',round(scores['6']['rgb_edge']['f1'],3),'GT F1',None if ref is None or ref['boundary'] is None else round(ref['boundary']['f1'],3),flush=True)
            np.savez_compressed(dest/'dino_labels.npz',**stored)
            if gt is not None:
                shown=rgb.copy();shown[p1.boundaries(gt)&valid]=[255,40,160];shown[~valid]=(shown[~valid]*.3).astype(np.uint8)
                p=Image.fromarray(shown);p.thumbnail((640,640));p.save(dest/'reference_overlay.jpg',quality=94)
            save(dest/'dino_metrics.json',{'id':row['id'],'methods':rowscores,'total_seconds':time.perf_counter()-start})
        done+=1;torch.cuda.empty_cache();print('DINO_DONE',row['id'],flush=True)
def masks_to_partition(masks,scores,h,w):
    labels=np.zeros((h,w),np.uint16)
    for label,i in enumerate(np.argsort(scores),1):labels[masks[i]]=label
    return labels
def run_sam(limit):
    import torch
    from segment_anything import sam_model_registry,SamAutomaticMaskGenerator
    torch.set_num_threads(2)
    model=sam_model_registry['vit_b'](checkpoint=str(ROOT/'sources/sam_vit_b_01ec64.pth')).cuda().eval()
    generator=SamAutomaticMaskGenerator(model,points_per_side=32,points_per_batch=64,pred_iou_thresh=.88,stability_score_thresh=.95,crop_n_layers=0)
    done=0
    for row in json.loads((ROOT/'manifest.json').read_text())['images']:
        dest,im,rgb,valid,gt=read_input(row)
        if (dest/'sam_metrics.json').exists():continue
        if limit and done>=limit:break
        st=time.perf_counter()
        with torch.inference_mode():m=generator.generate(rgb)
        masks=[x['segmentation'] for x in m];scores=[x['predicted_iou'] for x in m]
        labels=masks_to_partition(masks,scores,*rgb.shape[:2]);q=score(labels,rgb,valid,gt)
        preview(dest,'sam',labels,rgb,valid);np.savez_compressed(dest/'sam_labels.npz',labels=labels)
        save(dest/'sam_metrics.json',{'id':row['id'],'metrics':q,'mask_count':len(m),'seconds':time.perf_counter()-st,'quality_scores':scores})
        print('SAM_DONE',row['id'],len(m),round(time.perf_counter()-st,1),flush=True);done+=1
def run_sam3(limit):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(2)
    checkpoint=Path(str(research_path('work/sam3_checkpoint_path.txt'))).read_text().strip()
    model=build_sam3_image_model(device='cpu',checkpoint_path=checkpoint,load_from_HF=False)
    model.backbone._apply(lambda t:t.to(torch.bfloat16) if t.is_floating_point() else t)
    model=model.to('cuda').eval();processor=Sam3Processor(model,device='cuda',confidence_threshold=.5);done=0
    for row in json.loads((ROOT/'manifest.json').read_text())['images']:
        dest,im,rgb,valid,gt=read_input(row)
        if (dest/'sam3_metrics.json').exists():continue
        if limit and done>=limit:break
        st=time.perf_counter();masks=[];scores=[];names=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(im)
            for prompt in PROMPTS[row['domain']]:
                processor.reset_all_prompts(state);state=processor.set_text_prompt(prompt,state)
                a=state['masks'].squeeze(1).cpu().numpy();s=state['scores'].float().cpu().numpy()
                masks.extend(a>0.5);scores.extend(map(float,s));names.extend([prompt]*len(s))
        labels=masks_to_partition(masks,scores,*rgb.shape[:2]);q=score(labels,rgb,valid,gt)
        preview(dest,'sam3',labels,rgb,valid);np.savez_compressed(dest/'sam3_labels.npz',labels=labels)
        save(dest/'sam3_metrics.json',{'id':row['id'],'metrics':q,'mask_count':len(masks),'seconds':time.perf_counter()-st,'scores':scores,'mask_prompts':names,'prompts':PROMPTS[row['domain']]})
        print('SAM3_DONE',row['id'],len(masks),round(time.perf_counter()-st,1),flush=True);done+=1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dino','sam','sam3']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    save(ROOT/f'{a.mode}_protocol.json',{'code_sha256':p1.sha(Path(__file__)),'model':p1.MODEL if a.mode=='dino' else a.mode,
        'dino_revision':p1.REV,'dino_methods':METHODS,'pca':'Per-image PCA16 fitted to2048 token samples per method (two whole controls and four local scales); equal sampling, no whitening or explicit L2; no labels. Fourflip token vectors inverse-aligned before averaging.',
        'clustering':'Independent KMeans per method,K6 primary,K12 sensitivity,n_init10,seed42;8192 shared uniform coordinates/image.',
        'metrics':'Reference boundaryF1 tolerance2px Manhattan; ignore-label interiors eroded2px. ARI uses shared100000 randompixels. Oracle-majority-map mIoU maps each predicted partition to majorityGT class on sameimage: diagnostic upperbound-style representability, not zero-shotsemanticaccuracy. RGBedgeF1 top15% gradients is secondary appearanceproxy.',
        'sam':'Official SAM ViT-B, automatic32x32points,predIoU>=.88,stability>=.95,no added crops; highest predictedIoU wins overlaps; uncovered=0.',
        'sam3':'Cached SAM3 checkpoint revision3c879f39826c281e95690f02c7821c4de09afae7; confidence>=.5, fixed domain prompts,highest confidence wins overlaps,uncovered=0. Prompt-assisted comparison.',
        'sam3_prompts':PROMPTS,'sampling':'20 images selected before model outcomes; five/domain. No fine-tuning. Fungal reference scores withheld for unregistered masks.'})
    {'dino':run_dino,'sam':run_sam,'sam3':run_sam3}[a.mode](a.limit)
