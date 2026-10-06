"""Image-only Hugging Face SAM3 automatic mask generation. No human prompts."""
import json,time,inspect,platform,argparse,hashlib
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import pipeline
from PIL import Image
from threadpoolctl import threadpool_limits
import ecology75_benchmark as b
P=Path('/home/masahiro/kics-zert2');R=b.ROOT
DOMAINS=['moin','mycelium','fungal_network','bam','neon','coralscapes','tara','pmid']
def run():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',default=','.join(DOMAINS));a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    rev=json.loads((R/'sources/sam3_hf_revision.json').read_text())['revision']
    generator=pipeline('mask-generation',model='facebook/sam3',revision=rev,device=0,dtype=torch.float32)
    protocol={'implementation':'Hugging Face Transformers MaskGenerationPipeline with Sam3TrackerModel','transformers':transformers.__version__,'torch':torch.__version__,'python':platform.python_version(),'model':'facebook/sam3','revision':rev,'model_class':type(generator.model).__name__,'image_processor':generator.image_processor.to_dict(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'call':'generator(image, points_per_batch=4, output_bboxes_mask=True)','user_supplied_prompts':[],'automatic_sampling':'Pipeline generates its own default32x32 point grid; no text, manually selected points, boxes, masks, labels or domain-specific object requests are supplied. This is image-only automatic mask generation, not a model with no internal point prompts.','parameters':{'points_per_batch':4,'points_per_crop':32,'crops_n_layers':0,'pred_iou_thresh':0.88,'stability_score_thresh':0.95,'mask_threshold':0,'stability_score_offset':1,'crops_nms_thresh':0.7},'precision':'FP32 throughout, TF32 disabled. BF16 was rejected in a smoke test because pipeline NMS received mismatched box/score dtypes; no reduced-precision output is used.','cost':'Model load and one complete image-only pipeline warmup excluded. Timed stage covers the unmodified HF pipeline, including filtering/NMS and conversion to returned masks. GPU board energy sampled every200ms. Partition assembly, scoring, rendering and disk writes excluded and timed separately. Pipeline wrapper and precision differ from the historical custom-grid comparator.','overlap':'Highest predicted IoU wins primary; smaller-mask-first sensitivity. Background is label0. Both partitions use the same accepted automatic masks.','previous_comparator':'Original custom-grid sam3_metrics.json and sam3_* outputs remain untouched; the current report uses sam3_hf_metrics.json.'}
    b.mb.save(R/'sam3_hf_protocol.json',protocol)
    power=b.Power();warm=Image.open(R/'moin_01/input.png').convert('RGB')
    with torch.inference_mode():generator(warm,points_per_batch=4)
    torch.cuda.synchronize();print('WARMUP_COMPLETE',flush=True)
    for domain in a.datasets.split(','):
        for r in json.loads((R/f'{domain}_selection.json').read_text())['images']:
            d,im,rgb,gt,valid,rgbvalid,edges=b.inputs(r);outpath=d/'sam3_hf_metrics.json'
            if outpath.exists():continue
            w,h=im.size;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            # The only data input is the image. These kwargs control memory and output format.
            with torch.inference_mode():out=generator(im,points_per_batch=4,output_bboxes_mask=True)
            torch.cuda.synchronize();end=time.perf_counter()
            meta={'seconds':end-start,'gpu_board_joules':power.between(start,end),'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'mask_count':len(out['masks']),'user_supplied_prompt_count':0,'implementation':'hf_mask_generation','dtype':'float32'}
            post=time.perf_counter();masks=[np.asarray(x,dtype=bool) for x in out['masks']];assert all(m.shape==(h,w) for m in masks)
            quality=out['scores'].detach().cpu().float().numpy();assert len(quality)==len(masks) and np.all(quality>.88)
            area=np.array([int(m.sum()) for m in masks]);packed=np.stack([np.packbits(m) for m in masks]) if masks else np.empty((0,(w*h+7)//8),np.uint8)
            np.savez_compressed(d/'sam3_hf_proposals.npz',packed=packed,quality=quality,area=area,shape=[h,w],boxes=out['bounding_boxes'].detach().cpu().numpy())
            scores={}
            for variant,order in [('quality',np.argsort(quality,kind='stable')),('small_first',np.argsort(-area,kind='stable'))]:
                lab=np.zeros((h,w),np.uint16)
                for j in order:lab[masks[j]]=int(j)+1
                scores[variant]=b.score(lab,gt,valid,rgbvalid,edges);np.savez_compressed(d/f'sam3_hf_{variant}_labels.npz',labels=lab)
                b.previews(d,'sam3_hf' if variant=='quality' else 'sam3_hf_small_first',lab,rgb,valid)
            b.mb.save(outpath,{'id':r['id'],'asset_prefix':'sam3_hf','metrics':scores,'extraction':meta,'partition_scoring_preview_seconds':time.perf_counter()-post,'protocol':'sam3_hf_protocol.json'})
            print('HF_DONE',r['id'],json.dumps(meta),flush=True);del out,masks,packed;torch.cuda.empty_cache()
if __name__=='__main__':
    with threadpool_limits(limits=2):run()
