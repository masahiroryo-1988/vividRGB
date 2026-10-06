"""Evaluate reviewed leaf boundaries; optional independent leaf-instance predictions.
Usage: python score_leaf_references.py BENCHMARK_DIR references.json [--instances DIR]
Reference uint16 PNG: 0=background, 1..65534=visible leaf IDs, 65535=ignore.
Current K-means labels are appearance groups: they are NEVER scored as leaf instances.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
from scipy.optimize import linear_sum_assignment

def edge(a):
 b=np.zeros(a.shape,bool);b[:-1]|=a[:-1]!=a[1:];b[:,:-1]|=a[:,:-1]!=a[:,1:];return b
def boundary_score(pred,gt,valid,tol=2):
 valid=binary_erosion(valid,iterations=tol+1,border_value=0);a=edge(pred)&valid;b=edge(gt)&valid
 p=float((a&binary_dilation(b,iterations=tol)).sum()/a.sum()) if a.any() else 0.;r=float((b&binary_dilation(a,iterations=tol)).sum()/b.sum()) if b.any() else 0.
 return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
def instances(pred,gt,valid):
 pi=np.unique(pred[valid]);pi=pi[(pi>0)&(pi<65535)];gi=np.unique(gt[valid]);gi=gi[(gi>0)&(gi<65535)];iou=np.zeros((len(pi),len(gi)))
 for i,p in enumerate(pi):
  for j,g in enumerate(gi):
   a=(pred==p)&valid;b=(gt==g)&valid;iou[i,j]=(a&b).sum()/(a|b).sum()
 if iou.size:ii,jj=linear_sum_assignment(-iou);matched=iou[ii,jj]
 else:matched=np.array([])
 tp=int((matched>=.5).sum());precision=tp/len(pi) if len(pi) else 0.;recall=tp/len(gi) if len(gi) else 0.
 return {'predicted_leaves':len(pi),'reference_leaves':len(gi),'one_to_one_iou_sum':float(matched.sum()),'mean_iou_per_reference_leaf_unmatched_zero':float(matched.sum()/len(gi)) if len(gi) else None,'precision_iou50':precision,'recall_iou50':recall,'f1_iou50':2*precision*recall/(precision+recall) if precision+recall else 0.}
def main(args):
 root=Path(args.benchmark);refpath=Path(args.references);spec=json.loads(refpath.read_text());scores=[]
 for r in spec['windows']:
  if not r.get('reviewed'):continue
  mask=refpath.parent/r['mask'];gt=np.array(Image.open(mask));x,y,xx,yy=r['box'];assert gt.shape==(yy-y,xx-x)
  labels=np.load(root/r['image_id']/'labels.npz');valid=(gt!=65535)&labels['valid'][y:yy,x:xx];assert valid.sum()>0,'Reviewed mask is all-ignore';has_leaf=bool(np.any((gt>0)&(gt<65535)&valid));assert has_leaf or r.get('background_only') is True,'Explicitly mark reviewed leaf-free windows background_only=true'
  out={'window_id':r['id'],'image_id':r['image_id'],'boundary_accuracy':{}}
  for method in ['whole','crop384','crop_native']:
   out['boundary_accuracy'][method]={str(k):(boundary_score(labels[f'{method}_k{k}'][y:yy,x:xx],gt,valid) if has_leaf else {'f1':None,'false_boundary_density':float(edge(labels[f'{method}_k{k}'][y:yy,x:xx])[valid].mean()),'note':'Reviewed background-only window; no reference contours for F1'}) for k in [4,6,8]}
  if args.instances:
   p=Path(args.instances)/(r['id']+'.png')
   if p.exists():
    pred=np.array(Image.open(p));assert pred.shape==gt.shape;out['leaf_instance_accuracy']=instances(pred,gt,valid)
  scores.append(out)
 if not scores:raise ValueError('No reviewed reference windows. Accuracy cannot be calculated from empty/unreviewed masks.')
 result={'reference_windows':len(scores),'images':len({r['image_id'] for r in scores}),'results':scores,'note':'RGB-edge proxies are not used here. Appearance boundaries compared against reviewed leaf boundaries. Instance IoU only when separate leaf-instance predictions supplied. No leaf counting from K-means colors.'}
 path=root/'reference_evaluation.json';path.write_text(json.dumps(result,indent=2));print(path)
def test():
 gt=np.zeros((60,60),np.uint16);gt[5:25,5:25]=1;gt[35:55,35:55]=2;valid=np.ones(gt.shape,bool)
 assert boundary_score(gt,gt,valid)['f1']==1
 assert instances(gt,gt,valid)['mean_iou_per_reference_leaf_unmatched_zero']==1
 assert instances(np.zeros_like(gt),gt,valid)['recall_iou50']==0
 merged=(gt>0).astype(np.uint16);assert instances(merged,gt,valid)['recall_iou50']==.5
 print('PASS reference evaluator: perfect, empty, merged instances')
if __name__=='__main__':
 import sys
 if '--selftest' in sys.argv:test()
 else:
  p=argparse.ArgumentParser();p.add_argument('benchmark');p.add_argument('references');p.add_argument('--instances');main(p.parse_args())
