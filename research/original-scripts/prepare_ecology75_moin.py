"""Add ten original MOIN photographs under the same native-window protocol."""
import json,hashlib,shutil,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
P=Path('/home/masahiro/kics-zert2');R=P/'runs/eco_collection75_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
previous=P/'runs/moin_multiscale_pilot_v1/selection.json';source=json.loads(previous.read_text())
chosen=[r for r in source['images'] if r['selection_quintile'] in [1,5] or (r['selection_quintile']==3 and r['site'] in ['fessbach','heidesheim'])]
assert len(chosen)==10
names={'fessbach':'Feßbach','heidesheim':'Heidesheim','rettmer':'Rettmer','ribbeck':'Ribbeck'}
rows=[]
for i,old in enumerate(chosen,1):
    original=P/'datasets/moin/plot_all_inputs/originals'/f"{old['id']}.jpg";assert sha(original)==old['source_sha256']
    raw=Image.open(original);orientation=raw.getexif().get(274,1);im=ImageOps.exif_transpose(raw).convert('RGB');w,h=im.size;assert min(w,h)>=1024
    sw,sh=min(w,2048)//16*16,min(h,2048)//16*16;x,y=(w-sw)//2,(h-sh)//2;box=(x,y,x+sw,y+sh)
    d=R/f'moin_{i:02d}';d.mkdir(exist_ok=True);target=d/'input.png'
    if target.exists():
        assert np.array_equal(np.asarray(Image.open(target)),np.asarray(im.crop(box)))
    else:im.crop(box).save(target)
    thumb=im.crop(box);thumb.thumbnail((640,640));thumb.save(d/'original.jpg',quality=93)
    overview=im.copy();overview.thumbnail((720,720));draw=ImageDraw.Draw(overview);ratio=overview.width/w;draw.rectangle(tuple(round(v*ratio) for v in box),outline='#ffac32',width=3);overview.save(d/'source_overview.jpg',quality=90)
    r={'id':d.name,'dataset':'moin','plot_id':old['id'],'site':old['site'],'stratum':f"{names[old['site']]} · {old['id']} · prior rank {old['prior_visual_rank']}",'original_path':str(original),'original_dimensions':[w,h],'working_dimensions':[sw,sh],'native_window':list(box),'resize_ratio':1,'source_sha256':sha(original),'input_sha256':sha(target),'seed':20261001,'reference_available':False,'reference_kind':None,'reference_note':'No verified leaf/organism segmentation reference. Existing taxonomy and visual rankings are not pixel ground truth.','biological_occupancy':None,'occupancy_basis':'Not measured; no vegetation mask is applied.','smallest_resolved_structure':'Not measured; native pixel dimensions do not establish optical resolution.','physical_resolution':None,'source':'User-provided KICS_ZERT_2025.zip','source_member':old['archive_path'],'license':'User-provided research dataset; no public redistribution license asserted.','selection':'Frozen prior 20-photo pilot: highest and lowest within-site rank quintile members at all four sites, plus middle-quintile members at Feßbach and Heidesheim. This covers prior variation and is not a representative random sample of all439 photos. No new benchmark outcomes used.','prior_visual_rank':old['prior_visual_rank'],'prior_visual_quintile':old['selection_quintile'],'prior_selection_sha256':sha(previous),'exif_orientation_applied':orientation,'native_window_coordinate_system':'EXIF-oriented original pixels','original_note':'Original JPEG retained. EXIF orientation followed by a native center window; no inpainting, vegetation mask, upscaling, or annotation-driven segmentation prompt.'}
    (d/'source.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));rows.append(r)
(R/'moin_selection.json').write_text(json.dumps({'dataset':'moin','images':rows},ensure_ascii=False,indent=2))
canvas=Image.new('RGB',(2000,860),'#f0f4ed');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(rows):
    thumb=Image.open(R/r['id']/'original.jpg');thumb.thumbnail((390,385));x=i%5*400;y=i//5*430;canvas.paste(thumb,(x,y+28));draw.text((x+5,y+5),r['id']+' '+r['plot_id']+' '+r['site'],fill='black')
canvas.save(R/'moin_selection.jpg',quality=94)
print(json.dumps({'prepared':len(rows),'sites':dict(collections.Counter(r['site'] for r in rows)),'images':[r['plot_id'] for r in rows]}))
