"""Read partial results; export native-pixel center windows for visual inspection."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/masahiro/kics-zert2')
OUT=ROOT/'runs/moin_multiscale_pilot_v1'
METHODS={'s05':[5],'s10':[10],'s20':[20],'s40':[40],'s05_10':[5,10],
         's10_20':[10,20],'s05_10_20':[5,10,20],'s10_20_40':[10,20,40],
         's05_10_20_40':[5,10,20,40]}

ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--all',action='store_true');args=ap.parse_args()
selection=json.loads((OUT/'selection.json').read_text())
rows=[json.loads((OUT/r['id']/'metrics.json').read_text()) for r in selection['images'] if (OUT/r['id']/'metrics.json').exists()]
print('Completed',len(rows),'of',len(selection['images']))
print('Observed site counts:',{s:sum(r['site']==s for r in rows) for s in sorted({r['site'] for r in rows})})
for m in METHODS:
    vals=[r['methods'][m]['k']['6']['rgb_edge']['f1'] for r in rows]
    print(m,round(float(np.mean(vals)),4))
for key in ([args.id] if args.id else [r['id'] for r in rows] if args.all else []):
    dest=OUT/key
    rec=json.loads((dest/'metrics.json').read_text())
    im=Image.open(ROOT/'runs/moin_plot_all_20260928/clean_inputs'/f'{key}.png').convert('RGB')
    w,h=im.size;win=384;x0=(w-win)//2;y0=(h-win)//2;box=(x0,y0,x0+win,y0+win)
    frozen=np.load(ROOT/'runs/moin_phase1_benchmark_v1'/key/'transform.npz')
    fields={s:np.load(dest/f'field_s{s:02d}.npy',mmap_mode='r')[y0:y0+win,x0:x0+win].copy() for s in [5,10,20,40]}
    labels=np.load(dest/'labels.npz')
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    except OSError: font=ImageFont.load_default()
    views=[('Original RGB',im.crop(box))]
    for m,ss in METHODS.items():
        field=np.mean(np.stack([fields[s] for s in ss]),axis=0)
        rgb=np.uint8(np.clip((field[:,:,:3]-frozen['rgb_low'])/np.maximum(frozen['rgb_high']-frozen['rgb_low'],1e-6),0,1)*255)
        color=Image.fromarray(rgb)
        lab=' + '.join(str(s)+'%' for s in ss)
        views.append((f'{lab} | F1 {rec["methods"][m]["k"]["6"]["rgb_edge"]["f1"]:.3f}',color))
    cellw=win+20;cellh=win+55
    canvas=Image.new('RGB',(cellw*5,cellh*2+66),'#f3f5ef');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),f'{key} | center 384 x 384 working pixels, no resizing | common PCA colors | F1 scores use the full image',font=font,fill='#173c40')
    for i,(title,image) in enumerate(views):
        x=(i%5)*cellw+10;y=(i//5)*cellh+55
        draw.text((x,y),title,font=font,fill='#173c40');canvas.paste(image,(x,y+34))
    canvas.save(dest/'native_center_comparison.jpg',quality=96)
    print('Native-pixel window:',box,'saved',dest/'native_center_comparison.jpg')
