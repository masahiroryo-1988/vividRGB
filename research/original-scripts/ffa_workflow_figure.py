from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from PIL import Image,ImageOps
base=Path('/home/masahiro/kics-zert2');out=base/'work/ffa-preprint/figures'
im=Image.open(base/'runs/eco_collection75_v1/moin_01/input.png').convert('RGB')
crop=im.crop((770,920,975,1125)).resize((384,384),Image.Resampling.BICUBIC)
fig=plt.figure(figsize=(11.5,5.8));axs=[fig.add_axes(x) for x in [[.015,.52,.23,.38],[.285,.52,.2,.38],[.545,.57,.19,.3],[.775,.52,.22,.38]]]
axs[0].imshow(im);axs[0].add_patch(Rectangle((770,920),205,205,fill=False,color='#f2be43',lw=2));axs[0].set_title('1  Restrict context',fontsize=13)
axs[1].imshow(crop);axs[1].set_title('2  Enlarge to 384 × 384',fontsize=13)
views=[crop,ImageOps.mirror(crop),ImageOps.flip(crop),ImageOps.flip(ImageOps.mirror(crop))]
for j,v in enumerate(views):
    ax=fig.add_axes([.535+(j%2)*.095,.72-(j//2)*.185,.085,.165]);ax.imshow(v);ax.axis('off')
axs[2].axis('off');fig.text(.628,.92,'3  Frozen DINOv3',ha='center',fontsize=13);fig.text(.628,.47,'Four reflected inputs\nInverse-align\nand average\n24 × 24 patch tokens',ha='center',va='top',fontsize=13)
ffa=Image.open(base/'runs/eco_gmm_final50_v1/moin_01/s10_pc16_varimax.jpg');axs[3].imshow(ffa);axs[3].set_title('4  Reconstruct cues',fontsize=13)
for ax in [axs[0],axs[1],axs[3]]:ax.axis('off')
for x1,x2 in [(.248,.28),(.49,.528),(.735,.772)]:
    fig.add_artist(FancyArrowPatch((x1,.7),(x2,.7),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=16,color='#087f8c',lw=1.5))
fig.text(.125,.47,'Square crop\n5% or 10% of\nworking short side',ha='center',va='top',fontsize=13)
fig.text(.385,.47,'Bicubic interpolation\nRGB / 255\nthen mean/std',ha='center',va='top',fontsize=13)
fig.text(.885,.47,'Shared per-image PCA\nBilinear placement\nHann blending\n~75% overlap',ha='center',va='top',fontsize=13)
fig.text(.02,.26,'A change of scale and context at inference time',fontsize=14,color='#087f8c',weight='bold')
fig.text(.02,.14,'The frozen encoder sees enlarged local RGB crops. Overlap and aligned reflections combine local feature views.\nPCA or PCA + varimax displays cues. K-means or penalized GMM summarizes visual groups.',fontsize=12,linespacing=1.6)
fig.savefig(out/'workflow.pdf',bbox_inches='tight');fig.savefig(out/'workflow.png',dpi=200,bbox_inches='tight')
