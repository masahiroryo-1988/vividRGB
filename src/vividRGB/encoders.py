"""Optional pretrained backends; model access and model licenses remain upstream."""
from contextlib import nullcontext
import numpy as np

DINO_MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
DINO_REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'
SAM3_MODEL = 'facebook/sam3'
SAM3_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'


class DINOv3Encoder:
    """Raw patch tokens with RGB/255 and processor mean/std, without image resizing.

    CUDA uses FP16 autocast, FP16 returned tokens and disabled matmul TF32 as in
    the paper. CPU uses FP32 forward computation, with FP16 storage of tokens.
    Pass a local checkpoint path or request the pinned upstream model. Accept
    its gated model terms on Hugging Face before downloading.
    """
    interpolation_backend = 'torch'

    def __init__(self, model_id=DINO_MODEL, revision=DINO_REVISION, device=None,
                 local_files_only=False):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise ImportError('Install the encoder extra: pip install "vividRGB[dino]"') from error
        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_id, revision=revision,
                                              local_files_only=local_files_only,
                                              attn_implementation='sdpa').to(self.device).eval()
        self.patch_size = self.model.config.patch_size
        self.mean = torch.tensor(processor.image_mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(processor.image_std, device=self.device)[None, :, None, None]
        self.metadata = {'model': str(model_id), 'revision': revision, 'device': str(self.device),
                         'patch_size': self.patch_size, 'feature_normalization': 'raw tokens; no extra L2',
                         'precision': 'FP16 CUDA autocast' if self.device.type == 'cuda' else 'FP32 CPU',
                         'rgb_scale': 255, 'rgb_mean': processor.image_mean, 'rgb_std': processor.image_std}

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def warm_up(self):
        from PIL import Image
        self.encode([Image.new('RGB', (384, 384))])
        self.synchronize()

    def encode(self, images):
        torch = self.torch
        arrays = [np.asarray(im.convert('RGB')) for im in images]
        if not arrays or len({a.shape for a in arrays}) != 1:
            raise ValueError('An encoder batch must contain equally sized RGB images')
        arr = np.stack(arrays)
        h, w = arr.shape[1:3]
        # No implicit crop/resize/padding: the frozen benchmark used aligned working windows.
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f'DINO inputs must be multiples of {self.patch_size}; prepare an aligned working window')
        pixels = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).to(self.device).float() / 255
        amp = torch.autocast('cuda', dtype=torch.float16) if self.device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), amp:
            z = self.model(pixel_values=(pixels - self.mean) / self.std).last_hidden_state
        registers = getattr(self.model.config, 'num_register_tokens', 0)
        z = z[:, 1 + registers:]
        return z.reshape(len(images), h // self.patch_size, w // self.patch_size, -1).half().cpu().numpy()


class SAM3Automatic:
    """Image-only automatic masks; the pipeline generates its own internal point grid."""
    def __init__(self, device=None, revision=SAM3_REVISION, points_per_batch=4):
        try:
            import torch
            from transformers import pipeline
        except ImportError as error:
            raise ImportError('Install pip install "vividRGB[sam3]" and obtain upstream SAM3 access') from error
        self.torch = torch
        self.device = (0 if torch.cuda.is_available() else -1) if device is None else device
        self.points_per_batch = points_per_batch
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.generator = pipeline('mask-generation', model=SAM3_MODEL, revision=revision,
                                  device=self.device, dtype=torch.float32)

    def generate(self, image):
        from .features import open_rgb
        im = open_rgb(image)
        with self.torch.inference_mode():
            output = self.generator(im, points_per_batch=self.points_per_batch, output_bboxes_mask=True)
        masks = (np.stack([np.asarray(m, dtype=bool) for m in output['masks']])
                 if output['masks'] else np.empty((0, im.height, im.width), dtype=bool))
        raw_scores = output['scores']
        if hasattr(raw_scores, 'detach'):
            raw_scores = raw_scores.detach().cpu().float().numpy()
        scores = np.asarray(raw_scores, dtype=float)
        if masks.shape[1:] != (im.height, im.width) or len(scores) != len(masks):
            raise RuntimeError('SAM3 returned inconsistent mask dimensions or scores')
        labels = np.zeros((im.height, im.width), np.int32)
        for index in np.argsort(scores, kind='stable'):
            labels[masks[index]] = int(index) + 1
        return {'labels': labels, 'masks': masks, 'scores': scores,
                'user_supplied_prompts': [], 'overlap_policy': 'highest predicted IoU wins'}
