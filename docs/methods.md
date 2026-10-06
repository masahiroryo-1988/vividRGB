# Method and scientific scope

The frozen transformer weights and attention implementation are unchanged. Forced focused
attention is an inference-time intervention in source scale and supplied context:

1. Set crop side `max(16, round(focus * min(width, height)))`.
2. Use stride `max(1, round(crop_side / 4))`, adding a final edge-aligned window on each axis.
   This gives approximately 75% overlap, with slightly different realized overlap after rounding.
3. Bicubically enlarge each crop to 384 x 384 RGB pixels.
4. Encode four orientations: identity, horizontal, vertical and both reflections. Undo each
   reflection on its raw token grid, average in FP32, then store FP16 averaged tokens.
5. Center and project raw descriptors into the frozen per-image unwhitened PCA basis.
6. Bilinearly interpolate projected grids, multiply separable Hann weights with per-axis
   floor 0.05, accumulate overlapping fields and divide by strictly positive coverage.

The pinned DINOv3 ViT-B/16 provides 768-dimensional raw patch descriptors. Its class and
register tokens are discarded. RGB is divided by 255 and standardized by processor mean/std.
The default 224-pixel image-processor resize is bypassed. No further descriptor L2
normalization is applied. Model tokens have contextual dependence; the displayed dense
pixel field is interpolated and does not encode every output pixel independently.

PCA uses centered, unwhitened scores. Higher-variance directions therefore already contribute
larger Euclidean differences. Varimax rotates all retained projection axes orthogonally with
Kaiser row normalization only during optimization; it does not whiten image features.

The primary five-domain F1 study used independent K=6 K-means on PCA16 and a common 8192
sampled image-coordinate set, not the later automatic K or GMM maps. Automatic clustering
is a separate experiment using aspect-preserving cell-center grids with longest side 256.
GMM uses `S_lambda(K) = -2 log L_K + lambda p_K log n`, where
`p_K = K [d + d(d+1)/2] + K - 1`; lambda=4 is retained as an exploratory complexity choice.

Strong-RGB-edge boundary F1 compares partition boundaries to the strongest eligible 15%
of sigma=1 grayscale Sobel gradients, using a two-pixel Manhattan tolerance. It quantifies
appearance-boundary recovery, not leaf, organism, species or trait accuracy. Coralscapes
has a separate annotated class-boundary endpoint and gives a different model ordering.
The five-domain sample is illustrative and not a representative ecological population sample.

Energy is GPU board power integrated over declared stages at approximately 200-ms sampling.
It is not wall-plug power or carbon emissions. Efficiency is per-image `1/J`; reported means
average those reciprocals. Complete-stage DINOv3 energy was not measured in the frozen run.

The historical 439-image development study used different working inputs, exclusions,
approximately 33% overlap and no identical six-treatment calibration. It must not be
silently merged with the final 50-image, 75%-overlap comparison.

