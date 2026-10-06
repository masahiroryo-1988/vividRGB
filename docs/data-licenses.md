# Dataset provenance and terms

MIT applies to original vividRGB software and documentation; it does not replace
photograph, annotation or checkpoint licenses. Frozen numerical results record the
source licenses as they were recorded during acquisition. The new release manifest
records the owner's subsequent permission for MOIN publication, without rewriting history.

| Dataset | Provider | Terms and release treatment |
| --- | --- | --- |
| MOIN/KICS-Zert | Dataset owner, Masahiro Ryo / KICS-Zert collection | Public redistribution of the MOIN photographs was expressly authorized for this release on 6 October 2026. This permission is recorded separately; the code's MIT license is not asserted as a general photograph license. |
| Fungal Network Analysis | Mark Fricker, Carlos Aguilar-Trigueros, Lynne Boddy, Matthias Rillig; [Zenodo 5725751](https://zenodo.org/records/5725751) | CC BY 4.0. Cite the dataset and Aguilar-Trigueros et al. (2022), *Network traits predict ecological strategies in fungi*, DOI 10.1038/s43705-021-00085-1. Native cropped working inputs are derivatives; originals and crop coordinates are recorded. |
| NeonTreeEvaluation | Ben Weinstein, Sergio Marconi, Ethan White; [Zenodo 5914554](https://zenodo.org/records/5914554) | CC BY 4.0 in the provider's dataset metadata. This collection uses archived airborne RGB imagery, not a UAV acquisition. Native crop and geospatial resolution metadata are retained. |
| Coralscapes | Coralscapes authors / EPFL ECEO; [dataset card](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) | Apache-2.0 in the dataset card. Cite the dataset paper and retain provider attribution. Working images and published class references remain under the upstream terms. |
| PMID2019 | OUC Ocean Group; [original repository](https://github.com/ouc-ocean-group/PMID2019) | No explicit image redistribution license was found in the audited repository. Numerical evidence, source identifiers, frozen non-RGB descriptors and retrieval instructions are included; its RGB photographs are obtained directly from the provider instead of republished. |

The release does not include MyceliumSeg, BAMFORESTS or Tara Oceans, which were excluded
from the retained five-domain comparison. Public source availability alone is not treated
as a permission to relicense photographs. See the asset manifest for individual filenames,
SHA-256 digests, working windows and original source URLs.

The DINOv3 and SAM3 model checkpoints are not redistributed. Their terms and gated access
are controlled by Meta/Hugging Face. Third-party Python dependencies retain their own licenses.

Full dataset attribution, upstream license texts and provider metadata receipts are retained
in [third-party/NOTICE.md](third-party/NOTICE.md). The same notices accompany the research
release assets in a separate small archive.
