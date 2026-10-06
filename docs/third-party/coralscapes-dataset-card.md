---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: image
  splits:
  - name: train
    num_bytes: 4124816883.523
    num_examples: 1517
  - name: validation
    num_bytes: 405601142
    num_examples: 166
  - name: test
    num_bytes: 1175223789
    num_examples: 392
  download_size: 5853001394
  dataset_size: 5705641814.523
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
task_categories:
- image-segmentation
tags:
- biology
- ecology
- climate
license: apache-2.0
size_categories:
- 1K<n<10K
---

# Coralscapes Dataset

The Coralscapes dataset is the first general-purpose dense semantic segmentation dataset for coral reefs. Similar in scope and with the same structure as the widely used Cityscapes dataset for urban scene understanding, Coralscapes allows for the benchmarking of semantic segmentation models in a new challenging domain. 

<div align="center">
<table>
    <tr>
        <td><img src="figures/demo1.png" alt="Image 1" width="300"/></td>
        <td><img src="figures/demo2.png" alt="Image 2" width="300"/></td>
        <td><img src="figures/demo3.png" alt="Image 3" width="300"/></td>
    </tr>
</table>
</div>

## Dataset Structure
The Coralscapes dataset spans 2075 images at 1024×2048px resolution gathered from 35 dive sites in 5 countries in the Red Sea, labeled in a consistent and speculation-free manner containing 174k polygons over 39 benthic classes.
### Data Splits
The dataset provides three splits
- train - 1517 images across 27 dive sites 
- validation - 166 images across 3 dive sites
- test - 392 images across 5 dive sites 

<!-- ### Data categories -->

| **Id**  | **Name**                      | **Color**                | **Description** |
| --- | ------------------------- | -------------------- | ----------- |
| 1   | seagrass                  | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(125,222,125);border:1px solid black;"></span>  | Seagrass |
| 2   | trash                     | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,0,134);border:1px solid black;"></span>  | Includes all kinds of marine litter. Most common are plastic items including bags, cups, and bottles, aluminum cans and glass bottles, as well as abandoned fishing material, and parts of boats and machines. |
| 3   | other coral dead          | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(114,60,61);border:1px solid black;"></span>  | All other dead coral skeletons. |
| 4   | other coral bleached      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(250,224,225);border:1px solid black;"></span> | All other bleached coral, including bleached soft coral. |
| 5   | sand                      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(194,178,128);border:1px solid black;"></span>  | Loose, fine sand. |
| 6   | other coral alive         | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(224,118,119);border:1px solid black;"></span>  | All other live corals. Includes corals in thin plate or encrusting growth form, soft corals, and corals that can not be clearly classified into the other classes. |
| 7   | human                     | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,0,0);border:1px solid black;"></span>  | The human class includes divers and snorkelers, of which sometimes only a hand or a fin is in the frame. Some ambiguity can arise when a human carries a transect tool that is not laid out on the ground, like a transect reel. In these cases, we generally decide this tool becomes part of the human polygon. |
| 8   | transect tools            | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(8,205,12);border:1px solid black;"></span>  | This includes transect reels & spools, tags and markers placed on the reef, diving weights, surface marker buoys and the string attaching them to the ground or weights on the ground. |
| 9   | fish                      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,255,0);border:1px solid black;"></span>  | Fish of all kinds. |
| 10  | algae covered substrate   | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(125,163,125);border:1px solid black;"></span>  | Substrate covered in turf algae or other macroalgae, including fleshy algae and Turbinaria. |
| 11  | other animal              | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(0,255,255);border:1px solid black;"></span>  | Includes starfish (except the crown-of-thorns starfish), feather worms, sea turtles, and other non-identifiable invertebrates or animals of which there are not enough annotations to warrant a separate class. |
| 12  | unknown hard substrate    | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(125,125,125);border:1px solid black;"></span>  | Hard substrate that is part of the reef, which can not be identified into any of the other classes. Includes rocks and heavily decomposed coral skeletons. Also includes human-made structures (underwater infrastructure) such as pipes, pier columns, coral nursery tables, buoys and their lines, as well as boat anchors. |
| 13  | background                | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(29,162,216);border:1px solid black;"></span>  | This class is assigned to pixels that are too far away or too blurry to be classifiable into any other class. This includes the water surface, which is sometimes visible. |
| 14  | dark                      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(31,31,31);border:1px solid black;"></span>  | Parts of the image that are too dark to discern the benthic class. |
| 15  | transect line             | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(0,255,0);border:1px solid black;"></span>  | Rolled out transect tape. Excludes reels or other strands of rope that are laid out. |
| 16  | massive/meandering bleached | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,248,228);border:1px solid black;"></span> | Bleached massive corals or meandering corals that have decomposed enough so that the meandering structure is no longer visible. |
| 17  | massive/meandering alive  | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(236,150,21);border:1px solid black;"></span>  | Corals in a massive growth form. Prominently includes Porites, Favia, Favites, and many others. Includes some likely meandering corals (like Platygyra) where the meandering structure can not be clearly identified. |
| 18  | rubble                    | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(161,153,128);border:1px solid black;"></span>  | Small loose fragments of rocky substrate or dead coral. |
| 19  | branching bleached        | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(252,231,240);border:1px solid black;"></span>  | Bleached branching coral of all kinds, including bleached table acropora. |
| 20  | branching dead            | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(123,50,86);border:1px solid black;"></span>  | Other dead branching coral, including acropora, pocillopora, and stylophora. When overgrown by algae, labeled as ‘algae covered substrate’. |
| 21  | millepora                 | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(244,150,115);border:1px solid black;"></span>  | The ‘fire coral’ Millepora is technically not a coral, but a hydrozoan. Appears most commonly in a branching form (Millepora Dichotoma). |
| 22  | branching alive           | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(226,91,157);border:1px solid black;"></span>  | Branching corals that can surely be determined to be alive, but do not fit in to ‘table acropora alive‘, ‘acropora‘, ‘stylophora’, or ‘pocillopora’ because they are from a different genus or appear slightly blurred in the context. |
| 23  | massive/meandering dead   | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(134,86,18);border:1px solid black;"></span>  | Dead massive corals or meandering corals that have decomposed enough so that the meandering structure is no longer visible. |
| 24  | clam                      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(189,255,234);border:1px solid black;"></span>  | Live giant clams. |
| 25  | acropora alive            | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(236,128,255);border:1px solid black;"></span>  | Acropora that do not grow in tabular growth form. |
| 26  | sea cucumber              | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(0,231,255);border:1px solid black;"></span>  | Sea Cucumber |
| 27  | turbinaria                | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(228,255,119);border:1px solid black;"></span>  | Colloqually the ‘scroll’ coral. Does not include the macroalgae genus Turbinaria. |
| 28  | table acropora alive      | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(189,119,255);border:1px solid black;"></span>  | Acropora that grow in a tabular growth form. |
| 29  | sponge                    | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(240,80,80);border:1px solid black;"></span>  | Sponges of all kind. |
| 30  | anemone                   | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(0,255,189);border:1px solid black;"></span>  | Anemone |
| 31  | pocillopora alive         | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,146,150);border:1px solid black;"></span>  | Clearly identifiable Pocillopora. |
| 32  | table acropora dead       | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(85,53,116);border:1px solid black;"></span>  | Dead Acropora tables which are dead or collapsed but are not yet overgrown by algae. Parts visibly overgrown by algae are labeled as ‘algae covered substrate’. |
| 33  | meandering bleached       | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(251,243,216);border:1px solid black;"></span>  | Bleached meandering corals (Lobophyllia, Symphyllia, Platygyra, etc.). |
| 34  | stylophora alive          | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(255,111,194);border:1px solid black;"></span>  | Clearly identifiable Stylophora. |
| 35  | sea urchin                | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(0,142,255);border:1px solid black;"></span>  | Sea Urchin |
| 36  | meandering alive          | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(230,193,0);border:1px solid black;"></span>  | Corals of a meandering growth form. Includes Platygyra, Lobophyllia, Symphyllia. |
| 37  | meandering dead           | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(119,100,14);border:1px solid black;"></span>  | Dead meandering corals (Lobophyllia, Symphyllia, Platygyra, etc.). |
| 38  | crown of thorn            | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(179,245,234);border:1px solid black;"></span>  | Acanthaster planci, known to cause outbreaks in which it can severely reduce coral cover. |
| 39  | dead clam                 | <span style="display:inline-block;width:100px;height:15px;background-color:rgb(89,155,134);border:1px solid black;"></span>  | Dead giant clams. |

![figure](figures/polygon_counts_colored.png "Number of segmentation masks per class.")
*Number of annotated segmentation masks per class in the Coralscapes dataset splits for each of the 39 classes (shown with linear proportions on logarithmic scale).*

## License
This dataset is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Citation
If you find this project useful, please consider citing:
```bibtex
@misc{sauder2025coralscapesdatasetsemanticscene,
        title={The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs}, 
        author={Jonathan Sauder and Viktor Domazetoski and Guilhem Banc-Prandi and Gabriela Perna and Anders Meibom and Devis Tuia},
        year={2025},
        eprint={2503.20000},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2503.20000}, 
  }
```