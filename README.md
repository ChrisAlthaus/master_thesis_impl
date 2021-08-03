# Retrieval of Similar Paintings in Art using Arrangement of Persons & Objects

## Abstract
The interpretation of artworks is a complex task and requires art historians to analyze
information from the visual content and the context in which the artworks were created. In
order to understand the context of an artwork, general metadata such as information on the
historical origin of the artwork and the painting school of the artist have to be collected and
renovated. In addition to that, theme inspiration among artists and the reuse of stylistic
elements are represented visual links, which offer valuable information to understand an
artwork. Art search engines that have been designed, generally show good results for the
detection of visual similar images, such as replications and copies, however, fail to detect
visual links.
This thesis, addresses this challenge by utilizing information from human poses, and information
regarding the artwork's scene composition. A retrieval system was designed that
is able to detect visual links in artworks, which was shown by an user evaluation. A set
of geometric pose descriptors (GPD) was defined to encode the characteristic postures of
persons. Human poses were predicted by the state-of-the-art Mask R-CNN model, equipped
with a ResNet-50 feature extractor. The scene composition was encoded with scene graphs,
which were generated based on an existing deep learning architecture, Faster R-CNN +
MOTIFS, with unbiased predicate inference. A challenge was the lack of artwork datasets
with either visual link, human pose or scene graph annotations. Therefore, style transfer
was used, enabling the training of these architectures in the art domain. The human pose
detector showed 44.9 mAP on the style-transferred COCO dataset, where keypoint localization
errors contribute more than background errors. The scene graph detector showed
7.0 mR@50 on the style-transferred Visual Genome dataset. Quantitative and qualitative
analyses showed that the retrieval system is able to match characteristic poses and scene
graphs from artworks. An user study was conducted to compare descriptors and similarity
functions, which showed that the retrieval system successfully matches human poses and
the arrangement of objects in artworks.

## Notes
This is the codebase that implements human pose and scene graph detection, training and evaluation.


**This repository is build upon the following projects:**
- Detectron2 https://github.com/facebookresearch/detectron2
- Unbiased Scene Graph Generation https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
- Graph2Vec https://github.com/benedekrozemberczki/graph2vec
- Error Diagnosis for keypoint detection https://github.com/matteorr/coco-analyze

**Directory Structure:**
- flask : User Study (html, evaluation scripts)
- retrieval : Scripts for database querying (GPD, Graph2Vec) 
- scripts
  - branchgraphs: User interface for scene graph search (Jupyter notebook, utils,...)
  - branchkpts:  User interface for human pose search (Jupyter notebook, utils,...)
  - branchtogether: User interface for human pose + scene graph (merged) search (Jupyter notebook, utils,...)
  - detectron2: Mask R-CNN related content
  - graph_descriptors: Graph2Vec related content
  - openpose: - not implemented, just testing -
  - pose_descriptors: Geometric pose descriptor related content
  - posefix: PoseFix refinement related content (not used finally)
  - repofolder_layouts: Structure of the local repositories
  - scenegraph: SG related content
  - singularity: Scripts for GPU execution
  - statistics_dataset: Dataset statistics & visualization scripts
  - style_transfer: Style-transfer related content
  - utils: Utility functions for general filetypes

**General Tips:**
- A 'modified' folder indicates content that was edited from the extern projects.
- If you have file path problems, please look first for the folder structure in repofolder_layouts.
- If you have problems that you cannot solve, write me a message or create an issue.

**Links to Written Thesis & Presentation**

Written thesis: https://www.dropbox.com/s/7rfi25jpsddfg3p/Masterarbeit.pdf?dl=0 <br />
Presentation: https://www.dropbox.com/s/x51szh50gcl5jr2/presentation.pdf?dl=0





