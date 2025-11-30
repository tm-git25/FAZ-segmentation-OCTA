# FAZ Segmentation in OCTA Images

This repository contains the code associated with the paper:

> **Multitask Learning Approcha for Foveal Avascular Zone Segmentation in OCTA Images**  
> Tânia Melo, Ângela Carneiro, Aurélio Campilho, Ana Maria Mendonça, 
> Special Issue on IbPRIA 2025, Pattern Analysis and Applications Journal [under submission]

The work focuses on **deep learning-based segmentation** of the **foveal avascular zone (FAZ)** in **OCTA** images, with a particular emphasis on how **retinal vessel segmentation quality** affects FAZ boundary delineation.

---

## 🚀 Overview

The project has two main components:

1. **Preliminary study on blood vessel segmentation backbones**
   - Evaluation of **OCTA-Net** and **COSNet**, with and without several architectural and training modifications.
   - Experiments performed on multiple public OCTA datasets (e.g., **ROSE** and **OCTA-500**).
   - Selection of a **modified COSNet** as the vessel segmentation backbone.

2. **Two proposed FAZ segmentation approaches**
   - **Approach I – Multitask Learning Framework**  
     Joint segmentation of **vessels** and **FAZ** using a shared encoder and a modified COSNet-based vessel branch.
   - **Approach II – Conditionally Trained Network**  
     A **ResNeSt-based U-Net** for FAZ segmentation trained with a **vessel-aware loss function**, leveraging vessel masks as priors.
