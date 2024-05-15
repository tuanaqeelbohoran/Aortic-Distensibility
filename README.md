[info] More files will be added once we finish beta testing at Lester Hospital Glenfield, UK.

**Aortic Distensibility prediction by fully automated image segmentation**

**Abstract**:
Aortic distensibility (AD) is important for the prognosis of multiple cardiovascular diseases. We propose a novel resource-efficient deep learning (DL) model, inspired by the bi-directional ConvLSTM U-Net with densely connected convolutions, to perform end-to-end hierarchical learning of the aorta from cine cardiovascular MRI towards streamlining AD quantification. Unlike current DL aortic segmentation approaches, our pipeline: (i) performs simultaneous spatio-temporal learning of the video input, (ii) combines the feature maps from the encoder and decoder using non-linear functions, and (iii) takes into account the high class imbalance. By using multi-centre multi-vendor data from a highly heterogeneous patient cohort, we demonstrate that the proposed method outperforms the state-of-the-art method in terms of accuracy and at the same time it consumes 
 3.9 times less fuel and generates 
 2.8 less carbon emissions. Our model could provide a valuable tool for exploring genome-wide associations of the AD with the cognitive performance in large-scale biomedical databases. By making energy usage and carbon emissions explicit, the presented work aligns with efforts to keep DL’s energy requirements and carbon cost in check. The improved resource efficiency of our pipeline might open up the more systematic DL-powered evaluation of the MRI-derived aortic stiffness.

![satori1](https://github.com/tuanaqeelbohoran/Aortic-Distensibility/assets/120468459/f57fb3ff-1f47-40fe-8e75-8d0040a3d8a1)





![satori2](https://github.com/tuanaqeelbohoran/Aortic-Distensibility/assets/120468459/acad6823-55e7-4b40-bca6-3f38c3bb3b0a)

### References

Below is a citation for one of our research papers:

```bibtex
@article{Bohoran2023,
  author = {Bohoran, T. and Parke, K. and Graham-Brown, M. and Meisuria, M. and Singh, A. and Wormleighton, J. and Adlam, D. and Gopalan, D. and Davies, M. and Williams, B. and Brown, M. and McCann, G. and Giannakidis, A.},
  title = {Resource efficient aortic distensibility calculation by end to end spatiotemporal learning of aortic lumen from multicentre multivendor multidisease CMR images},
  journal = {Scientific Reports},
  volume = {13},
  pages = {21794},
  year = {2023},
  month = {12},
  day = {8},
  doi = {10.1038/s41598-023-48986-6},
}
