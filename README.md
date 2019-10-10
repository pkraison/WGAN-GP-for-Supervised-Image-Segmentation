# WGAN-GP-for-Supervised-Image-Segmentation

Code for the paper ['Generative Adversarial Networks to Synthetically Augment Data for Deep Learning based Image Segmentation'](https://diglib.tugraz.at/download.php?id=5b3619809d758&location=browse)

This version uses WGAN-GP as its GAN model. 

For more details, check out our [paper](https://diglib.tugraz.at/download.php?id=5b3619809d758&location=browse).

For citations ([bibtex](cite.bib)):
```
Neff, Thomas and Payer, Christian and Štern, Darko and Urschler, Martin (2018).
Generative Adversarial Networks to Synthetically Augment Data for Deep Learning based Image Segmentation.
In Proceedings of the OAGM Workshop 2018, pp. 22–29.
```

# Medical Data Augmentation Framework
Please also check out the work of my co-authors at https://github.com/christianpayer/MedicalDataAugmentationTool , which contains a fully-fledged medical data augmentation framework to use for deep learning. 


# Credits
Credits to [igul222](https://github.com/igul222) for their [wgan-gp tensorflow implementation](https://github.com/igul222/improved_wgan_training), which this code is based on.

# NOTE
This code was not tested since the paper was published, and I can not offer setup or usage instructions or maintain it, it is mainly here for completeness. It was mainly used with the Cityscapes dataset and the SCR Lung database dataset. 
The loading of files is similar to my other repo [here](https://github.com/thomasneff/Generative-Adversarial-Network-based-Synthesis-for-Supervised-Medical-Image-Segmentation).

The main model file is gan_cityscapes.py.

