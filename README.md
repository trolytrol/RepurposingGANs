# RepurposingGANs

This is the official code for:

#### Repurposing GANs for One-shot Semantic Part Segmentation

<sup>Nontawat Tritrong*, Pitchaporn Rewatbowornwong*, [Supasorn Suwajanakorn](https://www.supasorn.com/)<sup>

<sup>\* authors contributed equally <sup>

**CVPR-2021, Oral** **[[paper](https://arxiv.org/pdf/2103.04379.pdf)] [[Project Page](https://repurposegans.github.io/)]**

### Prerequisites
```
pip install -r requirements.txt
```

## Extracting GAN's feature
To reduce the computational cost, we precompute GAN's feature before training.
```
python gen_feature.py
```

## Few-shot training
```
python train.py --feature_dir fewshot/features --mask_dir fewshot/mask --cfg config/celeba_10c.yaml
```

## Auto-shot training
After few-shot training, you can generate image-mask pairs for auto-shot training.
```
python gen_dataset.py --outdir autoshot_dataset --num_pic 5000
```
Then, Auto-shot can be trained with:
```
python train_unet.py --image_dir autoshot_dataset/images --mask_dir autoshot_dataset/mask --checkpoint_dir checkpoints --cfg config/celeba_10c.yaml
```

## Credit
We'd like to thank the following implementations which we have used in this project:

- GANDissect (https://github.com/CSAILVision/GANDissect)
- StyleGAN2-pytorch (https://github.com/rosinality/stylegan2-pytorch)

## Citation

```bibtex
@inproceedings{tritrong2021repurposing,
  title={Repurposing gans for one-shot semantic part segmentation},
  author={Tritrong, Nontawat and Rewatbowornwong, Pitchaporn and Suwajanakorn, Supasorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4475--4485},
  year={2021}
}
```