# Weakly Supervised Semantic Segmentation based on MMSegmentation
## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb) on Colab.

To migrate from MMSegmentation 0.x, please refer to [migration](docs/en/migration).

## Tutorial

<details>
<summary>Get Started</summary>

- [MMSeg overview](docs/en/overview.md)
- [MMSeg Installation](docs/en/get_started.md)
- [FAQ](docs/en/notes/faq.md)

</details>

<details>
<summary>MMSeg Basic Tutorial</summary>

- [Tutorial 1: Learn about Configs](docs/en/user_guides/1_config.md)
- [Tutorial 2: Prepare datasets](docs/en/user_guides/2_dataset_prepare.md)
- [Tutorial 3: Inference with existing models](docs/en/user_guides/3_inference.md)
- [Tutorial 4: Train and test with existing models](docs/en/user_guides/4_train_test.md)
- [Tutorial 5: Model deployment](docs/en/user_guides/5_deployment.md)
- [Deploy mmsegmentation on Jetson platform](docs/zh_cn/user_guides/deploy_jetson.md)
- [Useful Tools](docs/en/user_guides/useful_tools.md)
- [Feature Map Visualization](docs/en/user_guides/visualization_feature_map.md)
- [Visualization](docs/en/user_guides/visualization.md)

</details>

<details>
<summary>MMSeg Detail Tutorial</summary>

- [MMSeg Dataset](docs/en/advanced_guides/datasets.md)
- [MMSeg Models](docs/en/advanced_guides/models.md)
- [MMSeg Dataset Structures](docs/en/advanced_guides/structures.md)
- [MMSeg Data Transforms](docs/en/advanced_guides/transforms.md)
- [MMSeg Dataflow](docs/en/advanced_guides/data_flow.md)
- [MMSeg Training Engine](docs/en/advanced_guides/engine.md)
- [MMSeg Evaluation](docs/en/advanced_guides/evaluation.md)

</details>

<details>
<summary>MMSeg Development Tutorial</summary>

- [Add New Datasets](docs/en/advanced_guides/add_datasets.md)
- [Add New Metrics](docs/en/advanced_guides/add_metrics.md)
- [Add New Modules](docs/en/advanced_guides/add_models.md)
- [Add New Data Transforms](docs/en/advanced_guides/add_transforms.md)
- [Customize Runtime Settings](docs/en/advanced_guides/customize_runtime.md)
- [Training Tricks](docs/en/advanced_guides/training_tricks.md)
- [Contribute code to MMSeg](.github/CONTRIBUTING.md)
- [Contribute a standard dataset in projects](docs/zh_cn/advanced_guides/contribute_dataset.md)
- [NPU (HUAWEI Ascend)](docs/en/device/npu.md)
- [0.x → 1.x migration](docs/en/migration/interface.md)，[0.x → 1.x package](docs/en/migration/package.md)

</details>


## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

# DeepLabV3+

> [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

<a href="https://github.com/tensorflow/models/tree/master/research/deeplab">[Official Repo]</a>
<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/sep_aspp_head.py#L30">[Code Snippet]</a>

## Training on a single GPU
Use `tools/train.py` to launch training jobs on a single GPU. The basic usage is as follows.
```bash
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512.py
```
```bash
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r101-d8_80k_mscoco-512x512.py
```

## Training on multiple GPUs
OpenMMLab2.0 implements distributed training with MMDistributedDataParallel. Use  `tools/dist_train.sh` to launch training on multiple GPUs.
```bash
sh tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512.py ${GPU_NUM}
```
## Results and models

### Pascal VOC 2012 + Aug
| Method     | Backbone | Crop Size | Lr schd | Mem (GB) |  Type  |  mIoU | mIoU(ms+flip) | 
| ---------- | -------- | --------- | ------: | -------- | ------ | ----: | ------------: | 
| DeepLabV3+ | R-101-D8 | 512x512   |   40000 | -        |  FSSS  | 78.62 |         79.53 |
| DeepLabV3+ | R-101-D8 | 512x512   |   40000 | 11       |  WSSS  | 7?    |         ?     |

### MSCOCO
| Method     | Backbone | Crop Size | Lr schd | Mem (GB) |  Type  |  mIoU | mIoU(ms+flip) | 
| ---------- | -------- | --------- | ------: | -------- | ------ | ----: | ------------: | 
| DeepLabV3+ | R-101-D8 | 512x512   |  160000 | -        |  FSSS  | 58.85 |               |
| DeepLabV3+ | R-101-D8 | 512x512   |  160000 | -        |  WSSS  | 5?    |               |

