## [CVPR 2026 Findings] Revisiting Image Manipulation Localization under Realistic Manipulation Scenarios

### Code & Dataset Release

We are gradually releasing the code, pretrained models, and datasets for this project.
Currently, we have released the inference code for **under the CAT-Net protocol**.

### Currently Available

* Inference code under the CAT-Net protocol
* Inference script using `test.sh`
* Pretrained model and related files via Baidu Netdisk

Please first download the required files from Baidu Netdisk:

```text
Baidu Netdisk link: https://pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn
```

Before running the inference script, please modify the dataset paths in `test.py` and `test_robustness.py` according to your local environment. Then replace the checkpoint path in `test.sh` with the downloaded pretrained model path.

After that, run:

```bash
bash test.sh
```


### TODO List

We will continue to update this repository with the following components:

* ✅ Inference code
* [ ] Training code
* [ ] Synthetic datasets
* [ ] HSIM dataset
* [ ] Autoregressive inference code

More code and data will be made publicly available as soon as possible.
