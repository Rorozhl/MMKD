# MMKD

This repo covers the implementation of the following ICME 2023 paper:
Adaptive Multi-Teacher Knowledge Distillation with Meta-Learning

## Installation
This repo was tested with Python 3.6, PyTorch 1.8.1, and CUDA 11.1.

## Running
Before distill the student, be sure to put the teacher model directory in setting.py.
``` shell
nohup python train_meta.py --model_s vgg8 --teacher_num 3 --distill inter --ensemble_method META --nesterov -r 1 -a 1 -b 100 --hard_buffer  --convs  --trial 0  --gpu_id 0&
```
where the flags are explained as:
* `--distill`: specify the distillation method
* `--model_s`: specify the student model, see 'models/__init__.py' to check the available model types.
* `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
* `-a`: the weight of the KD loss, default: `1`
* `-b`: the weight of other distillation losses, default: `0`
* `--teacher_num`: specify the ensemble size (number of teacher models)
* `--ensemble_method`: specify the ensemble_method
* `--hard_buffer`: whether a hard buffer is required
* `convs`: the way of feature alignment. If not, just use 1x1 convolution for alignment
  

## Citation
If you find this repository useful, please consider citing the following paper:
```

```

## Acknowledgement

The implementation of compared methods are mainly based on the author-provided code and the open-source benchmark https://github.com/HobbitLong/RepDistiller and https://github.com/alinlab/L2T-ww. 
