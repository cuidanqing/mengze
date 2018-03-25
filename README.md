# mengze

**The model is still under training.**

## Train

* You can train model from scratch or finetune from imagenet pretrained model.

* The models we trained are based on ResNet-50.

Before training, you should change the correspoding parameters in `train_from_scratch.py` or `train_from_imagenet.py`. (Such as `train_path`, `label_path`, `log_save_path`, `model_save_dir` and etc)

**train**
```bash
python train_from_imagenet.py
```
or
```bash
python train_from_scratch.py
```

## Inference

Change `model_path`, `train_path`, `label_path`, `infer_path` in `infer.py`, than run `python infer.py` to generate inference result, the result will be saved in `infer_path`.

## Test

Then, you can go `tools` directory, and use `mAP.py` to calculate the map value.


