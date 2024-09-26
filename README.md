
# Vision Transformer Paper Replication

This is a paper replication for the Vision Transformer model introduced in [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) by Dosovitsky et al. 2021

The code uses the ML python library PyTorch

A Google Colab version can be found [here](https://colab.research.google.com/drive/1xp1HRDoZkUbBG2vToQdcLhEHbea7GS0E?usp=sharing)
Or use VisionTransformer.ipynb


## Model

In the paper they split the training of the models between pretraining and finetuning. I used a model pretrained on ImageNet1K
and finetuned using the Oxford IIIT Pets Dataset that I accessed using [HuggingFace](https://huggingface.co/datasets/timm/oxford-iiit-pet)

Due to consideration for computational resouces, I implemented one of the smallest models, ViT-B 16, which has 86M parameters (see Table 1 of the paper for more details)

The model takes in a batch of images formatted as a tensor with shape [batch_size, colour_channels, height, width]

The model output is a tensor with shape [batch_size, num_labels]


## Results

In the paper, the Test Accuracy for the ViT-B 16 model pretrained on ImageNet1K and finetuned on Oxford Pets is 93.81% (see Table 5 in the paper)

I achieved a Test Accuracy of 92.46% - I believe this is due to me using a much smaller batch size (8 instead of 512) during finetuning training. 

I used a NVIDIA A100 Tensor Core GPU through Google Colab, but still reached memory limitations when using larger batch sizes. The training of the finetuning model took just over 2 hours. I ran the testing loop 3 times to get an average for the Test Accuracy.

## Installation

```bash
  conda create --name vit_paper_rep_env --file requirements.txt
```

    
## Usage/Examples
Run finetuning train pipeline using Oxford IIIT Pets dataset with the pretrained vitB16 model from torchvision (trained on ImageNet1K):  

```bash
python finetuning_train.py
```
Run finetuning test pipeline:

```bash
python finetuning_test.py
```
Run tests

```bash
python tests.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

