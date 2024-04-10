# Fine Tune GPT2 with Colossal AI
This example is reproducing the code from [Clossal-AI](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.py) on GPT2 fine tunning in distributed manners. 

## Introduction
This repository provides an approach to fine-tuning OpenAI's powerful GPT2 language model using Colossal AI, an efficient and user friendly platform for large-scale AI experimentation. Fine-tuning GPT2 allows you to test out practical experience of distributed training with Colosal-AI, unlocking its full potential in various natual language processing task. 

## Requirements

Before you can launch training, you need to install the following requirements

## Dataset Used
The dataset used in this repository example is from [GLUE](https://huggingface.co/datasets/nyu-mll/glue), the General Language Understanding Evaluation benchmark. The task used is *mrpc*: the Microsoft Research Paraphrase Corpus. This corpus of sentence pairs, extracted from online news sources, with human annotations to flag whether the sentences pairs are semantically equivalent. 

## Script Modification

GLUEDataBuilder modified with additioanl padding token --> GLUEDataBuilder_Modified.

Use **TorchDDPPlugin** only due to number of parameter consideration.

Pre-trained GPT2 model token embedding is resized (+1) with additional padding token. Padding token ID set to 0 in model configuration.

Modify **train()** function with loss and accuracy as return values. Used for visualization to see model performance improvement during training.

Modify and simplify **evaluate()** function to avoid out-of-memory issue. Keep loss and accuracy as return values for model evaluation metrics.

## Result
Batch Size: 32, Epoch: max 4 then no decrease in loss, loss: 0.52, Acc: 74.15%, learning rate:2.4e-5, weight decay: 0.01, warmup_fraction=0.1

Batch Size: 32, Epoch: max 4 then no decrease in loss, loss: 0.536, Acc: 72.7%, learning rate:1.2e-5, weight decay: 0.01, warmup_fraction=0.05

Batch Size: 16, Epoch: max 4 then no decrease in loss, loss: 0.54, Acc: 73.2%, learning rate:2.4e-5, weight decay: 0.01, warmup_fraction=0.1


**Final parameter:**
- Batch Size: 32
- Learning Rate: 2.4e-5
- Weight Decay: 0.01
- Warmup Fraction: 0.1

**Final Result:**
- Training Loss: 0.445
- Training Accuracy: 79.2%
- Testing Loss: 0.48
- Testing Accuracy: 77.4%

![image](https://github.com/ZhuSisi777/gpt2_fine_tune_with_colossalai/assets/115344451/c36cd24e-19ce-4b90-89a2-f7d8864d850b)
![image](https://github.com/ZhuSisi777/gpt2_fine_tune_with_colossalai/assets/115344451/f9c36971-dd6e-469a-831f-f01f2de14c51)



