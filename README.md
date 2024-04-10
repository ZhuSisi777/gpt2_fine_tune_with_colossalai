# Fine Tune GPT2 with Colossal AI
This example is reproducing the code from [Clossal-AI](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/gpt/hybridparallelism/finetune.py) on GPT2 fine tunning in distributed manners. 

## Introduction
This repository provides an approach to fine-tuning OpenAI's powerful GPT2 language model using Colossal AI, an efficient and user friendly platform for large-scale AI experimentation. Fine-tuning GPT2 allows you to test out practical experience of distributed training with Colosal-AI, unlocking its full potential in various natual language processing task. 

## Requirements

### Install PyTorch
```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### [Install Colossal-AI](https://github.com/hpcaitech/ColossalAI#installation)
```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
pip install -r ColossalAI/requirements/requirements.txt
pip install ColossalAI/
```

### Install requirements
```bash
pip install -r requirements.txt
```

## Dataset Used
The dataset used in this repository example is from [GLUE](https://huggingface.co/datasets/nyu-mll/glue), the General Language Understanding Evaluation benchmark. The task used is *mrpc*: the Microsoft Research Paraphrase Corpus. This corpus of sentence pairs, extracted from online news sources, with human annotations to flag whether the sentences pairs are semantically equivalent. 

## Script Modification

GLUEDataBuilder modified with additioanl padding token --> GLUEDataBuilder_Modified.

Use **TorchDDPPlugin** only due to number of parameter consideration.

Pre-trained GPT2 model token embedding is resized (+1) with additional padding token. Padding token ID set to 0 in model configuration.

Modify **train()** function with loss and accuracy as return values. Used for visualization to see model performance improvement during training.

Modify and simplify **evaluate()** function to avoid out-of-memory issue. Keep loss and accuracy as return values for model evaluation metrics.

## How to Use
1. Clone the Repository:
```bash
git clone https://github.com/ZhuSisi777/gpt2_fine_tune_with_colossalai.git
cd gpt2_fine_tune_with_colossalai
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare Data:
If you intend to fine-tune on a dataset from the GLUE benchmark, utilize the **data.py** helper package to load the data. Modify the script as necessary to suit your dataset requirements.

4. Exploring Intermediate Results:
The **finetune_gpt_modified.ipynb** notebook provides a convenient way to visualize and analyze intermediate results. Execute the notebook line by line to understand the execution flow and monitor progress.

5. Combile Python File:
**run_finetune_gpt2.py** is a combiled python file from **finetune_gpt_modified.ipynb**

6. Execute Fine-Tuning:
Utilize the **run.sh** shell script to execute the compiled python file and initiated the fine-tuning process:
```bash
bash run.sh
```
7. Monitor Progress:
During fine-tuning, monitor progress, and performance metrics to assess the effectiveness of the model adaptation. Adjust hyperparameters or dataset configurations as needed for optimal results.

8. Experiment and Iterate:
Fine-tuning is an iterative process. Experiment with different configurations, datasets, and hyperparameters to achieve desired performance levels for your specific task or application.

  
## Result
Batch Size: 32, Epoch: 9, loss: 0.52, Acc: 74.15%, learning rate:2.4e-5, weight decay: 0.01, warmup_fraction=0.1

Batch Size: 32, Epoch: 9, loss: 0.536, Acc: 72.7%, learning rate:1.2e-5, weight decay: 0.01, warmup_fraction=0.05

Batch Size: 16, Epoch: 9, loss: 0.54, Acc: 73.2%, learning rate:2.4e-5, weight decay: 0.01, warmup_fraction=0.1


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



