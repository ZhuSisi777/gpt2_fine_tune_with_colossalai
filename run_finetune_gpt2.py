from data import GLUEDataBuilder_Modified

import torch
import colossalai
from colossalai.context import Config

from typing import Callable, List, Union
# import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader
import datasets
import matplotlib.pyplot as plt


from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

output_transform_fn = lambda x: x
criterion = lambda x: x.loss


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}

def train_epoch(
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        _criterion: Callable,
        lr_scheduler: LRScheduler,
        train_dataloader: DataLoader,
        booster: Booster,
        coordinator: DistCoordinator,
    ):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_device = use_pipeline and booster.plugin.stage_manager.is_last_stage(ignore_chunk=True)
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_device)
    total_step = len(train_dataloader)
    predictions = []
    total_loss = 0
    total_correct_predictions = 0
    total_sample = 0

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(range(total_step), desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not print_flag) as pbar:
        # Forward pass
        for _ in pbar:
            data = next(train_dataloader_iter)
            data = move_to_cuda(data)
            outputs = model(**data)
            loss = _criterion(outputs, None)
            # Backward
            booster.backward(loss, optimizer)
            pbar.set_postfix({"loss": loss.item()})

            logits = outputs[1]
            preds = torch.argmax(logits, axis=1)
            predictions.append(preds)
            total_loss = total_loss + loss.item()

            correct_predictions = (preds == data['labels']).sum().item()
            total_correct_predictions += correct_predictions
            total_sample += preds.shape[0]

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
    avg_loss = total_loss / total_step
    avg_acc = total_correct_predictions / total_sample
    return avg_loss, predictions, avg_acc


def evaluate(
        model: nn.Module,
        _criterion: Callable,
        test_dataloader: DataLoader,
        booster: Booster,
    ):
    model.eval()
    test_predictions = []
    test_total_loss = 0
    test_total_correct_predictions = 0
    test_total_sample = 0
    test_dataloader_iter = iter(test_dataloader)

    with tqdm(range(len(test_dataloader))) as pbar:
        for _ in pbar:
            data = next(test_dataloader_iter)
            data = move_to_cuda(data)
            labels = data['labels']
            outputs = model(**data)

            val_loss = _criterion(outputs, None)
            pbar.set_postfix({"val loss": val_loss.item()})
            logits = outputs[1]

            preds = torch.argmax(logits, axis=1)
            test_predictions.append(preds)
            test_total_loss += val_loss.item()

            correct_predictions = (preds == labels).sum().item()
            test_total_correct_predictions += correct_predictions
            test_total_sample += preds.shape[0]
    test_avg_loss = test_total_loss / len(test_dataloader)
    test_avg_acc = test_total_correct_predictions / test_total_sample
    return test_avg_loss, test_avg_acc

def main():
    model_name = "gpt2"
    # ==============================
    # Launch Distributed Environment
    # ==============================
    # CONFIG = Config(
    #     dict(
    #         parallel=dict(
    #             data=dict(size=1),
    #             pipeline=dict(size=1),
    #             tensor=dict(size=1, mode=None),
    #             )
    #         )
    #     )
    # colossalai.launch(config=CONFIG,
    #                 rank=0,
    #                 world_size=1,
    #                 host='127.0.0.1',
    #                 port=8888,
    #                 backend='nccl')
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()
    
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    
    # ==============================
    # Prepare Dataloader
    # ==============================
    data_builder = GLUEDataBuilder_Modified(
        model_name, plugin, 'mrpc', train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
    )
    train_dataloader = data_builder.train_dataloader()
    test_dataloader = data_builder.test_dataloader()
    
    # ====================================
    # Prepare model, optimizer
    # ====================================
    cfg = AutoConfig.from_pretrained("gpt2", num_labels=2)
    model_gpt = GPT2ForSequenceClassification.from_pretrained("gpt2", config=cfg).cuda()
    
    # Modification
    model_gpt.resize_token_embeddings(len(data_builder.tokenizer))
    model_gpt.config.pad_token_id = 0
     
    lr = LEARNING_RATE * coordinator.world_size
    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_gpt.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model_gpt.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    
    # lr scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_FRACTION * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    
    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _criterion, _, lr_scheduler = booster.boost(
        model_gpt, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
    )
    
    
    # ==============================
    # Train model
    # ==============================
    train_losses=[]
    train_accs=[]
    for epoch in range(NUM_EPOCHS):
        train_loss, train_prediction, train_acc = train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'\nTraining Accuracy: {train_acc:.5f}')

        
    test_avg_loss, test_avg_acc = evaluate(model, _criterion, test_dataloader, booster)
    
    if coordinator.is_master():
        print(f'\nTesting Loss: {test_avg_loss:.3f}')
        print(f'\nTesting Accuracy: {test_avg_acc:.5f}')
    
    
    
if __name__ == "__main__":
    main()
