from __future__ import absolute_import, division, print_function

import argparse
import random
from pytorch_transformers.modeling_roberta import RobertaConfig
import torch
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from torch.nn import L1Loss, MSELoss
from pytorch_transformers import WarmupLinearSchedule , AdamW
from networks.SentiLARE import RobertaForSequenceClassification
from utils.databuilder import set_up_data_loader
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
from config.global_configs import DEVICE

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--data_path", type=str, default='./dataset/MOSI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
        default="roberta-base")                                                                     
    parser.add_argument("--model_name_or_path", default='D:\Source\Source Code of CASNet\pretrained_model\sentilare_model/', type=str,   
                        help="Path to pre-trained model or shortcut name")                                                 
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default='random', help="integer or 'random'")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    return parser.parse_args()

def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(DEVICE)
    #Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    cas_params = ['CAS']
    optimizer_grouped_parameters = [
        {
            "params": [
                 p for n, p in param_optimizer if not any(nd in n for nd in no_decay)  and not any(nd in n for nd in cas_params)
            ],
            "weight_decay": args.weight_decay,
        },
        {"params": model.roberta.encoder.CAS.parameters(), 'lr':args.learning_rate, "weight_decay": args.weight_decay},
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)  and not any(nd in n for nd in cas_params)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler

def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0

    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids, 
        )
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        tr_loss += loss.item()
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        logits = np.squeeze(logits).tolist()
        label_ids = np.squeeze(label_ids).tolist()
        preds.extend(logits)
        labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels

def evaluate_epoch(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    loss = 0
    nb_dev_examples, nb_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,        
                token_type_ids=segment_ids,  
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss += loss.item()
            nb_steps += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return loss / nb_steps, preds, labels

def train(
    args,
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,):
    valid_losses = []
    test_accuracies = []
    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_pre, train_label = train_epoch(args, model, train_dataloader, optimizer, scheduler)
        valid_loss, valid_pre, valid_label = evaluate_epoch(args, model, validation_dataloader)
        test_loss, test_pre, test_label = evaluate_epoch(args, model, test_data_loader)
        train_acc, train_mae, train_corr, train_f_score = score_model(train_pre, train_label)
        test_acc, test_mae, test_corr, test_f_score = score_model(test_pre, test_label)
        non0_test_acc, _, _, non0_test_f_score = score_model(test_pre, test_label, use_zero=True)
        valid_acc, valid_mae, valid_corr, valid_f_score = score_model(valid_pre, valid_label)
        print(
            "epoch:{}, train_loss:{}, train_acc:{}, valid_loss:{}, valid_acc:{}, test_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc
            )
        )
        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_acc": train_acc,
                    "train_corr": train_corr,
                    "valid_acc":valid_acc,
                    "valid_corr":valid_corr,
                    "test_loss":test_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,
                    "non0_test_acc": non0_test_acc,
                    "non0_test_f_score": non0_test_f_score,
                    "best_valid_loss": min(valid_losses),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )

def main():
    args = parser_args()
    wandb.init(project="CAS", reinit=True)

    set_random_seed(args.seed)
    wandb.config.update(args)

    (train_data_loader,
    dev_data_loader,
    test_data_loader,
    num_train_optimization_steps,
    ) = set_up_data_loader(args)

    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    train(
        args,
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )

if __name__ == "__main__":
    main()