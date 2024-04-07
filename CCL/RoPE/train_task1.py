#!/usr/bin/python3

import os
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from dataset_task1 import Dataset
from params import args
from model_task1 import Model


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        # 对抗性攻击 epsilon为攻击步长，emb_name为embedding层的名称
        for name, param in self.model.named_parameters():
            # 遍历参数，如果参数的requires_grad为True且包含emb_name，则备份参数
            if param.requires_grad and emb_name in name:
                # 备份参数
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                # 计算参数的梯度的二范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
                    # 判断梯度的二范数是否不为0，如果不为0，则更新参数

    def restore(self, emb_name='embeddings'):
        # 恢复参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def warmup_linear(x, warmup=0.002):
    # 线性调整学习率
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def get_model_input(data, device=None):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        # 填充数据至固定长度
        return d + [v] * (max_len - len(d))
    
    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    # 获取数据样本数量和最大context长度
    
    input_ids_list = []
    attention_mask_list = []
    target = []
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)
    labels = np.array(labels, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)
    labels = torch.from_numpy(labels).to(device)

    return input_ids, attention_mask, target, labels, sentence_id


def eval(model, val_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device, for_test=True)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct += (pred == labels).sum().item() 
            total += len(pred)
            
    return correct / (total + 1e-6)


def train(model, train_loader, val_loader):
    # 训练模型
    param_optimizer = list(model.named_parameters())
    # 优化器参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 不需要进行衰减的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 优化器参数组

    # *******************  NoisyTune  ****************

    noise_lambda = 0.15
    for name, para in param_optimizer:
        model.state_dict()[name][:] += \
            (torch.rand(para.size()).to(device) - 0.5) * \
            noise_lambda * torch.std(para)

    # ***********************************************


    total_steps = int(len(train_loader) * args.num_train_epochs /
                      args.accumulate_gradients)
    # 总训练步数
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr)
    # 优化器

    # adv = FGM(bert_model) if args.with_adv_train else None
    global_step = 0
    best_acc = 0.0
    fgm = FGM(model)
    # 训练过程
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        # 训练集
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        # 进度条
        model.train()
        # 训练模式
        for step, batch in enumerate(iter_bar):
            global_step += 1

            input_ids, attention_mask, target, labels, sentence_id = batch
            # 准备数据
            # 输入数据
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)

            loss = output['loss']

            total_loss += loss.item()

            if (step + 1) % 1000 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')

            loss.backward()

            # fgm.attack()  # embedding被修改了
            # # optimizer.zero_grad()  # 如果不想累加梯度，就把这里的注释取消
            # loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)['loss']
            # loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复Embedding的参数

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            # 梯度裁剪
            if (step + 1) % args.accumulate_gradients == 0:
                # 累计梯度
                lr_this_step = args.lr * \
                               warmup_linear(global_step / total_steps,
                                             args.warmup_proportion)
                # 线性调整学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                # 优化器更新参数
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # break

        acc = eval(model, val_loader)
        # 验证集

        if acc > best_acc:
            print(f'saved! new best acc {acc}, ori_acc {best_acc}')
            best_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs('saves', exist_ok=True)
            torch.save(model_to_save.state_dict(), f'saves/model_task1_best.bin')
        else:
            print(f'current acc: {acc}')
        # 保存模型
        # train_loader.dataset.gen_data()


def load_pretrained_bert(bert_model, init_checkpoint):
    # 加载预训练的bert模型
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', '').replace('roformer.', ''): v for k, v in state.items() if
                     not k.startswith('cls.')}
            # state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=False)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)
    # 这里的tokenizer是bert的tokenizer，用于分词

    train_dataset = Dataset("./dataset/cfn-train.json",
                            "./dataset/frame_info.json",
                            tokenizer)
    # 训练集
    dev_dataset = Dataset("./dataset/cfn-dev.json",
                          "./dataset/frame_info.json",
                          tokenizer)
    # 验证集

    config = BertConfig.from_json_file(args.config_file)
    # 这里的config是bert的config，用于构建模型
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = train_dataset.num_labels
    # 这里的num_labels是标签的数量
    model = Model(config)
    # 构建模型
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    # 加载预训练的bert模型
    msg = model.load_state_dict(state, strict=False)
    # 加载预训练的bert模型的参数
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)
    # 加载模型到device
    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=True
    )
    # 训练集的dataloader
    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    # 验证集的dataloader
    train(model, train_loader, val_loader)
    # 训练模型