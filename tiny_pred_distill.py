import logging

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from tiny_modeling_bert import BertModel

logging.basicConfig(filename='log/log.txt', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class TeacherModel(BertPreTrainedModel):
    """teacher模型"""

    def __init__(self, config):
        super(TeacherModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型
        # self.bert.eval()  # 不写效果好？

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=True, output_hidden_states=True)
        # sequence_output, pooler_output, hidden_states = outputs[0], outputs[1], outputs[2]

        return outputs.hidden_states, outputs.attentions


class StudentModel(BertPreTrainedModel):
    """student模型"""

    def __init__(self, config):
        super(StudentModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型
        self.fit_dense = nn.Linear(config.hidden_size, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=True, output_hidden_states=True)

        hidden_states = tuple()
        for s_id, sequence_layer in enumerate(outputs.hidden_states):
            hidden_states += (self.fit_dense(sequence_layer), )

        return hidden_states, outputs.attentions


class TrainDataset(Dataset):
    """自定义Dataset"""

    def __init__(self, dataframe):
        self._data = dataframe
        self._sentence1 = self._data.sentence1
        self._sentence2 = self._data.sentence2
        self._sentence3 = self._data.sentence3

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._sentence1.iloc[index], self._sentence2.iloc[index], self._sentence3.iloc[index]


class ValidDataset(Dataset):
    """自定义Dataset"""

    def __init__(self, dataframe):
        self._data = dataframe
        self._sentence1 = self._data.sentence1
        self._sentence2 = self._data.sentence2
        self._sentence3 = self._data.sentence3

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._sentence1.iloc[index], self._sentence2.iloc[index], self._sentence3.iloc[index]


class ModelCheckpoint(Callback):
    def __init__(self, save_path='output', mode='max', patience=10):
        super(ModelCheckpoint, self).__init__()
        self.path = save_path
        self.mode = mode
        self.patience = patience
        self.check_patience = 0
        self.best_value = 0.0 if mode == 'max' else 1e6  # 记录验证集最优值

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """
        验证集计算结束后检查
        :param trainer:
        :param pl_module:
        :return:
        """
        if self.mode == 'max' and pl_module.valid_metrics_history[-1] >= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_metrics_history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.tokenizer.save_pretrained(self.path)
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'max' and pl_module.valid_metrics_history[-1] < self.best_value:
            self.check_patience += 1

        if self.mode == 'min' and pl_module.valid_metrics_history[-1] <= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_metrics_history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.tokenizer.save_pretrained(self.path)
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'min' and pl_module.valid_metrics_history[-1] > self.best_value:
            self.check_patience += 1

        if self.check_patience >= self.patience:
            trainer.should_stop = True  # 停止训练


class TinySimCse(LightningModule):
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, teacher_path: str, student_path: str):
        super(TinySimCse, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(student_path)  # 加载分词器

        self.teacher_model = TeacherModel.from_pretrained(teacher_path)
        self.student_model = StudentModel.from_pretrained(student_path)

        self.train_dataset = TrainDataset(dataframe=train_data)  # 加载dataset
        self.valid_dataset = ValidDataset(dataframe=valid_data)

        self.max_length = 512  # 句子最大长度
        self.train_batch_size = 32
        self.valid_batch_size = 32

        self.valid_metrics_history = []

        self.automatic_optimization = True  # 最好关闭，训练速度变快
        if not self.automatic_optimization:
            self.optimizers, self.schedulers = self.configure_optimizers()
            self.optimizer = self.optimizers[0]  # 初始化优化器
            self.scheduler = self.schedulers[0]['scheduler']  # 初始化学习率策略

        self.scaler = torch.cuda.amp.GradScaler()  # 半精度训练

    def train_collate_batch(self, batch):
        """
        处理训练集batch，主要是文本转成相应的tokens
        :param batch:
        :return:
        """
        sentence_batch = []
        for (sentence1, sentence2, sentence3) in batch:
            sentence_batch.append(sentence1)
            sentence_batch.append(sentence2)
            sentence_batch.append(sentence3)
        outputs = self.tokenizer(sentence_batch, truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors='pt')
        return outputs['input_ids'], outputs['attention_mask'], outputs['token_type_ids']

    def valid_collate_batch(self, batch):
        """
        :param batch:
        :return:
        """
        sentence_batch = []
        for (sentence1, sentence2, sentence3) in batch:
            # sentence_batch.append(sentence1)
            sentence_batch.append(sentence2)
            # sentence_batch.append(sentence3)
        outputs = self.tokenizer(sentence_batch, truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors='pt')
        return outputs['input_ids'], outputs['attention_mask'], outputs['token_type_ids']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.valid_collate_batch,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        teacher_layer_num = self.teacher_model.config.num_hidden_layers
        student_layer_num = self.student_model.config.num_hidden_layers
        layers_per_block = int(teacher_layer_num / student_layer_num)

        teacher_hids, teacher_atts = self.teacher_model(input_ids, attention_mask, token_type_ids)
        student_hids, student_atts = self.student_model(input_ids, attention_mask, token_type_ids)

        # 加mask矩阵效果变差
        hid_loss = 0.0
        new_teacher_hids = [teacher_hids[i*layers_per_block] for i in range(student_layer_num+1)]
        for student_hid, teacher_hid in zip(student_hids, new_teacher_hids):
            hid_loss += F.mse_loss(student_hid, teacher_hid, reduction='mean')

        att_loss = 0.0
        new_teacher_atts = [teacher_atts[i*layers_per_block+layers_per_block-1] for i in range(student_layer_num)]
        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros(student_att), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros(teacher_att), teacher_att)
            att_loss += F.mse_loss(student_att, teacher_att, reduction='mean')

        loss = hid_loss + att_loss

        return loss, hid_loss, att_loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        loss, hid_loss, att_loss = self.forward(input_ids, attention_mask, token_type_ids)

        if not self.automatic_optimization:
            self.optimizer.zero_grad()  # 梯度置零

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()  # 学习率更新

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step}, train_step_loss: {loss:.5f}, '
                   f'train_step_hid_loss: {hid_loss:.5f}, train_step_att_loss: {att_loss:.5f}')

        return {'loss': loss, 'hid_loss': hid_loss, 'att_loss': att_loss}

    def training_epoch_end(self, outputs):
        loss, hid_loss, att_loss = 0.0, 0.0, 0.0
        for output in outputs:
            loss += output['loss'].item()
            hid_loss += output['hid_loss'].item()
            att_loss += output['att_loss'].item()
        loss /= len(outputs)
        hid_loss /= len(outputs)
        att_loss /= len(outputs)

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step}, train_loss: {loss:.5f}, '
                   f'train_hid_loss: {hid_loss:.5f}, train_att_loss: {att_loss:.5f}')
        logger.info(f'epoch: {self.current_epoch}, global_step: {self.global_step}, train_loss: {loss:.5f}, '
                    f'train_hid_loss: {hid_loss:.5f}, train_att_loss: {att_loss:.5f}')

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        loss, hid_loss, att_loss = self.forward(input_ids, attention_mask, token_type_ids)

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step}, valid_step_loss: {loss:.5f}, '
                   f'valid_step_hid_loss: {hid_loss:.5f}, valid_step_att_loss: {att_loss:.5f}')

        return {'loss': loss, 'hid_loss': hid_loss, 'att_loss': att_loss}

    def validation_epoch_end(self, outputs):
        loss, hid_loss, att_loss = 0.0, 0.0, 0.0
        for output in outputs:
            loss += output['loss'].item()
            hid_loss += output['hid_loss'].item()
            att_loss += output['att_loss'].item()
        loss /= len(outputs)
        hid_loss /= len(outputs)
        att_loss /= len(outputs)

        self.valid_metrics_history.append(loss)

        self.print(f'epoch: {self.current_epoch}, global_step: {self.global_step}, valid_loss: {loss:.5f}, '
                   f'valid_hid_loss: {hid_loss:.5f}, valid_att_loss: {att_loss:.5f}')
        logger.info(f'epoch: {self.current_epoch}, global_step: {self.global_step}, valid_loss: {loss:.5f}, '
                    f'valid_hid_loss: {hid_loss:.5f}, valid_att_loss: {att_loss:.5f}')

    def configure_optimizers(self, bert_lr=3e-5, other_lr=5e-5, total_step=10000):
        # teacher模型不训练
        for parm in self.teacher_model.parameters():
            parm.requires_grad = False

        # 设置优化器
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')], 'lr': bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('fit_dense')], 'lr': other_lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_step),
                                                    num_training_steps=total_step)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    train_data = pd.read_csv('./data/train_data.csv')
    valid_data = pd.read_csv('./data/valid_data.csv')

    checkpoint_callback = ModelCheckpoint(save_path=f'./simcse_pred_distill', mode='min')
    trainer = pl.Trainer(
        default_root_dir=f'pl_model',
        gpus=-1,
        precision=16,
        max_epochs=100,
        val_check_interval=1000,
        callbacks=[checkpoint_callback],
        logger=False,
        gradient_clip_val=0.0,
        distributed_backend=None,
        num_sanity_val_steps=-1,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=0,
    )
    cse = TinySimCse(train_data=train_data, valid_data=valid_data, teacher_path='./simcse', student_path='tiny_simbert_base')
    trainer.fit(cse)