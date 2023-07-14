import numpy as np
from numpy import inf
import torch
from torch import nn, optim
from utils.metrics import *
from utils.pytorchtools import *
from utils.util import *
from base.base_trainer import BaseTrainer
from logger.logger import *
from tqdm import tqdm
from sklearn.metrics import classification_report


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, model, criterion, optimizer, device, train_dataloader, valid_data):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        if self.config['trainer']['task'] == "user2item":
            self.task_index = 0
        elif self.config['trainer']['task'] == "item2item":
            self.task_index = 4
        elif self.config['trainer']['task'] == "vert_classify":
            self.task_index = 1
        elif self.config['trainer']['task'] == "pop_predict":
            self.task_index = 3
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.device = device

        self.train_dataloader = train_dataloader
        self.test_data = valid_data


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        all_loss = 0
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            batch = real_batch(batch)   #from utils
            out = self.model(batch['item1'], batch['item2'], self.config['trainer']['task'])[self.task_index]
            if self.config['trainer']['task'] == 'vert_classify':
                loss = self.criterion(out, torch.LongTensor(batch['label']).cuda())    
            else:
                loss = self.criterion(out, torch.FloatTensor(batch['label']).cuda())
            all_loss = all_loss + loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.save(self.model, './out/saved/models/KRED/checkpoint_{}.pt'.format(self.config['trainer']['task']))
        print("all loss: " + str(all_loss))


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        y_pred = []
        if self.config['trainer']['task'] == 'vert_classify' or self.config['trainer']['task'] == 'item2item':
            divide = 1
        else: 
            divide = 20
        start_list = list(range(0, int(len(self.test_data['label'])/divide), int(self.config['data_loader']['batch_size'])))
        for start in tqdm(start_list):
            if start + int(self.config['data_loader']['batch_size']) <= int(len(self.test_data['label'])/divide):
                end = start + int(self.config['data_loader']['batch_size'])
            else:
                end = int(len(self.test_data['label'])/divide)
            out = self.model(self.test_data['item1'][start:end], self.test_data['item2'][start:end], self.config['trainer']['task'])[
                self.task_index].cpu().data.numpy()

            y_pred.extend(out)
        n_truth = int(len(self.test_data['label'])/divide)
        truth = self.test_data['label'][:n_truth]
        if self.config['trainer']['task'] == 'vert_classify':
            y_class = []
            for list_i in y_pred:
                list_a = list_i.tolist()
                y_class.append(list_a.index(max(list_a)))
            
            acc = accuracy_score(truth, y_class)
            f1 = f1_score(truth, y_class, average='macro')
            print('ACC:%.6f \nF1:%.6f' % (acc, f1))
            return acc
        else:
            auc_score = cal_auc(truth, y_pred)
            print("auc socre: " + str(auc_score))
            return auc_score

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state_model = self.model.state_dict()
        filename_model = str(self.checkpoint_dir / 'checkpoint-model-epoch{}-{}.pth'.format(epoch, self.config['trainer']['task']))
        torch.save(state_model, filename_model)
        self.logger.info("Saving checkpoint: {} Task: {} ...".format(filename_model, self.config['trainer']['task']))


    def train(self):
        """
            Full training logic
        """
        logger_train = get_logger("train")

        logger_train.info("model training, task is {}".format(self.config['trainer']['task']))
        if self.config['model']['kgat'] == False:
            logger_train.info("kgat deactivated")
        if self.config['model']['context'] == False:
            logger_train.info("context deactivated")
        if self.config['model']['distillation'] == False:
            logger_train.info("distillation deactivated")
        valid_scores = []
        early_stopping = EarlyStopping(patience=self.config['trainer']['early_stop'], verbose=True)
        # from pytorchtools.py
        for epoch in range(self.start_epoch, self.epochs+1):
            print(f'start training epoch : {epoch}')
            self._train_epoch(epoch)
            print(f'validation of epoch {epoch}')
            valid_socre = self._valid_epoch(epoch)
            valid_scores.append(valid_socre)
            early_stopping(valid_socre, self.model)
            if early_stopping.early_stop:
                logger_train.info("Early stopping")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

