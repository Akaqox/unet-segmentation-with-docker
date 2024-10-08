import sys
import torch
from utils.utils import EarlyStopping, train_report
from utils.model_utils import  save_model
from utils.plot import plot_loss, plot_acc
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

report_dir = 'results/reports'
model_dir = 'results/models'

earlys_tolerance = 2
earlys_delta = 0.01


class Train():

    def __init__(self, train_loader, val_loader, hyper_p, model=None, criterion = None, optimizer = None, EPOCH = 25):


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hyper_p = hyper_p
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.EPOCH = EPOCH
        self.unique_name = f'{self.hyper_p[-3]}_date_{timestamp}_model'
        self.train_list = []
        self.val_list = []

        # self.train_loader = [(images.to(device), labels.to(device)) for images, labels in train_loader]
        # self.val_loader = [(images.to(device), labels.to(device)) for images, labels in val_loader]

    def train_one_epoch(self) -> tuple:
        running_loss = 0.0
        batch_accs = []

        for images, labels in  self.train_loader:

            images, labels = images.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs > 0.5
            batch_accs.append(torch.sum(preds.float() == labels).item()/labels.numel())
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = sum(batch_accs)/len(batch_accs)
        return epoch_loss, epoch_acc


    def validate(self) -> tuple:
        val_loss = 0.0
        batch_accs = []
        image_count = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
            preds = outputs > 0.5

            batch_accs.append(torch.sum(preds.float() == labels).item()/labels.numel()*images.size(0))

            image_count += images.size(0)
        val_loss = val_loss / len(self.val_loader.dataset)
        val_acc = sum(batch_accs)/(len(batch_accs)*image_count)
        return val_loss, val_acc
    
    def plot(self)-> None:
        if len(self.train_list) == 0:
            raise Exception("Model not trained yet")
        
        train_loss  = []
        train_acc = []
        val_loss = []
        val_acc = []
        for ([loss, acc], [vloss, vacc]) in zip(self.train_list, self.val_list):
            train_loss.append(loss)
            train_acc.append(acc)
            val_loss.append(vloss)
            val_acc.append(vacc)

        plot_loss(self.unique_name, train_loss, val_loss)
        plot_acc(self.unique_name, train_acc, val_acc)
        
    def fit(self)->None:
        i = 0
        best_loss = float('inf')
        # early_stopping = EarlyStopping(tolerance=earlys_tolerance, min_delta=earlys_delta)
        
        for i in range(self.EPOCH):
            #train stage
            self.model.train()
            epoch_loss, epoch_acc  = self.train_one_epoch()
            self.train_list.append((epoch_loss, epoch_acc))

            #validation stage 
            self.model.eval()
            val_loss, val_acc = self.validate()
            self.val_list.append((val_loss, val_acc))

            print(f'Epoch {i}/{self.EPOCH - 1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_loss < best_loss:
                name = f'{self.model.__class__.__name__}-best_model.tch'
                save_model(self.model, model_dir, name)
                print(f" Loss decreased from {best_loss:.4f} to {val_loss:.4f}. Model Saved!")
                best_loss = val_loss

            # early_stopping(epoch_loss, val_loss)
            # if early_stopping.early_stop:
            #     print("Early stopping at epoch: ", i)
            #     break

        save_model(self.model, model_dir, name=self.unique_name, best=False)

        train_p = [i, best_loss]
        train_report(report_dir, self.unique_name, self.hyper_p, train_p)
    