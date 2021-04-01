import os
import torch
import torch.nn as nn
import datetime

class BaseModule(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = path
        os.system('mkdir -p %s'%path)
        self.model_lst = [x for x in sorted(os.listdir(self.path)) if x.endswith('.pkl')]
        self.best_model = None
        self.best_accuracy = -float('inf')
    
    def _loadName(self, epoch=None):
        if len(self.model_lst) == 0:
            print("Empty model folder! Using initial weights")
            return None, 0

        if epoch is not None:
            for i, name in enumerate(self.model_lst):
                if name.endswith('%.5d.pkl'%epoch):
                    print("Use %s"%name)
                    return name, i
            print ('Epoch not found, use initial weights')
            return None, 0
        else:
            print ('Use last epoch, %s'%self.model_lst[-1])
            return self.model_lst[-1], len(self.model_lst)-1

    def Load(self, epoch=None):
        name, _ = self._loadName(epoch)
        if name is not None:
            params = torch.load('%s/%s'%(self.path, name))
            self.load_state_dict(params, strict=False)
            self.best_model = name
            epoch = int(self.best_model.split('_')[-1].split('.')[0]) + 1
        else:
            epoch = 0

        return epoch

    def Save(self, epoch, accuracy=None, replace=False):
        if accuracy is None or replace==False:
            aaa = '%.5d'%epoch
            now = 'model_%s.pkl'%datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
            params = self.state_dict()
            name = '%s/%s'%(self.path, now)
            torch.save(params, name)
            self.best_model = now
        else:
            if accuracy > self.best_accuracy:
                aaa = '%.5d'%epoch
                now = 'model_%s.pkl'%datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
                params = self.state_dict()
                name = '%s/%s'%(self.path, now)
                if self.best_model is not None: os.system('rm %s/%s'%(self.path, self.best_model))
                torch.save(params, name)
                self.best_model = now
                self.best_accuracy = accuracy
                print ('Save %s'%name)
