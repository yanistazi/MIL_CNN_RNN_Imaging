import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

# Plot imports
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn  as sns
import matplotlib.pyplot as  plt

import json

parser = argparse.ArgumentParser(description='')
parser.add_argument('--test_lib', type=str, default='filelist', help='path to data file')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--model_path', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')


# New arguments : We can now specify :
        # which predefined model we want to use from pytorch models
        # Create our own model from scratch using a class (I provided a class customized_CNN1 as example)
        # Chose proba  threshold for classification
        # Chose whether or not to plot the metrics
        
parser.add_argument('--model_name', type=str, default='', help='model name (used for available pytorch models)')        
parser.add_argument('--customized_model', type=str, default='', help='create your own CNN (you have to create a class and add it see customized_CNN1)')
parser.add_argument('--proba_threshold', type=float, default=0.5, help='probability threshold for validation')  
parser.add_argument('--plot_metrics', type=str, default="True", help='whether to plot the results ')   



    
def main():
    global args, best_acc
    args = parser.parse_args()
    
    for arg in vars(args):
        print (arg, getattr(args, arg))

        
    #cnn

    if(args.customized_model==""):
        model = eval("models."+args.model_name+"(True)")
        if(any(substring in args.model_name for substring in ["resnet","inception_v3"])):
            model.fc = nn.Linear(model.fc.in_features, 2)
        if("densenet" in args.model_name):
            model.classifier = nn.Linear(model.classifier.in_features, 2)
        if(any(substring in args.model_name for substring in ["vgg","alexnet"])):
            model.classifier[6]=nn.Linear(model.classifier[6].in_features,2)
    else :
        model = eval(args.customized_model+"()")
        
    ch = torch.load(args.model_path)
    model.load_state_dict(ch['state_dict'])    

        
    model.cuda()

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])


    dset = MILdataset(args.test_lib, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(loader, model)
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))

    fp = open(os.path.join(args.output, 'MIL_predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()
    
    # Plot  metrics
    
    if(args.plot_metrics=="True"):
        df = pd.read_csv(args.output+"MIL_predictions.csv")
        y_test = np.array(df)[:,1].astype('float')
        y_pred = np.array(df)[:,2].astype('float')
        probas = np.array(df)[:,3].astype('float')
        # probas
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : %f" % roc_auc)
        
        fig, axs = plt.subplots(1,2,figsize=(10,5))
        sns.lineplot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc,ax=axs[0])
        axs[0].plot([0, 1], [0, 1], 'k--')
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.0])
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver operating characteristic')
        axs[0].legend(loc="lower right")
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,ax=axs[1])
        axs[1].set_title("Confusion Matrix")
        fig.savefig(args.output+"MIL_plot_metrics.png")
        print("Plot  Done!")
        
        
def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return list(out)


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
            
class customized_CNN1(Module):   
    def __init__(self):
        super(customized_CNN1, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = Sequential(
            Linear(200704, 1000),
            Linear(1000, 2))

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        

if __name__ == '__main__':
    main()