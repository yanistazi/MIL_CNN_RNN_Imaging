import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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


parser = argparse.ArgumentParser(description='RNN aggregator testing script')
parser.add_argument('--test_lib', type=str, default='', help='path to test MIL library binary, same number of slides per patient')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
parser.add_argument('--rnn', type=str, help='path to trained RNN model checkpoint')

# New arguments : We can now specify :
        # which pretrained model for the embedding
        # provide RNN architecture we want to use for the RNN model : default , RNN , GRU, LSTM classes are provided in the code below
        # whether or not we want a bidirectional RNN 
        # how many layers in the RNN
        # Chose whether or not to plot the metrics
parser.add_argument('--pretrained_model', type=str, default='resnet34', help='pretrained model name, can be customized and if the case needs to add its class, we provided a customized_CNN1 class example') #Yanis
parser.add_argument('--RNN_model', type=str, default='rnn_single', help='RNN model to use (rnn_single is the default one from the paper)') #Yanis
parser.add_argument('--bidirectional', type=str, default="False", help='whether or not  to use a bidirectional  RNN (available for RNN; RNN GRU; RNN LSTM)') #Yanis
parser.add_argument('--num_layers',  default= 3, type=int, help='number of layers to use in the RNN') #Yanis
parser.add_argument('--plot_metrics', type=str, default="True", help='whether to plot the results ')   #Yanis

def main():
    global args
    args = parser.parse_args()
    
    for arg in vars(args):
        print (arg, getattr(args, arg))
    
    # with open(args.output+'arguments.json', 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    args.bidirectional = eval(args.bidirectional)
    print("RNN Model:",args.RNN_model)
    print("Bidirectional:",args.bidirectional)
    
    
    #load libraries
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    dset = rnndata(args.test_lib, args.s, False, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    

    #make model
    embedder = modelEncoder(args.model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()
    
    if(args.RNN_model=="rnn_single"):
        rnn = rnn_single(args.ndims)
    else :
        rnn = eval(args.RNN_model+"(num_layers=args.num_layers, hidden_size=args.ndims,bidirectional=args.bidirectional)")
        
    rnn_dict = torch.load(args.rnn)
    rnn.load_state_dict(rnn_dict['state_dict'])
    rnn = rnn.cuda()
    
    
    cudnn.benchmark = True

    probs = test_single(embedder, rnn, loader)

    fp = open(os.path.join(args.output, 'MIL_RNN_predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, probs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()
        
    # Plot  metrics
    
    if(args.plot_metrics=="True"):
        df = pd.read_csv(args.output+"MIL_RNN_predictions.csv")
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
        fig.savefig(args.output+"MIL_RNN_plot_metrics.png")
        print("Plot  Done!")


def test_single(embedder, rnn, loader):
    rnn.eval()

    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Testing - Batch: [{}/{}]'.format(i+1,len(loader)))
            
            batch_size = inputs[0].size(0)
            
            if  (args.RNN_model == "rnn_single") :
                state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                if  (args.RNN_model == "rnn_single") :
                    output, state = rnn(input, state)
                else :
                    output = rnn(torch.unsqueeze(input,0))    
                    
            output = torch.squeeze(output,0)#Yanis
            output = F.softmax(output, dim=1)
            probs[i*args.batch_size:i*args.batch_size+batch_size] = output.detach()[:,1].clone()

    return probs.cpu().numpy()
# target = target.cuda()
#             loss = criterion(torch.squeeze(output,0),target)
            
#             running_loss += loss.item()*target.size(0)
#             fps, fns = errors(output.detach(), target.cpu())
#             running_fps += fps
#             running_fns += fns




class modelEncoder(nn.Module):

  def __init__(self, path):
      super(modelEncoder, self).__init__()
      if(any(substring in args.pretrained_model for substring in ["resnet","inception_v3"])):
          temp = eval("models."+args.pretrained_model+"()")
          temp.fc = nn.Linear(temp.fc.in_features, 2)
      elif("densenet" in args.pretrained_model):
          temp = eval("models."+args.pretrained_model+"()")
          temp.classifier = nn.Linear(temp.classifier.in_features, 2)
      elif(any(substring in args.pretrained_model for substring in ["vgg","alexnet"])):
          temp = eval("models."+args.pretrained_model+"()")
          temp.classifier[6]=nn.Linear(temp.classifier[6].in_features,2)
      else:
          temp = eval(args.pretrained_model+"()")
          
      ch = torch.load(path)
      temp.load_state_dict(ch['state_dict'])
      self.features = nn.Sequential(*list(temp.children())[:-1])
      # self.fc = temp.fc

  def forward(self,x):
      x = self.features(x)
      x = x.view(x.size(0),-1)
      return "not_used", x

class rnn_single(nn.Module):

  def __init__(self, ndims):
      super(rnn_single, self).__init__()
      self.ndims = ndims
      input_size = 512
      if(any(substring in args.pretrained_model for substring in ["resnet18","resnet34"])):
        input_size = 512
      if("vgg" in args.pretrained_model):
        input_size = 25088
      if("resnet50" in args.pretrained_model):
        input_size = 2048
      if("alexnet" in args.pretrained_model):
        input_size = 9216
      if("densenet" in args.pretrained_model):
        input_size = 108192  
      if("customized_CNN1" in args.pretrained_model):
        input_size = 200704             
      self.fc1 = nn.Linear(input_size, ndims)
      self.fc2 = nn.Linear(ndims, ndims)

      self.fc3 = nn.Linear(ndims, 2)

      self.activation = nn.ReLU()

  def forward(self, input, state):
      input = self.fc1(input)
      state = self.fc2(state)
      state = self.activation(state+input)
      output = self.fc3(state)
      return output, state

  def init_hidden(self, batch_size):
      return torch.zeros(batch_size, self.ndims)

class RNN(nn.Module):
    def __init__(self,  hidden_size, num_layers, bidirectional=False, num_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in args.pretrained_model for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in args.pretrained_model):
          input_size = 25088
        if("resnet50" in args.pretrained_model):
          input_size = 2048
        if("alexnet" in args.pretrained_model):
          input_size = 9216
        if("densenet" in args.pretrained_model):
          input_size = 108192 
        if("customized_CNN1" in args.pretrained_model):
          input_size = 200704 
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)

    def forward(self, input):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, input.size(0), self.hidden_size).to("cuda")
        # Forward propagate LSTM
        out, _ = self.rnn(input, h0)
        # out = out.reshape(out.shape[0], -1)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

class RNN_LSTM(nn.Module):
    def __init__(self,num_layers,  hidden_size, bidirectional=False, num_classes=2):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in args.pretrained_model for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in args.pretrained_model):
          input_size = 25088
        if("resnet50" in args.pretrained_model):
          input_size = 2048
        if("alexnet" in args.pretrained_model):
          input_size = 9216
        if("densenet" in args.pretrained_model):
          input_size = 108192 
        if("customized_CNN1" in args.pretrained_model):
          input_size = 200704 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)

    def forward(self, input):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, input.size(0), self.hidden_size).to("cuda")
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, input.size(0), self.hidden_size).to("cuda")

        # Forward propagate LSTM
        out, _ = self.lstm(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
class RNN_GRU(nn.Module):
    def __init__(self, num_layers, hidden_size, bidirectional=False, num_classes=2):
        super(RNN_GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in args.pretrained_model for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in args.pretrained_model):
          input_size = 25088
        if("resnet50" in args.pretrained_model):
          input_size = 2048
        if("alexnet" in args.pretrained_model):
          input_size = 9216
        if("densenet" in args.pretrained_model):
          input_size = 108192  
        if("customized_CNN1" in args.pretrained_model):
          input_size = 200704 
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)
    
    def forward(self, input):
        hidden_state = torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, input.size(1), self.hidden_size).to("cuda")
        output, hidden_state = self.gru(input, hidden_state)
        output = self.fc(output[-1])
        return output


class rnndata(data.Dataset):

    def __init__(self, path, s, shuffle=False, transform=None):

        lib = torch.load(path)
        self.s = s
        self.transform = transform
        self.slidenames = lib['slides']
        self.targets = lib['targets']
        self.grid = lib['grid']
        self.level = lib['level']
        self.mult = lib['mult']
        self.size = int(224*lib['mult'])
        self.shuffle = shuffle

        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        self.slides = slides

    def __getitem__(self,index):

        slide = self.slides[index]
        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid,len(grid))

        out = []
        s = min(self.s, len(grid))
        for i in range(s):
            img = slide.read_region(grid[i], self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        
        return out, self.targets[index]

    def __len__(self):
        
        return len(self.targets)
        
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