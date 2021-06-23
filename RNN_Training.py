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

import json

parser = argparse.ArgumentParser(description='RNN aggregator training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary, same number of slides per patient')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present, same number of slides per patient')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence, default is False (I think  it is  useful for bidirectional)')

# New arguments : We can now specify :
        # from which MIL model it comes from
        # provide RNN architecture we want to use for the RNN model : default , RNN , GRU, LSTM classes are provided in the code below
        # whether or not we want a bidirectional RNN 
        # how many layers in the RNN
        # which optimizer we want for the RNN
        # optimizer learning rate for the RNN
        # optimizer weight decay for the RNN
        
parser.add_argument('--pretrained_model', type=str, default='resnet34', help='pretrained model name, can be customized and if the case needs to add its class, we provided a customized_CNN1 class example') 
parser.add_argument('--RNN_model', type=str, default='rnn_single', help='RNN model to use (rnn_single is the default one from the paper)') 
parser.add_argument('--bidirectional', type=str, default="False", help='whether or not  to use a bidirectional  RNN (available for RNN; RNN GRU; RNN LSTM)') 
parser.add_argument('--num_layers',  default= 3, type=int, help='number of layers to use in the RNN') 
parser.add_argument('--optimizer', type=str, default='default', help='optimizer to use for gradient descent')   
parser.add_argument('--lr', type=float, default=0.1, help='learning rate of the optimizer to use for gradient descent')   
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of the optimizer to use for gradient descent')   
 

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    
    for arg in vars(args):
        print (arg, getattr(args, arg))
    
    with open(args.output+'RNN_arguments.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.bidirectional = eval(args.bidirectional)
    print("RNN Model:",args.RNN_model)
    print("Bidirectional:",args.bidirectional)
    
    
    #load libraries
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dset = rnndata(args.train_lib, args.s, args.shuffle, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.val_lib, args.s, False, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
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
    rnn = rnn.cuda()
    
    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    if(args.optimizer =="default"):
        optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    else :
        if(any(substring in args.optimizer for substring in ["LBFGS","Rprop"])):
            optimizer = eval("optim."+args.optimizer+"(rnn.parameters(), lr="+str(args.lr)+")")
        else :
            optimizer = eval("optim."+args.optimizer+"(rnn.parameters(), lr="+str(args.lr)+", weight_decay="+str(args.weight_decay)+")")

    cudnn.benchmark = True

    fconv = open(os.path.join(args.output, 'RNN_convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    for epoch in range(args.nepochs):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, val_loader, criterion)

        fconv = open(os.path.join(args.output,'RNN_convergence.csv'), 'a')
        fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
        fconv.close()

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc:
            best_acc = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(args.output,'RNN_checkpoint_best.pth'))

def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        if  (args.RNN_model == "rnn_single") :
            state = rnn.init_hidden(batch_size).cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            if  (args.RNN_model == "rnn_single") :
                output, state = rnn(input, state)
            else :
                output = rnn(torch.unsqueeze(input,0))

        target = target.cuda()
        loss = criterion(torch.squeeze(output,0), target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
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
            
            target = target.cuda()
            loss = criterion(torch.squeeze(output,0),target)
            
            running_loss += loss.item()*target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns
            
    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred!=real
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return fps,fns

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