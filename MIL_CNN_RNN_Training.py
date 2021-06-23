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

import json

# Initial MIL CNN parameters
parser = argparse.ArgumentParser(description='MIL + RNN tile classifier training script')
parser.add_argument('--train_lib_MIL', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib_MIL', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of  root output file (output_model/) usually')
parser.add_argument('--batch_size_MIL', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs_MIL', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights_MIL', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

# New  parameters for MIL CNN

parser.add_argument('--model_MIL', type=str, default='resnet34', help=' model to start with')   
parser.add_argument('--pretrained', type=str, default='True', help=' whether or not to use the model with pretrained weights')   
parser.add_argument('--customized_model', type=str, default='', help='create your own CNN (you have to create a class and add it see customized_CNN1)')   
parser.add_argument('--pretrained_path', type=str, default='', help='path of checkpoint for pretrained model to use')  
parser.add_argument('--optimizer_MIL', type=str, default='RMSprop', help='optimizer to use for gradient descent')   
parser.add_argument('--lr_MIL', type=float, default=1e-4, help='learning rate of the optimizer to use for gradient descent')   
parser.add_argument('--weight_decay_MIL', type=float, default=1e-4, help='weight decay of the optimizer to use for gradient descent')   
parser.add_argument('--proba_threshold', type=float, default=0.5, help='probability threshold for validation')   

# Initial RNN parameters
parser.add_argument('--train_lib_RNN', type=str, default='', help='path to train MIL library binary, same number of slides per patient')
parser.add_argument('--val_lib_RNN', type=str, default='', help='path to validation MIL library binary. If present, same number of slides per patient')
parser.add_argument('--batch_size_RNN', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs_RNN', type=int, default=100, help='number of epochs')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--weights_RNN', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence, default is False (I think  it is  useful for bidirectional)')

# New  parameters for RNN
parser.add_argument('--RNN_model', type=str, default='rnn_single', help='RNN model to use (rnn_single is the default one from the paper)') 
parser.add_argument('--bidirectional', type=str, default="False", help='whether or not  to use a bidirectional  RNN (available for RNN; RNN GRU; RNN LSTM)') 
parser.add_argument('--num_layers',  default= 3, type=int, help='number of layers to use in the RNN') 
parser.add_argument('--optimizer_RNN', type=str, default='default', help='optimizer to use for gradient descent')   
parser.add_argument('--lr_RNN', type=float, default=0.1, help='learning rate of the optimizer to use for gradient descent')   
parser.add_argument('--weight_decay_RNN', type=float, default=1e-4, help='weight decay of the optimizer to use for gradient descent')   

parser.add_argument('--folder_output', type=str, default='', help='we usually keep it  empty and the  script creates it by  creating  folders and  subfolders with model info for gridsearch')

best_acc_MIL = 0
def main_MIL():


    global args, best_acc_MIL
    args = parser.parse_args()
    model_name = args.model_MIL if args.customized_model=="" else args.customized_model
    print("MIL Model name:",model_name)
    
    #Handle folder creation for  gridsearch
    if args.folder_output == "":
        max = 0
        for root, dirs, files in os.walk(args.output+"/"+model_name+"/gridsearch"):
          for dirname in sorted(dirs):
                if(int(dirname)>max):
                  max=int(dirname)
        os.makedirs(args.output+"/"+model_name+"/gridsearch/"+str(max+1))
        args.folder_output = args.output+"/"+model_name+"/gridsearch/"+str(max+1)+"/"
    else:
        if not os.path.exists(args.folder_output):
            os.makedirs(args.folder_output)   
            
    for arg in vars(args):
        print (arg, getattr(args, arg))
    
    with open(args.folder_output+'arguments.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    print("################################################Starting MIL part################################################")
    #cnn
    if(args.customized_model==""):
        model = eval("models."+args.model_MIL+"(args.pretrained)")
        if(any(substring in args.model_MIL for substring in ["resnet","inception_v3"])):
            model.fc = nn.Linear(model.fc.in_features, 2)
        if("densenet" in args.model_MIL):
            model.classifier = nn.Linear(model.classifier.in_features, 2)
        if(any(substring in args.model_MIL for substring in ["vgg","alexnet"])):
            model.classifier[6]=nn.Linear(model.classifier[6].in_features,2)
    else :
        model = eval(args.customized_model+"()")
        
    if(args.pretrained_path!=""):
        checkpoint= torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['state_dict'])
        
    model.cuda()

    if args.weights_MIL==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights_MIL,args.weights_MIL])
        criterion = nn.CrossEntropyLoss(w).cuda()
    if(any(substring in args.optimizer_MIL for substring in ["Rprop"])):
        optimizer = eval("optim."+args.optimizer_MIL+"(model.parameters(), lr="+str(args.lr_MIL)+")")
    else :
        optimizer = eval("optim."+args.optimizer_MIL+"(model.parameters(), lr="+str(args.lr_MIL)+", weight_decay="+str(args.weight_decay_MIL)+")")
    

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib_MIL, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size_MIL, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib_MIL:
        val_dset = MILdataset(args.val_lib_MIL, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size_MIL, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file

    fconv = open(os.path.join(args.folder_output,'MIL_convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs_MIL):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs_MIL, loss))
        fconv = open(os.path.join(args.folder_output, 'MIL_convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val_lib_MIL and (epoch+1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= args.proba_threshold else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs_MIL, err, fpr, fnr))
            fconv = open(os.path.join(args.folder_output, 'MIL_convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc_MIL:
                best_acc_MIL = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc_MIL,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.folder_output,'MIL_checkpoint_best.pth'))
                
                
def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs_MIL, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size_MIL:i*args.batch_size_MIL+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

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
    return out

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
            
best_acc_RNN = 0

def main_RNN():
    global args, best_acc_RNN
    # args = parser.parse_args() don't parse  because  we lose the updated value   of folder_output
    
    args.bidirectional = eval(args.bidirectional)
    print("################################################Starting RNN part################################################")
    print("RNN Model:",args.RNN_model)
    if(args.RNN_model!="rnn_single"):
        print("Bidirectional:",args.bidirectional)
    model_name = args.model_MIL if args.customized_model=="" else args.customized_model
    print("MIL Pretained Model name:",model_name)
    
    #load libraries
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dset = rnndata(args.train_lib_RNN, args.s, args.shuffle, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size_RNN, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.val_lib_RNN, args.s, False, trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size_RNN, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model
    print(args.folder_output+"MIL_checkpoint_best.pth")
    embedder = modelEncoder(args.folder_output+"MIL_checkpoint_best.pth")
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
    if args.weights_RNN==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights_RNN,args.weights_RNN])
        criterion = nn.CrossEntropyLoss(w).cuda()
    if(args.optimizer_RNN =="default"):
        optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    else :
        if(any(substring in args.optimizer_RNN for substring in ["Rprop"])):
            optimizer = eval("optim."+args.optimizer_RNN+"(rnn.parameters(), lr="+str(args.lr_RNN)+")")
        else :
            optimizer = eval("optim."+args.optimizer_RNN+"(rnn.parameters(), lr="+str(args.lr_RNN)+", weight_decay="+str(args.weight_decay_RNN)+")")

    cudnn.benchmark = True

    fconv = open(os.path.join(args.folder_output, 'RNN_convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    for epoch in range(args.nepochs_RNN):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, val_loader, criterion)

        fconv = open(os.path.join(args.folder_output,'RNN_convergence.csv'), 'a')
        fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
        fconv.close()

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc_RNN:
            best_acc_RNN = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(args.folder_output,'RNN_checkpoint_best.pth'))
            
            
def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs_RNN, i+1, len(loader)))

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
    print('Training - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs_RNN, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs_RNN,i+1,len(loader)))
            
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
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs_RNN, running_loss, running_fps, running_fns))
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
      model_name = args.model_MIL if args.customized_model=="" else args.customized_model
      if(any(substring in model_name for substring in ["resnet","inception_v3"])):
          temp = eval("models."+model_name+"()")
          temp.fc = nn.Linear(temp.fc.in_features, 2)
      elif("densenet" in model_name):
          temp = eval("models."+model_name+"()")
          temp.classifier = nn.Linear(temp.classifier.in_features, 2)
      elif(any(substring in model_name for substring in ["vgg","alexnet"])):
          temp = eval("models."+model_name+"()")
          temp.classifier[6]=nn.Linear(temp.classifier[6].in_features,2)
      else:
          temp = eval(model_name+"()")
          
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
      model_name = args.model_MIL if args.customized_model=="" else args.customized_model
      self.ndims = ndims
      input_size = 512
      if(any(substring in model_name for substring in ["resnet18","resnet34"])):
        input_size = 512
      if("vgg" in model_name):
        input_size = 25088
      if("resnet50" in model_name):
        input_size = 2048
      if("alexnet" in model_name):
        input_size = 9216
      if("densenet" in model_name):
        input_size = 108192  
      if("customized_CNN1" in model_name):
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
        model_name = args.model_MIL if args.customized_model=="" else args.customized_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in model_name for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in model_name):
          input_size = 25088
        if("resnet50" in model_name):
          input_size = 2048
        if("alexnet" in model_name):
          input_size = 9216
        if("densenet" in model_name):
          input_size = 108192 
        if("customized_CNN1" in model_name):
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
        model_name = args.model_MIL if args.customized_model=="" else args.customized_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in model_name for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in model_name):
          input_size = 25088
        if("resnet50" in model_name):
          input_size = 2048
        if("alexnet" in model_name):
          input_size = 9216
        if("densenet" in model_name):
          input_size = 108192 
        if("customized_CNN1" in model_name):
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
        model_name = args.model_MIL if args.customized_model=="" else args.customized_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        input_size = 512
        if(any(substring in model_name for substring in ["resnet18","resnet34"])):
          input_size = 512
        if("vgg" in model_name):
          input_size = 25088
        if("resnet50" in model_name):
          input_size = 2048
        if("alexnet" in model_name):
          input_size = 9216
        if("densenet" in model_name):
          input_size = 108192  
        if("customized_CNN1" in model_name):
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
    main_MIL()
    main_RNN()