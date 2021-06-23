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

parser = argparse.ArgumentParser(description='MIL tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

# New arguments : We can now specify :
        # which predefined model we want to use from pytorch models
        # decide whether or not the model is pretrained on ImageNet
        # Create our own model from scratch using a class (I provided a class customized_CNN1 as example)
        # Use any downloaded pretrained model that is not available on pytorch using its .pth downloaded file
        # Chose any optimizer : Adam, RMSprop, SGD, Rprop,...
        # Chose optimizer learning rate if param is available
        # Chose optimizer weight decay if available
        #  Chose proba  threshold for classification
parser.add_argument('--model', type=str, default='resnet34', help=' model to start with')
parser.add_argument('--pretrained', type=str, default='True', help=' whether or not to use the model with pretrained weights') 
parser.add_argument('--customized_model', type=str, default='', help='create your own CNN (you have to create a class and add it see customized_CNN1)')
parser.add_argument('--pretrained_path', type=str, default='', help='path of checkpoint for pretrained model to use')
parser.add_argument('--optimizer', type=str, default='RMSprop', help='optimizer to use for gradient descent')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of the optimizer to use for gradient descent')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of the optimizer to use for gradient descent') 
parser.add_argument('--proba_threshold', type=float, default=0.5, help='probability threshold for validation')


best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    
    for arg in vars(args):
        print (arg, getattr(args, arg))
        
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.output+'MIL_CNN_arguments.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    model_name = args.model if args.customized_model=="" else args.customized_model
    print("MIL Model name:",model_name)
    #cnn

    if(args.customized_model==""):
        model = eval("models."+args.model+"(args.pretrained)")
        if(any(substring in args.model for substring in ["resnet","inception_v3"])):
            model.fc = nn.Linear(model.fc.in_features, 2)
        if("densenet" in args.model):
            model.classifier = nn.Linear(model.classifier.in_features, 2)
        if(any(substring in args.model for substring in ["vgg","alexnet"])):
            model.classifier[6]=nn.Linear(model.classifier[6].in_features,2)
    else :
        model = eval(args.customized_model+"()")
        
    if(args.pretrained_path!=""):
        checkpoint= torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['state_dict'])
        
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    if(any(substring in args.optimizer for substring in ["Rprop"])):
        optimizer = eval("optim."+args.optimizer+"(model.parameters(), lr="+str(args.lr)+")")
    else :
        optimizer = eval("optim."+args.optimizer+"(model.parameters(), lr="+str(args.lr)+", weight_decay="+str(args.weight_decay)+")")
    

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'MIL_convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'MIL_convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val_lib and (epoch+1) % args.test_every == 0:
            
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= args.proba_threshold else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'MIL_convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            print(1-err)
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'MIL_checkpoint_best.pth'))
                print("model saved at epoch " + str((epoch+1)))
def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
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