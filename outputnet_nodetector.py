import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnx2pytorch import ConvertModel
from tqdm import tqdm
from collections import OrderedDict
from models.vgg_cif10 import VGG
from models.vgg import vgg16_bn
# from .SpectralAdversarialDefense.models import *
from TRADES.models import wideresnet as wd
# from Cleanwrnet.wrnet.wide_resnet import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('Load modules...')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import argparse

# removes all warnings
import warnings
warnings.filterwarnings("ignore")

# torch.nn.Module.dump_patches = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# code for detector extraction (setup)
#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='InputMFS', help="the detector youz want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
args = parser.parse_args()
# choose attack
attack_method = args.attack
detector = args.detector
net = args.net

#load model vgg16
print('Loading model...')
if net == 'cif10':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
elif net == 'cif100':
    model = vgg16_bn()
    model.load_state_dict(torch.load('./models/vgg_cif100.pth'))
else:
    print('unknown model')
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#get a list of all feature maps of all layers
model_features = model.features
def get_layer_feature_maps(X, layers):
    X_l = []
    for i in range(len(model_features)):
        X = model_features[i](X)
        if i in layers:
            Xc = torch.Tensor(X.cpu())
            X_l.append(Xc.cuda())
    return X_l

#normalization
def cifar_normalize(images):
    if net == 'cif10':
        images[:,0,:,:] = (images[:,0,:,:] - 0.4914)/0.2023
        images[:,1,:,:] = (images[:,1,:,:] - 0.4822)/0.1994
        images[:,2,:,:] = (images[:,2,:,:] - 0.4465)/0.2010
    elif net == 'cif100':
        images[:,0,:,:] = (images[:,0,:,:] - 0.5071)/0.2675
        images[:,1,:,:] = (images[:,1,:,:] - 0.4867)/0.2565
        images[:,2,:,:] = (images[:,2,:,:] - 0.4408)/0.2761
    return images

#indice of activation layers
act_layers= [2,5,9,12,16,19,22,26,29,32,36,39,42]
fourier_act_layers = [9,16,22,29,36,42]


################Sections for each different detector

#######Fourier section

def calculate_fourier_spectrum(im, typ='MFS'):
    im = im.float()
    im = im.cpu()
    im = im.data.numpy() #transorm to numpy
    fft = np.fft.fft2(im)
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))
    if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
        fourier_spectrum *= 1/np.max(fourier_spectrum)
    return fourier_spectrum


def calculate_spectra(images, typ='MFS'):
    fs = []   
    for i in range(len(images)):
        image = images[i]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("in forward of model A")
        # print("x is:")
        # print(type(x))
        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        x = preprocess(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class LogisticRegressionWrapper(nn.Module): #otherwise cannot freeze the weights of Spectral Defense
    def __init__(self, detector):
        super(LogisticRegressionWrapper, self).__init__()
        self.detector = detector

    def forward(self, x):
        x_cpu = x.cpu() if x.is_cuda else x
        proba_cpu = self.detector.predict_proba(x_cpu)

        if x.is_cuda:
            proba_gpu = torch.tensor(proba_cpu, device=x.device)
            return proba_gpu
        else:
            return proba_cpu
        # return self.detector.predict_proba(x)

class OutputNetwork(nn.Module):
    def __init__(self, freeze_others=True): # freeze TRADES and standard while training output network
        super(OutputNetwork, self).__init__()
        self.mixing_layer_A = nn.Linear(10, 512, bias=True) # mix detector output with standard
        self.mixing_layer_B = nn.Linear(640, 512, bias=True) # mix detector output with TRADES
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        
        print("Loading Model A")
        self.model_A = Wide_ResNet(28, 10, 0.3, 10) # wrn_28_10
        self.model_A = nn.DataParallel(self.model_A, device_ids=[0])
        # self.model_A = torch.load('./Cleanwrnet/wrnet/best_model.pt')
        A_ckpt = torch.load("./wide-resnet.pytorch/checkpoint/cifar10/wide-resnet-28x10.pt", map_location="cuda:0")
        self.model_A.load_state_dict(A_ckpt)
        self.model_A.cuda()
        self.model_A.eval()
        
        print("Loading Model B")
        self.model_B = wd.WideResNet()# TRADES
        B_ckpt = torch.load("./TRADES/checkpoints/model_cifar_wrn.pt", map_location="cuda:0")
        self.model_B.load_state_dict(B_ckpt)
        self.model_B.cuda()
        self.model_B.eval()
        
        print("Loading Detector")
        # input_variable_name = "X"
        # input_shape = [None, 3072]
        # input_type = FloatTensorType(input_shape)
        # self.detector = pickle.load(open('./SpectralAdversarialDefense/data/detectors/LR_fgsm_InputMFS_test_cif10.sav', 'rb'))
        # onnx_model = convert_sklearn(detector, initial_types=[(input_variable_name, input_type)])
        # self.detector = ConvertModel(onnx_model)
        # self.detector = self.detector.to(torch.device("cuda:0")) # does not work
        
        self.freeze_others = freeze_others


    def forward(self, X):
        if(self.freeze_others):
            # for param in self.detector.parameters():
            #     param.requires_grad = False
            for param in self.model_A.parameters(): # 暫時這樣，測試 unfrozen
                param.requires_grad = True
            for param in self.model_B.parameters():
                param.requires_grad = True
            pass
        
        def detector_preprocess(X):
            X = calculate_spectra(X) 
            characteristics = np.asarray(X, dtype=np.float32)
            return characteristics

        # detector_input = detector_preprocess(X)
        model_A_output = self.model_A(X)
        model_B_output = self.model_B(X)
        # Flatten the model_A_output and model_B_output to 1D tensors
        model_A_output_flat = model_A_output.view(-1, 10)
        model_B_output_flat = model_B_output.view(-1, 640)
        # Pass through the mixing layers
        # mixed_output_A = torch.tensor(model_A_output_flat, dtype=torch.float32)
        mixed_output_A = model_A_output_flat.to(torch.float32)
        # mixed_output_B = torch.tensor(model_B_output_flat, dtype=torch.float32)
        mixed_output_B = model_B_output_flat.to(torch.float32)
        mixed_output_A = self.mixing_layer_A(mixed_output_A)
        mixed_output_B = self.mixing_layer_B(mixed_output_B)
        fc_output = torch.cat((mixed_output_A, mixed_output_B), dim=1)
        fc_output = self.fc1(fc_output)
        fc_output = self.fc2(fc_output)

        return fc_output

# remove this?
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Using GPU?")
print(use_cuda)

trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, **kwargs)
testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, **kwargs)
print(len(trainset))
print(len(testset))
print(len(train_loader))
print(len(val_loader))

net = OutputNetwork().to(device)
net = nn.DataParallel(net, device_ids=[0]) # edit

num_epochs = 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
print_every = 100
def train_clean_network():
    bestcorrect = 0
    for epoch in range(num_epochs):
        net.train()  # Set the network to training mode
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            # print("outputs:")
            # print(outputs)
            # print(len(outputs))
            # print(labels)
            # print(len(labels))
            # exit(0)
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every few iterations
            if i % print_every == print_every - 1:
                print(f'Epoch [{epoch+1}, {i+1}/{len(train_loader)}], Loss: {running_loss/print_every:.4f}')
                running_loss = 0.0

        # Validation
        net.eval()  # Set the network to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if correct > bestcorrect:
                    bestcorrect = correct
                    torch.save(net.state_dict(), './test_checkpoints_nodetector/best_checkpoint_unfreeze_08221400.pt')
        print(f'Epoch [{epoch+1}], Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100*correct/total:.2f}%')

    print('Finished Training')


# main function
if __name__ == '__main__':
    train_clean_network()


# net = OutputNetwork()
# ckpt = torch.load('./test_checkpoints_nodetector/best_checkpoint3.pt', )
# net.load_state_dict(ckpt)
# from autoattack import AutoAttack
# adversary = AutoAttack(net, norm='Linf', eps=0.031, version='standard')

