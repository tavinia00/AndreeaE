    


import matplotlib.pyplot as plt
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse


# Train:
def Model_build(train_dir, test_dir, valid_dir, arch,gpu):
	train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
	image_datasets_train = datasets.ImageFolder(train_dir+'/',  transform=train_transforms)
	trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)

# Test

	test_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       
                                       transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
	image_datasets_test = datasets.ImageFolder(test_dir+'/',   transform=test_transforms)
	testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=64, shuffle=True)

# Valid
	image_datasets_valid = datasets.ImageFolder(valid_dir+'/',   transform=test_transforms)

	validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64, shuffle=True)

	if gpu=="cuda":
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	if arch == "densenet121":
		model = models.densenet121(pretrained=True)
	else:
		model = models.vgg13(pretrained=True)    

# replace classifier with own classifier
	classifier = nn.Sequential(OrderedDict([
                          	('fc1', nn.Linear(1024, 500)),
                          	('relu', nn.ReLU()),
                          	('fc2', nn.Linear(500, 102)),
                          	('output', nn.LogSoftmax(dim=1))
                          	]))
# attach classifier to model
	model.classifier = classifier

	criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
	model.to(device)
	return{"model":model,"criterion":criterion,"optimizer":optimizer,"device":device,"trainloader":trainloader, "testloader":testloader, "validloader":validloader}
    

def train(dir, folder="/",arch="densenet121", no_epochs=3,gpu="cpu"):
    data_dir = os.getcwd()+dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    epochs=int(no_epochs)
    steps = 0
    trainloader=Model_build(train_dir, test_dir, valid_dir, arch,gpu)["trainloader"]
    validloader = Model_build(train_dir, test_dir, valid_dir, arch,gpu)["validloader"]
    model=Model_build(train_dir, test_dir, valid_dir,arch,gpu)["model"] 
    criterion=Model_build(train_dir, test_dir, valid_dir,arch,gpu)["criterion"] 
    optimizer=Model_build(train_dir, test_dir, valid_dir,arch,gpu)["optimizer"] 
    train_losses,valid_losses=[],[]
    for e in range(epochs):
        running_loss=0
        for images, labels in trainloader:
            optimizer.zero_grad()
            logps = model(images)#shorter way to run forward
            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
        else:
            valid_loss=0
            accuracy=0
            with torch.no_grad():
                for images,labels in validloader:
                    log_ps=model(images)
                    valid_loss+=criterion(log_ps,labels)
                    ps=torch.exp(log_ps)
                    top_p,top_class=ps.topk(1,dim=1)
                    equals=top_class==labels.view(*top_class.shape)
                    accuracy+=torch.mean(equals.type(torch.cuda.FloatTensor))
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            print("epoch:{}/{}..".format(e+1,epochs),
                 "Training loss:{:.3f}..".format(running_loss/len(trainloader)),
                 "valid loss:{:.3f}..".format(valid_loss/len(validloader)),
                 "valid accuracy:{:.3f}".format(accuracy/len(validloader)))
    plt.plot(train_losses,label="training loss")
    plt.plot(valid_losses,label="validation loss")
    plt.legend(frameon=False)
    model.class_to_idx = image_datasets_train.class_to_idx
    checkpoint = {
                "state_dict": model.state_dict(),
                "class_to_idx": model.class_to_idx}
    torch.save(checkpoint,os.getcwd()+folder+"checkpoint.pth")
    
def save_checkpoint(image_folder,folder):
    train(image_folder)
    model.class_to_idx = image_datasets_train.class_to_idx
    checkpoint = {
                "state_dict": model.state_dict(),
                "class_to_idx": model.class_to_idx}
    torch.save(checkpoint,folder+"checkpoint.pth")

parser = argparse.ArgumentParser(description='.')
parser.add_argument('image_folder',
                    help='image folder')

parser.add_argument('--save_dir', dest='training', action='store_const',
                    const=train, default=train,
                    help='(default: train model)')

parser.add_argument('checkpoint_folder',
                    help='checkpoint folder')
parser.add_argument('--arch', help='architecture torch')
parser.add_argument('--epochs',type=int, help='no epochs')
parser.add_argument('--gpu', help='gpu type')

args = parser.parse_args()
args.training(args.image_folder,args.checkpoint_folder,args.arch,args.epochs,args.gpu)



#to run:
#cd ImageClassifier
#python train.py "/flowers" "/"
#python train.py data_dir --save_dir save_directory
#python train.py "/flowers"  --save_dir "/flowers" --gpu "cuda"
# python train.py "/flowers" --save_dir "/"  --arch "densenet121" --epochs 3 --gpu "cpu"
