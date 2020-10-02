
import matplotlib.pyplot as plt
import pandas as pd

import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import numpy as np
import os



def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model




def Model_build(train_dir, test_dir, valid_dir,gpu):
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

	model = models.densenet121(pretrained=True)

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



def predict_image(image_folder,checkpoint, topk=5, category_names ="cat_to_name.json",gpu="cpu" ):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    data_dir = os.getcwd()+image_folder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    image_index=0
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       
                                       transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    image_datasets_train = datasets.ImageFolder(train_dir+'/',  transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    images, labels  =  next(iter(trainloader))
    model=Model_build(train_dir, test_dir, valid_dir,gpu)["model"] 
    model = load_checkpoint(checkpoint, model)


    label = labels[image_index].item()#get label of chosen image
    print("Name flower:",cat_to_name[str(model.class_to_idx[str(label)])])

    ps = torch.exp(model(images))#get class probabilities
    probs, top_class = ps.topk(topk,dim=1)#returns k highest value, since I want the most likely value I use ps,topk(1)
    probs =probs[image_index]
    print("Probabilities(%) for top most likely class:",probs)
    top_class=top_class[image_index]
    print("Top most likely classes:",top_class)
    class_={}
    name=[]
    prob=[]
    for i in top_class:
        for cls, ind in model.class_to_idx.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if (ind) == int(i):
                class_[i] = cat_to_name[cls]
                name.append(cat_to_name[cls])
    for i in probs:
        prob.append(i.item())  
        

    for i in range(topk):
        print("The prediction for Flower: ", name[i], " has ", round(prob[i]/100,4), " probability.")


    
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--sum', dest='predict_im', action='store_const',
                    const=sum, default=predict_image,
                    help='sum the integers (default: find the max)')
parser.add_argument('image_folder', help='image folder')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('--top_k', type=int,help='top k most likely classes')
parser.add_argument('--category_names', help='category names file')
parser.add_argument('--gpu', help='gpu(cuda or cpu')


in_args = parser.parse_args()

in_args.predict_im(in_args.image_folder,in_args.checkpoint,in_args.top_k, in_args.category_names,in_args.gpu)

#examples of how to run 
#cd ImageClassifier
#python predict.py "/flowers" 
#python predict.py "/flowers" --top_k 5
#python predict.py "/flowers" "checkpoint.pth" --top_k 5
#python predict.py "/flowers" "checkpoint.pth" --top_k 2 --category_names "cat_to_name.json"

