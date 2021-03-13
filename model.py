import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import average_precision_score, accuracy_score
import copy

def get_correct(preds, labels):
    a = np.sum([torch.equal(y,t) for (y, t) in zip(preds, labels)])
    return a

def train_model(model, dataloaders, loss_fn, optimizer, num_epochs, scheduler, device, class_num):

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_epoch = -1
    best_acc = 0

    trainlosses = []
    testlosses = []
    testperfs = []
  
    for epoch in range(num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            for batch_idx, (images, labels) in enumerate(dataloaders["train"]):
                # print("batch no: ", batch_idx)
                images, labels = images.to(device), labels.to(device)

                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                running_loss += loss.item() * images.size(0)
                running_corrects += get_correct(preds, labels)
                
            if phase == "train":
                trainlosses.append(running_loss)
                print("train loss: ", running_loss)
            else:
                testlosses.append(running_loss)
                accuracy = running_corrects #/ len(image_datasets['val'])
                testperfs.append(accuracy)
                print("val loss: {}, accucary: {}".format(running_loss,accuracy))
                
            # if phase == "val" and running_corrects / len(image_datasets['val']) > best_acc:
            #     best_acc = running_corrects / len(image_datasets['val'])
            #     best_model_wts = copy.deepcopy(model.state_dict())
    
    return model

def test_model(model, dataloader, criterion, device, class_num, saveResult):

    model.eval()
 
    curcount = 0
    corrects = 0
    losses = []
    
    # prediction scores for each class. 
    # each numpy array is a list of scores. one score per image
    concat_pred = [[] for _ in range(class_num)] 
    # labels scores for each class. 
    # each numpy array is a list of labels. one label per image
    concat_labels = [[] for _ in range(class_num)]
    # average precision for each class
    avgprecs = np.zeros(class_num) 
    # filenames as they come out of the dataloader
    fnames = dataloader.dataset.images
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            
            inputs, labels = images.to(device), labels.to(device)        
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
          
            # this was an accuracy computation
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()
            labels = labels.float()
            corrects += np.sum([torch.equal(y,t) for (y, t) in zip(preds, labels)])
            curcount += labels.shape[0]
            
            for output in outputs:
                for cls, pred in enumerate(output):
                    concat_pred[cls].append(pred.item())
            for label in labels:
                for cls, lbl in enumerate(label):
                    concat_labels[cls].append(lbl.item())

    accuracy = corrects / curcount
    
    for c in range(class_num):   
        avgprecs[c] = average_precision_score(y_true = concat_labels[c], y_score = concat_pred[c])
    mean_avgprecs = np.mean(avgprecs)

    print("accuracy: ", accuracy)
    print("loss: ", np.mean(losses))
    print("mean average precision: ", mean_avgprecs)
    print("average precision: ", avgprecs)
    
    if saveResult:
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

        for i in range(class_num):
            file_name = "./results/{}.csv".format(object_categories[i])
            file = open(file_name, mode = 'w')
            args = np.argsort(concat_pred[i])[::-1]
            for j in args:
                file.write('{} {}\n'.format(fnames[j], concat_pred[i][j]))
            file.close()

    return avgprecs, np.mean(losses)