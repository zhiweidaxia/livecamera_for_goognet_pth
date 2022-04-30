import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import tqdm
import numpy as np
import argparse
import os.path as osp
import os
# from pytorch_toolbelt.inference import tta
import shutil
import glob
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

import cv2
from tracemalloc import start
from multiprocessing import Process, Queue
import os
import time
import random
import multiprocessing
#加载预编译网络
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

#net, input_size = initialize_model(model_name = "inception" , num_classes=21, feature_extract=True, use_pretrained=True)

def processFun(q,k,share_var,share_lock):
        import torch.nn.functional as F
        share_lock.acquire()
    
        net, input_size = initialize_model(model_name = "inception" , num_classes=21, feature_extract=True, use_pretrained=True)
        from torchvision.datasets import ImageFolder
        test_transform=transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        print(os.listdir('./val'))
        print('----------------------------------------')
        testset = ImageFolder('./val', test_transform)

        dictNum=testset.class_to_idx
        classes = []
        for i in dictNum:
            classes.append(i)
        net.eval()
        net.load_state_dict(torch.load('./inception/inception_025.pth'))
        net = net

        while True:
            try:
                print('开始评估+++++++++++++++++++++++++++++++++++')
                

                img = 0
                img = np.array(share_var[0])

                img = Image.fromarray(np.uint8(img))

                to_tensor = transforms.Compose([
                                transforms.Resize(299),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                img = to_tensor(img)

                img = torch.unsqueeze(img, 0) 
                img1 = img

                outputs = net(img1)
                outputs = F.softmax(outputs, dim=1)
                predicted = torch.max(outputs, dim=1)[1].cpu().item()
                outputs = outputs.squeeze(0)

                confidence=outputs[predicted].item()
                result = classes[predicted]
                print(result)

                confidence=round(confidence*100, 3)
                res = result+' '+str(confidence)+"%%"
                print("识别的水果是",res)
                q.put(res)
                time.sleep(10)
                if str(k.get_nowait()) == 'q':
                    break
            except:
                pass
            
cap=cv2.VideoCapture(1)

if __name__ == '__main__':
    #定义网络
    

    
    i=0
    queue = Queue()         #开启线程队列，主要为了共享数据
    queue1 = Queue()
    # 创建子进程
    share_var = multiprocessing.Manager().list()    #创建表格数据，传输图片
    share_var.append(np.empty((480, 640, 3)))                             #以防万一，设置一个空值
    share_lock = multiprocessing.Manager().Lock()   #线程上锁
    process1 = multiprocessing.Process(target=processFun, args=(queue,queue1,share_var,share_lock)) #设置线程
    process1.start()#开启线程
    
    class_accs =''      #通过queue获取精度
    ii =0               #计数器
    
    while True:
        ret ,frame = cap.read()
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转化图片格式为RGB，默认为BRG
        rets, thresh = cv2.threshold(imgray, 150, 255, 0)#二值化，src图片，150为阈值
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#查找边界

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w*h>8000 and w*h<120000:
                cv2.putText(frame, "refresh%s"%(ii%30), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0, 255), 2)
                fff=frame[x-5:x+w+5,y-5:y+h+5]
                print(ii)
                if ii%30 == 0:
                    try:
                        cv2.imshow("imgs",fff)
                        share_var[0] = fff.tolist()
                        #time.sleep(10)
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++---")
                        class_accs = str(queue.get_nowait())
                        print("水果是data---",class_accs)
                        im2 = cv2.resize(fff, (299,299), interpolation=cv2.INTER_CUBIC)
                        f=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
                        ii = 0

                    except:
                        pass
                
                cv2.putText(frame, class_accs, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    
        k=cv2.waitKey(1)
        
        cv2.imshow("capture", frame)
        
        
        

        if k==27:
            queue1.put('q')
            break
        ii=ii+1
    process1.join()
    cap.release()
    cv2.destroyAllWindows()
    exit()
    
