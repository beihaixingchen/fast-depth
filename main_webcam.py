import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils



from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

'''
from torchviz import make_dot
import onnx
import onnxruntime as ort
'''

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        # print(checkpoint)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        #print(model)

        model.eval()

        cap = cv2.VideoCapture(1)#cap表示视频，函数内部参数表示调用的摄像头编号

        while True:

            start = time.time()

            #图片输入
            #filename = 'image.jpg'
            ret, frame = cap.read()#frame表示每一帧的图片

            cv2.imshow('frame', frame)

            image = Image.fromarray(frame)
            #image = Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
            image = image.resize((224,224),Image.ANTIALIAS) #将其转换为要求的输入大小224*224
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(image) #转为Tensor
            x = img.resize(1,3,224,224) #如果存在要求输入图像为4维的情况，使用resize函数增加一维

            #深度推理
            #x = torch.rand(1,3,224,224)
            x_torch = x.type(torch.cuda.FloatTensor)
            depth = model(x_torch)

            #图片输出
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8))-1

            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape, dtype=depth.type)
            
            out = out.cpu().detach().numpy()  # tensor转化为np.array
            out = out.reshape(224,224)  # (1,1,224,224)转为(224,224)
            #print(out)
            out = Image.fromarray(out) 
            out = out.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            
            out = np.array(out)

            cv2.imshow('out', out)
            #plt.imshow(out, cmap=plt.cm.inferno)

            end = time.time()
            print(1/(end-start))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #out.save('depth.png')
        
        cap.release()
        cv2.destroyAllWindows()

        #output_directory = os.path.dirname(args.evaluate)
        #validate(val_loader, model, args.start_epoch, write_to_file=False)
        return


if __name__ == '__main__': 
    main()
