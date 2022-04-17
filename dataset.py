import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
#os.path.exists(file)检测file路径是否有文件或者目录，如果没有，就在file路径创建一个空文件夹


file_path = 'E:/Program Files/Alexnet/data_name'
flower_class = [cla for cla in os.listdir(file_path)] #os.listdir会返回一个list,包含了该路径下所有文件名和子目录名

mkfile('./data/train')  #相对地址，相对Alexnet文件夹而言
for cla in flower_class:
    mkfile('data/train/' + cla)

mkfile('./data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)

split_rate = 0.2 #划分比例
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/' #路径拼接，把cla里面的拿出来
    images = os.listdir(cla_path)  #图片名
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate)) #随机截取images中k个元素组成新列表
    for index, image in enumerate(images):#index是默认变量，代表自增索引
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)#从老路径copy到新路径
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print('Now the processing of class[{}] is [{}]/[{}]\n'.format(cla, index+1, num))
print('Processing files over')
