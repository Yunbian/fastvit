import glob
import os
import shutil
from sklearn.model_selection import train_test_split
from data_split import read_split_data

'''
data
├─Black-grass
├─Charlock
├─Cleavers
├─Common Chickweed
├─Common wheat
├─Fat Hen
├─Loose Silky-bent
├─Maize
├─Scentless Mayweed
├─Shepherds Purse
├─Small-flowered Cranesbill
└─Sugar beet

如果自己的数据是上面的，通过makedata方法，将其结构转换为下面的结构类型

├─data
│  ├─val
│  │   ├─Black-grass
│  │   ├─Charlock
│  │   ├─Cleavers
│  │   ├─Common Chickweed
│  │   ├─Common wheat
│  │   ├─Fat Hen
│  │   ├─Loose Silky-bent
│  │   ├─Maize
│  │   ├─Scentless Mayweed
│  │   ├─Shepherds Purse
│  │   ├─Small-flowered Cranesbill
│  │   └─Sugar beet
│  └─train
│      ├─Black-grass
│      ├─Charlock
│      ├─Cleavers
│      ├─Common Chickweed
│      ├─Common wheat
│      ├─Fat Hen
│      ├─Loose Silky-bent
│      ├─Maize
│      ├─Scentless Mayweed
│      ├─Shepherds Purse
│      ├─Small-flowered Cranesbill
│      └─Sugar beet
'''

image_list=glob.glob('data1/*/*')
print(image_list)
file_dir='data'
if os.path.exists(file_dir):
    print('true')
    #os.rmdir(file_dir)
    shutil.rmtree(file_dir)#删除再建立
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

trainval_files, val_files = train_test_split(image_list,test_size=0.3,random_state=42)
#train_data, train_label, val_data, val_label = read_split_data(image_list, test_size=0.3, random_state=42)   #image_list划分为训练集和验证集，并将结果保存在trainval_files和val_files中供后续使用。

train_dir='train'
val_dir='val'

train_root=os.path.join(file_dir, train_dir)
val_root=os.path.join(file_dir, val_dir)

print("train_root   " + train_root)
print("val_root   " + val_root)
count = 0

for file in trainval_files:
    count += 1
    file_class=file.replace("\\", "/").split('/')[-2]
    file_name=file.replace("\\", "/").split('/')[-1]
    file_class=os.path.join(train_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)

print(count)

for file in val_files:
    file_class=file.replace("\\", "/").split('/')[-2]
    file_name=file.replace("\\", "/").split('/')[-1]
    file_class=os.path.join(val_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)