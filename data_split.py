import random
import os
import json
import matplotlib.pyplot as plt



def read_split_data(root: str, val_rate: float = 0.2):  # root代表图片的地址
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    '''
        遍历文件夹，一个文件夹对应一个类别
        os.listdir(root) 会返回指定路径root下的所有文件和文件夹的名称
        os.path.isdir(os.path.join(root, cla)) 用于判断某个名称是否是一个文件夹
        os.path.join(root, cla) 将路径root和文件夹名称cla连接起来形成完整的路径

        返回每个文件中的名称cla，如果它是一个文件夹，就将其添加到列表flower_class中去
    '''
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    '''
        排序，保证各平台顺序一致
        生成类别名称以及对应的数字索引
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    '''
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    '''
        {
            "0": "daisy",
            "1": "dandelion",
            "2": "roses",
            "3": "sunflowers",
            "4": "tulips"
        }
    '''
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    '''
        open('class_indices.json', 'w') 打开一个名为 class_indices.json 的文件.
        使用写入模式 'w'。如果文件不存在，则会创建新文件；如果文件已存在，则会清空文件内容并从头开始写入
    '''
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)  # 将整体路径连接起来
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本  images 中随机抽取一定数量的元素作为验证集的路径
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:  # images存储的是路径
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label