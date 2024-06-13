import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import distributed
from models.fastvit import fastvit_t8
from torch.autograd import Variable
from torchvision import datasets
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = ["0", "1", "2", "3"]
os.environ['CUDA_LAUNCH_BLOCKING']="1"

# def seed_everything(seed=42):
#     os.environ['PYHTONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix typo: 'PYHTONHASHSEED' -> 'PYTHONHASHSEED'
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure reproducibility

'''
    # 定义学习集 DataLoader
    train_data = torch.utils.data.DataLoader(各种设置...) 
    # 将数据喂入神经网络进行训练
    for i, (input, target) in enumerate(train_data): 
        循环代码行......
'''

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch,model_ema):
    model.train()
    print(device)
    #计算并存储损失和准确率的平均值。
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)   #train_loader事包含训练数据的数据集加载器，在每个迭代中，从加载器中获取批量的数据和对应的目标标签
    print(total_num, len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        # 打印当前数据和模型所在设备信息
        print(f"pre-Batch {batch_idx + 1}/{len(train_loader)} - Data on {data.device}, Model on {next(model.parameters()).device}")
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        print(f"doing-Batch {batch_idx + 1}/{len(train_loader)} - Data on {data.device}, Model on {next(model.parameters()).device}")
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        print("zero_grad")

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = torch.nan_to_num(criterion_train(output, targets))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            loss = criterion_train(output, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)
        print("synchronized here?")
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss =loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc

# 验证过程
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []

    for data, target in test_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg,  acc,  acc5_meter.avg))

    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), file_dir + '/' + 'best.pth')
        else:
            torch.save(model.state_dict(), file_dir + '/' + 'best.pth')
        Best_ACC = acc
    if isinstance(model, torch.nn.DataParallel):
        state = {

            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'Best_ACC':Best_ACC
        }
        if use_ema:
            state['state_dict_ema']=model.module.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    else:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema']=model.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    return val_list, pred_list, loss_meter.avg, acc

if __name__ == '__main__':
    #创建保存模型的文件夹
    file_dir = 'checkpoints/FastVit/cifar10'
    if os.path.exists(file_dir):
        print('true')
        os.makedirs(file_dir,exist_ok=True)
    else:
        os.makedirs(file_dir)
    # 设置全局参数
    model_lr = 1e-4
    BATCH_SIZE = 8
    EPOCHS = 300
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(DEVICE)
    # torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend='nccl')
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # global device
    # device = torch.device("cuda", local_rank)
    use_amp = True  # 是否使用混合精度
    use_dp = True #是否开启dp方式的多卡训练
    classes = 18
    resume =None
    CLIP_GRAD = 5.0
    Best_ACC = 0 #记录最高得分
    use_ema=True
    model_ema_decay=0.9998
    start_epoch=1
    seed=42
    seed_everything(seed)

    # 数据预处理7
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654])
    ])
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=classes)

    # 读取数据
    dataset_train = datasets.ImageFolder('data/train', transform=transform)
    dataset_test = datasets.ImageFolder("data/val", transform=transform_test)
    print(dataset_train)

    train_sampler = distributed.DistributedSampler(dataset_train)
    val_sampler = distributed.DistributedSampler(dataset_test)
    # dataset_train = datasets.CIFAR10("./cifar10", download=True, train=True, transform=transform)
    # dataset_test = datasets.CIFAR10("./cifar10", download=True, train=False, transform=transform_test)
    with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler= train_sampler, pin_memory=True, shuffle=True,
                                               drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, shuffle=False,num_workers=4)

    # 实例化模型并且移动到GPU
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = torch.nn.CrossEntropyLoss()

    # 设置模型
    model_ft = fastvit_t8(pretrained=False)

    # 多GPU操作，使用 DataParallel 包装模型
    # if torch.cuda.device_count() > 1:
    #     model_ft = nn.DataParallel(model_ft)
    # print(model_ft)
    num_fr = model_ft.head.in_features
    model_ft.head = nn.Linear(num_fr, classes)

    if resume:
        model = torch.load(resume)
        print(model['state_dict'].keys())
        model_ft.load_state_dict(model['state_dict'])
        Best_ACC = model['Best_ACC']
        start_epoch = model['epoch'] + 1
    # model_ft.to(DEVICE)
    # print(model_ft)

    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)

    device_ids = [0, 1, 2, 3]
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    # if torch.cuda.device_count() > 1 and use_dp:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model_ft = torch.nn.DataParallel(model_ft, device_ids=device_ids)
    # model_ft.to(DEVICE)
    model_ft.to(device)
    model_ft = torch.nn.DistributedDataParallel(model_ft, device_ids=device_ids, output_device=device, find_unused_parameters=True)
    if use_ema:
        model_ema = ModelEma(
            model_ft,
            decay=model_ema_decay,
            device=device,
            resume=resume)
    else:
        model_ema = None

    # 训练与验证
    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    if resume and os.path.isfile(file_dir + "result.json"):
        with open(file_dir + 'result.json', 'r', encoding='utf-8') as file:
            logs = json.load(file)
            train_acc_list = logs['train_acc']
            train_loss_list = logs['train_loss']
            val_acc_list = logs['val_acc']
            val_loss_list = logs['val_loss']
            epoch_list = logs['epoch_list']
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_list.append(epoch)
        log_dir['epoch_list'] = epoch_list
        train_loss, train_acc = train(model_ft, device, train_loader, optimizer, epoch, model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        if use_ema:
            val_list, pred_list, val_loss, val_acc = val(model_ema.ema, device, test_loader)
        else:
            val_list, pred_list, val_loss, val_acc = val(model_ft, device, test_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))
        if epoch < 600:
            continue
            # cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)

