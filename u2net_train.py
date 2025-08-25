import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from data_loader import MyDataset
from data_loader import NormalizeSeparate
from model import U2NET

# CONFIG
epoch_num = 100 # 100000 NOTEs
batch_size_train = 8 # 6 NOTE
batch_size_val = 1
train_num = 0
val_num = 0
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations
learning_rate = 0.001

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------
model_name = 'u2net'
current_dir = os.getcwd()
saved_dir = os.path.join(current_dir, 'saved_models')
dataset_dir = os.path.join(current_dir, 'dataset')
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

ext = '.png'
pattern = os.path.join(dataset_dir,'*' + ext)
all_files = glob.glob(pattern)

imgs_path_list = [f for f in all_files if not f.endswith('_MASK.png')]
labels_path_list = [f.replace(ext, '_MASK' + ext) for f in imgs_path_list]

train_num = len(imgs_path_list)
print("---")
print("train images: ", train_num)
print("---")

# transform 一般是放在 __getitem__ 中的,因此每个样本被取出来的时候,都会单独经过一次 transform
my_dataset = MyDataset(
    img_name_list=imgs_path_list,
    lbl_name_list=labels_path_list,
    transform= A.Compose([
        A.RandomSizedCrop((310,330),320,320),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.AdvancedBlur(),
        A.RandomCrop(288,288),
        NormalizeSeparate(),
        ToTensorV2()
    ])
)

my_dataloader = DataLoader(
    my_dataset,
    batch_size=batch_size_train,
    shuffle=True, # 每个 epoch 取出来的顺序是不一样的
    num_workers=0) # NOTE num_workers=0 for Windows

# ------- 3. define model --------
# define the net
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")

for epoch in range(0, epoch_num):
    net.train() # 进入训练模式

    for i, data in enumerate(my_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        labels = labels.permute(0,3,1,2) # Tensor 格式

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            save_path = os.path.join(
                saved_dir,
                "{}_bce_itr_{}_train_{:.3f}_tar_{:.3f}.pth".format(
                    model_name, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val
                )
            )
            torch.save(net.state_dict(), save_path)
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

