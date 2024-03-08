import numpy as np
import torch
import matplotlib.pyplot as plt

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) # 设置优化器，初始化学习率
# scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6) # 设置你的lr策略
# plt.figure()
# max_epoch=3000
# iters=5
# cur_lr_list = []
# for epoch in range(max_epoch):
#     for batch in range(iters):
#         '''
#         训练代码
#         '''
#     scheduler.step()  # 每个epoch执行一次该句，即可更新学习率
#     cur_lr=optimizer.param_groups[-1]['lr']
#     cur_lr_list.append(cur_lr)
# x_list = list(range(len(cur_lr_list)))
# plt.plot(x_list, cur_lr_list)
# plt.show()

# lr = 0.0001
# epoch_start = 0
# max_epoch = 500
# iters = 403
# def adjust_learning_rate(epoch, MAX_EPOCHES, INIT_LR, power=0.9):
#     return round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
# plt.figure()
# cur_lr_list = []
# for epoch in range(epoch_start, max_epoch):
#     # for batch in range(iters):
#     cur_lr = adjust_learning_rate(epoch, max_epoch, lr, power=0.9)
#     cur_lr_list.append(cur_lr)
# x_list = list(range(len(cur_lr_list)))
# plt.plot(x_list, cur_lr_list)
# plt.show()

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine import MessageHub
from mmengine.optim import LinearLR, CosineAnnealingLR, MultiStepLR

message_hub = MessageHub.get_instance('reduce_lr')


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


model = ToyModel()
optimizer = optim.SGD(
    model.parameters(), lr=1, momentum=0.01, weight_decay=5e-4)

scheduler = LinearLR(optimizer)

epoch_start = 0
max_epoch = 500
iters = 403

plt.figure()
cur_lr_list = []
for epoch in range(epoch_start, max_epoch):
    # for batch in range(iters):
    # cur_lr = adjust_learning_rate(epoch, max_epoch, lr, power=0.9)
    # scheduler.last_step = 1
    # message_hub.update_scalar('value', 1)
    optimizer.step()
    scheduler.step()
    cur_lr = scheduler._get_value()
    cur_lr_list.append(cur_lr)

print(cur_lr_list[:5])
print(cur_lr_list[-5:])
x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()
