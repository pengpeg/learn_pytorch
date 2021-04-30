# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 21:20
# @Author  : Chen
# @File    : tensorboard.py
# @Software: PyCharm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .model import Net
from .utils import matplotlib_imshow, plot_classes_preds
from .dataset import get_dataset, get_dataloader


def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainset, testset, classes = get_dataset()
    trainloader, testloader, classes = get_dataloader(trainset, testset)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    ################################################################
    # 使用make_grid将图像写入TensorBoard
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    matplotlib_imshow(img_grid, one_channel=True)
    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    #################################################################
    # 执行：tensorboard --logdir=runs

    #################################################################
    # 使用TensorBoard检查模型
    writer.add_graph(net, images)
    writer.close()

    #################################################################
    # 在 TensorBoard 中添加“投影仪”，可视化高维数据的低维表示
    # helper function
    def select_n_random(data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)
        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]
    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)
    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]
    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images.unsqueeze(1))
    writer.close()

    #################################################################
    # 使用 TensorBoard 跟踪模型训练
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs, labels, classes),
                                  global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')

    ###########################################################################
    # 使用 TensorBoard 评估经过训练的模型
    # 1\. gets the probability predictions in a test_size x num_classes Tensor
    # 2\. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # helper function
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(classes[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)
        writer.close()

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)


if __name__ == '__main__':
    main()