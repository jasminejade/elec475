import numpy as np
import torch


def test(network, test_loader, device='cuda'):
    network.to(device=device)
    network.eval()
    totalOut = []
    totalLabel = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = network(imgs) # forward method
            totalOut.extend(outputs)
            totalLabel.extend(labels)
            # for each j array in output, get the index of the largest value
            # this returns an 1 dim array with the class prediction for each j

    min, max, mean, STD = eval(totalOut, totalLabel)




def eval(outputs, labels):
    # outputs.numpy(force=True)
    temp = np.zeros((len(outputs),2))
    euclid_dist = np.zeros((len(outputs),1))
    assert(len(outputs)==len(labels))
    for i in range(len(outputs)):
        temp[i] = (outputs[i].cpu() - labels[i].cpu())
        euclid_dist[i] = np.sqrt(np.dot(temp[i].T, temp[i]))
    # outputs = np.array(outputs.cpu())
    # labels = np.array(labels)
    # temp = np.array(temp)
    # euclid_dist = np.sqrt(np.dot(temp.T, temp))
    mean = np.average(euclid_dist)
    STD = np.std(euclid_dist)
    minimum = min(euclid_dist)
    maximum = max(euclid_dist)

    print(f'Min:{minimum}')
    print(f'Max:{maximum}')
    print(f'Mean:{mean}')
    print(f'STD:{STD}')

    return minimum, maximum, mean, STD