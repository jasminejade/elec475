from matplotlib import pyplot as plt
from torchvision import transforms
def getAccuracy(total, figtitle):
    labels = ["accuracy", "error"]
    acc = [total, 10000 - total]


    # plt.subplot(1, 2, 1)
    plt.pie(acc, labels=labels)
    plt.title("Accuracy %" + str(total * 100 // 10000))


    plt.savefig(figtitle)
    blessed = total/10000

    return blessed

def train_transformer():
    transform_list = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform_list


def testImage(model, eval_set):
    model.to(device='cuda')
    model.eval()

    index = int(input("Pleas enter a number between 1-10000: "))
    img = eval_set[index][0]
    label = eval_set[index][1]

    output = model(img)

    fig, ax = plt.subplots(1,1)
    fig.suptitle(f'test image is class: {label} , netSales guess is:{output}')
    ax[0].imshow(img)

    plt.show()

