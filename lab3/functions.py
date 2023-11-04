from matplotlib import pyplot as plt

def getAccuracy(total, top5total, figtitle):
    labels = ["accuracy", "error"]
    acc = [total, 10000 - total]
    top5acc = [top5total, 10000 - top5total]

    plt.subplot(1, 2, 1)
    plt.pie(acc, labels=labels)
    plt.title("Accuracy %" + str(total * 100 // 10000))

    plt.subplot(1, 2, 2)
    plt.pie(top5acc, labels=labels)
    plt.title("Top5Accuracy %" + str(top5total * 100 // 10000))

    plt.savefig(figtitle)
    blessed = total/10000
    top5bless = top5total/10000

    return blessed, top5bless