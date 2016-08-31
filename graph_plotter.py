import matplotlib.pyplot as plt


def cost_vs_epoch(cost_list):
    plt.plot(list(range(len(cost_list))),cost_list)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()

def accuracy_vs_epoch(accuracy_list):
    plt.plot(list(range(len(accuracy_list))), accuracy_list)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()