import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plotting_history(history):
    """plot accuracy and loss of train and test data
    good for observing overfitting
    """
    # If you want to control which colors matplotlib cycles through, use ax.set_color_cycle:
    fig, ax = plt.subplots()
    # ax.set_color_cycle(['red', 'black', 'yellow','blue'])#"steelblue",
    colors = ["steelblue","saddlebrown","seagreen","mediumorchid"]

    number = 4
    cmap = plt.get_cmap('Set3')
    # colors = [cmap(i) for i in np.linspace(0, 1, number)]
    # ax.set_color_cycle(colors)
    max_acc = []
    min_loss = []
    full_time = []
    for i in range(history.shape[0]):
        model_hist = history[i][0:50]
        acc_values = model_hist[:,1]
        loss_values = model_hist[:, 2]
        val_acc_values = model_hist[:, 3]
        val_loss_values = model_hist[:,4]
        time = model_hist[:,5]
        # acc_vote = model_hist[:, 6]
        # acc_mean = model_hist[:, 7]
        # val_acc_vote = model_hist[:, 8]
        # val_acc_mean = model_hist[:, 9]
        epochs = range(1, len(loss_values) + 1)
        max_acc.append(max(val_acc_values))
        min_loss.append(min(val_loss_values))
        full_time.append(time[-1])

        plt.plot(time, acc_values, c= colors[i], label='train accuracy')
        # plt.plot(epochs, loss_values, c= colors[i],label='train loss')
        plt.plot(time, val_acc_values, c= colors[i],linestyle = "dashed",label='validation accuracy')
        # plt.plot(epochs, val_loss_values, c= colors[i],linestyle = "dashed",label='validation loss')
        # plt.plot(epochs, acc_vote, c=colors[i+1], label='train accuracy voting')
        # plt.plot(epochs, acc_mean, c=colors[i+2], label='train accuracy mean-predict')
        # plt.plot(epochs, val_acc_vote, c=colors[i+1], linestyle="dashed", label='validation accuracy voting')
        # max_acc.append(max(val_acc_vote))
        # plt.plot(epochs, val_acc_mean, c=colors[i+2], linestyle="dashed", label='validation accuracy mean-predict')
        # max_acc.append(max(val_acc_mean))
        # plt.yticks(list(plt.yticks()[0])+[0,max(val_acc_values)])
        # plt.plot((0,time[-1]/5),(max(val_acc_values),max(val_acc_values)),  color = colors[i])#, linestyle = "dashed")
        # plt.plot((time[-1], time[-1]), (0.62, acc_values[-1]), color = colors[i])
    ax.set_ylim(ax.get_ylim())
    for m in range(0,len(max_acc)):
        plt.plot((0, ax.get_xlim()[1]/10),(max_acc[m],max_acc[m]),color = colors[m])
    #     plt.plot((full_time[m], full_time[m]), (ax.get_ylim()[0], ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])/5), color=colors[m])
    print(max_acc)
    print(min_loss)
    print(full_time)
    # ax.text(3, 0.52, r'Model 1', fontsize=10)
    # ax.text(24, 0.52, r'Model 2', fontsize=10)
    # ax.text(44, 0.52, r'Model 3', fontsize=10)
    # ax.text(64, 0.52, r'Model 4', fontsize=10)
    # ax.text(84, 0.52, r'Model 5', fontsize=10)
    # plt.plot((20, 20), (0.5, 0.61), 'k-')
    # plt.plot((40, 40), (0.5, 0.61), 'k-')
    # plt.plot((60, 60), (0.5, 0.61), 'k-')
    # plt.plot((80, 80), (0.5, 0.61), 'k-')
    patch = mpatches.Patch(color=colors[0], label="16 Nodes")
    patch2 = mpatches.Patch(color=colors[1], label="32 Nodes")
    patch3 = mpatches.Patch(color=colors[2], label="64 Nodes")
    patch4 = mpatches.Patch(color=colors[3], label="100 Nodes")
    line = mlines.Line2D([], [], linestyle="solid", color="black", label = "train set")
    line2 = mlines.Line2D([], [], linestyle="dashed", color="black", label="validation set")
    # plt.legend()
    ax.set_title('Accuracy with different Number of Nodes per Layer')
    # ax.set_title('Loss value over Epochs')
    ax.set_xlim(left=0)
    plt.legend(handles=[patch,patch2,patch3, patch4, line,line2],fontsize='large',ncol=1,loc=5)
    # plt.ylim(0.8,1.01)
    # plt.xlabel('Time in s')
    plt.xlabel('Time in s')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(left=0.08, right=0.995, top=0.94, bottom=0.1)
    # plt.savefig("/home/florian/Dropbox/Masterarbeit/latex/figures/"+file[:-4]+".pdf")
    # plt.savefig("/home/florian/Dropbox/Masterarbeit/latex/figures/"+file[:-4]+".pdf", bbox_inches='tight',pad_inches = 0)
    plt.show()


# history = pd.read_csv("historyFlavi_myVersion.csv")
# history = pd.read_csv("historyFlavi_OnOff_Timesteps.csv")
path = "/home/florian/Dropbox/Masterarbeit/ML/"
file = "historyinflu-nodes-runtime-corected.csv"
history = pd.read_csv(path+file)
# history = pd.read_csv("historyDesign.csv")
# os.remove("history.csv")
max_evals = 4
history = np.reshape(history.values,(max_evals , int(history.values.shape[0]/max_evals),history.values.shape[1]))
# print(history)
plotting_history(history)
