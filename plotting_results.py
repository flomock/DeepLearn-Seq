import matplotlib

matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def plotting_history(path, file, titel, x_axes='', y_axes='', accuracy=False, loss=False, voting=False, runtime=False,
                     label1='', label2='', label3='', label4=''):
    """plot accuracy and loss of train and test data
    good for observing overfitting
    """
    history = pd.read_csv(path +"/"+ file)
    totalEpochs = len(history['Unnamed: 0'])
    uniqueEpochs = len(history['Unnamed: 0'].unique())
    max_evals = totalEpochs // uniqueEpochs
    history = np.reshape(history.values, (max_evals, int(history.values.shape[0] / max_evals), history.values.shape[1]))

    # If you want to control which colors matplotlib cycles through, use ax.set_color_cycle:
    fig, ax = plt.subplots()
    # ax.set_color_cycle(['red', 'black', 'yellow','blue'])#"steelblue",
    colors = ["steelblue", "saddlebrown", "seagreen", "mediumorchid"]

    # number = 4
    # cmap = plt.get_cmap('Set3')
    # colors = [cmap(i) for i in np.linspace(0, 1, number)]
    # ax.set_color_cycle(colors)
    max_acc = []
    min_loss = []
    full_time = []
    for i in range(history.shape[0]):
        model_hist = history[i][0::]
        acc_values = model_hist[:, 1]
        loss_values = model_hist[:, 2]
        val_acc_values = model_hist[:, 3]
        val_loss_values = model_hist[:, 4]
        time = model_hist[:, 5]
        if voting:
            acc_vote = model_hist[:, 6]
            acc_mean = model_hist[:, 7]
            val_acc_vote = model_hist[:, 8]
            val_acc_mean = model_hist[:, 9]

        epochs = range(1, len(loss_values) + 1)
        max_acc.append(max(val_acc_values))
        min_loss.append(min(val_loss_values))
        full_time.append(time[-1])

        if x_axes == '':
            if runtime:
                x_axes = 'Runtime in s'
            else:
                x_axes = 'Epochs'

        if y_axes == '':
            if accuracy and loss:
                y_axes = 'Accuracy and Loss'
            elif accuracy:
                y_axes = 'Accuracy'
            elif loss:
                y_axes = 'Loss'
            else:
                y_axes = 'no defined label'

        if runtime:
            epochs = time

        if accuracy:
            plt.plot(epochs, acc_values, c=colors[i], label='train accuracy')
            plt.plot(epochs, val_acc_values, c=colors[i], linestyle="dashed", label='validation accuracy')
        if loss:
            plt.plot(epochs, loss_values, c=colors[i], label='train loss')
            plt.plot(epochs, val_loss_values, c=colors[i], linestyle="dashed", label='validation loss')
        if voting:
            plt.plot(epochs, acc_vote, c=colors[i + 1], label='train accuracy voting')
            plt.plot(epochs, acc_mean, c=colors[i + 2], label='train accuracy mean-predict')
            plt.plot(epochs, val_acc_vote, c=colors[i + 1], linestyle="dashed", label='validation accuracy voting')
            max_acc.append(max(val_acc_vote))
            plt.plot(epochs, val_acc_mean, c=colors[i + 2], linestyle="dashed",
                     label='validation accuracy mean-predict')
            max_acc.append(max(val_acc_mean))
            plt.yticks(list(plt.yticks()[0]) + [0, max(val_acc_values)])

        # if runtime:
        #     plt.plot((0,time[-1]/5),(max(val_acc_values),max(val_acc_values)),  color = colors[i])#, linestyle = "dashed")
        #     plt.plot((time[-1], time[-1]), (0.62, acc_values[-1]), color = colors[i])
    ax.set_ylim(ax.get_ylim())
    for m in range(0, len(max_acc)):
        plt.plot((0, ax.get_xlim()[1] / 10), (max_acc[m], max_acc[m]), color=colors[m])
        if runtime:
            plt.plot((full_time[m], full_time[m]),
                     (ax.get_ylim()[0], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 5), color=colors[m])
    print(max_acc)
    print(min_loss)
    print(full_time)

    handels = []
    for index, label in enumerate([label1, label2, label3, label4]):
        if label == '':
            break
        else:
            patch = mpatches.Patch(color=colors[index], label=label)
            handels.append(patch)

    line = mlines.Line2D([], [], linestyle="solid", color="black", label="training set")
    line2 = mlines.Line2D([], [], linestyle="dashed", color="black", label="validation set")
    handels.append(line)
    handels.append(line2)
    # plt.legend()
    ax.set_title(titel)
    # ax.set_title('Loss value over Epochs')
    # ax.set_xlim(left=0)
    plt.legend(handles=handels, fontsize='large', ncol=1)  # ,loc=4)
    # plt.ylim(0.8,1.01)
    # plt.xlabel('Time in s')
    plt.xlabel(x_axes)
    plt.ylabel(y_axes)
    # plt.subplots_adjust(left=0.12, right=0.995, top=0.94, bottom=0.1)
    # plt.tight_layout()
    plt.savefig(path +"/"+ file[:-4] + ".pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("/home/florian/Dropbox/Masterarbeit/latex/figures/"+file[:-4]+".pdf", bbox_inches='tight',pad_inches = 0)
    # plt.show()


# path = '/home/go96bix/projects/nanocomb/nanocomb/plots/'
# file = 'historynanocomb_test_100samples_len10000_TBTT.csv'
#
# plotting_history(path, file, accuracy=True, loss=True, runtime=False, label1="test", voting=True,
#                  titel='Accuracy with different Sample sizes')
