import os

import matplotlib
matplotlib.use('Agg')          # save figures without needing a display
import matplotlib.pyplot as plt

FIGURE_DIR = os.environ.get('EIGENSCORE_FIGURE_DIR', './Figure')


def VisAUROC(tpr, fpr, AUROC, method_name, file_name="CoQA"):
    if "coqa" in file_name:
        file_name = "CoQA"
    if "nq" in file_name:
        file_name = "NQ"
    if "trivia" in file_name:
        file_name = "TriviaQA"
    if "SQuAD" in file_name:
        file_name = "SQuAD"
    plt.plot(fpr, tpr, label="AUC-{}=".format(method_name)+str(round(AUROC,3)))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('ROC Curve on {} Dataset'.format(file_name), fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, "AUROC_{}.png".format(file_name)),
                dpi=300, bbox_inches='tight')


def reset_figure():
    plt.close('all')


if __name__ == "__main__":
    pass
