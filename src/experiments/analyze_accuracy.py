import os
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import re


def main():
    data_dir = '/Users/applepie/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/pilot1/merged/'
    filepaths = [filepath for filepath in os.listdir(data_dir) if filepath.endswith('reformatted.csv')]

    img_exp_averages = []
    img_noexp_averages = []
    word_exp_averages = []
    word_noexp_averages = []

    subjects = []
    for filepath in filepaths:
        match = re.search('^[0-9]*', filepath)
        subject = match.group()

        subjects.append(subject)
        df = pd.read_csv(data_dir + filepath, index_col=0)
        bars_exp = np.array([block_exp_acc(df, block_num) for block_num in range(8)])
        bars_exp_len = np.array([block_exp_len(df, block_num) for block_num in range(8)])
        bars_noexp = np.array([block_noexp_acc(df, block_num) for block_num in range(8)])
        bars_noexp_len = np.array([block_noexp_len(df, block_num) for block_num in range(8)])

        img_exp_averages.append(np.mean(bars_exp[:4]))
        img_noexp_averages.append(np.mean(bars_noexp[:4]))
        word_exp_averages.append(np.mean(bars_exp[4:]))
        word_noexp_averages.append(np.mean(bars_noexp[4:]))

        width = .35
        z_score = 1.96
        conf_intervals_exp = z_score * np.sqrt(bars_exp * (1 - bars_exp) / bars_exp_len)
        conf_intervals_noexp = z_score * np.sqrt(bars_noexp * (1 - bars_noexp) / bars_noexp_len)
        plt.bar(np.arange(8) - width / 2, bars_exp, width, label='Exp Acc')
        plt.bar(np.arange(8) + width / 2, bars_noexp, width, label='No Exp Acc')

        xticks = [df[df['block_number'] == block_num]['block_type'].iloc[0] + f'\n{block_num+1}' for block_num in range(8)]
        plt.xticks(np.arange(8), xticks)
        plt.errorbar(np.arange(8) - width / 2, bars_exp, yerr=conf_intervals_exp, fmt='o', color='r', capsize=5)
        plt.errorbar(np.arange(8) + width / 2, bars_noexp, yerr=conf_intervals_noexp, fmt='o', color='r', capsize=5)
        plt.legend(loc='upper right')
        plt.ylabel('accuracy')
        # plt.axhline(y=0.5, color='black', linestyle=':')
        plt.title(f'Subject {subject}')

        plt.savefig(f'figures/bars/acc/{subject}')
        # plt.show()
        # break
        plt.clf()

    subjects = np.array(subjects).astype(int)
    indices = np.argsort(subjects)
    plt.bar(np.arange(len(filepaths)) - width / 2, np.array(img_exp_averages)[indices], width, label='Exp Acc')
    plt.bar(np.arange(len(filepaths)) + width / 2, np.array(img_noexp_averages)[indices], width, label='No Exp Acc')
    plt.xlabel('subject')
    plt.ylabel('accuracy')
    plt.title('Avg Accuracy for Images')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(subjects)), subjects[indices], rotation=45, ha="right")
    plt.savefig('figures/bars/acc/avg-img')
    plt.show()

    plt.bar(np.arange(len(filepaths)) - width / 2, word_exp_averages, width, label='Exp Acc')
    plt.bar(np.arange(len(filepaths)) + width / 2, word_noexp_averages, width, label='No Exp Acc')
    plt.xlabel('subject')
    plt.ylabel('accuracy')
    plt.title('Avg Accuracy for Words')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(subjects)), subjects[indices], rotation=45, ha="right")
    plt.savefig('figures/bars/acc/avg-word')
    plt.show()

    avg_img_acc_exp = np.mean(img_exp_averages)
    avg_img_acc_noexp = np.mean(img_noexp_averages)
    avg_word_acc_exp = np.mean(word_exp_averages)
    avg_word_acc_noexp = np.mean(word_noexp_averages)

    avg_accs = [np.mean(img_exp_averages), np.mean(img_noexp_averages), np.mean(word_exp_averages), np.mean(word_noexp_averages)]
    avg_std = [np.std(img_exp_averages), np.std(img_noexp_averages), np.std(word_exp_averages), np.std(word_noexp_averages)]

    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    print(colors.shape)
    plt.bar(np.arange(4), avg_accs, color=colors)
    plt.title('Average Accuracy Across All Subjects')
    plt.errorbar(np.arange(4), avg_accs, yerr=avg_std, fmt='o', color='r', capsize=5)
    plt.xticks(np.arange(4), ['i+e', 'i-e', 'w+e', 'w-e'])
    plt.savefig('figures/bars/acc/avg-word')
    plt.show()

def block_exp_acc(df, block_num):
    block = df[df['is_experienced'] & (df['block_number'] == block_num)]['is_correct']
    return block.mean()


def block_exp_len(df, block_num):
    block = df[df['is_experienced'] & (df['block_number'] == block_num)]['is_correct']
    return len(block)


def block_noexp_acc(df, block_num):
    block = df[(1 - df['is_experienced']) & (df['block_number'] == block_num)]['is_correct']
    return block.mean()


def block_noexp_len(df, block_num):
    block = df[(1 - df['is_experienced']) & (df['block_number'] == block_num)]['is_correct']
    return len(block)


if __name__ == '__main__':
    main()
