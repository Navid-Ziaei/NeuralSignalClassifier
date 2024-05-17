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




        df = pd.read_csv(data_dir + filepath, index_col=0)

        bars_exp = np.array([block_exp_rt(df, i) for i in range(8)])
        bars_noexp = np.array([block_noexp_rt(df, i) for i in range(8)])
        bars_exp_std = np.array([block_exp_rt_std(df, i) for i in range(8)])
        bars_noexp_std = np.array([block_noexp_rt_std(df, i) for i in range(8)])


        img_exp_averages.append(np.mean([el for el in bars_exp[:4] if not np.isnan(el)]))
        img_noexp_averages.append(np.mean([el for el in bars_noexp[:4] if not np.isnan(el)]))
        word_exp_averages.append(np.mean([el for el in bars_exp[4:] if not np.isnan(el)]))
        word_noexp_averages.append(np.mean([el for el in bars_noexp[4:] if not np.isnan(el)]))

        width = .35
        plt.bar(np.arange(8) - width / 2, bars_exp, width, label='Exp RT')
        plt.bar(np.arange(8) + width / 2, bars_noexp, width, label='No Exp RT')
        plt.errorbar(np.arange(8) - width / 2, bars_exp, yerr=bars_exp_std, fmt='o', color='r', capsize=5)
        plt.errorbar(np.arange(8) + width / 2, bars_noexp, yerr=bars_noexp_std, fmt='o', color='r', capsize=5)
        plt.legend(loc='upper right')
        # plt.xlabel('go trigger')
        plt.ylabel('response time')
        plt.title(f'Subject {subject}')
        xticks = [df[df['block_number'] == block_num]['block_type'].iloc[0] + f'\n{block_num}' for block_num in range(8)]
        plt.xticks([r for r in range(8)], xticks)

        plt.savefig(f'figures/bars/rt/{subject}')
        plt.clf()

        subjects.append(subject)

    subjects = np.array(subjects).astype(int)
    indices = np.argsort(subjects)
    plt.bar(np.arange(len(filepaths)) - width / 2, np.array(img_exp_averages)[indices], width, label='Exp RT')
    plt.bar(np.arange(len(filepaths)) + width / 2, np.array(img_noexp_averages)[indices], width, label='No Exp RT')
    plt.xlabel('subject')
    plt.ylabel('response time')
    plt.title('Avg RT for Images')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(subjects)), subjects[indices], rotation=45, ha="right")
    plt.savefig(f'figures/bars/rt/avg-img')
    plt.show()
    plt.clf()

    plt.bar(np.arange(len(filepaths)) - width / 2, np.array(word_exp_averages)[indices], width, label='Exp RT')
    plt.bar(np.arange(len(filepaths)) + width / 2, np.array(word_noexp_averages)[indices], width, label='No Exp RT')
    plt.xlabel('subject')
    plt.ylabel('response time')
    plt.title('Avg RT for Words')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(subjects)), subjects[indices], rotation=45, ha="right")
    plt.savefig(f'figures/bars/rt/avg-word')




def block_exp_rt(df, block_num):
    block = df[df['is_experienced'].astype(bool)]
    block = block[block['block_number'] == block_num]
    block = block['response_time']
    # block = block.replace(0, 1000)
    block = block[block > 0]
    return block.mean()


def block_noexp_rt(df, block_num):
    block = df[(1 - df['is_experienced']).astype(bool) & (df['block_number'] == block_num)]['response_time']
    # block = block.replace(0, 1000)
    block = block[block > 0]
    return block.mean()


def block_exp_rt_std(df, block_num):
    block = df[df['is_experienced'].astype(bool) & (df['block_number'] == block_num)]['response_time']
    # block = block.replace(0, 1000)
    block = block[block > 0]
    return block.std()


def block_exp_rt_hist(df, block_num):
    block = df[df['is_experienced'].astype(bool) & (df['block_number'] == block_num)]['response_time']
    # block = block.replace(0, 1000)
    block = block[block > 0]
    block.hist(bins=50)
    return block


def block_noexp_rt_std(df, block_num):
    block = df[(1 - df['is_experienced']).astype(bool) & (df['block_number'] == block_num)]['response_time']
    # block = block.replace(0, 1000)
    block = block[block > 0]
    return block.std()


if __name__ == '__main__':
    main()