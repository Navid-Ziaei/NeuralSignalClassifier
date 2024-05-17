import seaborn as sns
import matplotlib.pyplot as plt


def result_box_plot(result, metric='F1-Score'):
    # Create a box plot with scattered points
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5), gridspec_kw={'wspace': .1, 'hspace': 0.8})

    sns.boxplot(y=result, ax=ax, palette='Set2')

    # Add scattered points
    sns.stripplot(y=result, jitter=True, linewidth=1, palette='Set2', ax=ax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel('Patients')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} for Each Patient', fontsize=10, fontweight='bold', color='dimgray')

    plt.show()
