import matplotlib.pyplot as plt

AXIS_LABEL_FONTSIZE = 16
TITLE_LABEL_FONTSIZE = 18
TICKS_FONTSIZE = 14

def plot_histogram(values, xlabel, ylabel, title=None):
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(values, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black',
                                linewidth=1.2)

    # Title and labels with appropriate font sizes
    if title is not None:
        plt.title(title, fontsize=TITLE_LABEL_FONTSIZE, fontweight='bold')
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)

    # Adjusting tick parameters for readability
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)

    # Enhancing the grid for better readability while keeping it unobtrusive
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Removing top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()