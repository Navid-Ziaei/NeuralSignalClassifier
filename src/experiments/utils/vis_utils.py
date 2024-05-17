import numpy as np
import matplotlib.pyplot as plt

def visualize_class_imbalance(x_train, y_train, x_val, y_val, x_test, y_test):
    # plot the data imbalance in train test and validation
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].bar(['Not Experienced', 'Experienced'], np.unique(np.squeeze(y_train), return_counts=True)[1], color='blue')
    axs[0].set_title('Train')
    axs[1].bar(['Not Experienced', 'Experienced'], np.unique(np.squeeze(y_val), return_counts=True)[1], color='orange')
    axs[1].set_title('Validation')
    axs[2].bar(['Not Experienced', 'Experienced'], np.unique(np.squeeze(y_test), return_counts=True)[1], color='green')
    axs[2].set_title('Test')
    plt.show()

    print("Class imbalance ratio (Experienced/Not Experienced) in Train: ",
          np.unique(np.squeeze(y_train), return_counts=True)[1][1] /
          np.unique(np.squeeze(y_train), return_counts=True)[1][
              0])
    print("Class imbalance ratio (Experienced/Not Experienced) in Validation: ",
          np.unique(np.squeeze(y_val), return_counts=True)[1][1] / np.unique(np.squeeze(y_val), return_counts=True)[1][
              0])
    print("Class imbalance ratio (Experienced/Not Experienced) in Test: ",
          np.unique(np.squeeze(y_test), return_counts=True)[1][1] /
          np.unique(np.squeeze(y_test), return_counts=True)[1][0])
