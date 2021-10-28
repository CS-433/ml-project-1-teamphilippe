import matplotlib.pyplot as plt

def plot_histogram(x, col_idx):
    """
    Plot a histogram for the givne column
    Parameters 
        x: 
            The data matrix
        col_idx:
            The index of the column for which we want to plot the histogram
    """
    # Plot histograms
    plt.hist(x[:, col_idx], bins=100, log=True)
    
    # Set titles and axis labels
    plt.title(f'Histogram of feature index {col_idx}')
    plt.ylabel('Counts (log scale)')
    plt.xlabel('Values')
    plt.show()
    
def double_histograms(x, idx_1, idx_2):
    """
    Plot a histogram for the givne column
    Parameters 
        x: 
            The data matrix
        idx_1:
            The index of the first column for which we want to plot the histogram
        idx_2:
            The index of the second column for which we want to plot the histogram
        
    """
    fig, ax = plt.subplots(1, 2, sharey='row',figsize=(15,5))
    # Plot histograms
    ax[0].hist(x[:, idx_1], bins=100, log=True)
    ax[1].hist(x[:, idx_2], bins=100, log=True)
    plt.tight_layout()


    # Set titles and axis labels
    ax[0].set_title(f'Histogram of feature index {idx_1}')
    ax[0].set_ylabel('Counts (log scale)')
    ax[0].set_xlabel('Values')

    ax[1].set_title(f'Histogram of feature index {idx_2}')
    ax[1].set_ylabel('Counts (log scale)')
    ax[1].set_xlabel('Values')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

def quad_histograms(x, idx_1, idx_2, idx_3, idx_4):
    """
    Plot a histogram for the givne column
    Parameters 
        x: 
            The data matrix
        idx_1:
            The index of the first column for which we want to plot the histogram
        idx_2:
            The index of the second column for which we want to plot the histogram
        idx_3:
            The index of the third column for which we want to plot the histogram
        idx_4:
            The index of the fourth column for which we want to plot the histogram
        
    """
    fig, ax = plt.subplots(2, 2, sharey='row',figsize=(15,10))

    # Plot histograms
    ax[0,0].hist(x[:, idx_1], bins=100, log=True)
    ax[0,1].hist(x[:, idx_2], bins=100, log=True)
    ax[1,0].hist(x[:, idx_3], bins=100, log=True)
    ax[1,1].hist(x[:, idx_4], bins=100, log=True)
    plt.tight_layout()


    # Set titles and axis labels
    ax[0, 0].set_title(f'Histogram of feature index {idx_1}')
    ax[0, 0].set_ylabel('Counts (log scale)')
    ax[0, 0].set_xlabel('Values')

    ax[0, 1].set_title(f'Histogram of feature index {idx_2}')
    ax[0, 1].set_ylabel('Counts (log scale)')
    ax[0, 1].set_xlabel('Values')

    ax[1, 0].set_title(f'Histogram of feature index {idx_3}')
    ax[1, 0].set_ylabel('Counts (log scale)')
    ax[1, 0].set_xlabel('Values')
    
    ax[1, 1].set_title(f'Histogram of feature index {idx_4}')
    ax[1, 1].set_ylabel('Counts (log scale)')
    ax[1, 1].set_xlabel('Values')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()