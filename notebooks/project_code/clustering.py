import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

def plot_tsne_clusters(data, 
                       labels, 
                       perplexity = 30,
                       s = 0.5, 
                       figsize = (10,10),
                       random_state = None):
    """Plot tsne 2D composition
    """
    # calculate tsne
    tsne = TSNE(random_state = random_state,
                perplexity = perplexity)
    tsne_comp = tsne.fit_transform(data)
    
    # create plot
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))
    plt.figure(figsize = figsize)
    plt.scatter(tsne_comp[:, 0], tsne_comp[:, 1],
                c = palette[labels], s = s);