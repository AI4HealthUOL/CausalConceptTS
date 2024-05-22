from sklearn.cluster import KMeans
import numpy as np

def kmeans_explore(x_train, x_val, x_test, ks=10):
    # shape == N,L,K
    n_samples_train, n_timesteps, n_features = x_train.shape
    n_samples_val, n_timesteps, n_features = x_val.shape
    n_samples_test, n_timesteps, n_features = x_test.shape

    x_train_2d = x_train.reshape(-1, n_features)
    x_val_2d = x_val.reshape(-1, n_features)
    x_test_2d = x_test.reshape(-1, n_features)

    data_2d = np.concatenate([x_train_2d, x_val_2d, x_test_2d])
    
    inertia = []  # List to store inertia values

    for k in range(1, ks):
        print(f'----:clusering at {k} clusters:----')
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_2d)
        inertia.append(kmeans.inertia_)
    
    plt.figure(dpi=300)
    plt.plot(np.diff(inertia), label='Inertia increement every additional cluster', linestyle='dashed', linewidth=5)
    plt.legend()
    plt.xticks(np.arange(0,ks-1), range(1,ks));
    
      
def kmeans_predict(x_train, x_val, x_test, k=None):
    
    train_concepts = []
    val_concepts = []
    test_concepts = []
    
    n_samples_train, n_timesteps, n_features = x_train.shape
    n_samples_val, n_timesteps, n_features = x_val.shape
    n_samples_test, n_timesteps, n_features = x_test.shape

    x_train_2d = x_train.reshape(-1, n_features)
    x_val_2d = x_val.reshape(-1, n_features)
    x_test_2d = x_test.reshape(-1, n_features)

    data_2d = np.concatenate([x_train_2d, x_val_2d, x_test_2d])
    
    
    def _create_mask(array, value):
        mask = np.ones_like(array)
        mask[array == value] = 0

        return mask

    def _reshape_to_sets(array):

        x_train = array[:n_samples_train*n_timesteps]
        x_train = x_train.reshape(n_samples_train, n_timesteps, n_features)

        x_val = array[n_samples_train*n_timesteps:n_samples_train*n_timesteps+n_samples_val*n_timesteps]
        x_val = x_val.reshape(n_samples_val, n_timesteps, n_features)

        x_test = array[n_samples_train*n_timesteps+n_samples_val*n_timesteps:]
        x_test = x_test.reshape(n_samples_test, n_timesteps, n_features)

        return x_train, x_val, x_test
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_2d)
    
    x = kmeans.predict(data_2d)
    
    for i in range(k):
        m = _create_mask(x, i)
        mask = np.tile(m[:, np.newaxis], (1, n_features))
        train_concept,val_concept,test_concept = _reshape_to_sets(mask)
        train_concepts.append(train_concept)
        val_concepts.append(val_concept)
        test_concepts.append(test_concept)
        
    return train_concepts, val_concepts, test_concepts


def plot_concepts(max_length, leadsn, index, data, concepts, savefig=True, savename='concepts.pdf'):

    sample = data[index]
    
    masks = [concepts.transpose(0, 2, 1)[index].astype(float) for concept in concepts]
    
    samples = [sample.copy() for _ in range(len(masks))]
    
    for sample, mask in zip(samples, masks):
        sample[mask == 1] = np.nan
    
    fig, axes = plt.subplots(len(leadsn), 1, figsize=(20, 16), dpi=600)
    
    if len(leadsn) == 1:
        axes = [axes]
    
    colors = ['b', 'm', 'c', 'g', 'r', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'navy', 'teal']
    colors = colors[:len(masks)]  
    labels = [chr(i) for i in range(ord('A'), ord('A') + len(masks))]

    for ax, lead, channel, *masked_samples in zip(axes, leadsn, sample, *samples):
        ax.plot(np.arange(max_length), channel[:max_length], c='k', lw=2, zorder=0, alpha=1.0)
        
        ax.set_yticks([])
        ax.legend([lead], loc='upper right', fontsize=22, handlelength=0)
        ax.set_xticks([])
        ax.set_xlim([0, max_length + 20])
        ax.set_ylim([channel.min() - 0.08, channel.max() + 0.08])
        
        for concept, color, label in zip(masked_samples, colors, labels):
            single_time_steps = np.where(~np.isnan(concept))[0]
            single_time_steps = single_time_steps[np.logical_and(single_time_steps > 0, single_time_steps < max_length)]
            ax.scatter(single_time_steps, concept[single_time_steps], c=color, marker='o', s=35, zorder=2, label=label)
    
    plt.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', fontsize=36, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.12), markerscale=3)
    
    plt.xticks(np.arange(0, max_length+1, 30), np.arange(0, max_length+1, 30), color='k', fontsize=30)
    plt.xlabel('Time steps', size=40)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if savefig:
        plt.savefig(savename, bbox_inches='tight', transparent=True)
    
    plt.show()