import numpy as np
import torch


def compute_significance(con, q_threshold=0.05):
    lower_quantile = np.quantile(con, q_threshold)
    upper_quantile = np.quantile(con, 1 - q_threshold)
    # Check if both quantiles are negative or both positive
    significant = np.sign(lower_quantile) == np.sign(upper_quantile)
    
    return significant



def get_global(concepts, target_generations, l_channels, target_class='class'):
    target_disease = target_class
    target_samples = 20
    target_generations = target_generations
    channels = range(l_channels) 

    global_do0_u = {}
    global_do0_c = {}

    global_sig_dou_s = {}
    global_sig_doc_s = {}

    global_sig_dou_g = {}
    global_sig_doc_g = {}

    for mask_s in tqdm.tqdm(concepts):
        do1s = []
        do0s_u = []
        do0s_c = []

        for g in range(1,target_generations+1):
            outfile = f'{target_disease}_{mask_s}_{g}.npy'
            do1 = np.load(f'/user/leal6863/causality/causality3/imputer/results/DROUGHT/DROUGHT/T200_beta00.0001_betaT0.02/{outfile}')
            do0_u = np.load(f'/user/leal6863/causality/causality3/imputer/results/DROUGHT/UNCOND/T200_beta00.0001_betaT0.02/{outfile}')
            do0_c = np.load(f'/user/leal6863/causality/causality3/imputer/results/DROUGHT/NORM/T200_beta00.0001_betaT0.02/{outfile}')

            y_pred_do1 = torch.sigmoid(model.forward(torch.from_numpy(do1).float().cuda())).detach().cpu().numpy()
            y_pred_do0_u = torch.sigmoid(model.forward(torch.from_numpy(do0_u).float().cuda())).detach().cpu().numpy()
            y_pred_do0_c = torch.sigmoid(model.forward(torch.from_numpy(do0_c).float().cuda())).detach().cpu().numpy()

            do1s.append(y_pred_do1[:,0])
            do0s_u.append(y_pred_do0_u[:,0])
            do0s_c.append(y_pred_do0_c[:,0])

        do1s_ite = np.log([np.mean(k) for k in zip(*do1s)]) # length = samples
        do0s_u_ite = np.log([np.mean(k) for k in zip(*do0s_u)])
        do0s_c_ite = np.log([np.mean(k) for k in zip(*do0s_c)])

        global_do0_u[mask_s] = np.mean(do1s_ite - do0s_u_ite)
        global_do0_c[mask_s] = np.mean(do1s_ite - do0s_c_ite)


        # SIGNIFICANCE
        sig_sam_u = []
        sig_sam_c = []

        num_iterations = 1000

        for i in range(num_iterations):
            random_indices_samples = np.random.choice(range(target_samples), size=target_samples, replace=True)

            # samples
            do1s_boots = np.array(do1s)[:,random_indices_samples]
            do0s_u_boots = np.array(do0s_u)[:,random_indices_samples]
            do0s_c_boots = np.array(do0s_c)[:,random_indices_samples]

            do1s_ite = np.log([np.mean(k) for k in zip(*do1s_boots)]) # length = samples
            do0s_u_ite = np.log([np.mean(k) for k in zip(*do0s_u_boots)])
            do0s_c_ite = np.log([np.mean(k) for k in zip(*do0s_c_boots)])

            sig_sam_u.append(np.mean(do1s_ite - do0s_u_ite))
            sig_sam_c.append(np.mean(do1s_ite - do0s_c_ite))


        # sig samples
        global_sig_dou_s[mask_s] = compute_significance(sig_sam_u)
        global_sig_doc_s[mask_s] = compute_significance(sig_sam_c)
        
    return global_sig_dou_s, global_sig_doc_s




def get_channel_wise(concepts, target_generations, l_channels, target_class='class'):
    
    target_disease = target_class 
    target_generations = target_generations
    channels = list(range(l_channels))

    channel_sig_dou_s = {c: {} for c in channels} 
    channel_sig_doc_s = {c: {} for c in channels}  

    channel_sig_dou_g = {c: {} for c in channels} 
    channel_sig_doc_g = {c: {} for c in channels}

    channel_do0_u = {c: {} for c in channels}  # Dictionary to store unconditioned data
    channel_do0_c = {c: {} for c in channels}  # Dictionary to store conditioned data

    for mask_s in tqdm.tqdm(concepts):
        for c in channels:  # Iterate over channels
            do1s = []  # List to store mean predictions for each channel
            do0s_u = []  # List to store unconditioned data for each channel
            do0s_c = []  # List to store conditioned data for each channel

            for g in range(1, target_generations + 1):
                outfile = f'{target_disease}_{mask_s}_{g}.npy'
                do1 = np.load(f'/class/T200_beta00.0001_betaT0.02/{outfile}')
                do0_u = np.load(f'/uncond/T200_beta00.0001_betaT0.02/{outfile}')
                do0_c = np.load(f'/norm/T200_beta00.0001_betaT0.02/{outfile}')

                ch = channels.copy()
                ch.remove(c)
                do1[:,ch,:] = real[:,ch,:].copy()
                do0_u[:,ch,:] = real[:,ch,:].copy()
                do0_c[:,ch,:] = real[:,ch,:].copy()

                y_pred_do1 = torch.sigmoid(model.forward(torch.from_numpy(do1).float().cuda())).detach().cpu().numpy()
                y_pred_do0_u = torch.sigmoid(model.forward(torch.from_numpy(do0_u).float().cuda())).detach().cpu().numpy()
                y_pred_do0_c = torch.sigmoid(model.forward(torch.from_numpy(do0_c).float().cuda())).detach().cpu().numpy()

                do1s.append(y_pred_do1[:,0])
                do0s_u.append(y_pred_do0_u[:,0])
                do0s_c.append(y_pred_do0_c[:,0])

            do1s_ite = np.log([np.mean(k) for k in zip(*do1s)])
            do0s_u_ite = np.log([np.mean(k) for k in zip(*do0s_u)])
            do0s_c_ite = np.log([np.mean(k) for k in zip(*do0s_c)])

            channel_do0_u[c][mask_s] = np.mean(do1s_ite - do0s_u_ite)
            channel_do0_c[c][mask_s] = np.mean(do1s_ite - do0s_c_ite)

            # SIGNIFICANCE
            sig_sam_u = []
            sig_sam_c = []

            num_iterations = 1000

            for i in range(num_iterations):
                random_indices_samples = np.random.choice(range(target_samples), size=target_samples, replace=True)

                # samples
                do1s_boots = np.array(do1s)[:,random_indices_samples]
                do0s_u_boots = np.array(do0s_u)[:,random_indices_samples]
                do0s_c_boots = np.array(do0s_c)[:,random_indices_samples]

                do1s_ite = np.log([np.mean(k) for k in zip(*do1s_boots)])
                do0s_u_ite = np.log([np.mean(k) for k in zip(*do0s_u_boots)])
                do0s_c_ite = np.log([np.mean(k) for k in zip(*do0s_c_boots)])

                sig_sam_u.append(np.mean(do1s_ite - do0s_u_ite))
                sig_sam_c.append(np.mean(do1s_ite - do0s_c_ite))

            # sig samples
            channel_sig_dou_s[c][mask_s] = compute_significance(sig_sam_u)
            channel_sig_doc_s[c][mask_s] = compute_significance(sig_sam_c)
            
    return channel_sig_dou_s, channel_sig_doc_s


def get_plot_drought(global_sig_do, global_sig_do_s, global_do0,
                channel_sig_do, channel_sig_do_s, channel_do0,
                 channels,
                  concepts
                name):
    
    plt.rcParams['font.size'] = 12
    fig, axs = plt.subplots(1, 2, figsize=(18, 3), dpi=600)

    data_channel_do0 = np.array([[channel_do0[c][mask_s] for c in sorted(channel_do0.keys())] for mask_s in ['0', '1', '2', '3', '4']])
    
    
    extreme_value = max( abs(np.min(data_channel_do0)), abs(np.max(data_channel_do0)) )  
    
    ax0 = axs[0]
    im0 = ax0.imshow(data_channel_do0, vmin=-extreme_value, vmax=extreme_value, cmap='seismic')

    ax0.set_xticks(np.arange(len(channels)))
    ax0.set_xticklabels(channels, rotation=70)
    ax0.set_yticks(np.arange(len(concepts)))
    ax0.set_yticklabels(concepts)

    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    
    v0 = [-extreme_value,  0,  extreme_value]
    plt.colorbar(im0, cax=cax0, ticks=v0, format=FuncFormatter(lambda x, _: f'{x:.4f}'), ax=ax0)

    
    global_do0 = np.array(list(global_do0.values())).reshape(-1,1)    
    
    extreme_value = max( abs(np.min(global_do0)), abs(np.max(global_do0)) )
    
    ax1 = axs[1]
    im1 = ax1.imshow(global_do0, vmin=-extreme_value, vmax=extreme_value, cmap='seismic')

    channels_global = ['Global']  # Channel for the global plot
    ax1.set_xticks(np.arange(len(channels_global)))
    ax1.set_xticklabels(channels_global)
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', which='both', left=False)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="50%", pad=0.05)  # Adjusted size
    v1 = [-extreme_value,  0, extreme_value]
    plt.colorbar(im1, cax=cax1, ticks=v1, format=FuncFormatter(lambda x, _: f'{x:.4f}'), ax=ax1)

            
    for i,c in enumerate(channels):
        for concept,concept_plot in zip(channel_sig_do_s[0].keys(), concepts):
            sig = channel_sig_do_s[i][concept]
            if sig:
                ax0.scatter(i, concepts.index(concept_plot), marker='*', color='gray', s=100)

    for concept,concept_plot in zip(global_sig_do_s.keys(),concepts):
        sig = global_sig_do_s[concept]
        if sig:
            ax1.scatter(0, concepts.index(concept_plot), marker='*', color='gray', s=100)

    plt.tight_layout(pad=0)  

    plt.savefig(name, bbox_inches='tight', pad_inches=0.1)
    plt.show()