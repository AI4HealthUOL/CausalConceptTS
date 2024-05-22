from imputer.SSSDS4Imputer import SSSDS4Imputer
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams


def load_imputer(path_json, ckpt_iter):
    
    with open(path_json) as f:
        data = f.read()
    config = json.loads(data)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                                  diffusion_config["beta_0"],
                                                  diffusion_config["beta_T"])

    output_directory = gen_config['output_directory']
    output_directory = os.path.join(output_directory, local_path)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)

    print("output directory:", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    net = SSSDS4Imputer(**model_config).cuda()
    print_size(net)

    # load checkpoint
    ckpt_path = gen_config['ckpt_path']
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')
        
        
    return net