import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import random, os
import copy

from utils.data import NoiseDataset, get_mnist_data, GratingDataset, ChunkDataset
from utils.model import bernoulli_fn, zero_fn, random_step, get_model
from utils.training_evaluation import get_pc_trainer, get_mcpc_trainer, KLdivergence, get_paired_stat
from utils.plotting import setup_fig


random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


    
def similarity_increase_digit(ax=None, data=None, epochs=[0, 5, 10, 15], path_models="models"):
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        print("Using GPU")
    

    if data is None:
        config = {
            #
            "EPOCHS":50,
            "batch_size_train":256,
            "batch_size_val": 1024,
            "batch_size_test": 256,
            #
            "input_size": 20,
            "hidden_size": 128,
            "hidden2_size": 128,
            "output_size": 784,
            "loss_fn": bernoulli_fn ,
            "activation_fn": 'relu',
            #
            "T_pc":1000,
            "optimizer_x_fn_pc": optim.Adam,
            "optimizer_x_kwargs_pc":{"lr": 0.7},
            #
            "mixing":50,
            "sampling":100,
            "optimizer_x_kwargs_mcpc":{"lr": 0.1},
            #
            "optimizer_p_fn_mcpc": optim.Adam,
            "optimizer_p_kwargs_mcpc": {"lr": 0.01, "weight_decay":0.},
            "input_var":None
        }

        # Load MNIST data
        train_loader, val_loader, test_loader = get_mnist_data(config)
        # make gratings
        grating_dataset = GratingDataset(config["batch_size_test"], size=28, num_orientations=16)
        grating_loader = DataLoader(grating_dataset, batch_size=config["batch_size_test"], shuffle=True)
        # make noise
        noise_dataset = NoiseDataset(config["batch_size_test"], size=28)
        noise_loader = DataLoader(noise_dataset, batch_size=config["batch_size_test"], shuffle=True)
        
        # select 5 neurons randomly
        rand_idx = random.sample(range(config["hidden2_size"]), 5)
        
        kls_seed=[]
        seeds = range(10)
        for seed in seeds:
            model_name_base= path_models + "\\epoch_save\\mcpc_aging_"+str(seed)+"_"

            indent = 20 # indent to keep only part of data to prevent memory overload
            kls = np.zeros((3,len(epochs)))
            for idx, epoch in enumerate(epochs):
                # get model
                gen_pc = get_model(config, use_cuda)
                gen_pc.train()

                # create trainers for genaration
                config_gen = copy.deepcopy(config)
                config_gen["mixing"] = 500
                config_gen["sampling"] = 9500
                config_gen["optimizer_x_kwargs_mcpc"] = {"lr": 0.05}

                pc_trainer_gen = get_pc_trainer(gen_pc, config_gen, is_mcpc=True, training=False)
                mcpc_trainer_gen = get_mcpc_trainer(gen_pc, config_gen, training=False)
                
                model_name = model_name_base + "epoch"+str(epoch) if epoch!=0 else model_name_base + "epoch_init"
                gen_pc.load_state_dict(torch.load(model_name, map_location='cuda:0'), strict=False)

                # load all data
                data, _ = list(test_loader)[0]
                grating_data = list(grating_loader)[0]
                noise_data = list(noise_loader)[0]
                pseudo_input = torch.zeros(data.shape[0], config_gen["input_size"])
                if use_cuda:
                    pseudo_input, data, grating_data, noise_data = pseudo_input.cuda(), data.cuda(), grating_data.cuda(), noise_data.cuda()

                # get spontaneous activity
                pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=zero_fn,loss_fn_kwargs={})
                res_mcpc_prior = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=zero_fn, loss_fn_kwargs={},callback_after_t=random_step,
                                            callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                            is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
                prior = torch.concatenate([r[2] for r in res_mcpc_prior["xs"][config_gen["mixing"]::]])

                ## get evoked activity
                # for natural stimuli
                pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
                res_mcpc = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': data}, 
                                            callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                            is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
                posterior_natural = torch.concatenate([r[2] for r in res_mcpc["xs"][config_gen["mixing"]::]])
                # for gratings
                pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': grating_data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
                res_mcpc = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': grating_data}, 
                                            callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                            is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
                posterior_gratings = torch.concatenate([r[2] for r in res_mcpc["xs"][config_gen["mixing"]::]])
                # for noise
                pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': noise_data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
                res_mcpc = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': noise_data}, 
                                            callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                            is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
                posterior_noise = torch.concatenate([r[2] for r in res_mcpc["xs"][config_gen["mixing"]::]])
                
                ## data proprocessing
                prior = prior[:,rand_idx]
                posterior_natural = posterior_natural[:,rand_idx]
                posterior_gratings = posterior_gratings[:, rand_idx]
                posterior_noise = posterior_noise[:, rand_idx]
                # compute KL divergence 
                kls[0,idx] = KLdivergence(prior[::indent], posterior_natural[::indent])
                kls[1,idx] = KLdivergence(prior[::indent], posterior_noise[::indent])
                kls[2,idx] = KLdivergence(prior[::indent], posterior_gratings[::indent])
                
                print("KLs : ", kls)

                del prior, posterior_noise, posterior_gratings, posterior_natural

            kls_seed.append(kls)

        # concatenate of data across seeds
        kls_np = np.concatenate([k.reshape(k.shape[0],k.shape[1], 1) for k in kls_seed], axis=2)
    else:
        kls_np = data

    # find mean and s.e.m.
    kls_mean = kls_np.mean(-1)
    kls_sem =  kls_np.std(-1)/kls_np.shape[-1]

    # Set up the bar chart
    conditions = [str(i) for i in epochs]
    types = ['natural ', 'noise', 'gratings']
    colors = ['C0', 'C1', 'C2']

    if ax is None:
        fig, ax = plt.subplots()
        plot=True
    else:
        plot=False

    bar_width = 0.2
    index = np.arange(len(conditions))

    for i, type_label in enumerate(types):
        ax.bar(index + i * bar_width, kls_mean[i, :], bar_width, label=type_label, color=colors[i], yerr=[np.zeros_like(kls_sem[i, :])+0.05, kls_sem[i, :]], zorder=2, error_kw=dict(capsize = 4,zorder=1))

    ax.set_xlabel('epoch')
    ax.set_ylabel('KL divergence')
    ax.set_xticks(index + (bar_width * (len(types) - 1)) / 2)
    ax.set_xticklabels(conditions)  
    # ax.set_xlim([-bar_width/2, 3+5*bar_width/2])

    if kls_np.shape[2]>2:
        # find p-value for noise
        p = get_paired_stat(kls_np[0,-1,:], kls_np[1,-1,:], type="less")
        print(p)
        if p < 0.05:
            text = "*"
        if p < 0.01:
            text = "**"
        if p < 0.001:
            text = "***"
        if p >= 0.05:
            text = f"{p:.2e}"

        x1, x2 = 3, 3 + bar_width
        y, h, col = (kls_mean + kls_sem)[:,-1].max()+0.1, 0.1, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col)
        ax.set_ylim(bottom=0)

        # find p-value for grating
        p = get_paired_stat(kls_np[0,-1,:], kls_np[2,-1,:], type="less")
        print(p)
        if p < 0.05:
            text = "*"
        if p < 0.01:
            text = "**"
        if p < 0.001:
            text = "***"
        if p >= 0.05:
            text = f"{p:.2e}"

        x1, x2 = 3, 3 + 2*bar_width
        y, h, col = (kls_mean + kls_sem)[:,-1].max()+0.5, 0.1, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col)
        ax.set_ylim(bottom=0)

    if plot:
        plt.show()

def berkes_2011(ax=None):
    
    kls_bar = np.array([[497.1496437,490.4988124,462.787015],
        [176.8012668,207.2842439,203.9588282],
        [99.76247031,71.49643705,273.2383215],
        [76.48456057,127.4742676,208.3927158]])
    kls_sem_bar = np.array([[606.3341251,560.8867775,467.7751386],
        [196.7537609,220.5859066,220.5859066],
        [126.9200317,78.70150435,275.4552652],
        [85.35233571,158.5114806,247.189232]])

    kls_sem_bar -= kls_bar # data is given in coordinates, remove mean to find error bar

    # Set up the bar chart
    conditions = ['29-30', '44-45', '83-92', '129-151']
    types = ['natural stimuli', 'noise', 'gratings']
    colors = ['C0', 'C1', 'C2']
    index = np.arange(len(conditions))

    if ax is None:
        fig, ax = plt.subplots()
        plot=True
    else:
        plot=False

    bar_width = 0.2
    
    for i, type_label in enumerate(types):
        ax.bar(index + i * bar_width, kls_bar[:, i], bar_width, label=type_label, color=colors[i], yerr=[np.zeros_like(kls_sem_bar[:, i])+50,kls_sem_bar[:, i]], zorder=2, error_kw=dict(capsize = 4,zorder=1))

    ax.set_xlabel('postnatal age (days)')
    ax.set_ylabel('KL divergence')
    # ax.set_title('Bar Chart with SEM Error Bars')
    ax.set_xticks(index + (bar_width * (len(types) - 1)) / 2)
    ax.set_xticklabels(conditions)
    ax.legend()

    x1, x2 = 3, 3 + bar_width
    y, h, col = (kls_bar)[-1,:].max(), 20, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    x1, x2 = 3, 3 + 2*bar_width
    y, h, col = (kls_bar+ kls_sem_bar)[-1,:].max()+20, 20, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

    if plot:
        plt.show()

    return ax

def variability_stimulus_onset_nonlinear(axs, path_models):
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        print("Using GPU")
        
    seed = 0
    config = {
        "model_type": "mcpc",
        "dataset": "mnist",
        "model_name": "mcpc_fid_"+str(seed),
        "loss_fn": bernoulli_fn ,
        "EPOCHS":10,
        "batch_size_train":256,
        "batch_size_val": 256,
        "batch_size_test": 256,
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "input_var":0.3,
        "activation_fn": 'relu',
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.7},
        "mixing":0,
        "sampling":8000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.05},
        "optimizer_p_fn_mcpc": optim.Adam,
        "optimizer_p_kwargs_mcpc": {"lr": 0.01, "weight_decay":0.}
    }

    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config)

    # get model
    gen_pc = get_model(config, use_cuda)
    gen_pc.train()
    model_name = path_models + "/mcpc_fid_1"
    gen_pc.load_state_dict(torch.load(model_name), strict=False)

    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True, training=False)
    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    data,_ = list(test_loader)[0]
    pseudo_input = torch.zeros(data.shape[0],config["input_size"])
    if use_cuda:
        pseudo_input, data = pseudo_input.cuda(), data.cuda()

    # initialise sampling
    pc_trainer.train_on_batch(inputs=pseudo_input)
    # mc inference
    # without input
    mcpc_trainer.train_on_batch(inputs=pseudo_input, callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_return_results_every_t=False, is_return_representations=False, is_return_outputs=False)
    mc_results_noinput = mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=zero_fn, loss_fn_kwargs={},callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer}, is_sample_x_at_batch_start=False, is_return_results_every_t=True, is_return_xs=True)
    # with input
    mc_results_input = mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config["input_var"]},callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_return_results_every_t=True, is_return_xs=True)
    
    xs = mc_results_noinput["xs"] + mc_results_input["xs"]
    xs_cat = []
    for idx in range(len(xs[0])):
        xs_cat.append(torch.concatenate([xs[i][idx].unsqueeze(-1) for i in range(len(xs))], dim=2).transpose(0,2))
    xs = torch.concatenate(xs_cat,dim=1)
    xs = xs.view(xs.shape[0],-1)

    chunk_size = 1000
    tensorset = ChunkDataset(xs.T, chunk_size)
    data_loader = DataLoader(tensorset, batch_size=1, shuffle=True)
    means=[]
    stdss=[]
    weights=[]
    n = 1000
    for batch in tqdm(data_loader):
        batch.squeeze_()
        series = pd.DataFrame(batch.T)
        moving_std = series.rolling(window=n).std()
        mean = moving_std.mean(axis=1)
        stds = moving_std.std(axis=1)
        means.append(mean.values)
        stdss.append(stds.values)
        weights.append(batch.shape[0]-1)
    
    means_df = pd.DataFrame(means).values    
    stdss_df = pd.DataFrame(stdss).values    
    weights_df = pd.DataFrame(weights).values    

    mean = (means_df*(weights_df.reshape(-1,1)+1)).sum(0)/((weights_df+1).sum())
    stds = (stdss_df*(weights_df.reshape(-1,1))).sum(0)/((weights_df.reshape(-1,1)+1).sum()-1)
    sem = stds/np.sqrt(xs.shape[1])

    time = np.linspace(0, len(mean)*config["optimizer_x_kwargs_mcpc"]['lr'], len(mean)) - 203

    axs.plot(time, mean, "C0", linewidth=2, label="MCPC")
    axs.fill_between(time, mean+sem, mean-sem, interpolate=True, alpha=0.2, color='C0')
    axs.vlines(config["sampling"]*config["optimizer_x_kwargs_mcpc"]["lr"]-203, np.nanmin(mean-sem)-0.1 ,np.nanmax(mean+sem)+0.1, linestyles="dashed", colors="grey", linewidth=2.)
    axs.set_xlabel("time (AU)")
    axs.set_ylabel(r'variance of $x$')
    axs.set_xlim(0,600)
    axs.set_yticks([1,1.1])
    axs.set_ylim(np.nanmin(mean-sem)-0.02 ,np.nanmax(mean+sem)+0.02)
    axs.legend()

def variability_stimulus_onset_nonlinear_pc(axs, path_models):
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        print("Using GPU")
        
    seed = 0
    config = {
        "model_type": "mcpc",
        "dataset": "mnist",
        "model_name": "mcpc_fid_"+str(seed),
        "loss_fn": bernoulli_fn ,
        "EPOCHS":10,
        "batch_size_train":256,
        "batch_size_val": 256,
        "batch_size_test": 100,
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "input_var":0.3,
        "activation_fn": 'relu',
        "T_pc":8000,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.05},
        "optimizer_p_fn": optim.Adam,
        "optimizer_p_kwargs": {"lr": 0.01, "weight_decay":0.1},
    }

    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config)

    # get model
    gen_pc = get_model(config, use_cuda)
    gen_pc.train()
    model_name = path_models + "/mcpc_fid_1"
    gen_pc.load_state_dict(torch.load(model_name), strict=False)

    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=False, training=False)

    data,_ = list(test_loader)[0]
    pseudo_input = torch.zeros(data.shape[0],config["input_size"])
    if use_cuda:
        pseudo_input, data = pseudo_input.cuda(), data.cuda()

    # initialise sampling
    pc_trainer.train_on_batch(inputs=pseudo_input)
    # mc inference
    # without input
    pc_results_noinput = pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=zero_fn,loss_fn_kwargs={},is_sample_x_at_batch_start=False,is_return_results_every_t=True, is_return_xs=True)
    # with input
    pc_results_input = pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': data},is_sample_x_at_batch_start=False,is_return_results_every_t=True,  is_return_xs=True)
    
    xs = pc_results_noinput["xs"] + pc_results_input["xs"]
    xs_cat = []
    for idx in range(len(xs[0])):
        xs_cat.append(torch.concatenate([xs[i][idx].unsqueeze(-1) for i in range(len(xs))], dim=2).transpose(0,2))
    xs = torch.concatenate(xs_cat,dim=1)
    xs = xs.view(xs.shape[0],-1)

    chunk_size = 1000
    tensorset = ChunkDataset(xs.T, chunk_size)
    data_loader = DataLoader(tensorset, batch_size=1, shuffle=True)
    means=[]
    stdss=[]
    weights=[]
    n = 1000
    for batch in tqdm(data_loader):
        batch.squeeze_()
        series = pd.DataFrame(batch.T)
        moving_std = series.rolling(window=n).std()
        mean = moving_std.mean(axis=1)
        stds = moving_std.std(axis=1)
        means.append(mean.values)
        stdss.append(stds.values)
        weights.append(batch.shape[0]-1)
    
    means_df = pd.DataFrame(means).values    
    stdss_df = pd.DataFrame(stdss).values    
    weights_df = pd.DataFrame(weights).values    

    mean = (means_df*(weights_df.reshape(-1,1)+1)).sum(0)/((weights_df+1).sum())
    stds = (stdss_df*(weights_df.reshape(-1,1))).sum(0)/((weights_df.reshape(-1,1)+1).sum()-1)
    sem = stds/np.sqrt(xs.shape[1])

    time = np.linspace(0, len(mean)*config["optimizer_x_kwargs_pc"]['lr'], len(mean)) - 203

    axs.plot(time, mean, "r", linewidth=2, label="PC")
    axs.fill_between(time, mean+sem, mean-sem, interpolate=True, alpha=0.2, color='r')
    axs.vlines(config["T_pc"]*config["optimizer_x_kwargs_pc"]["lr"]-203, np.nanmin(mean-sem)-0.1 ,np.nanmax(mean+sem)+0.1, linestyles="dashed", colors="grey", linewidth=2.)
    axs.set_xlabel("time (AU)")
    axs.set_ylabel(r'variance of $x$')
    axs.set_xlim(0,600)
    axs.set_yticks([0,0.2])
    axs.set_ylim(np.nanmin(mean-sem) - 0.02 ,np.nanmax(mean+sem) + 0.01)
    axs.legend()
    
def churchland_2010(axs=None):
    if axs is None:
        fig, axs = plt.subplots()
        plot=True
    else:
        plot=False
    mean_x = [-335.8884668,-320.1656688,-310.7920771,-301.4184854,-292.0448937,-282.671302,-273.2977103,-263.9241185,-254.5505268,-245.1769351,-235.8033434,-226.4297517,-217.05616,-207.6825683,-198.3089766,-188.9353849,-179.5617932,-170.1882015,-160.8146098,-153.0032833,-146.7542222,-140.5051611,-134.2560999,-126.4447735,-118.6334471,-112.905086,-109.8066276,-107.0987148,-104.5209565,-101.2662715,-98.32399837,-95.5639826,-93.16848164,-91.16365022,-88.62649655,-88.16927402,-85.64361869,-82.20667635,-78.66552794,-70.20322325,-60.82963154,-51.45603983,-42.08244812,-32.70885641,-23.33526471,-13.961673,-4.588081289,3.223245134,11.03457156,20.40816327,29.78175497,39.15534668,48.52893839,57.9025301,67.27612181,76.64971351,86.02330522,95.39689693,104.7704886,114.1440803,123.5176721,132.8912638,142.2648555,150.0761819,157.8875083,167.2611,176.6346917,186.0082834,195.3818751,204.7554669,214.1290586,223.5026503,232.876242,242.2498337,251.6234254,260.9970171,270.3706088,279.7442005,289.1177922,298.4913839,307.8649756,317.2385674,326.6121591,335.9857508,345.3593425,354.7329342,364.1065259,373.4801176,382.8537093,392.227301,401.6008927,410.9744844,420.3480761,429.7216678,439.0952596,448.4688513,457.842443,467.2160347,476.5896264,485.9632181,495.3368098,503.1481362]
    mean = [14.67470499,14.69391284,14.77896951,14.81299217,14.81299217,14.81299217,14.81299217,14.8810375,14.91506017,14.8810375,14.77896951,14.77896951,14.74494684,14.67690151,14.59184484,14.33667485,14.09851618,13.84334619,13.75828952,13.97093118,14.27713518,14.63437317,14.94057717,15.22126417,15.29781517,14.96609417,14.55782218,14.04748218,13.61369319,12.99277953,12.44841687,11.94488141,11.56382755,11.25762355,10.90400091,10.90889123,10.49551583,10.04131324,9.79464666,9.743614908,10.01579624,10.27096623,10.37303423,10.39004557,10.42406823,10.27096623,10.01579624,9.760626241,9.403388246,9.216263582,9.250286248,9.335342914,9.284308915,9.199252249,9.046150251,8.842014254,8.586844258,8.365696927,8.229606262,8.212594929,8.331674261,8.484776259,8.637878257,8.816497254,9.09718425,9.301320248,9.437410912,9.454422246,9.403388246,9.301320248,9.14821825,8.995116252,8.87603692,8.671900923,8.399719593,8.246617596,8.229606262,8.229606262,8.229606262,8.297651595,8.365696927,8.399719593,8.43374226,8.399719593,8.331674261,8.348685594,8.586844258,8.961093586,9.080172917,9.14821825,9.09718425,8.995116252,8.842014254,8.637878257,8.467764926,8.263628929,8.212594929,8.195583596,8.246617596,8.331674261,8.280640262,8.229606262]

    sem_pos_x=[-334.9027765,-315.4788729,-299.8562201,-287.3580978,-271.735445,-259.2373227,-246.7392004,-234.2410781,-221.7429559,-209.2448336,-196.7467113,-184.248589,-171.7504667,-159.2523445,-146.7542222,-132.6938346,-121.7579776,-113.9466512,-110.1972971,-108.7563506,-106.0312839,-103.0890107,-100.4591906,-97.82937048,-94.70483991,-90.66902233,-87.41429611,-83.30650872,-77.99488933,-70.85420152,-58.9549132,-46.76924398,-34.2711217,-21.77299942,-9.274877143,0.098714565,14.78400824,28.21948969,40.71761197,53.73648934,68.83838709,89.14783579,101.6459581,113.8614774,125.8904659,135.003336,147.5396468,160.4912838,172.885255,184.4460182,196.9441404,209.9630178,220.3781197,232.876242,245.9992704,259.4347518,271.9328741,287.0060095,306.3027104,315.4381643,325.0498938,337.027261,350.0461383,365.9812442,373.760533,382.8537093,396.3933418,409.4122191,422.4310965,434.4084637,453.6764022,465.1330143,476.5537079,486.0310928,493.5661324,504.8665042]
    sem_pos=[15.37436616,15.45091716,15.57850216,15.57850216,15.52746816,15.62953616,15.62953616,15.50195116,15.47643416,15.37436616,15.24678117,14.91506017,14.53230518,14.43023718,15.00969739,15.73160416,16.03780816,15.68057016,15.14471317,14.68030377,14.26182498,13.73107139,13.28197219,12.781839,12.22046501,11.64888422,11.17681972,10.75578923,10.37303423,10.25990977,10.57929727,10.85785723,10.88337423,10.88337423,10.62820423,10.32200023,9.721357949,9.684075242,9.760626241,9.692580909,9.454422246,8.944082253,8.620866924,8.58204161,8.736838897,8.915451151,9.132562215,9.568828134,9.8607127,9.964762239,9.86269424,9.658558243,9.488444912,9.318331581,8.926940063,8.625119757,8.637878257,8.641579028,8.707839701,8.820191282,8.842014254,8.807991588,8.739946255,9.144900105,9.427472946,9.556490244,9.624535577,9.505456245,9.250286248,9.046150251,8.671900923,8.637878257,8.653486511,8.760653864,8.676153756,8.637878257]

    sem_neg_x=[-335.1473923,-326.4147299,-313.9166077,-304.5430159,-295.1694242,-285.7958325,-277.9845061,-265.4863838,-255.3287982,-245.1769351,-232.6788129,-223.3052211,-215.4938947,-201.4335072,-192.0599154,-182.6863237,-173.312732,-162.376875,-154.6485261,-148.3164875,-140.5051611,-134.2560999,-129.5693041,-120.2958576,-113.9446208,-110.4598329,-107.1246631,-105.3564193,-103.271103,-101.6069826,-99.65183441,-97.9053501,-95.82429134,-94.07976866,-92.69976077,-90.77298068,-86.98469924,-81.13908024,-76.45228438,-65.51642739,-59.41043084,-48.33150926,-38.95791755,-24.89752999,-13.961673,-4.588081289,2.467864117,11.84145583,23.53269383,34.46855083,46.9666731,61.02706067,73.52518294,82.0861678,92.81028921,104.7704886,114.1440803,129.7667332,139.1403249,148.5139166,157.8875083,170.3856306,179.7592223,189.132814,198.5064057,207.8799974,218.8158544,226.6271808,239.1253031,248.9795918,257.8724865,267.2460782,282.8687311,289.1177922,300.0536492,311.5646259,323.4876285,331.2989549,345.3593425,353.1706689,360.9819953,369.6145125,379.7291787,387.5405052,400.0386274,409.4122191,420.3480761,426.5971373,437.6417234,450.0311165,463.0385488,471.9028305,481.2764222,495.3368098,501.5858709]
    sem_neg=[14,13.94541418,14.04748218,14.04748218,14.04748218,14.04748218,14.04748218,14.14955018,14.17777778,14.14955018,14.04748218,14.04748218,14.04748218,13.94541418,13.74127819,13.46909686,13.26496086,13.1288702,13.28888889,13.53714219,13.94541418,14.20058418,14.45575418,14.61708143,14.28989368,13.94541418,13.47590139,13.16289286,12.85668887,12.4552214,12.08096668,11.69991821,11.39371422,11.08751022,10.80682323,10.47510223,10.01579624,9.556490244,9.250286248,9.301320248,9.614814815,9.811660241,9.86269424,9.964762239,9.760626241,9.556490244,9.248650543,8.932585582,8.739946255,8.842014254,8.842014254,8.739946255,8.484776259,8.251851852,8.059492931,7.821334268,7.821334268,8.076504265,8.178572263,8.331674261,8.637878257,8.893048253,8.944082253,8.944082253,8.893048253,8.790980255,8.637878257,8.535810258,8.331674261,8.074074074,7.923402267,7.821334268,7.821334268,7.821334268,7.923402267,8.014814815,8.025470265,8.025470265,7.923402267,7.923402267,8.025470265,8.37037037,8.586844258,8.637878257,8.637878257,8.535810258,8.38270826,8.229606262,8.074074074,7.821334268,7.837037037,7.821334268,7.923402267,7.872368267,7.821334268]

    onset = -139

    mean_x, mean, sem_pos_x, sem_pos, sem_neg_x, sem_neg = np.array(mean_x), np.array(mean), np.array(sem_pos_x), np.array(sem_pos), np.array(sem_neg_x), np.array(sem_neg)

    time_min = min([mean_x.min(), sem_neg_x.min(), sem_pos_x.min()])
    mean_x, sem_pos_x, sem_neg_x, onset =mean_x - time_min, sem_pos_x- time_min, sem_neg_x- time_min, onset- time_min 

    # fill between sem
    verts = [(sem_pos_x[i], sem_pos[i]) for i in range(len(sem_pos_x))] + \
            [(sem_neg_x[i], sem_neg[i]) for i in range(len(sem_neg_x)-1, -1, -1)]
    poly = Polygon(verts, facecolor='k', alpha=0.2)

    # Add the polygon to the plot
    axs.plot(mean_x, mean, "k", linewidth=2, label="membrane potential")
    axs.add_patch(poly)
    axs.vlines(onset, 7 ,17, linestyles="dashed", colors="grey", linewidth=2., label="stimulus onset")
    axs.set_xlabel("time (ms)")
    axs.set_ylabel(r'variance of $V_m$ $(mV^2)$')
    axs.set_xlim(0, 600)
    axs.set_ylim(7.65, 16.47)
    axs.legend()

    if plot==True:
        plt.show()
    
def similarity_increase(path_models, path_figures):
    setup_fig()
    f, axs = plt.subplots(2, 1, figsize=(5.8,4.8))
    berkes_2011(axs[0])
    similarity_increase_digit(axs[1], path_models=path_models)
    plt.tight_layout()
    plt.savefig(path_figures + "//5b.svg")
    plt.show()
 
def variability_quenching(path_models, path_figures):
    setup_fig()
    f, axs = plt.subplots(3, 1, figsize=(5.8,4.8))
    churchland_2010(axs[0])
    variability_stimulus_onset_nonlinear(axs[1], path_models)
    variability_stimulus_onset_nonlinear_pc(axs[2], path_models)
    plt.tight_layout()
    plt.savefig(path_figures + "//5a.svg")
    plt.show()


if __name__ == "__main__":
    pwd = os.getcwd()
    path_models = pwd + "//models"
    path_figures = pwd + "//figures"


    # variability_quenching(path_models, path_figures)
    similarity_increase(path_models, path_figures)