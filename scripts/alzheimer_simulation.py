import numpy as np
import symengine as sym
from math import pi
from alzheimer_functions import plot_spreading, spectral_properties, plot_spectral_properties, delay_matrix, compile_hopf, solve_dde, alzheimer
import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# Here we simulate Alzheimers' disease
# and plot the results
# -------------------------------------
# file paths, where to save dynamics (oscillations) and spreading (heterodimer model) solutions
file_name = 'alzheimers_default'
dyn_save_path = './' + file_name +  '_neural.p'
spread_save_path = './'+ file_name + '_spread.p'
solve = True  # whether to run and save simulation (True), or load simulation (False)

# dynamical (oscillator) settings
dyn_step = 1/1250; dyn_atol = 10**-6; dyn_rtol = 10**-4
dyn_tspan = (0,11); t_stamps = np.linspace(0,35,11)
dyn_cutoff = 1  # time to cutoff for analysis
trials = 10  # number of repetitions
bands = [[8,12]]  # frequency bands to analyze 
delay_dim = 40  # discretization dimension of delay matrix

# spread (heterodimer) settings
tau_nodes = [26, 67]  # left and right entorhinal cortex, toxic initialization
beta_nodes = [0, 41, 3, 44, 13, 54, 14, 55, 19, 60, 33, 74]  # Mattson et. al (2019) stage I, toxic initialization
seed_amount = 0.01; spread_atol = 10**-6; spread_rtol = 10**-4
spread_tspan = (0,35); spread_y0 = False  # False gives default setting

# plot settings
lobe_names = ['frontal', 'parietal', 'occipital', 'temporal', 'limbic', 'basal-ganglia', 'brain-stem']
lobe_file = '../data/LobeIndex.csv'; xlimit = t_stamps[-1]; plt.style.use('seaborn-muted')
colours = sns.color_palette('hls', len(lobe_names)+4) 
wiggle = 0.1  # wiggles trials on same x tick

# Read Budapest coupling matrix
with open('../data/CouplingMatrixInTime-000.csv', 'r') as f:
    W = list(csv.reader(f, delimiter=','))
W = np.array(W, dtype=float)
W = np.maximum( W, W.transpose() )
N = np.shape(W)[0]  # number of nodes

# HOPF PARAMETERS
w = [sym.var(f'wf_{n}') for n in range(N)]  # natural frequency
a = [sym.var(f'a_{n}') for n in range(N)]  # excitatory strength
b = [sym.var(f'b_{n}') for n in range(N)]  # inhibitory strengh
decay = -0.01; kappa = 1; h = 5; transmission_speed = 130.0
control_pars = [*a, *b, *w]  # parameters to be overwriteable in C++ compilation 

freqss = np.random.normal(10,1, size=(N,trials))  # samples of frequencies
freqss *= 2*pi

# SPREADING PARAMETERS
rho = 1 * 10**(-3)
# AB
a0 = 1  * 2; ai = 1  * 2; aii = 1 * 2; api = 0.75 * 2
# tau
b0 = 1 * 2; bi = 1 * 2; bii = 1 * 2; biii = 6 * 2; bpi = 1.33 * 2
    # concentration-to-damage
k1 = 1; k2 = 1; k3 = 0; gamma = 0.0
# damage-to-NNM
c1 = 0.8 ; c2 = 1.8; c3 = 0.4 
# NNM variable parameters 
a_init = 1; b_init = 1; a_min = 0.05; a_max = 1.95; b_min = 0.05
delta = False

# define brain regions (LobeIndex_I.txt)
regions = [[] for _ in range(len(lobe_names))]
with open('../data/LobeIndex_I.txt') as f:
    node = 0
    for line in f:
        lobe = int(float(line.strip()))-1
        regions[lobe].append(node)
        node += 1

# create delay matrix per Bick & Goriely
distances = []
with open('../data/LengthFibers33.csv', 'r') as file:
    reader = csv.reader(file)
    node_i = 0
    for row in reader:
        node_j = 0
        for col in row:
            if float(col) > 0:
                distances.append((node_i,node_j, float(col)/10))
            node_j += 1
        node_i += 1
delays = delay_matrix(distances, transmission_speed, N, discretize=delay_dim)

# SOLVE
if solve:
    # compile hopf model
    print('\nCompiling...')
    DE = compile_hopf(N, a=a, b=b, delays=delays, t_span=dyn_tspan, \
                 kappa=kappa, w=w, decay=decay, random_init=True, \
                 h=h, control_pars=control_pars)
    print('Done.')

    # randomize initial values
    dyn_y0 = np.zeros((trials, 2*N))
    theta0 = np.random.uniform(0, 2*3.14, (trials,N))
    R0 = np.random.uniform(0,1,(trials,N))
    dyn_y0[:,::2] = R0 * np.cos(theta0) 
    dyn_y0[:,1::2] = R0 * np.sin(theta0) 

    # simulate alzheimer's progression
    print('\nSolving alzheimer model...')
    spread_sol, dyn_sols =  alzheimer(W, DE, dyn_y0, tau_seed=tau_nodes, beta_seed=beta_nodes, \
             seed_amount=seed_amount, trials=trials, \
            t_spread=t_stamps, spread_tspan=spread_tspan, \
            spread_y0=spread_y0, a0=a0, ai=ai, api=api, aii=aii, b0=b0, bi=bi, \
            bii=bii, biii=biii, gamma=gamma, \
            delta=delta, bpi=bpi, c1=c1, c2=c2, c3=c3, k1=k1, k2=k2, k3=k3, \
            rho=rho, a_min=a_min, a_max=a_max, b_min=b_min, a_init=a_init, b_init=b_init, \
            freqss=freqss, method='RK45', \
            spread_atol=spread_atol, spread_rtol=spread_rtol, dyn_atol=dyn_atol, dyn_rtol=dyn_rtol, \
            dyn_step=dyn_step, dyn_tspan=dyn_tspan, dyn_cutoff=dyn_cutoff, display=True, \
            )
    print('Done.')

    # SAVE SOLUTIONS
    # dump
    print('\nSaving solutions...')
    pickle.dump( dyn_sols, open( dyn_save_path, "wb" ) )
    pickle.dump( spread_sol, open( spread_save_path, "wb" ) )
    print('Done.')

# LOAD SOLUTIONS
# dump
print('\nLoading solutions...')
dyn_sols = pickle.load( open( dyn_save_path, "rb" ) )
spread_sol = pickle.load( open( spread_save_path, "rb" ) )
print('Done.')

# PLOT
# plot spreading
t_spread = spread_sol['disc_t']
print('\nPlotting...')
figs, axs = plot_spreading(spread_sol, colours, lobe_names, xlimit=xlimit, regions=regions, averages=True)

# analyze dynamics and plot 
bandpowers, freq_peaks = spectral_properties(dyn_sols, bands, 0, freq_tol=0, relative=False)

sns.set_context(font_scale=2, rc={"axes.labelsize":18,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":8})   
figs_PSD, figs_peaks = plot_spectral_properties(t_spread, bandpowers, freq_peaks, bands, wiggle, '', lobe_names[:-1], colours, regions=regions[:-1], only_average=False, n_ticks=6)
print('Done.')

# save figues
figs[0].savefig('../plots/ab_damage.pdf', dpi=300)
figs[1].savefig('../plots/protein_damage.pdf', dpi=300)
figs[2].savefig('../plots/toxic_concentration.pdf', dpi=300)
figs[3].savefig('../plots/weight_damage.pdf', dpi=300)
figs[4].savefig('../plots/healthy_concentration.pdf', dpi=300)
figs_PSD[0].savefig('../plots/oscillatory_power.pdf', dpi=300)
figs_peaks[0].savefig('../plots/oscillatory_frequency.pdf', dpi=300)

# show figures
plt.show()

# WE'RE DONE
