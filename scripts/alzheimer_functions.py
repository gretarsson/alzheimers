from scipy.integrate import solve_ivp, simps, trapz
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft, rfft, rfftfreq
from scipy.interpolate import interp1d 
from scipy.integrate import solve_ivp 
from scipy.spatial.distance import hamming 
from scipy.stats import pearsonr
from math import e, cos, sin, pi, ceil, log10, floor
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pprint import pprint
import random
import seaborn as sns
import numpy as np
import copy
from numba import jit
import symengine as sym
import time as timer

# -----------------------------------------
# helper functions to simulate Alzheimers!
# ----------------------------------------

# -----------------------------------------
# compile hopf normal form model into
# C++ wrapper
# INPUT:
# Hopf normal form parameters (parameters to be changed must be symengine variables)
# control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
# DDE - JiTCDDE object
# y0 - numpy array (initial conditions)
# -----------------------------------------
def compile_hopf(N, a=False, b=False, delays=False, t_span=(0,10), \
             kappa=10, h=1, w=False, decay=-0.01, inter_idx=[], inter_c=1,  \
             random_init=True, delay_c=1, max_delay=None, decay0=0, decay1=1, \
             only_a=False, control_pars=()):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t

    # set default parameter values
    if delays is False:
        delays = np.zeros((N,N))
    if not a:
        a = 1
    if not b:
        b = 1

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # interhemispheric coupling matrix (scales interhemispheric coupling by inter_c)
    inter_mat = [ [1 for _ in range(N)] for _ in range(N) ]
    for e1, e2 in inter_idx:
        inter_mat[e1][e2] = inter_c

    # if a or b not list then make list (list necessary for symengine variables)
    if not isinstance(a,list):
        a_val = a
        a = [a_val for _ in range(N)]
    if not isinstance(b,list):
        b_val = b
        b = [b_val for _ in range(N)]
    if not isinstance(decay,list):
        decay_val = decay
        decay = [decay_val for _ in range(N)]
    if not isinstance(h,list):
        h_val = h
        h = [h_val for _ in range(N)]

    # TEST DISCARDING B SEMIAXIS
    if only_a:
        b = a

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # define input to node
            afferent_input = kappa * sum( inter_mat[j][k] * W[j][k] * y(2*j+0, t-delay_c*delays[j,k]) for j in range(N) )

            # transform decays
            decay[k] = decay1*(decay[k]-decay0)

            # dynamics of node k
            yield decay[k]*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) \
                     - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) \
                         + h[k] * sym.tanh(afferent_input)
            yield decay[k]*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0)  \
                     - y(2*k+1)*(y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)

    # set up initial conditions
    if random_init:
        theta0 = np.random.uniform(0, 2*3.14, N)
        R0 = np.random.uniform(0,1,N)
    else:
        R0 = np.full((N),1)
        theta0 = np.full((N),0)
    y0 = np.zeros((2*N)) 
    y0[::2] = R0 * np.cos(theta0)
    y0[1::2] = R0 * np.sin(theta0)
    
    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    DDE = jitcdde(neural_mass, n=2*N, control_pars=control_pars, max_delay=max_delay)  
    DDE.compile_C(do_cse=True, chunk_size=int(N*2))  # after vacation this is suddenly slow

    # add number of nodes and initial conditions to DDE object
    DDE.N = N
    DDE.y0 = y0

    return DDE


# ----------------------------------------------------------------
# solve DDE 
# INPUT:
# DDE - a jitcdde object 
# y0 - numpy array (initial conditions)
# parameterss -  numpy array shape: (#runs, #parameters)
# -> each row is a parameter setting with a parameter in each
# column
# OUTPUT:
#   sols: (#runs) array with solutions stored as dictionaries
# ----------------------------------------------------------------
def solve_dde(DDE, y0, W, t_span=(0,10), step=10**-4, atol=10**-6, rtol=10**-4, parameterss=False, display=False, discard_y=False, cutoff=0):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t

    # check if parameter array given
    if parameterss is False:
        parameterss = np.array([[]])
        parN, num_par = (1, 0)
    else:
        parameterss = np.array( parameterss )
        parN, num_par = parameterss.shape

    # initialize 
    sols = np.empty((parN), dtype='object')

    # set number of nodes and flatten values of adjacency matrix
    N = DDE.N
    flat_num_W = list(W.flatten())

    # set integration parameters
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    #DDE.set_integration_parameters(rtol=1e12,atol=1e12, first_step=10**-4, max_step=10**-4, min_step=10**-4)  # test fixed step size

    # start clock
    if display:
        start = timer.time()

    # loop over parameters
    for i in range(parN):
        # add numeric adj. matrix and add model parameters
        parameters = [*flat_num_W, *parameterss[i,:]]

        # set past history
        DDE.constant_past(y0, time=0.0)

        # set model parameters (only if set by user)
        try:
            DDE.set_parameters(parameters)
        except:
            print(f'\nThe number of implicit parameters is {num_par}. Make sure that this is reflected in the JiTCDDE compilation.\n')
            return None, None

        # handle initial discontinuities
        DDE.adjust_diff()
        #DDE.step_on_discontinuities(propagations=1)
        #DDE.integrate_blindly(0.01, step=step)

        # solve
        data = []
        t = []
        for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
            data.append( DDE.integrate(time) )
            t.append(time)

        # organize data
        data = np.array(data)
        data = np.transpose(data)
        t = np.array(t)

        # store solution as dictionary, potentially discard y and cut off transients
        sol = {}
        sol['x'] = data[0:2*N:2,t>cutoff]
        if discard_y:
            sol['y'] = []
        else:
            sol['y'] = data[1:2*N:2,t>cutoff]
        sol['t'] = t[t>cutoff]

        # purge past history
        DDE.purge_past()

        # store solution in grid array
        sols[i] = sol

    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for all DDE simulations: {end-start} seconds\nElapsed time per DDE simulation: {round((end-start)/parN,4)}')
    
    # we're done
    return sols 


# ----------------------------------------------------------------
# simulate multi-timescale alzheimer's model 
# INPUT:
#   W0 - numpy array (N,N), initial adjacency matrix
#   DE - a JiTC*DE object, has to be compiled with
#           2*N implicit parameters
#   dyn_y0 - numpy array (#trials, #variables), initial values for DE
#   optional arguments are spreading parameters and
#       integration parameters
# OUTPUT:
#   spread_sol: dictionary, solutions of spreading model
#   dyn_sols: array of dictionaries, solutions of dynamical model
#               at different time points
# ----------------------------------------------------------------
def alzheimer(W0, DE, dyn_y0, tau_seed=False, beta_seed=False, seed_amount=0.1, t_spread=False, \
        spread_tspan=False, \
        spread_y0=False, a0=0.75, ai=1, api=1, aii=1, b0=1, bi=1, bii=1, biii=1, gamma=0, delta=0.95, \
        bpi=1, c1=1, c2=1, c3=1, k1=1, k2=1, k3=1, c_init=0, c_min=0,
        rho=10**(-3), a_min=False, a_max=False, b_min=False, a_init=1, b_init=1, \
        freqss=np.empty([1,1]), method='RK45', spread_max_step=0.125, as_dict=True, \
        spread_atol=10**-6, spread_rtol=10**-3, dyn_atol=10**-6, dyn_rtol=10**-4, \
        dyn_step=1/100, dyn_tspan=(0,10), display=False, trials=1, SDE=False,  \
        normalize_row=False, dyn_cutoff=0, feedback=False, kf=1, bii_max=2, adaptive=False):
    # imports
    from math import e

    # set t_spread if not provided, and add end points if not inluded by user
    if t_spread.size == 0:
        t_spread = [0,spread_tspan[-1]]
    else:
        if 0 not in t_spread:
            t_spread = [0] + t_spread
    Ts_final = t_spread[-1]

    # initialize dynamical solutions
    #dyn_sols = np.empty((len(t_spread)), dtype='object')
    dyn_sols = []  

    # if only one initial condition given, repeat it for all trials
    if len(dyn_y0.shape) == 1:
        n_vars = dyn_y0.shape[0]
        new_dyn_y0 = np.empty((trials,n_vars))
        for l in range(trials):
            new_dyn_y0[l,:] = dyn_y0
        dyn_y0 = new_dyn_y0

    # construct laplacian, list of edges, and list of neighours
    N = W0.shape[0]  
    M = 0
    edges = []
    neighbours = [[] for _ in range(N)]
    w0 = []
    for i in range(N):
        for j in range(i+1, N):
            if W0[i,j] != 0:
                M += 1
                edges.append((i,j))
                neighbours[i].append(j)
                neighbours[j].append(i)
                w0.append(W0[i,j])

    # construct spreading initial values, spread_y0
    if not spread_y0:
        u = np.array([a0/ai for _ in range(N)])
        up = np.array([0 for _ in range(N)])
        v = np.array([b0/bi for _ in range(N)])
        vp = np.array([0 for _ in range(N)])
        qu = np.array([0 for _ in range(N)])
        qv = np.array([0 for _ in range(N)])
        a = np.array([a_init for _ in range(N)])
        b = np.array([b_init for _ in range(N)])
        c = np.array([c_init for _ in range(N)])
        spread_y0 = [*u, *up, *v, *vp, *qu, *qv, *a, *b, *c, *w0]

    # seed tau and beta
    if beta_seed:
        for index in beta_seed:
            beta_index = N+index
            if seed_amount:
                spread_y0[beta_index] = seed_amount
            else:
                spread_y0[beta_index] = (10**(-2)/len(beta_seed))*a0/ai
    if tau_seed:
        for index in tau_seed:
            tau_index = 3*N+index 
            if seed_amount:
                spread_y0[tau_index] = seed_amount
            else:
                spread_y0[tau_index] = (10**(-2)/len(tau_seed))*b0/bi

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # make pf a list (necessary, in case of feedback)
    pf = np.ones((N))

    # initialize spreading solution
    t0 = t_spread[0]
    empty_array = np.array([[] for _ in range(N)])
    empty_arraym = np.array([[] for _ in range(M)])
    spread_sol = {'t': np.array([]), 'u':empty_array, 'up':empty_array, 'v':empty_array, \
                     'vp':empty_array, 'qu':empty_array, 'qv':empty_array, 'a':empty_array, \
                     'b':empty_array, 'c':empty_array, 'w':empty_arraym, 'w_map': edges, \
                     'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)], \
                     'pf':np.transpose(np.array([pf])), 'disc_t':[0]}

    # spreading dynamics
    def rhs(t, y):
        # set up variables as lists indexed by node k
        u = np.array([y[i] for i in range(N)])
        up = np.array([y[i] for i in range(N, 2*N)])
        v = np.array([y[i] for i in range(2*N, 3*N)])
        vp = np.array([y[i] for i in range(3*N, 4*N)])
        qu = np.array([y[i] for i in range(4*N, 5*N)])
        qv = np.array([y[i] for i in range(5*N, 6*N)])
        a = np.array([y[i] for i in range(6*N, 7*N)])
        b = np.array([y[i] for i in range(7*N, 8*N)])
        c = np.array([y[i] for i in range(8*N, 9*N)])

        # update laplacian from m weights
        w = np.array([y[i] for i in range(9*N, 9*N+M)])
        L = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            # set (n,m) in l
            L[n,m] = -w[i]
            L[m,n] = L[n,m]
            # update (n,n) and (m,m) in l
            L[n,n] += w[i]
            L[m,m] += w[i]

        # check if l is defined correctly
        for i in range(N):
            if abs(sum(L[i,:])) > 10**-10:
                print('L is ill-defined')
                print(sum(L[i,:]))
    
        # scale Laplacian by diffusion constant
        L = rho*L
        
        # nodal dynamics
        du, dup, dv, dvp, dqu, dqv, da, db, dc = [[] for _ in range(9)]
        for k in range(N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k,l]*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
            dupk = sum([-L[k,l]*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]
            dvk = pf[k]*sum([-L[k,l]*v[l] for l in neighbours_k]) + b0 - bi*v[k] \
                     - bii*v[k]*vp[k] - biii*up[k]*v[k]*vp[k]
            dvpk = pf[k]*sum([-L[k,l]*vp[l] for l in neighbours_k]) - bpi*vp[k] \
                     + bii*v[k]*vp[k] + biii*up[k]*v[k]*vp[k]
            ## append
            du.append(duk)
            dup.append(dupk)
            dv.append(dvk)
            dvp.append(dvpk)

            # damage dynamics
            dquk = k1*up[k]*(1-qu[k])
            dqvk = k2*vp[k]*(1-qv[k]) + k3*up[k]*vp[k]
            ## append
            dqu.append(dquk)
            dqv.append(dqvk)

            # excitatory-inhibitory dynamics
            dak = c1*qu[k]*(a_max-a[k])*(a[k]-a_min) - c2*qv[k]*(a[k]-a_min)
            dbk = -c3*qu[k]*(b[k]-b_min)
            dck = -c3*qu[k]*(c[k]-c_min)
            ## append
            da.append(dak)
            db.append(dbk)
            dc.append(dck)

        # connecctivity dynamics
        dw = []
        for i in range(M):
            # extract edge
            n, m = edges[i]
            
            # axonopathy dynamcs
            dwi = -gamma*w[i]*(qv[n] + qv[m])
            ## append
            dw.append(dwi)

        # pack right-hand side
        rhs = [*du, *dup, *dv, *dvp, *dqu, *dqv, *da, *db, *dc, *dw]

        return rhs 

    # measure computational time
    if display:
        start = timer.time()

    # set initial dynamical model parameters
    W_t = W0

    # SOLVE MULTI-SCALE MODEL FOR TIME>0
    t = 0;  i = 0 
    while t < Ts_final + 1:
        # SOLVE DYNAMICAL MODEL AT T0
        # initialize storage for trial simulations
        dyn_x = []
        dyn_y = []

        # solve dynamical model for each trial
        for l in range(trials):
            # set initial values
            dyn_y0_l = dyn_y0[l,:]
    
            # update dynamical parameters
            if freqss.size > 0:
                freqs = freqss[:,l] 
                dyn_pars = [[*a, *b, *freqs]]
            else:
                dyn_pars = [[*a, *b]]

            # if told, normalize adj. matrix
            if normalize_row:
                for n in range(N):
                    W_t[n,:] = W_t[n,:] / np.sum(W_t[n,:])

            # solve dynamical model at time 0
            print(f'\tSolving dynamical model at time {t0} (trial {l+1} of {trials}) ...')
            if SDE:
                dyn_sol = solve_sde(DE, dyn_y0_l, W_t, t_span=dyn_tspan, step=dyn_step,  \
                     atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
            else:
                dyn_sol = solve_dde(DE, dyn_y0_l, W_t, t_span=dyn_tspan, step=dyn_step, \
                     atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
            print('\tDone')

            # store each trial
            dyn_x_l = dyn_sol[0]['x'] 
            dyn_y_l = dyn_sol[0]['y'] 
            dyn_x.append(dyn_x_l)
            dyn_y.append(dyn_y_l)

        # store all trials in tuple and add to dyn_sols
        dyn_t = dyn_sol[0]['t']
        dyn_x = np.array( dyn_x )
        dyn_y = np.array( dyn_y )
        dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
        dyn_sols.append(dyn_sol_tup)
        #dyn_sols[i] = dyn_sol_tup  

        # SPREADING MODEL FROM T0 to T
        # if only one time-point, return the spreading initial conditions
        if len(t_spread) == 1:
            print('\tOnly one time point in spreading simulation')
            spread_sol['t'] = np.concatenate((spread_sol['t'], [0]))
            spread_sol['u'] = np.concatenate((spread_sol['u'], np.reshape(spread_y0[0:N], (N,1))), \
                                                 axis=1)
            spread_sol['up'] = np.concatenate((spread_sol['up'], np.reshape(spread_y0[N:2*N], (N,1))), \
                                                 axis=1)
            spread_sol['v'] = np.concatenate((spread_sol['v'], np.reshape(spread_y0[2*N:3*N], (N,1))), \
                                                 axis=1)
            spread_sol['vp'] = np.concatenate((spread_sol['vp'], np.reshape(spread_y0[3*N:4*N], \
                                                 (N,1))), axis=1)
            spread_sol['qu'] = np.concatenate((spread_sol['qu'], np.reshape(spread_y0[4*N:5*N], \
                                                (N,1))), axis=1)
            spread_sol['qv'] = np.concatenate((spread_sol['qv'], np.reshape(spread_y0[5*N:6*N], \
                                                (N,1))), axis=1)
            spread_sol['a'] = np.concatenate((spread_sol['a'], np.reshape(spread_y0[6*N:7*N], \
                                                (N,1))), axis=1)
            spread_sol['b'] = np.concatenate((spread_sol['b'], np.reshape(spread_y0[7*N:8*N], \
                                                (N,1))), axis=1)
            spread_sol['c'] = np.concatenate((spread_sol['c'], np.reshape(spread_y0[8*N:9*N], \
                                                (N,1))), axis=1)
            spread_sol['w'] = np.concatenate((spread_sol['w'], np.reshape(spread_y0[9*N:9*N+M], \
                                                (M,1))), axis=1)
        # end simulation at last time point
        if t >= Ts_final:
            break

        # set time interval to solve (if adaptive, analyze dynamics here)
        if feedback:
            mods = (dyn_x_l**2 + dyn_y_l**2)**(1/2) 
            avg_mod = np.mean(mods, axis=1) 
            if t0==0:
                mod0 = np.mean(avg_mod)
                pf_0 = pf - 1e-5
            if adaptive:
                eqs = kf*(-mod0+avg_mod-pf+pf_0)
                funcs = 1 / (kf*(mod0 + pf - pf_0))
                step_size = np.amin( funcs )
                t = t + step_size
                print(f'\t\tAdaptive step size = {step_size}')
        if not adaptive:
            t = t_spread[i+1]
        spread_tspan = (t0, t)

        # solve spreading from time t_(i-1) to t_(i)
        print(f'\n\tSolving spread model for {spread_tspan} ...')
        sol = solve_ivp(rhs, spread_tspan, spread_y0, method=method, \
                         max_step=spread_max_step, atol=spread_atol, rtol=spread_rtol)
        print('\tDone.')

        # append spreading solution
        spread_sol['t'] = np.concatenate((spread_sol['t'], sol.t))
        spread_sol['u'] = np.concatenate((spread_sol['u'], sol.y[0:N,:]), axis=1)
        spread_sol['up'] = np.concatenate((spread_sol['up'], sol.y[N:2*N,:]), axis=1)
        spread_sol['v'] = np.concatenate((spread_sol['v'], sol.y[2*N:3*N,:]), axis=1)
        spread_sol['vp'] = np.concatenate((spread_sol['vp'], sol.y[3*N:4*N,:]), axis=1)
        spread_sol['qu'] = np.concatenate((spread_sol['qu'], sol.y[4*N:5*N,:]), axis=1)
        spread_sol['qv'] = np.concatenate((spread_sol['qv'], sol.y[5*N:6*N,:]), axis=1)
        spread_sol['a'] = np.concatenate((spread_sol['a'], sol.y[6*N:7*N,:]), axis=1)
        spread_sol['b'] = np.concatenate((spread_sol['b'], sol.y[7*N:8*N,:]), axis=1)
        spread_sol['c'] = np.concatenate((spread_sol['c'], sol.y[8*N:9*N,:]), axis=1)
        spread_sol['w'] = np.concatenate((spread_sol['w'], sol.y[9*N:9*N+M,:]), axis=1)
        spread_sol['disc_t'].append(t)

        # extract the parameters for the dynamic model
        a = sol.y[6*N:7*N,-1]
        b = sol.y[7*N:8*N,-1]
        w = sol.y[9*N:9*N+M,-1]

        # construct adjacency matrix at time t
        W_t = np.zeros((N,N))
        for j in range(M):
            n, m = edges[j]
            weight = w[j]
            W_t[n,m] = weight
            W_t[m,n] = weight

        # append dynamic model parameters to rhythms list
        rhythms_i = (W_t, a, b, t)
        spread_sol['rhythms'].append(rhythms_i)
        
        # in the future, potential feedback changes from dynamics to spreading here
        if feedback:
            # update parameters
            t_res = t - t0
            # euler
            #pf = pf + t_res * kf * (pf - pf_0) * ((avg_mod - mod0) - (pf - pf_0))
            # RK4
            rk1 = kf * (pf - pf_0) * ((avg_mod - mod0) - (pf - pf_0))
            rk2 = kf * ((pf+t_res*rk1/2) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk1/2) - pf_0))
            rk3 = kf * ((pf+t_res*rk2/2) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk2/2) - pf_0))
            rk4 = kf * ((pf+t_res*rk3) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk3) - pf_0))
            pf = pf + 1/6 * (rk1 + 2*rk2 + 2*rk3 + rk4) * t_res
            print(pf)
            # append parameters
            pf_save = np.transpose(np.array([pf]))  # need correct np dimensions
            spread_sol['pf'] = np.concatenate((spread_sol['pf'], pf_save), axis=1)

        # update spreading initial values, spread_y0, and start of simulation, t0
        spread_y0 = sol.y[:,-1]
        t0 = t
        i += 1

    # display computational time
    if display:
        end = timer.time()
        print(f'\nElapsed time for alzheimer simulations: {end-start} seconds\nElapsed time per time step: {round((end-start)/len(t_spread),4)}')

    # done
    return spread_sol, dyn_sols



# ----------------------------------------
# Matrix of delays. Input for rhythms series
# ----------------------------------------

def delay_matrix(distances, transmission_speed, N, discretize=40):
    # distances should be a list of 3-tuples like (node1, node2, distance)
    delay_matrix = np.zeros((N,N))
    for n,m,distance in distances:
        delay_matrix[n,m] = distance/transmission_speed
        delay_matrix[m,n] = delay_matrix[n,m]

    if discretize:
        nonzero_inds = np.nonzero(delay_matrix)
        max_delay = np.amax(delay_matrix)
        n_delays = np.count_nonzero(delay_matrix)

        lower_bounds = np.arange(0, discretize) * max_delay/discretize
        upper_bounds = np.zeros((discretize))
        for i in range(discretize):
            upper_bounds[i] = (i+1) * max_delay/discretize
        for l in range(len(nonzero_inds[0])):
            i = nonzero_inds[0][l]
            j = nonzero_inds[1][l]
            for k in range(discretize):
                w_ij = delay_matrix[i,j]
                if w_ij <= upper_bounds[k]: 
                    w_ij = upper_bounds[k]  # round to upper (GorielyBick)
                    delay_matrix[i,j] = w_ij
                    delay_matrix[j,i] = w_ij
                    break

    return delay_matrix

# ---------------------------------------
# plot spreading
# Input
#   regions : list of lists, each list contains nodes to average over
# --------------------------------------

def plot_spreading(sol, colours, legends, xlimit=False, regions=[], averages=True, plot_c=False):
    # extract solution
    a = sol['a']
    b = sol['b']
    c = sol['c']
    qu = sol['qu']
    qv = sol['qv']
    u = sol['u']
    v = sol['v']
    up = sol['up']
    vp = sol['vp']
    w = sol['w']
    t = sol['t']

    # N of x-ticks
    nx = 5

    # find N
    N = a.shape[0]

    # if regions not given, plot all nodes
    if len(regions) == 0:
        regions = [[i] for i in range(N)]

    # plot 1-by-2 plot of all nodes'/regions a and b against time
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions tau and Abeta damage
    fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions toxic tau and Abeta concentration
    fig3, axs3 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of average weight
    fig4, axs4 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions healthy tau and Abeta concentration
    fig5, axs5 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plot settings
    if xlimit:
        plt.xlim((0, xlimit))
    axs[0].set_xlabel('$t_{spread}$')
    axs[0].set_ylabel('Excitatory semiaxis, $a$')
    axs[1].set_xlabel('$t_{spread}$')
    axs[1].set_ylabel('Inhibitory semiaxis, $b$')

    axs2[0].set_xlabel('$t_{spread}$')
    axs2[0].set_ylabel('Amyloid-$\\beta$ damage, $q^{(\\beta)}$')
    axs2[1].set_xlabel('$t_{spread}$')
    axs2[1].set_ylabel('Tau damage, $q^{(\\tau)}$')
    axs2[0].set_ylim([-0.1, 1.1])
    axs2[1].set_ylim([-0.1, 1.1])

    axs3[0].set_xlabel('$t_{spread}$')
    axs3[0].set_ylabel('Toxic amyloid-$\\beta$ concentration, $\\tilde{u}$')
    axs3[1].set_xlabel('$t_{spread}$')
    axs3[1].set_ylabel('Toxic tau concentration, $\\tilde{v}$')

    axs5[0].set_xlabel('$t_{spread}$')
    axs5[0].set_ylabel('Healthy amyloid-$\\beta$ concentration, $u$')
    axs5[1].set_xlabel('$t_{spread}$')
    axs5[1].set_ylabel('Healthy tau concentration, $v$')

    # plot a, b, damage and concentrations against time
    for r in range(len(regions)):
        # initialize
        region = regions[r]
        avg_region_a = []
        avg_region_b = []
        avg_region_c = []
        avg_region_qu = []
        avg_region_qv = []
        avg_region_up = []
        avg_region_vp = []
        avg_region_u = []
        avg_region_v = []

        # compute averages over regions
        for node in region:
            avg_region_a.append(a[node,:])
            avg_region_b.append(b[node,:])
            avg_region_c.append(c[node,:])
            avg_region_qu.append(qu[node,:])
            avg_region_qv.append(qv[node,:])
            avg_region_up.append(up[node,:])
            avg_region_vp.append(vp[node,:])
            avg_region_u.append(u[node,:])
            avg_region_v.append(v[node,:])

        # convert lists to arrays
        avg_region_a = np.array(avg_region_a)
        avg_region_b = np.array(avg_region_b)
        avg_region_c = np.array(avg_region_c)
        avg_region_qu = np.array(avg_region_qu)
        avg_region_qv = np.array(avg_region_qv)
        avg_region_up = np.array(avg_region_up)
        avg_region_vp = np.array(avg_region_vp)
        avg_region_u = np.array(avg_region_u)
        avg_region_v = np.array(avg_region_v)

        # plot a, b
        axs[0].plot(t, np.mean(avg_region_a, axis=0), c=colours[r], label=legends[r])
        axs[1].plot(t, np.mean(avg_region_b, axis=0), c=colours[r], label=legends[r])
        if plot_c:
            axs[1].plot(t, np.mean(avg_region_c, axis=0), c=colours[r], label=legends[r])

        # plot damage
        axs2[0].plot(t, np.mean(avg_region_qu, axis=0), c=colours[r], label=legends[r])
        axs2[1].plot(t, np.mean(avg_region_qv, axis=0), c=colours[r], label=legends[r])

        # plot concentration
        axs3[0].plot(t, np.mean(avg_region_up, axis=0), c=colours[r], label=legends[r])
        axs3[1].plot(t, np.mean(avg_region_vp, axis=0), c=colours[r], label=legends[r])

        axs5[0].plot(t, np.mean(avg_region_u, axis=0), c=colours[r], label=legends[r])
        axs5[1].plot(t, np.mean(avg_region_v, axis=0), c=colours[r], label=legends[r])

    # plot averages over all nodes
    if averages:
        # a and b
        axs[0].plot(t, np.mean(a, axis=0), c='black', label='average')
        axs[1].plot(t, np.mean(b, axis=0), c='black', label='average')
        if plot_c:
            axs[1].plot(t, np.mean(c, axis=0), c='black', label='average')

        # damage
        axs2[0].plot(t, np.mean(qu, axis=0), c='black', label='average')
        axs2[1].plot(t, np.mean(qv, axis=0), c='black', label='average')

        # toxic concentratio
        axs3[0].plot(t, np.mean(up, axis=0), c='black', label='average')
        axs3[1].plot(t, np.mean(vp, axis=0), c='black', label='average')

        # healthy concentration
        axs5[0].plot(t, np.mean(u, axis=0), c='black', label='average')
        axs5[1].plot(t, np.mean(v, axis=0), c='black', label='average')


    # plot average weights over time
    axs4.plot(t, np.mean(w, axis=0), c='black')
    axs4.set_ylabel('Average link weight')
    axs4.set_xlabel('$t_{spread}$')

    # show
    axs[1].legend(loc='best')
    axs3[0].legend(loc='best')
    plt.tight_layout()

    # we're done
    figs = (fig, fig2, fig3, fig4, fig5)   
    axss = (axs, axs2, axs3, axs4, axs5)
    return figs, axss


# -----------------------------------------------
# compute power-spectal properties
# -----------------------------------------------

def spectral_properties(solutions, bands, fourier_cutoff, modified=False, functional=False, db=False, freq_tol=0, relative=False, window_sec=None):
    #from mne.connectivity import spectral_connectivity

    # find size of rhythms
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions) 

    # find average power (and frequency peaks) in bands for each node over time stamps
    bandpowers = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]
    freq_peaks = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]

    # functional connectivity parameters and initializations
    if functional:
        functional_methods = ['coh', 'pli', 'plv']
        average_strengths = [[[[] for _ in range(len_rhythms)] for _ in range(len(functional_methods))] for _ in range(len(bands))]

    for b in range(len(bands)):
        for i in range((len_rhythms)):
            t, x, y = solutions[i]

            # iterate through trials
            for l in range(L):
                xl = x[l]

                # find last 10 seconds
                inds = [s for s in range(len(t)) if t[s]>fourier_cutoff]
                t = t[inds]
                x_cut = xl[:,inds]
                tot_t = t[-1] - t[0]
                sf = len(x_cut[0])/tot_t

                # compute spectral connectivity
                if functional:
                    functional_connectivity = spectral_connectivity([x_cut], method=functional_methods, sfreq=sf, fmin=bands[b][0], fmax=bands[b][1], mode='fourier', faverage=True, verbose=False)
                    
                    # get average link strength
                    for j in range(len(functional_methods)):
                        functional_matrix = functional_connectivity[0][j]  # lower triangular
                        n_rows, n_cols, _ = functional_matrix.shape

                        # compute average strength
                        average_strength = 0
                        for c in range(n_cols):
                                for r in range(c+1, n_cols): 
                                    average_strength += functional_matrix[r,c][0]
                        average_strength /= N*(N-1)/2

                        # append
                        average_strengths[b][j][i].append(average_strength)

                # find PSD and peak
                for j in range(N):
                    # PSD
                    bandpower_t = bandpower(x_cut[j], sf, bands[b], modified=modified, relative=relative, window_sec=window_sec)
                    if db:
                        bandpower_t = 10*log10(bandpower_t)
                    bandpowers[b][j][i].append(bandpower_t)

                    # frequency peaks
                    freq_peak_t = frequency_peaks(x_cut[j,:], sf, band=bands[b], tol=freq_tol, modified=modified, window_sec=window_sec)
                    freq_peaks[b][j][i].append(freq_peak_t)

    # package return value
    spectral_props = [bandpowers, freq_peaks]
    if functional:
        spectral_props.append(average_strengths)

    return spectral_props




# ---------------------------------------------
# plot spectral properties
# -------------------------------------------
def plot_spectral_properties(t_stamps, bandpowers, freq_peaks, bands, wiggle, title, legends, colours, bandpower_ylim = False, only_average=False, regions=[], n_ticks=5, relative=False):
    # find N and length of rhythms
    N = len(bandpowers[0])
    L = len(bandpowers[0][0][0])
    len_rhythms = len(bandpowers[0][0])

    # initialize
    figs_PSD = []
    figs_peaks = []
    axs_PSD = []
    axs_peaks = []
    for b in bands:
        fig_PSD = plt.figure()  # PSD
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = n_ticks) )
        figs_PSD.append(fig_PSD)

        fig_peaks = plt.figure()  # peaks
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = n_ticks) )
        #plt.xticks(np.arange(0, 30+1, 10))
        figs_peaks.append(fig_peaks)

    #  wiggle points in x-direction
    if regions:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*len(regions))
        wiggled = [np.array(t_stamps) + (i - len(regions)/2)*wiggle for i in range(len(regions)+1)]
    else:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*N)
        wiggled = [np.array(t_stamps) + (i - N/2)*wiggle for i in range(N+1)]

    # plot average power versus timestamps (one pipeline for regional input, one without)
    for b in range(len(bands)):
        if regions:
            # compute average over regions, and variance of region average over trials
            avg_powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
            avg_peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]

            for r in range(len(regions)):
                region = regions[r]
                powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                for ts in range(len(t_stamps)):
                    for l in range(L):
                        zero_peaks = 0
                        for node in region:
                            powers[ts][l] += bandpowers[b][node][ts][l]
                            node_peak = freq_peaks[b][node][ts][l]
                            if not np.isnan(node_peak):
                                peaks[ts][l] += freq_peaks[b][node][ts][l]
                            else: 
                                zero_peaks += 1

                        powers[ts][l] = powers[ts][l]/len(region)
                        if zero_peaks < len(region)/4:
                            peaks[ts][l] = peaks[ts][l]/(len(region)-zero_peaks)
                        else:
                            peaks[ts][l] = float("NaN")

                        # compute average over entire brain
                        if r == 0:
                            # average of trial
                            zero_peaks = 0
                            for n in range(N):
                                avg_powers[ts][l] += bandpowers[b][n][ts][l] 
                                
                                node_peak = freq_peaks[b][n][ts][l]
                                if not np.isnan(node_peak):
                                    avg_peaks[ts][l] += freq_peaks[b][n][ts][l] 
                                else:
                                    zero_peaks += 1
                            avg_powers[ts][l] /= N
                            if zero_peaks < N/4:
                                avg_peaks[ts][l] /= N - zero_peaks
                            else:
                                avg_peaks[ts][l] = float("NaN")


                # plot regions
                if not only_average:
                    # plot power
                    mean = np.mean(powers,axis=1)
                    std = np.std(powers,axis=1)
                    sns.despine()  # remove right and upper axis line

                    # plot peaks
                    mean = np.mean(peaks,axis=1)
                    std = np.std(peaks,axis=1)
                    sns.despine()  # remove right and upper axis line

                    figs_PSD[b].axes[0].spines['right'].set_visible(False)
                    figs_PSD[b].axes[0].spines['top'].set_visible(False)
                    figs_PSD[b].axes[0].errorbar(wiggled[r], np.mean(powers, axis=1), c=colours[r], \
                            label=legends[r], marker='o', linestyle='--', alpha=0.75, yerr=np.std(powers, axis=1), capsize=6, capthick=2)
                    sns.despine()
                    figs_peaks[b].axes[0].errorbar(wiggled[r], np.nanmean(peaks, axis=1), c=colours[r], \
                            label=legends[r], marker='o', linestyle='--', alpha=0.75, yerr=np.std(peaks,axis=1), capsize=6, capthick=2)
                    sns.despine()


            # plot average        
            # global power
            mean = np.mean(avg_powers,axis=1)
            std = np.std(avg_powers,axis=1)
            #sns.despine()  # remove right and upper axis line

            # global peaks
            mean = np.mean(avg_peaks,axis=1)
            std = np.std(avg_peaks,axis=1)
            #sns.despine()  # remove right and upper axis line

            figs_PSD[b].axes[0].errorbar(wiggled[-1], np.mean(avg_powers, axis=1), c='black', \
                    label='average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(powers, axis=1), capsize=6, capthick=2)
            sns.despine()
            figs_peaks[b].axes[0].errorbar(wiggled[-1], np.nanmean(avg_peaks, axis=1), c='black', \
                    label='average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(peaks,axis=1), capsize=6, capthick=2)
            sns.despine()

        else: 
            avg_power = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            avg_peak = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            for i in range(N):
                # power
                power = bandpowers[b][i]
                avg_power[:,i] = np.mean(power, axis=1) 
                if not only_average:
                    figs_PSD[b].axes[0].errorbar(wiggled[i], np.mean(power, axis=1), c=colours[i], label=legends[i], marker='o', linestyle='--', alpha=0.75, yerr=np.std(power, axis=1), capsize=6, capthick=2)

                # peak
                peak = freq_peaks[b][i]
                avg_peak[:,i] = np.mean(peak, axis=1) 
                if not only_average:
                    figs_peaks[b].axes[0].errorbar(wiggled[i], np.nanmean(peak, axis=1), c=colours[i], label=legends[i], marker='o', linestyle='--', alpha=0.75, yerr=np.std(peak,axis=1), capsize=6, capthick=2)

            # average power/peak over nodes
            avg_power = np.array(avg_power, dtype=np.float64)  # not included -> error due to sympy float values 
            figs_PSD[b].axes[0].errorbar(t_stamps, np.mean(avg_power, axis=1), c='black', label='Node average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_power, axis=1), capsize=6, capthick=2)
            
            avg_peak = np.array(avg_peak, dtype=np.float64)  # not included -> error due to sympy float values 
            figs_peaks[b].axes[0].errorbar(t_stamps, np.nanmean(avg_peak, axis=1), c='black', label='Node average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_peak, axis=1), capsize=6, capthick=2)

            
    # set labels 
    for b in range(len(bands)):
        # power
        figs_PSD[b].axes[0].set_title(title)
        if relative:
            figs_PSD[b].axes[0].set_ylabel(f'Relative power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        else:
            figs_PSD[b].axes[0].set_ylabel(f'Absolute power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        #figs_PSD[b].axes[0].set_xlabel('Speading time (years)')
        figs_PSD[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_PSD[b].axes[0].set_xlim([np.amin(wiggled)-wiggle, np.amax(wiggled)+wiggle])

        if bandpower_ylim:
            figs_PSD[b].axes[0].set_ylim([-0.05, bandpower_ylim])

        # peak
        figs_peaks[b].axes[0].set_title(title)
        figs_peaks[b].axes[0].set_ylabel(f'Peak frequency (${bands[b][0]} - {bands[b][1]}$ Hz)')
        #figs_peaks[b].axes[0].set_xlabel('Spreading time (years)')
        figs_peaks[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_peaks[b].axes[0].set_xlim([np.amin(wiggled)-wiggle, np.amax(wiggled)+wiggle])
        figs_peaks[b].axes[0].set_ylim([bands[b][0] - 0.5, bands[b][-1] + 0.5])
        figs_peaks[b].axes[0].set_ylim([8,11])

        # legends
        figs_PSD[b].axes[0].legend()
        plt.tight_layout()
        figs_peaks[b].axes[0].legend()
        plt.tight_layout()

    # we're done
    return figs_PSD, figs_peaks


def power_spectrum(y, t, plot=False):
    y = np.array(y)
    y = np.add(y, -np.mean(y))  # remove 0 frequency
    ps = np.abs(np.fft.fft(y))**2
    ps = ps
    time_step = abs(t[1] - t[0])
    freqs = np.fft.fftfreq(y.size, time_step)
    idx = np.argsort(freqs)


    if plot:
        plt.plot(freqs[idx], ps[idx])
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlim((-100, 100))
        #plt.ylim((0, 0.1))
        plt.show()

    # find largest power
    argmax = np.argmax(ps[idx])


    return freqs[idx], ps[idx], ps[idx][argmax]

# taken from https://raphaelvallat.com/bandpower.html
def bandpower(data, sf, band, window_sec=None, relative=False, modified=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch, periodogram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Compute the (modified) periodogram 
    if modified:
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        glob_idx = np.logical_and(freqs >= 0, freqs <= 40)
        bp /= simps(psd[glob_idx], dx=freq_res)
    return bp

# modified the above function to return spectrogram peaks
def frequency_peaks(data, sf, band=None, window_sec=None, tol=10**-3, modified=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    tol : float
        tolerance for ignoring maximum peak and set frequency to zero

    Return
    ------
    peak : float
        Largest PSD peak in frequency.
    """
    from scipy.signal import welch, periodogram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Compute the (modified) periodogram 
    if modified:
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2/low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # plot periodigram
    #plt.plot(freqs, psd)
    #plt.xlim([0, 14])
    #plt.show()

    # find peaks in psd
    if band.any():
        low, high = band
        filtered = np.array([i for i in range(len(freqs)) if (freqs[i] > low and freqs[i] < high)])
        psd = psd[filtered]
        freqs = freqs[filtered]

    max_peak = np.argmax(abs(psd))
    if max_peak is None or abs(psd[max_peak]) < tol:
        #freq_peak = 0
        freq_peak = float("NaN")
    else:
        freq_peak = freqs[max_peak]
    # we're done
    return freq_peak
