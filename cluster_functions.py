# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:43:59 2025

@author: Ruan
"""

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from scipy.stats import bootstrap
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from scipy.spatial import distance
import statsmodels.api as sm

from astropy.coordinates import SkyCoord
import astropy.units as u


labelsize=12
palette = 'viridis'
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 14}
font_cb = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 12}

#___________________________________________________________________________________________________-

def bin_frac(data, q=0):
    binaries = len(data[data['q']>q])
    
    return binaries/len(data)


def find_k_nearest_cluster(df, idx, k=5, av_lim=.5, dist_lim=1.5):
    params = ['age', 'FeH', 'mass_total', 'n_members']
    mask_ref = (df.Av < av_lim) & (df.dist < dist_lim) & (df.index != idx)
    ref_sample = df[mask_ref]
    dists = []

    for cluster in ref_sample.index:
        d = distance.euclidean(ref_sample.loc[cluster, params],
                               df.loc[idx, params])
        dists.append((d, cluster))

    dists.sort(key=lambda x: x[0])
    return dists[:k] 


def get_probabilities(log_m2, mask_m1, mask_m2):
    
    N_stars = mask_m1.sum() + mask_m2.sum()
    
    N_prim = ((mask_m1) & (~np.isinf(log_m2))).sum()
              
    P_prim = N_prim / N_stars
    
    N_comp = (mask_m2).sum()
    P_comp = N_comp / N_stars 
    
    
    N_BS = N_prim + N_comp
    P_BS = N_BS / N_stars

    return P_prim, P_comp, P_BS, N_prim, N_comp, N_BS





def get_new_masses(mass, er_mass, comp_mass, er_comp_mass, n_boots = 1000, random_state=None, distribution = 'uniform'):
     

    rng = np.random.default_rng(random_state)
    
    new_mass = []
    new_comp_mass = []
    
    mass = np.asarray(mass)
    er_mass = np.asarray(er_mass)
    comp_mass = np.asarray(comp_mass)
    er_comp_mass = np.asarray(er_comp_mass)

    i=0
    for _ in range(0,n_boots):
        
#========================================normal distribution=================================================================================        
        if distribution.lower() == 'gaussian':
            
            mass_boot = rng.normal(mass, er_mass)
            mass_validation = (mass_boot<0) | (mass_boot< mass-er_mass) | (mass_boot>mass+er_mass)
            j=0
            while len(mass_boot[mass_validation])>0:
                if j >100:
                    mass_boot[mass_validation] = mass[mass_validation]
                    break
                mass_boot[mass_validation] = rng.normal(mass[mass_validation], er_mass[mass_validation])
                mass_validation = (mass_boot<0) | (mass_boot< mass-er_mass) | (mass_boot>mass+er_mass)
                j+=1
            comp_mass_boot = rng.normal(comp_mass, er_comp_mass)
            comp_mass_validation = (comp_mass_boot<0) | (comp_mass_boot>mass_boot) | (comp_mass_boot<comp_mass - er_comp_mass) | (comp_mass_boot>comp_mass + er_comp_mass)
            j=0
            while len(comp_mass_boot[comp_mass_validation])>0:
                if j >100:
                    comp_mass_boot[comp_mass_validation] = comp_mass[comp_mass_validation]
                    break        
                comp_mass_boot[comp_mass_validation] = rng.normal(comp_mass[comp_mass_validation], er_comp_mass[comp_mass_validation])
                comp_mass_validation = (comp_mass_boot<0) | (comp_mass_boot>mass_boot) | (comp_mass_boot<comp_mass - er_comp_mass) | (comp_mass_boot>comp_mass + er_comp_mass)
                j+=1
#============================================================================================================================================

##=======================================uniform distribution================================================================================                
        else:
            mass_boot = rng.uniform(mass-er_mass, mass+er_mass)
            mass_validation = (mass_boot<0) 
            
            j=0
            while len(mass_boot[mass_validation])>0:
                if j>100:
                    mass_boot[mass_validation] = mass[mass_validation]
                    break
                
                mass_boot[mass_validation] = rng.uniform(mass[mass_validation]-er_mass[mass_validation], 
                                                         mass[mass_validation]+er_mass[mass_validation])     
                mass_validation = (mass_boot<0)
                j+=1


            comp_mass_boot = rng.uniform(comp_mass-er_comp_mass, comp_mass+er_comp_mass)

            comp_mass_validation = (comp_mass_boot<0) | (comp_mass_boot>mass_boot) 
            j=0
            while len(comp_mass_boot[comp_mass_validation])>0:
                if j>100:
                    comp_mass_boot[comp_mass_validation] = np.minimum(comp_mass[comp_mass_validation],
                                                                      mass_boot[comp_mass_validation])
                    break
                comp_mass_boot[comp_mass_validation] = rng.uniform(comp_mass[comp_mass_validation]-er_comp_mass[comp_mass_validation], 
                                                             comp_mass[comp_mass_validation]+er_comp_mass[comp_mass_validation])
                
                comp_mass_validation = (comp_mass_boot < 0) | (comp_mass_boot > mass_boot)
                j+=1   
#============================================================================================================================================
        new_mass.append(mass_boot)
        new_comp_mass.append(comp_mass_boot)
        i+=1
        
    return (np.array(new_mass), np.array(new_comp_mass))




def sigma_fb(df, idx,q=0, k=5, av_lim=.5, dist_lim=1.5, alpha=2, eps=1e-6):
    nearest = find_k_nearest_cluster(df, idx, k, av_lim, dist_lim)
    if q==0:
        col = 'bin_frac'
    else:
        col = 'bin_frac_'+ str(q)
    fb_k = df.loc[idx, col]

    dists = np.array([d for d, _ in nearest])
    refs = np.array([df.loc[c, col] for _, c in nearest])
    diffs = refs - fb_k

    sigma = np.sqrt(np.sum(diffs**2) /k)
    return sigma

def n_members(data):
    return len(data) + len(data[data['comp_mass']>0])

def half_mass_ratio(data, dist):
    
    aux = data.copy(deep=True)
    #Converter coordenadas astronômicas em coordenadas tridimensionais, x,y,z
    coords = SkyCoord(ra=aux['RA_ICRS'].values * u.degree,
                      dec=aux['DE_ICRS'].values * u.degree,
                      distance= dist * u.pc*1000,
                      frame='icrs')

    cartesian = coords.cartesian
    x, y, z = cartesian.x.value, cartesian.y.value, cartesian.z.value

    #Define o centro do aglomerado como a média em cada direção
    x_center = x.mean()
    y_center = y.mean()
    z_center = z.mean()
    
    #Obtém a distância de cada sistema em relação ao centro do aglomerado
    aux['r'] = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
    
    #Calcula a massa do sistema já propagando o erro associado
    mass_system = unp.uarray(aux['mass'], aux['er_mass'])+unp.uarray(aux['comp_mass'], aux['er_comp_mass'])
    aux['mass_system'] = unp.nominal_values(mass_system)
    aux['e_mass_system'] = unp.std_devs(mass_system)
    
    total_mass =aux['mass_system'].sum()
    
    
    aux.sort_values(by='r', inplace=True)
    
    mass = 0
    for _, row in aux.iterrows():
        mass = mass + row['mass_system']
        if mass >= total_mass/2:
            rh = row['r']
            break
    #calcula o raio normalizado r/rh
    aux['r/rh'] = aux['r']/rh
    
    
    return aux['r/rh'], rh



#Função para estimar o erro de rh via bootstrap
def get_rh(sample, dist):
    _, rh, = half_mass_ratio(sample, dist)
    return rh

def bootstrap_rh(data, dist, n_resamples=1000, ci=95, random_state=None, verbose=False):
    rng = np.random.default_rng(seed=random_state)
    n = len(data)
    rh_samples = []
    errors = 0

    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = data.iloc[sample_idx].reset_index(drop=True)

        try:
            rh = get_rh(sample, dist)
            rh_samples.append(rh)
        except Exception as e:
            errors += 1
            if verbose:
                print(f"[{i}] Erro na reamostragem: {e}")
            continue

    if verbose:
        print(f"\nTotal de amostras bem-sucedidas: {len(rh_samples)}")
        print(f"Amostras com erro: {errors}")

    if len(rh_samples) == 0:
        raise RuntimeError("Nenhuma amostra válida foi gerada. Verifique os dados ou a função half_mass_ratio.")

    rh_samples = np.array(rh_samples)

    std_rh = np.std(rh_samples)
    alpha = 100 - ci
    lower = np.percentile(rh_samples, alpha / 2)
    upper = np.percentile(rh_samples, 100 - alpha / 2)
    return std_rh,
        #'rh_mean': np.mean(rh_samples),
        
        #'rh_ci': (lower, upper),
        #'rh_samples': rh_samples

def relaxation_time(df):
    """
    t_relax = (8.9*10**5*(N*rh**3)**0.5)/(m**0.5*log(0.4*N))
    N = number of members
    m = mean stellar mass
    rh =  half-mass radius
    
    """
    aux = df.copy(deep=True)
    cte = 8.9*10**5
    rh = unp.uarray(aux['rh'], aux['e_rh'])
    N = aux['n_members']
    m = unp.uarray(aux['mass_total'], aux['e_mass_total'])/N
    
    t_relax = (cte*(N*rh**3)**0.5)/(unp.log10(0.4*N)*m**0.5)
    t_relax = t_relax/1e6 #Tempo de relaxamento em Myr
    
    e_t_relax = unp.std_devs(t_relax)
    t_relax = unp.nominal_values(t_relax)
    

    return t_relax, e_t_relax

def stellar_density(data, radius_col):
    r = unp.uarray(data[radius_col], data['e_'+radius_col])
    V = (4/3)*np.pi*r**3
    n_stars = data['n_members']
    density = n_stars/V
    
    return unp.nominal_values(density), unp.std_devs(density)
   
def format_erro(valor, erro):
    return f"{valor:.2f} ± {erro:.2f}" 

def save_results(df):
        
    df = df.round(2)
    
    tabela_formatada = pd.DataFrame({
        'f_bin': df.apply(lambda x: format_erro(x['bin_frac'], x['e_bin_frac']), axis=1),
        'f_bin_0.5': df.apply(lambda x: format_erro(x['bin_frac_0.5'], x['e_bin_frac_0.5']), axis=1),
        'r_h': df.apply(lambda x: format_erro(x['rh'], x['e_rh']), axis=1),
        't_relax (Myr)': df.apply(lambda x: format_erro(x['t_relax'], x['e_t_relax']), axis=1),
        'τ': df.apply(lambda x: format_erro(x['tau'], x['e_tau']), axis=1)
    })
    
    # Exporta para LaTeX
    tabela_latex = tabela_formatada.to_latex(index=True, escape=False)
    with open("tabela_resultados.tex", "w", encoding="utf-8") as f:
        f.write(tabela_latex)

def lowess(x, y, f=1./3.):
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 *
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr


    
    