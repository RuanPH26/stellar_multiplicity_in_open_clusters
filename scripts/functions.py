#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:44:56 2026

@author: ruan
"""

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from scipy.stats import bootstrap
from scipy import stats as stats
from astropy.coordinates import SkyCoord
import astropy.units as u


labelsize=12
palette = 'viridis'
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 14}
font_cb = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 12}

#___________________________________________________________________________________________________-


# Relationated to fraction of binaries
#===================================================================================================

def bin_frac(data, q=0):
    total_sys = len(data)
    bin_sys = len(data[data['q']>q])
    return bin_sys/total_sys

def corr_fb_by_similarity(df, col, idx, k=5, av_lim=.5, dist_lim=1.5,):
    """
    Retorna o valor corrigido de fb. O valor de fb corrigido considera a média ponderada dos k aglomerados mais próximos dentro da subamostra 
    de referência. Os pesos são inversamente proporcionais as distâncias (no espaço dos parâmetros) entre o aglomerado analisado e seus k-aglomerados
    semelhantes, dando maior peso aos que possuem uma distância menor.
    """
    nearest = find_k_nearest_cluster(df, idx, k, av_lim, dist_lim,) #Encontra os k semelhantes   
    dists, clusters = zip(*nearest)    
    dists = np.array(dists)
    pesos = 1/dists
    pesos = pesos/pesos.sum() #normaliza os pesos
    fb_ref = np.array([df.loc[c, col] for _, c in nearest]) #obtem os valores de fb dos k-aglomerados semelhantes
    return (fb_ref*pesos).sum()

def get_max(target, ref):
    max_param = np.maximum(np.abs(target), np.abs(ref).max())
    return max_param

def find_k_nearest_cluster(df, idx, k=5, av_lim=.5, dist_lim=1.5):

    params = ['age', 'FeH', 'mass_total', 'n_members']
    
    mask_ref = (df.Av < av_lim) & (df.dist < dist_lim) & (df.index != idx)
    ref_sample = df[mask_ref]
    
    if ref_sample.empty:
        return []

   
    ref_coords = ref_sample[params].values.astype(float)
    target_coords = df.loc[idx, params].values.astype(float)
    

    ref_coords[:, 0] = 10 ** ref_coords[:, 0]
    target_coords[0] = 10 ** target_coords[0]
    
    for i, _ in enumerate(params):
        max_param  = get_max(target_coords[i], ref_coords[:,i])
        ref_coords[:,i] = ref_coords[:,i]/max_param
        target_coords[i] = target_coords[i]/max_param
        
    
    dists = np.linalg.norm(ref_coords - target_coords, axis=1)
    
    
    if len(dists) > k:
        
        k_idx = np.argpartition(dists, k)[:k]
        k_idx = k_idx[np.argsort(dists[k_idx])]
    else:
        k_idx = np.argsort(dists)
        
    closest_clusters = ref_sample.index[k_idx]
    closest_dists = dists[k_idx]
    
    return list(zip(closest_dists, closest_clusters))

def corr_mean(df, q=0):
    benchmark = df[(df['Av']<0.5) & (df['dist']<1.5)]
    if q==0.5:
        col = 'bin_frac_05'
    else:
        col = 'bin_frac'
    mean_bf_bench = benchmark[col].mean()
    bin_range = 0.5
    Av = 0
    while Av <= df['Av'].max():
        mask = (df['Av']>= Av) & (df['Av']< Av+bin_range)
        aux = df[mask]
        if len(aux) > 0:
            mean_bf_aux = aux[col].mean()
            shift = mean_bf_aux - mean_bf_bench
            df.loc[mask, col+'_corr'] = aux[col] - shift
    
        Av += bin_range
    return df[col+'_corr']
    

def calcula_sigma_sys(df, data, col, cluster, n_bootstrap=1000, random_state=None):
    
    rng = np.random.default_rng(random_state)

    sigma_sys = []
    
    df_cluster = data.copy()
    idx_bin = df_cluster[df_cluster["flag_binary"] == 1].index 
    idx_single = df_cluster[df_cluster["flag_binary"] == 0].index
    bootstrap_fb = []
    N = calcula_N(df, data, col, cluster)
    print(f'N={N}')
    
    for _ in range(n_bootstrap):
        aux = df_cluster.copy(deep=True)       
        if N > 0:
            N_boots = rng.integers(0, N)
            if N_boots>len(idx_bin):
                N_boots=len(idx_bin)
            idx = rng.choice(idx_bin, size=np.abs(N_boots), replace=False)
            aux.loc[idx, "flag_binary"] = 0
        else:
            N_boots = rng.integers(N, 0)
            if np.abs(N_boots)>len(idx_single):
                N_boots=len(idx_single)
            idx = rng.choice(idx_single, size= np.abs(N_boots), replace=False)
            aux.loc[idx, "flag_binary"] = 1
            
        fb = aux["flag_binary"].sum() / len(aux)
        bootstrap_fb.append(fb)
        
    sigma_sys = np.std(bootstrap_fb)

    return  sigma_sys

def calcula_sigma_bin(df, data, col, cluster):
    
    fb = df.loc[cluster, col]
    n_systems = len(data.loc[cluster])
    sigma_bin = np.sqrt(fb*(1-fb)/n_systems)

    return np.array(sigma_bin)
        
def calcula_N(df, data, col, cluster):
    
    n_systems = n_systems = len(data.loc[cluster])
    
    N = (df.loc[cluster, col]-df.loc[cluster, col+'_corr'])*n_systems

    N = np.where((N>=0)&(N<=1),1,N)
    N = np.where((N<=0)&(N>=-1),-1,N)
    return np.array(N)
    
def calcula_er_fb(df, data, col, cluster, q=0, n_bootstrap=1000, random_state=1):
      
    data['flag_binary'] = np.where(data["q"] > q, 1, 0) 

    sigma_sys = calcula_sigma_sys(df, data, col, cluster, n_bootstrap, random_state)

    sigma_bin = calcula_sigma_bin(df, data, col, cluster)

    sigma_fb = np.sqrt(sigma_bin**2 + sigma_sys**2)
    
    return sigma_fb


#=============================================================================================

def get_probabilities(log_m2, mask_m1, mask_m2):
    
    N_stars = mask_m1.sum() + mask_m2.sum()
    
    N_prim = ((mask_m1) & (~np.isinf(log_m2))).sum()
              
    P_prim = N_prim / N_stars
    
    N_comp = (mask_m2).sum()
    P_comp = N_comp / N_stars 
    
    
    N_BS = N_prim + N_comp
    P_BS = N_BS / N_stars

    return P_prim, P_comp, P_BS, N_prim, N_comp, N_BS


def get_new_masses(mass, er_mass, comp_mass, er_comp_mass, n_boots = 1000, random_state=None, distribution = 'gaussian'):
     

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

def n_members(data, q=0):
    n_stars = len(data) + len(data[data['q']>q])
    return n_stars

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
    benchmark_flag = (df['dist']<1.5) & (df['Av']<0.5)
    
    tabela_formatada = pd.DataFrame({
        'f_bin': df.apply(lambda x: format_erro(x['bin_frac_corr'], x['er_bin_frac']), axis=1),
        'f_bin_0.5': df.apply(lambda x: format_erro(x['bin_frac_05_corr'], x['er_bin_frac_05']), axis=1),
        'r_h': df.apply(lambda x: format_erro(x['rh'], x['e_rh']), axis=1),
        't_relax (Myr)': df.apply(lambda x: format_erro(x['t_relax'], x['e_t_relax']), axis=1),
        'τ': df.apply(lambda x: format_erro(x['tau'], x['e_tau']), axis=1),
        'benchmark_flag': benchmark_flag,
    })
    
    # Exporta para LaTeX
    tabela_latex = tabela_formatada.to_latex(index=True, escape=False)
    with open("..\tabela_resultados.tex", "w", encoding="utf-8") as f:
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

                             
def loop_mass_ratio(log_m1, q, min_mass, max_mass, dm=0.05):
        
        m_min = min_mass
        m_max = m_min+dm
        
        mass_bins = []
        qs = []
        
        while m_min <= max_mass:
            
            if m_max> max_mass:
                break
                
            if m_min < np.log10(4):
                dm = 0.05
            else:
                dm = np.log10(1 + 1 / (10**m_min))
                
            mask_m1 = (log_m1 >=m_min) & (log_m1 <m_max)            
            total_stars = mask_m1.sum()
            
            if total_stars>=100:               

                mass_bins.append(np.median(log_m1[mask_m1])) 
                qs.append(q[mask_m1].mean())
                m_min = m_min + dm 
            m_max = m_max + dm

        return (np.array(mass_bins),np.array(qs))


def loop_probabilities(log_m1, log_m2,min_mass, max_mass, dm=0.05):
        
        m_min = min_mass
        m_max = m_min+dm
        
        mass_bins = []
        P_prim_arr = []
        P_comp_arr = []
        P_BS_arr = []         
        
        while m_min <= max_mass:
            
            if m_max> max_mass:
                break
                
            if m_min < np.log10(4):
                dm = 0.05
            else:
                dm = np.log10(1 + 1 / (10**m_min))
                
            mask_m1 = (log_m1 >=m_min) & (log_m1 <m_max)
            mask_m2 = (log_m2 >=m_min) & (log_m2 < m_max)
            
            total_stars = mask_m1.sum() + mask_m2.sum()
                
            
            if total_stars>=100:
                P_prim, P_comp, P_BS, N_prim, N_comp, N_BS = get_probabilities(log_m2, mask_m1, mask_m2)
                
                masses = np.concatenate([log_m1[mask_m1], log_m2[mask_m2]]) 
                mass_bins.append(np.median(masses)) 
                P_prim_arr.append(P_prim) 
                P_comp_arr.append(P_comp) 
                P_BS_arr.append(P_BS)

                m_min = m_min + dm 
            m_max = m_max + dm

        return (np.array(mass_bins),np.array(P_prim_arr),np.array(P_comp_arr),np.array(P_BS_arr))


def statistical_test(sample_1, sample_2):
    # Kolmogorov–Smirnov test
    ks_test = stats.ks_2samp(sample_1, sample_2)
    print(f"KS test: stat = {ks_test.statistic:.4f}, p = {ks_test.pvalue:.2e}")
    if ks_test.pvalue<0.05:
        print('Distribuições distintas\n')
    else:
        print('Não há evidência de que são diferentes\n')
        
    # Mann–Whitney U test    
    mw_test = stats.mannwhitneyu(sample_1, sample_2, alternative='two-sided')
    print(f"Mann–Whitney U test: U = {mw_test.statistic:.4f}, p = {mw_test.pvalue:.4e}")
    
    if mw_test.pvalue<0.05:
        print('Distribuições distintas\n')
    else:
        print('Não há evidência de que são diferentes\n')
    
    # Anderson–Darling test  
    AD_test = stats.anderson_ksamp([sample_1, sample_2])
    print(f"Anderson–Darling test: stat = {AD_test.statistic:.4f}, p = {AD_test.pvalue:.4e}")
    if AD_test.pvalue < 0.05: 
        print('Distribuições distintas\n')
    else:
        print('Não há evidência de que são diferentes\n')

def cliffs_delta(x, y):
    nx = len(x)
    ny = len(y)
    return ((np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (nx * ny)).round(2)

    