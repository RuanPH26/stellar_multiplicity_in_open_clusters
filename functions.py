# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:43:59 2025

@author: Ruan
"""

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from scipy.stats import bootstrap

from astropy.coordinates import SkyCoord
import astropy.units as u


labelsize=12
palette = 'viridis'
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 14}
font_cb = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 12}

#___________________________________________________________________________________________________-

def bin_frac(data):
    binaries = len(data[data['q']>0])
    return binaries/len(data)

def relaxation_time(df):
    """
    t_relax = (8.9*10**5*(N*rh**3)**0.5)/(m**0.5*log(0.4*N))
    N = number of members
    m = mean stellar mass
    rh =  half-mass radius
    
    """
    cte = 8.9*10**5
    rh = df['rh']
    N = df['n_members']
    m = df['mass_total']/N
    
    t_relax = (cte*(N*rh**3)**0.5)/(np.log10(0.4*N)*m**0.5)
    
    
    return

def half_mass_ratio(data):
    
    aux = data.copy(deep=True)
    #Converter coordenadas astronômicas em coordenadas tridimensionais, x,y,z
    coords = SkyCoord(ra=aux['RA_ICRS'].values * u.degree,
                      dec=aux['DE_ICRS'].values * u.degree,
                      distance=aux['dist'].values * u.pc*1000,
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
    aux['er_mass_system'] = unp.std_devs(mass_system)
    
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
    
    return aux, rh



#Função para estimar o erro de rh via bootstrap
def get_rh(sample):
    _, rh = half_mass_ratio(sample)
    return rh

def bootstrap_rh(data, n_resamples=5000, ci=95, random_state=None, verbose=True):
    rng = np.random.default_rng(seed=random_state)
    n = len(data)
    rh_samples = []
    errors = 0

    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = data.iloc[sample_idx].reset_index(drop=True)

        try:
            rh = get_rh(sample)
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

    return {
        'rh_mean': np.mean(rh_samples),
        'rh_std': std_rh,
        'rh_ci': (lower, upper),
        'rh_samples': rh_samples
    }
    
    
    