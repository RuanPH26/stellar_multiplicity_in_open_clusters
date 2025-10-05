# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 00:13:06 2025

@author: Ruan
"""

from functions import *
import os
#Caminho para dados dos aglomerados
path = './Aglomerados/'

run =False
if run:
    
    df = pd.read_csv('Dados/results.csv', index_col='Cluster')
    for cluster in df.index:
        data = pd.read_csv(f"{path}{cluster}.csv")
        dist= df.loc[cluster, 'dist']
    
        # Calcula rh e e_rh
        data, df.loc[cluster, 'rh'], = half_mass_ratio(data, dist)
        df.loc[cluster, 'e_rh'] = bootstrap_rh(data, dist,)
        data.to_csv(f"{path}{cluster}.csv", index=False) #Atualiza dados do aglomerados para incluir as posições dos sistemas

        # Calcula t_relax e densidade estelar
        df.loc[cluster, 't_relax'], df.loc[cluster, 'e_t_relax'] = relaxation_time(df.loc[cluster])
        df.loc[cluster, 'stellar_dens'], df.loc[cluster, 'e_stellar_dens'] = stellar_density(df.loc[cluster])
        
        # Calcula as frações de binárias 
        df.loc[cluster, 'bin_frac'] = bin_frac(data)          #caso geral
        df.loc[cluster, 'bin_frac_0.5'] = bin_frac(data, q=0.5)   #apenas sistemas com q>=0.5
    
    # Calcula tau
    age = 10**unp.uarray(df['age'], df['e_age'])              # idade em anos
    t_relax = unp.uarray(df['t_relax'], df['e_t_relax']) * 1e6  # t_relax em anos
    
    # Tau = age / t_relax
    tau = age / t_relax
    df['tau'] = unp.nominal_values(tau)
    df['e_tau'] = unp.std_devs(tau)
    
    # Calcula incertezas da fração binária
    for cluster in df.index:
        df.loc[cluster, 'e_bin_frac'] = sigma_fb(df, cluster, q=0)
        df.loc[cluster, 'e_bin_frac_0.5'] = sigma_fb(df, cluster, q=0.5)
