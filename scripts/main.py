#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:51 2026

@author: ruan
"""

from functions import *
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    
    # Setup paths
    path = 'membership_data_edr3/'
    output_path = 'results/'
    cluster_data = pd.read_csv('Data/log-results-eDR3-MF_integrada.csv', 
                               sep = ';', index_col='Cluster')
    
    list_data = []
    for cluster in cluster_data.index:
        
        file = path+cluster+'_data_stars.npy'
        try:
            data = pd.DataFrame(np.load(file, allow_pickle=True))
            data['Cluster'] = cluster
            q = data['comp_mass'] / data['mass']
            er_q = q*np.sqrt((data['er_mass']/data['mass'])**2 + 
                             (data['er_comp_mass']/data['comp_mass'])**2)
            data['q'] = q
            data['er_q'] = er_q
            cluster_data.loc[cluster, 'n_members'] = len(data) + len(data[data['q']>0]) 
            list_data.append(data)
        except FileNotFoundError: 
            print(f"Warning: File {cluster}.csv not found in path {path}")
            cluster_data = cluster_data.drop(index = cluster)
        except Exception as e:
            print(f"Error reading {cluster}.csv: {e}")
            
    if list_data: 
        data_final = pd.concat(list_data, ignore_index=True)
    else:
        print("No valid files were found.")
        data_final = pd.DataFrame() 
        
    
    os.makedirs('results', exist_ok=True)
    data_final.to_csv('results/data.csv', index=False)
    cluster_data.to_csv('results/results.csv', index=True)
            
    
    # Toggle execution
    RUN = True 
    
    if RUN:
        
        df = pd.read_csv('results/results.csv', index_col='Cluster')
        data = pd.read_csv('results/data.csv', index_col='Cluster')
        aux = data.copy()
        mask = ((aux['er_comp_mass']/aux['comp_mass'])<0.5) | (aux['comp_mass']==0)
        aux = aux[mask]
        
        i=1
        for cluster in df.index:
            

            
            dist= df.loc[cluster, 'dist']
        
            # Calcula rh e e_rh
            data.loc[cluster, 'r/rh'], df.loc[cluster, 'rh'], = half_mass_ratio(data.loc[cluster], dist)
            df.loc[cluster, 'e_rh'] = bootstrap_rh(data.loc[cluster], dist,)
            data.loc[cluster].to_csv(f"{path}{cluster}.csv", index=False) #Atualiza dados do aglomerados para incluir as posições dos sistemas
    
            # Calcula t_relax e densidade estelar
            df.loc[cluster, 't_relax'], df.loc[cluster, 'e_t_relax'] = relaxation_time(df.loc[cluster])

            
            df.loc[cluster, 'bin_frac'] = bin_frac(aux.loc[cluster])
                                       
            df.loc[cluster, 'bin_frac_05'] = bin_frac(aux.loc[cluster], q = 0.5)
            

        for cluster in df.index:
            

            
            df.loc[cluster, 'bin_frac_corr'] = corr_fb_by_similarity(df, 'bin_frac', cluster, k=5, av_lim=.5, dist_lim=1.5,)
            df.loc[cluster, 'er_bin_frac'] = calcula_er_fb(df, aux.loc[cluster], 'bin_frac', cluster, q=0)
            
            df.loc[cluster, 'bin_frac_05_corr'] = corr_fb_by_similarity(df, 'bin_frac_05', cluster,  k=5, av_lim=.5, dist_lim=1.5,)
            df.loc[cluster, 'er_bin_frac_05'] = calcula_er_fb(df, aux.loc[cluster], 'bin_frac_05', cluster, q = 0.5)
            
            
        # Calcula tau
        age = 10**unp.uarray(df['age'], df['e_age'])              # idade em anos
        t_relax = unp.uarray(df['t_relax'], df['e_t_relax']) * 1e6  # t_relax em anos
        
        # Tau = age / t_relax
        tau = age / t_relax
        df['tau'] = unp.nominal_values(tau)
        df['e_tau'] = unp.std_devs(tau)
        
        df.to_csv(output_path+'results.csv')
        mask = ((data['er_comp_mass']/data['comp_mass'])<0.5) | (data['comp_mass']==0)
        data = data[mask]
        data.to_csv(output_path+'data.csv')

        mass, er_mass = data['mass'].values, data['er_mass'].values
        comp_mass, er_comp_mass = data['comp_mass'].values, data['er_comp_mass'].values
        new_masses, new_comp_masses = get_new_masses(mass, er_mass, comp_mass, er_comp_mass, n_boots=2)

        np.save(output_path+'new_masses.npy', new_masses)
        np.save(output_path+'new_comp_masses.npy', new_comp_masses)
        
        
        
