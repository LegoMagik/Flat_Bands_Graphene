# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:54:34 2023

@author: LB274087
"""

import numpy as np
import time
import kwant
import matplotlib.pyplot as plt
import scipy
from kwant.wraparound import wraparound,plot_2d_bands

import UniformStrainAux as me


#========================================Make system - Monolayer Graphene uniformly strained=====================================================

def make_system(t0=3.0, a0=0.14, theta = 0, eps=0, beta=3.37, L = -1):
    
    """
    t0: Hopping coefficient
    a0: Lattice distance in nm
    theta: applied strain angle in °
    eps: Deformation ratio on Axis at angle theta (eps <= 1)
    beta: decaying ratio
    L: -1 if infinite system else you choose the size of it (square)
    
    Returns:
        System built that can be used by kwant functions
        Pseudo vector A defined by the hamiltonian of in-plane strained graphene H = h vF (p-A)
    """
    
    h = 6.626e-34
    e = 1.602e-19
    sig = 0.165 #Poisson's ratio
    
    #========Pre-making========
    
    def square(pos):
        x,y = pos
        return (0 <= abs(x) < L) and (0 <= abs(y) < L)
    
    def circle(pos):
        x,y = pos
        return x**2 + y**2 <= L*L

    rotation_rad = theta*np.pi/180

    eps_tiers = eps*np.array(([(np.cos(rotation_rad))**2-sig*(np.sin(rotation_rad))**2,(1+sig)*np.cos(rotation_rad)*np.sin(rotation_rad)],[(1+sig)*np.cos(rotation_rad)*np.sin(rotation_rad),(np.sin(rotation_rad))**2-sig*(np.cos(rotation_rad))**2]))
    eps_arr = np.identity(2) + eps_tiers #Cf Peirera 2009

    Ca = (beta/(2*a0)) #Not really effective (h/(2*np.pi*e))
    A_pmf = Ca*np.array([-2*eps_arr[1,0],eps_tiers[1,1]-eps_tiers[0,0]]) #Cf "Emergent Horava gravity in graphene"
    print("Pseudo Potential vector = ", A_pmf)
    
    #===Designing the lattice=====

    #Primitive vectors
    a1,a2 = np.array([a0*3/2,a0*np.sqrt(3)/2]),np.array([a0*3/2,-a0*np.sqrt(3)/2])
    a1_rot,a2_rot,fir_atom,sec_atom = np.dot(eps_arr,a1),np.dot(eps_arr,a2),np.dot(eps_arr,np.array([0,0])),np.dot(eps_arr,np.array([a0,0]))

    #Translational vectors
    d1,d2,d3 = np.array([a0,0]),np.array([-a0*1/2,a0*np.sqrt(3)/2]),np.array([-a0*1/2,-a0*np.sqrt(3)/2])
    d1_rot,d2_rot,d3_rot = np.dot(eps_arr,d1),np.dot(eps_arr,d2),np.dot(eps_arr,d3)
    l1,l2,l3 = np.linalg.norm(d1_rot),np.linalg.norm(d2_rot),np.linalg.norm(d3_rot)
    print("Interatomic distances= ",np.array([l1,l2,l3]))
    
    #On-plane jump coef (Vpp_pi)
    def decaying_t(trans_vec):
        s = []
        for l in trans_vec: s.append(t0*np.exp(beta*(1-l/a0)))
        return np.array(s)
    
    T_arr = decaying_t(np.array([l1,l2,l3]))
    print("Hopping coefficients= ",T_arr)
    
    t1,t2,t3 = T_arr[0],T_arr[1],T_arr[2]
        
    
    #Building system=========START============
    
    lat = kwant.lattice.general([(a1_rot[0],a1_rot[1]),(a2_rot[0],a2_rot[1])],[(fir_atom[0],fir_atom[1]),(sec_atom[0],sec_atom[1])])

    a,b = lat.sublattices

    if L == -1: 
        sym = kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1)))
        sys = kwant.Builder(sym)
        sys[lat.shape(lambda p: True, (0, 0))] = 0
        
    else: 
        sys = kwant.Builder()
        sys[lat.shape(square, (0, 0))] = 0
    
    sys[[kwant.builder.HoppingKind((0,0),a,b)]] = t1
    sys[[kwant.builder.HoppingKind((0,1),a,b)]] = t2
    sys[[kwant.builder.HoppingKind((1,0),a,b)]] = t3
    
    #Building system==========END=============
  
    return sys,A_pmf 

def main():
    
    sys,A = make_system(theta=90,eps =0.19)
    final_sys = wraparound(sys).finalized()
    
    # me.plot_path_bands(final_sys,0.8*A,'G', 'K1', 'M1','G')
    
    # sys,A = make_system(theta=30,eps = 0.19)
    # kwant.plot(sys)
    # final_sys = wraparound(sys).finalized()
    # me.plot_isocolor_bands(final_sys,extend_bbox=1,sym_points = True)
    me.plot_isocolor_bands(final_sys,A,extend_bbox=1,sym_points = True)
    
    # sys = make_system(theta=30,eps = 0.05)
    # final_sys = wraparound(sys).finalized()
    # me.plot_isocolor_bands(final_sys,extend_bbox=1.5)
    
    # sys = make_system(theta=30,eps = 0.19)
    # final_sys = wraparound(sys).finalized()
    # me.plot_isocolor_bands(final_sys,extend_bbox=1.5,sym_points = True)
    
    # sys = make_system(theta=0,eps = 0)
    # final_sys = wraparound(sys).finalized()
    # me.plot_isocolor_bands(final_sys,extend_bbox=1.5,fig_title='Band structure for theta = 0° and eps = 0.199')
    
    # sys,A,A2 = make_system(theta=0,eps = 0,L=5)
    # kwant.plot(sys)
    
    # sys,A= make_system(theta=30,eps = 0.15,L=200)
    # fsyst = sys.finalized()
    # me.plot_ldos(fsyst)
    

    
main() 

