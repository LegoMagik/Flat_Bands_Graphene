# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:11:29 2023

@author: LB274087
"""

import numpy as np
import time
import kwant
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
from kwant.wraparound import wraparound,plot_2d_bands

#Fonction pour retrouver les points de symétries et leurs coo

def rot(angle,vec): 
    a = np.cos(angle)
    b = np.sin(angle)
    
    if a < 1e-11: a = 0
    elif b < 1e-11: b = 0 
    
    rota = np.array([[a,-b],[b,a]])
    return np.dot(rota,vec)

def plot_path_bands(sys,A_pmf = np.array([0,0]),*args):
    """
    Application ex: plot_path_bands(finalised_syst,'G', 'K', 'M','G')
    
    sys: Finalized bulk
    A_pmf = np.array([0,0]) : Potentiel vector defined by the hamiltonian of in-plane strained graphene H = h vF (p-A)
    *arg: pathing point
    Possible pôints ['G','K1','K2','K3','K1_bis','K2_bis','K3_bis','M1','M2','M3']
    
    NB: Seems it is not taking into account strain and angle (Symmetry points change)
    """
    
    #=======================================First Brillouin Zone==========================================
    
    # columns of B are lattice vectors
    B = np.array(sys._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T
    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = kwant.linalg.lll.lll(A.T)
    neighbors = np.dot(kwant.linalg.lll.voronoi(reduced_vecs), transf)
    lat_ndim, space_ndim = sys._wrapped_symmetry.periods.shape

    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * lat_ndim], neighbors))

    # Transform to cartesian coordinates and rescale.
    # Will be used in 'outside_bz' function, later on.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)

    # Calculate the Voronoi cell vertices
    vor = scipy.spatial.Voronoi(klat_points)
    around_origin = vor.point_region[0]
    bz_vertices = vor.vertices[vor.regions[around_origin]]
    
    bz_vertices_bis = []
    for vec in bz_vertices: bz_vertices_bis.append(rot(0,vec))
    bz_vertices_bis = np.array(bz_vertices_bis)
    
    # fig, ax = plt.subplots()
    
    # scipy.spatial.voronoi_plot_2d(vor)
    # ax = fig.add_subplot()
    # plt.xlim(-40,40)
    # plt.ylim(-40,40)
    # ax.set_aspect('equal')

    #=======================================Solving Eigenvalues==========================================

    def symm_point(text:str,bz_ver):
        if text == 'G': return np.array([0,0])
        elif text == 'K1' : return bz_ver[5]-A_pmf
        elif text == 'K1_bis': return bz_ver[4]+A_pmf
        elif text == 'K2': return bz_ver[1]-A_pmf
        elif text == 'K2_bis': return bz_ver[0]+A_pmf
        elif text == 'K3': return bz_ver[3]-A_pmf
        elif text == 'K3_bis': return bz_ver[2]+A_pmf
        
        elif text == 'M2': return 0.5 * (bz_ver[0] + bz_ver[1])
        elif text == 'M3': return 0.5 * (bz_ver[2] + bz_ver[3])
        elif text == 'M1': return 0.5 * (bz_ver[4] + bz_ver[5])
        
    def k_path(bz_vertices,*args1):
        dummy_1 = list(args1[0])
        dummy_2 = dummy_1[1:]
        points = zip(dummy_1[:-1], dummy_2)
        k = []
        for p1, p2 in points:
           point1 = symm_point(p1,bz_vertices_bis)
           point2 = symm_point(p2,bz_vertices_bis)
           kx=np.linspace(point1[0],point2[0],1000)
           ky=np.linspace(point1[1],point2[1],1000)
           
           k.append(np.array(list(zip(kx,ky))))

        return np.concatenate(k)

    def momentum_to_lattice(syst, k):
        B=np.array(sys._wrapped_symmetry.periods).T
        A = np.linalg.pinv(B).T
        k, residuals = scipy.linalg.lstsq(A, k)[:2]
        if np.any(abs(residuals) > 1e-7):
            raise RuntimeError("Probleme")
        return k

    def ham(sys, k_x, k_y=None, **params):
        # transform into the basis of reciprocal lattice vectors
        k = momentum_to_lattice(sys, [k_x] if k_y is None else [k_x, k_y])
        p = dict(zip(sys._momentum_names, k), **params)
        return sys.hamiltonian_submatrix(params=p, sparse=False)

    k_paths = k_path(bz_vertices_bis,args)
    
    energy = [] 
    for kx, ky in k_paths: energy.append(np.sort(np.real(np.linalg.eig(ham(sys, kx,ky))[0])))

    dummy  = np.linspace(0, len(args) - 1, len(energy))
    
    #==================================================Display==================================================
    
    plt.figure(figsize=(10,5))
    plt.xticks(list(range(0,len(args))), list(args))
    plt.xlabel("k")
    plt.ylabel("energy")
    plt.grid()
    
    for n in range(len(args)):
        plt.axvline(x = list(range(0,len(args)))[n], color='black', linestyle = "--", linewidth = 1)
    for n in (np.array(energy)).T: 
        plt.plot(dummy, n)


def plot_isocolor_bands(sys,A_pmf = np.array([0,0]),n_kx=50,n_ky=50,extend_bbox=0,normalize = 7.5*np.pi, fig_title = 'Band structure intensity map',sym_points = False):
    """
    Application ex: plot_isocolor_bands(final_sys,extend_bbox=1,fig_title='Band structure for theta = 90° and eps = 0.2')
    
    sys: Finalized bulk
    A_pmf: Potentiel vector defined by the hamiltonian of in-plane strained graphene H = h vF (p-A)
    n_kx/n_ky: discretize axis number
    extend_bbox: number of BZ to appear
    normalize: axis normalisiation
    fig_title: Figure title

    Returns:
        Isocolor mapping of the dispersion relation
    """
    
    #=======================================First Brillouin Zone==========================================
    
    # columns of B are lattice vectors
    B = np.array(sys._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T
    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = kwant.linalg.lll.lll(A.T)
    neighbors = np.dot(kwant.linalg.lll.voronoi(reduced_vecs), transf)
  
    lat_ndim, space_ndim = sys._wrapped_symmetry.periods.shape

    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * lat_ndim], neighbors))
    # print(klat_points)
    # Transform to cartesian coordinates and rescale.
    # Will be used in 'outside_bz' function, later on.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)
        

    # Calculate the Voronoi cell vertices
    vor = scipy.spatial.Voronoi(klat_points)
    around_origin = vor.point_region[0]
    bz_vertices = vor.vertices[vor.regions[around_origin]]
    # print(bz_vertices)
    
    bz_vertices_bis = []
    for vec in bz_vertices: bz_vertices_bis.append(rot(0,vec))
    bz_vertices_bis = np.array(bz_vertices_bis)
    
    # print(bz_vertices_bis)
    
    # fig, ax = plt.subplots()
    
    # scipy.spatial.voronoi_plot_2d(vor)
    # ax = fig.add_subplot()
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # ax.set_aspect('equal')

    #=======================================Solving Eigenvalues==========================================

    def symm_point(text:str,bz_ver):
        if text == 'G': return np.array([0,0])
        elif text == 'K1' : return bz_ver[5]-A_pmf
        elif text == 'K1_bis': return bz_ver[4]+A_pmf
        elif text == 'K2': return bz_ver[1]-A_pmf
        elif text == 'K2_bis': return bz_ver[0]+A_pmf
        elif text == 'K3': return bz_ver[3]-A_pmf
        elif text == 'K3_bis': return bz_ver[2]+A_pmf
        
        elif text == 'M2': return 0.5 * (bz_ver[0] + bz_ver[1])
        elif text == 'M3': return 0.5 * (bz_ver[2] + bz_ver[3])
        elif text == 'M1': return 0.5 * (bz_ver[4] + bz_ver[5])

    k_max = np.max(np.abs(bz_vertices_bis), axis=0)

    ## build grid along each axis, if needed
    ks = []
    for k, km in zip((n_kx, n_ky), k_max):
        k = np.array(k)
        if not k.shape:
            if extend_bbox:
                km += km * extend_bbox
            k = np.linspace(-km, km, k)
        ks.append(k)

    def momentum_to_lattice(k):
        B=np.array(sys._wrapped_symmetry.periods).T
        A = np.linalg.pinv(B).T
        k, residuals = scipy.linalg.lstsq(A, k)[:2]
        if np.any(abs(residuals) > 1e-7):
            raise RuntimeError("Probleme")
        return k

    def ham(sys, k_x, k_y=None, **params):
        # transform into the basis of reciprocal lattice vectors
        k = momentum_to_lattice([k_x] if k_y is None else [k_x, k_y])
        p = dict(zip(sys._momentum_names, k), **params)
        return sys.hamiltonian_submatrix(params=p, sparse=False)
    
    kx_l,ky_l = ks[0],ks[1]
    
    energy = np.zeros((n_kx,n_ky))
    
    for i,k_x in enumerate(kx_l):
        for j,k_y in enumerate(ky_l):
            energy[i,j] = np.abs((np.sort(np.real(np.linalg.eig(ham(sys,k_x,k_y))[0])))[0])
#NB: Since there can be multiple eigen values this line chooses only one of them by using no specific criteria (hyp: symetric solutions)

    #=======================================Display==========================================
    
    fig, ax = plt.subplots()
    
    #Display isocoloring of absolute value of the one of the eigen value
    ax.contourf((1/normalize)*kx_l,(1/normalize)*ky_l,energy.T,8, cmap = (plt.cm.get_cmap('Reds')).reversed()) #8
    
    #ax.contour(kx_l,ky_l,energy.T,colors='k')
    
    if sym_points:
        s_points = ['G','K1','K2','K3','K1_bis','K2_bis','K3_bis','M1','M2','M3']
        for p in s_points:
            a = (1/normalize)*rot(0,symm_point(p,bz_vertices_bis))
            ax.scatter(a[0],a[1],c='k')
            ax.annotate(p,(a[0],a[1]),textcoords="offset points",xytext=(0,10),ha='center')
            if p in ['M1','M2','M3']:
                ax.scatter(-a[0],-a[1],c='k')
                ax.annotate(p,(-a[0],-a[1]),textcoords="offset points",xytext=(0,10),ha='center')

    ax.set_box_aspect(1)
    plt.title(fig_title)
    plt.xlabel('kx/π')
    plt.ylabel('ky/π')
    
    
