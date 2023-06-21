# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:11:29 2023

@author: LB274087
"""

import numpy as np
import time
import kwant
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tinyarray as ta
import scipy
import scipy.linalg as la
from kwant.wraparound import wraparound,plot_2d_bands

from scipy.sparse.linalg import eigsh

h_CONST = 6.626e-34
h_bar = h_CONST/(2*np.pi)

e_CONST = 1.6e-19

nm_CONST = 1e-9

beta = 3.37
a0 = 0.14

N_POINTS_BAND_CONST = 500

#===========================PMF functions==========================
#Always follow the structure of z_PMF_triangle(SC_size,z0,r)


def z_PMF_triangle(SC_size,z0,r): #r = (x,y)
    """
    Parameters:
        SC_size: PMF peridocity (nm)
        z0: Height amplitude of the strain (nm)
        r: Position of an atome in the (x,y) plane
            
    Returns:
        Triangle lattice of out-of-plane strain at position r=(x,y) 
    """

    #Periodicity of the triangle mode on the direction of a1 and a2
    K = 2*np.pi/(SC_size)
    K1,K2 = K*np.array([0,2/(np.sqrt(3))]),K*np.array([1,-1/(np.sqrt(3))])
    K3 = K1+K2
    
    z,dzx,dzy = 0,0,0
    
    for k in [K1,K2,K3]: 
        z += np.cos(k[0]*r[0]+ k[1]*r[1])
        a = np.sin(k[0]*r[0]+ k[1]*r[1])
        if k[0]: dzx += k[0]*a
        elif k[0] == 0: z*r[0]
        if k[1]: dzy += k[1]*a
        elif k[1] == 0: z*r[1]
        
    return z0*z/4.5,-z0*dzx/4.5,-z0*dzy/4.5


def z_PMF_lamel(SC_size,z0,r): #r = (x,y)
    """
    Parameters:
        SC_size: PMF peridocity (nm)
        z0: Height amplitude of the strain (nm)
        r: Position of an atome in the (x,y) plane
            
    Returns:
        Triangle lattice of out-of-plane strain at position r=(x,y) 
    """

    #Periodicity of the triangle mode on the direction of a1 and a2
    K = 2*np.pi/(SC_size)
    K1 = K*np.array([0,2/(np.sqrt(3))])
    
    z,dzx,dzy = 0,0,0
    
    z = np.cos(K1[0]*r[0]+ K1[1]*r[1])
    a = np.sin(K1[0]*r[0]+ K1[1]*r[1])
    if K1[0]: dzx += K1[0]*a
    elif K1[0] == 0: z*r[0]
    if K1[1]: dzy += K1[1]*a
    elif K1[1] == 0: z*r[1]
        
    return z0*z,-z0*dzx,-z0*dzy


def PVF_func(z_func,SC_size,z0,r):
    """
    Parameters
        z_func: height mode function which returns 3 values at position r=(x,y)
        z_func(r) -> (z,dz_x,dz_y)
        
        SC_size: PMF peridocity (nm)
        z0: Height amplitude of the strain (nm)
        r: Position of an atome in the (x,y) plane
    
    Returns:
        Array of Pseudo vector field depending on position r=(x,y)
        
    NB: VF stands for pseudo vector field
    """
    
    eps_xx = 0.5*(list(z_func(SC_size,z0,r))[1])**2
    eps_yy = 0.5*(list(z_func(SC_size,z0,r))[2])**2
    eps_xy = 0.5*(list(z_func(SC_size,z0,r))[1])*(list(z_func(SC_size,z0,r))[2])
    
    return (beta/a0)*np.array([eps_xx-eps_yy,-2*eps_xy])


def PVF_PMF_matrix(width,func_A,*args):
        """
        Parameters:
            width: Size of the axis (Square image)
            func_A: vectorial function of the field that depends on r = (x,y)
            *args: arguments needed to process
            
        Returns:
            Plot the vector field of func_A on the (x,y) plane
        
        e.g. me.vector_field_plot(4*width,PMF_func,beta,a0,z_PMF,norm_SC,z0)
        """
        
        nc = 600
        x,y = np.linspace(0,2*width,nc),np.linspace(0,2*width,nc)
        
        h = 2*width/nc
        
        A_repr_x = np.zeros((nc,nc))
        A_repr_y = np.zeros((nc,nc))
        
        for i,ix in enumerate(x):
            for j,jy in enumerate(y):
                A_repr_x[i,j] = (func_A(*args,r=np.array([ix,jy])))[0]
                A_repr_y[i,j] = (func_A(*args,r=np.array([ix,jy])))[1]
                
        B_repr = np.zeros((nc,nc))
        for i,ix in enumerate(x[1:-1]):
            for j,jy in enumerate(y[1:-1]):
                B_repr[i,j] = (A_repr_y[i+1,j] - A_repr_y[i,j] - A_repr_x[i,j+1] + A_repr_x[i,j])
                
        B_repr *= 1/h
        
        print("Amplitude of the Pseudo magnetic field = %s [T]" % np.max((B_repr)))
        # print(B_repr)
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        ax.contourf(x,y,B_repr,100, cmap = (plt.cm.get_cmap('coolwarm')))
        plt.xlabel('x [nm]')
        plt.ylabel('y [nm]')

        # return A_repr_x,A_repr_y,B_repr
        
        
    

def PMF_plot(width,A_funk,z_func,SC_size,z0):
    n = 100
    x,y = np.linspace(0,2*width,n),np.linspace(0,2*width,n)
    
    B_repr = np.zeros((n,n))
    h = 2*width/n
    
    for i,ix in enumerate(x[1:]):
        for j,jy in enumerate(y[1:]):
            B_repr[i,j] = (A_funk(z_func,SC_size,z0,r=np.array([x[i],jy]))[1]-A_funk(z_func,SC_size,z0,r=np.array([x[i-1],jy]))[1])-A_funk(z_func,SC_size,z0,r=np.array([ix,y[j]]))[0]+A_funk(z_func,SC_size,z0,r=np.array([ix,y[j-1]]))[0]
    
    B_repr[i,j] *= 1/h
    print("Amplitude of the Pseudo magnetic field = %s [T]" % np.max((B_repr)))
    
    fig, ax = plt.subplots()    
    ax.contourf(x,y,B_repr,20, cmap = (plt.cm.get_cmap('Reds')).reversed())


#================Building system: auxiliar functions==============

def inlayer(lat, n):
    """Return list of n shortest HoppingKinds.

       Can be used directly as a Kwant builder key.
    """
    ret = []
    for i in range(1, n + 1):
        ret.extend(lat.neighbors(i))
        l = len(ret)
        if l >= n:
            break
    if l != n:
        raise ValueError(f"The {n} nearest neighbors are not symmetric. "
                         f"Consider {l}.")
    return ret


def rot(angle,vec): 
    a = np.cos(angle)
    b = np.sin(angle)
    
    if a < 1e-11: a = 0
    elif b < 1e-11: b = 0 
    
    rota = np.array([[a,-b],[b,a]])
    return np.dot(rota,vec)


    
#===========================================DIAGONALIZATION and DISPLAY===========================================

def norm_field_plot(width,func_A,*args):
    """
    Parameters:
        width: Size of the axis (Square image)
        func_A: function of the field that depends on r = (x,y)
        *args: arguments needed to process
        
    Returns:
        Plot the norm of the function func_A on the (x,y) plane
    
    e.g. me.norm_field_plot(4*norm_SC,me.PMF_func,beta,a0,z_PMF,norm_SC,z0)
    """
    
    n = 250
    x,y = np.linspace(0,2*width,n),np.linspace(0,2*width,n)
    
    A_repr = np.zeros((n,n))
    
    for i,ix in enumerate(x):
        for j,jy in enumerate(y):
            A_repr[i,j] = list(func_A(*args,r=np.array([ix,jy])))[0]

    fig, ax = plt.subplots()    
    ax.set_box_aspect(1)
    ax.contourf(x,y,A_repr,20, cmap = (plt.cm.get_cmap('coolwarm')))
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')



def vector_field_plot(width,func_A,*args):
    """
    Parameters:
        width: Size of the axis (Square image)
        func_A: vectorial function of the field that depends on r = (x,y)
        *args: arguments needed to process
        
    Returns:
        Plot the vector field of func_A on the (x,y) plane
    
    e.g. me.vector_field_plot(4*width,PMF_func,beta,a0,z_PMF,norm_SC,z0)
    """
    
    n = 40
    x,y = np.linspace(0,2*width,n),np.linspace(0,2*width,n)
    
    A_repr_x = np.zeros((n,n))
    A_repr_y = np.zeros((n,n))
    
    for i,ix in enumerate(x):
        for j,jy in enumerate(y):
            A_repr_x[i,j] = (func_A(*args,r=np.array([ix,jy])))[0]
            A_repr_y[i,j] = (func_A(*args,r=np.array([ix,jy])))[1]

    fig, ax = plt.subplots()    
    ax.quiver(x,y,A_repr_x,A_repr_y)
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')

def plot_path_bands(sys,params,*args):
    """
    Application ex: plot_path_bands(finalised_syst,'G', 'K', 'M','G')
    
    Parameters:
        sys: Finalized bulk
        fig_title: Figure title
        *arg: pathing points (at least 2 points)
        
    Returns:
        Plot of the dispersion relation depending on the choosen path
        
    Possible points ['G','K1','K2','K3','K1_bis','K2_bis','K3_bis','M1','M2','M3']
    
    """
    
    #First Brillouin Zone==========================================
    
    # columns of B are lattice vectors
    B = np.array(sys._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T
    # print(A)
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
    # print(bz_vertices_bis)

    #Solving Eigenvalues=========================================
    
    def symm_point(text:str,bz_ver):
        if text == 'G': return np.array([0,0])
        elif text == 'K1': return bz_ver[5] 
        elif text == 'K2': return bz_ver[4]
        elif text == 'K2_bis': return bz_ver[3]
        elif text == 'K1_bis': return bz_ver[2]
        elif text == 'K3_bis': return bz_ver[1]
        elif text == 'K3': return bz_ver[0]
        
        elif text == 'M1': return 0.5 * (bz_ver[5] + bz_ver[4])
        elif text == 'M2': return 0.5 * (bz_ver[3] + bz_ver[2])
        elif text == 'M3': return 0.5 * (bz_ver[1] + bz_ver[0])
        
    def k_path(bz_vertices,*args1):
        dummy_1 = list(args1[0])
        dummy_2 = dummy_1[1:]
        points = zip(dummy_1[:-1], dummy_2)
        k = []
        for p1, p2 in points:
           point1 = symm_point(p1,bz_vertices_bis)
           point2 = symm_point(p2,bz_vertices_bis)
           kx=np.linspace(point1[0],point2[0],N_POINTS_BAND_CONST)
           ky=np.linspace(point1[1],point2[1],N_POINTS_BAND_CONST)
           
           k.append(np.array(list(zip(kx,ky))))

        return np.concatenate(k)

    def momentum_to_lattice(syst, k):
        k, residuals = scipy.linalg.lstsq(A, k)[:2]
        if np.any(abs(residuals) > 1e-7):
            raise RuntimeError("Probleme")
        return k

    def ham(sys, k_x, k_y=None, **params):
        # transform into the basis of reciprocal lattice vectors
        k = momentum_to_lattice(sys, [k_x] if k_y is None else [k_x, k_y])
        p = dict(zip(sys._momentum_names, k), **params)
        return sys.hamiltonian_submatrix(params=p, sparse=True)

    alpha = 1.5
    k_paths = k_path(alpha*bz_vertices_bis,args)
    
    dz_t0 = time.time()
    
    energy = []
    
    N_eg = 28
    for kx, ky in k_paths: 
        energy.append(np.sort(np.real(eigsh(ham(sys, kx,ky),k=N_eg,which='SM')[0])))
        
    dz_t = time.time() - dz_t0
        
    print("Diagonalization time = %s seconds" % (dz_t))
    
    #vF measurements====================================================
    
    firstBand = ((np.array(energy)).T)[1+(N_eg-1)//2]
    
    E0 = firstBand.min()
    ind_E0 = np.where(firstBand == firstBand.min())

    E1 = firstBand[ind_E0[0][0] - N_POINTS_BAND_CONST//10]
    ind_E1 = np.where(firstBand == E1)
    
    # print(E0)
    k0 = np.linalg.norm(k_paths[ind_E0,:])
    k1 = np.linalg.norm(k_paths[ind_E1,:])
    
    # print(E1,E0,k1,k0)
    
    vf = (1/h_bar)*np.abs((E1-E0)/(k1-k0))*e_CONST*nm_CONST
    
    vf = "{:e}".format(vf)
    
    #Display the Eigenvalues============================================
    
    dummy  = np.linspace(0, len(args) - 1, len(energy))
    
    plt.figure(figsize=(10,5))
    plt.xticks(list(range(0,len(args))), list(args))

    white_rect = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    if len(params) == 5: 
        aSC,N_at,z0,SCangle,fig_title = params[0],params[1],params[2],params[3],params[4]
        rect = [white_rect, white_rect, white_rect,white_rect,white_rect]
        legendary = ("Supercell size = %s [nm]" % aSC, "Deformation amplitude along z axis = %s [nm]" % z0,"Number of atoms along one supercell axis = %s" % N_at, "Supercell angle = %s °" % SCangle, "vF = %s [m/s]" % vf)
        plt.legend(rect, legendary,loc='lower right',fontsize='small')
    
    else: 
        print("Number of parameters is not equal to 5")
        rect = [white_rect]
        legendary = ("vF = %s [m/s]" % vf)
        plt.legend(rect, legendary,loc='lower right',fontsize='small')
    
    plt.title(fig_title)
    plt.xlabel("k [nm-1]")
    plt.ylabel("energy [eV]")
    plt.grid()
    
    for n in range(len(args)):
        plt.axvline(x = list(range(0,len(args)))[n], color='black', linestyle = "--", linewidth = 1)
    for n in (np.array(energy)).T:
        plt.plot(dummy, n)

def plot_isocolor_bands(sys,A_pmf = np.array([0,0]),n_kx=50,n_ky=50,extend_bbox=0,normalize = np.pi, fig_title = 'Band structure intensity map',sym_points = False):
    """
    Application ex: plot_isocolor_bands(final_sys,extend_bbox=1,fig_title='Band structure for theta = 90° and eps = 0.2')
    
    sys: Finalized bulk
    n_kx/n_ky: discretize axis number
    extend_bbox: number of BZ to appear
    normalize: axis normalisiation
    fig_title: Figure title
    
    Problem: Voronoi vertices are not aligning with the dispersion relation
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
        elif text == 'K1': return bz_ver[5] + A_pmf
        elif text == 'K2': return bz_ver[4] - A_pmf
        elif text == 'K2_bis': return bz_ver[3] + A_pmf
        elif text == 'K1_bis': return bz_ver[2] - A_pmf
        elif text == 'K3_bis': return bz_ver[1] + A_pmf
        elif text == 'K3': return bz_ver[0] - A_pmf
        
        elif text == 'M1': return 0.5 * (bz_ver[5] + bz_ver[4])
        elif text == 'M2': return 0.5 * (bz_ver[3] + bz_ver[2])
        elif text == 'M3': return 0.5 * (bz_ver[1] + bz_ver[0])
    

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
            energy[i,j] = np.abs((np.sort(np.real(np.linalg.eig(ham(sys,k_x,k_y))[0])))[-1])
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

    # fig.colorbar(im,ax=ax)

def plot_ldos(sys, fig_title = 'LDOS'):
    """
    Parameters
    ----------
    sys : Finalized system
    fig_title : TYPE, optional
        DESCRIPTION. The default is 'LDOS'.

    Returns
    -------
    Display LDOS
    """

    spectrum = kwant.kpm.SpectralDensity(sys,rng=0) #,energy_resolution=0.05
    # spectrum.add_moments(100)
    # spectrum.add_vectors(5)
    energy,density = spectrum()
    
    fig_ldos, ax_ldos = plt.subplots()
    
    plt.grid()
    plt.title(fig_title)
    plt.plot(energy,density)


def rot(angle,vec): 
    a = np.cos(angle)
    b = np.sin(angle)
    
    if a < 1e-11: a = 0
    elif b < 1e-11: b = 0 
    
    rota = np.array([[a,-b],[b,a]])
    return np.dot(rota,vec)