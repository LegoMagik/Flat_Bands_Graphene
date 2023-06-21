

import numpy as np
import time
import kwant
import matplotlib.pyplot as plt
import scipy
import math
import tinyarray as ta
from kwant.wraparound import wraparound,plot_2d_bands

import PMFStrainPositionAux as me

def make_system_PMF_unloc(N_at = 30, beta = 3.37, t0=3.7, a0=0.14, aSC = 0.8, z0 = 0.1, SCangle = 0, SCrepr = False):
    """
    Parameters:
        N_at: Number of atoms on one side of the superlattice
        t0: Hopping coefficient for graphene (3.7 eV)
        a0: Lattice distance (nm)
        aSC: PMF peridocity (nm)
        z0: Height amplitude of the strain (nm)
        SCangle: Supercell angle from the chair config of graphene
        SCrepr: Representation of the supercell and the applied strain
        
    Returns:
        System of graphene with out-of-plane straining
        with no leads -> Only dispersion relation allowed
        
    NB: "unloc" stands for unlocalized
    """
    
    #========Constants=========
    
    nm = 1.0e-9 #Metric conversion
    phi0 = 1 #4e-15 #h/e
    
    #========Pre-making========
    
    if N_at%2: N_at += 1

    #============Super cell description===========
    
    n1 = math.floor(N_at*np.cos(SCangle)+(N_at/np.sqrt(3))*np.sin(SCangle))
    m1 = - math.floor((2*N_at/np.sqrt(3))*np.sin(SCangle))
    n2 = math.floor((2*N_at/np.sqrt(3))*np.sin(SCangle))
    m2 = math.floor(N_at*np.cos(SCangle)-(N_at/np.sqrt(3))*np.sin(SCangle))
    
    true_angle = np.arccos((m1+2*n1)/(2*N_at))
    
    print("Transformation matrix=",n1,m1)
    print("                      ",n2,m2)
    
    #==================================== Basis definition =========================================
    
    prim_vecs = ta.array(((np.sqrt(3),0),(np.sqrt(3)/2,3/2)))
    basis = ta.array([(0,0),(0,1)])

    a1,a2 = a0*np.array([np.sqrt(3),0]),a0*np.array([np.sqrt(3)/2,3/2])
    norm_SC = N_at*np.sqrt(3)*a0

    #==================Height straining and hoppings definitions====================
    
    z_PMF = me.z_PMF_triangle
    
    #To plot the PMF variations
    
    # me.vector_field_plot(1.5*norm_SC,me.PVF_func,z_PMF,norm_SC,z0)
    # me.norm_field_plot(1.5*norm_SC,z_PMF,norm_SC,z0)
    # me.PMF_plot(norm_SC,me.PVF_func,z_PMF,norm_SC,z0)
    
    def tij(rij): return t0*np.exp(-beta*(rij/a0 - 1))

    def hopt(site1,site2): #Asking for a height strain with the dérivation on x and y axis
        x2,y2 = site2.pos
        x1,y1 = site1.pos
        
        r1 = np.array([x1,y1])
        r2 = np.array([x2,y2])
        
        #Calculating local straining
        eps = np.array([[0.5*((z_PMF(norm_SC,z0,r2))[1])**2,0.5*(z_PMF(norm_SC,z0,r2))[1]*(z_PMF(norm_SC,z0,r2))[2]],[0.5*(z_PMF(norm_SC,z0,r2))[1]*(z_PMF(norm_SC,z0,r2))[2],0.5*((z_PMF(norm_SC,z0,r2))[2])**2]])
            
        r12 = np.linalg.norm(np.dot(np.eye(2)+eps,(r2-r1)))
        
        return  tij(r12)
    
    #Building system=========START============
    
    build_t0 = time.time()
    
    prim_vecs_real = a0 * prim_vecs
    basis_real = a0 * basis
    
    lat_init = kwant.lattice.general(prim_vecs_real,basis_real,norbs=1)
    
    
    comm_vect1 = (n1,m1)
    comm_vect2 = (n2,m2)
    
    #Defining the superlattice with spacial dependant hopping
    sym_SC = kwant.TranslationalSymmetry(lat_init.vec(comm_vect1), lat_init.vec(comm_vect2))
    
    sys = kwant.Builder(sym_SC)
    
    sys[lat_init.shape(lambda r: True, (0, 0))] = 0
    sys[me.inlayer(lat_init, 3)] = hopt
    
    build_t = time.time() - build_t0
    
    print("Filling hoppings duration = %s seconds" % (build_t))
    
    #Building system==========END=============
    
    #Kwant-plot==========START=============
    #To plot the super lattice
    
    if SCrepr:
        # def hoppings_color(site1,site2):
        #     x2,y2 = site2.pos
        #     x1,y1 = site1.pos
            
        #     r1 = np.array([x1,y1])
        #     r2 = np.array([x2,y2])
            
        #     r12 = (z_PMF(norm_SC,1.9,r2))[0]-(z_PMF(norm_SC,1.9,r1))[0]

        #     if r12 >= 0: return (r12,0,0)
        #     else : return (0,0,-r12)
            
        def site_color_pmf(site1):
            x1,y1 = site1.pos
            
            r1 = np.array([x1,y1])

            
            dz = (z_PMF(norm_SC,0.9,r1))[0]

            if dz >= 0: return (dz,0,0)
            else : return (0,0,-dz)
            
        # def hoppings_edge(site1,site2):
        #     x2,y2 = site2.pos
        #     x1,y1 = site1.pos
            
        #     r1 = np.array([x1,y1])
        #     r2 = np.array([x2,y2])
            
        #     r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + ((z_PMF(norm_SC,0.9,r1))[0]-(z_PMF(norm_SC,0.9,r2))[0])**2)

        #     return np.abs(t0-tij(r12))/(4*t0)

        
        kwant.plot(sys,site_color=site_color_pmf)
        
    #Kwant-plot==========END=============
    
    return sys,norm_SC,N_at,z0,true_angle

def make_system_PMF_loc(systSize = 1, N_at = 30, beta = 3.37, t0=3.7, a0=0.14, aSC = 0.8, z0 = 0.1, SCangle = 0, SCrepr = False):
    """
    Parameters:
        t0: Hopping coefficient for graphene (3.7 eV)
        a0: Lattice distance (nm)
        aSC: PMF peridocity (nm)
        z0: Height amplitude of the strain (nm)
        SCangle: Supercell angle from the chair config of graphene
        SCrepr: Representation of the supercell and the applied strain
        
    Returns:
        System of graphene with out-of-plane straining
        with no leads -> Only dispersion relation allowed
        
    NB: "unloc" stands for unlocalized
    """
    
    #========Constants=========
    
    nm = 1.0e-9 #Metric conversion
    phi0 = 1 #4e-15 #h/e
    
    #========Pre-making========
     
    # N_at = 30 #math.floor(aSC/(np.sqrt(3)*a0))  #47 - 70 (n_atom =  2*(N**2))
    
    if N_at%2: N_at += 1
    
    # print("Number of atoms along one main supercell axis = ",N_at)
    
    #============Super cell description===========

    # n1 = math.floor(N_at*np.cos(SCangle)+(N_at/np.sqrt(3))*np.abs(np.sin(SCangle)))
    # m1 = - math.floor((2*N_at/np.sqrt(3))*np.abs(np.sin(SCangle)))
    # n2 = math.floor((2*N_at/np.sqrt(3))*np.abs(np.sin(SCangle)))
    # m2 = math.floor(N_at*np.cos(SCangle)-(N_at/np.sqrt(3))*np.abs(np.sin(SCangle)))
    
    n1 = math.floor(N_at*np.cos(SCangle)+(N_at/np.sqrt(3))*np.sin(SCangle))
    m1 = - math.floor((2*N_at/np.sqrt(3))*np.sin(SCangle))
    n2 = math.floor((2*N_at/np.sqrt(3))*np.sin(SCangle))
    m2 = math.floor(N_at*np.cos(SCangle)-(N_at/np.sqrt(3))*np.sin(SCangle))
    
    true_angle = np.arccos((m1+2*n1)/(2*N_at))
    
    print("Transformation matrix=",n1,m1)
    print("                      ",n2,m2)
    
    #==================================== Basis definition =========================================
    
    prim_vecs = ta.array(((np.sqrt(3),0),(np.sqrt(3)/2,3/2)))
    basis = ta.array([(0,0),(0,1)])

    a1,a2 = a0*np.array([np.sqrt(3),0]),a0*np.array([np.sqrt(3)/2,3/2])
    norm_SC = N_at*np.sqrt(3)*a0

    #==================Height straining and hoppings definitions====================
    
    z_PMF = me.z_PMF_triangle
    
    # me.vector_field_plot(1.5*norm_SC,me.PVF_func,z_PMF,norm_SC,z0)
    
    # me.PMF_plot(norm_SC,me.PVF_func,z_PMF,norm_SC,z0)
    
    # me.PVF_PMF_matrix(3*norm_SC,me.PVF_func,z_PMF,norm_SC,z0)
    
    def tij(rij): return t0*np.exp(-beta*(rij/a0 - 1))

    def hopt(site1,site2): #Asking for a height strain with the dérivation on x and y axis
        x2,y2 = site2.pos
        x1,y1 = site1.pos
        
        r1 = np.array([x1,y1])
        r2 = np.array([x2,y2])
        
        r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + ((z_PMF(norm_SC,z0,r1))[0]-(z_PMF(norm_SC,z0,r2))[0])**2)
        
        return  tij(r12)
    
    #Building system=========START============
    
    build_t0 = time.time()
    
    prim_vecs_real = a0 * prim_vecs
    basis_real = a0 * basis
    
    lat_init = kwant.lattice.general(prim_vecs_real,basis_real,norbs=1)
    
    
    comm_vect1 = (n1,m1)
    comm_vect2 = (n2,m2)
    
    sym_SC = kwant.TranslationalSymmetry(lat_init.vec(comm_vect1), lat_init.vec(comm_vect2))
    
    sys = kwant.Builder()
    
    def square(pos):
        (x,y) = pos
        
        return (-systSize/2 < x < systSize/2) and (-systSize/2 < y < systSize/2)
    
    sys[lat_init.shape(square, (0, 0))] = 0
    sys[me.inlayer(lat_init, 3)] = hopt
    
    build_t = time.time() - build_t0
    
    print("Filling hoppings duration = %s seconds" % (build_t))
    
    #Building system==========END=============
    
    #Kwant-plot==========START=============
            
    if SCrepr:
        # def hoppings_color(site1,site2):
        #     x2,y2 = site2.pos
        #     x1,y1 = site1.pos
            
        #     r1 = np.array([x1,y1])
        #     r2 = np.array([x2,y2])
            
        #     r12 = (z_PMF(norm_SC,1.9,r2))[0]-(z_PMF(norm_SC,1.9,r1))[0]

        #     if r12 >= 0: return (r12,0,0)
        #     else : return (0,0,-r12)
            
        def site_color_pmf(site1):
            x1,y1 = site1.pos
            
            r1 = np.array([x1,y1])

            
            dz = (z_PMF(norm_SC,0.9,r1))[0]

            if dz >= 0: return (dz,0,0)
            else : return (0,0,-dz)
            
        # def hoppings_edge(site1,site2):
        #     x2,y2 = site2.pos
        #     x1,y1 = site1.pos
            
        #     r1 = np.array([x1,y1])
        #     r2 = np.array([x2,y2])
            
        #     r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + ((z_PMF(norm_SC,0.9,r1))[0]-(z_PMF(norm_SC,0.9,r2))[0])**2)

        #     return np.abs(t0-tij(r12))/(4*t0)

        
        kwant.plot(sys,site_color=site_color_pmf)
        
    #Kwant-plot==========END=============
    
    return sys


def main():
    main_t0 = time.time()

    def solve_PMF_system(N_at = 30, beta = 3.14, t0=3.7, a0=0.14, aSC = 0.8, z0 = 0.1, SCangle = 0, SCrepr = False):
        sys = make_system_PMF_unloc(N_at = N_at, beta = beta, t0=t0, a0=a0, aSC = aSC, z0 = z0, SCangle = SCangle, SCrepr = SCrepr)
        sys0 = sys[0]
        params=list(sys[1:])+["Band diagram for periodic strain"]
        
        final_sys = wraparound(sys0).finalized()
        
        me.plot_path_bands(final_sys,params,'G', 'K1', 'K2','G')
        
    solve_PMF_system(N_at = 8,z0=0.5)
    
    
    main_t = time.time() - main_t0
    
    print("Main program duration = %s seconds" % (main_t))


main()
        