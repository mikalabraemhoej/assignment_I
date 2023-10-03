import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w0, w1, phi0, phi1,vbeg_a_plus,vbeg_a,a,c,l1, l0):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i in nb.prange(par.Nz):
        
            ## i. labor supply times wage
            if i_fix <3:
                l0[i_fix,i,:] =  phi0 * par.z_grid[i] #(w0 * phi0 * par.eta[i_fix,0] + w1 * phi1 * par.eta[i_fix,1]) * par.z_grid[i]
            else:
                l1[i_fix,i,:] =  phi1 * par.z_grid[i]
            
            ## ii. cash-on-hand
            if i_fix <3:
                m = (1+r)*par.a_grid + w0 *  l0[i_fix,i,:]
            else:
                m = (1+r)*par.a_grid + w1 * l1[i_fix,i,:]
            
            #print(m)
            # iii. EGM
            c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i])**(-1/par.sigma)
            #print(c_endo)
            m_endo = c_endo + par.a_grid # current consumption + end-of-period assets
            #print(m_endo)
            # iv. interpolation to fixed grid
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i])
            a[i_fix,i,:] = np.fmax(a[i_fix,i,:],0.0) # enforce borrowing constraint
            c[i_fix,i] = m-a[i_fix,i]

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a