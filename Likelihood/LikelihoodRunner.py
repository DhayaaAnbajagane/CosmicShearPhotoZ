import numpy as np
from scipy import interpolate, stats
import joblib
import os, sys
from tqdm import tqdm

import pyccl as ccl


class WZLikelihoodRunner(object):
    
    def __init__(self, Wz, z, zmsk, C_Wz, b_r, alpha_r, s_mu, s_sig, b_u, b_u_sig, alpha_u, alpha_u_sig,
                 R_min, R_max, R_bins, Niter = 10, lnlk_tolerance = 0.01):
        
        #Measurements from WZ
        self.Wz   = Wz
        self.z    = z
        self.zmsk = zmsk 
        self.C_Wz = C_Wz
        self.b_r  = b_r
        self.alpha_r = alpha_r
        
        self.R_min  = R_min
        self.R_max  = R_max
        self.R_bins = R_bins
        
        #Systematic params to marginalize over
        self.s_mu    = s_mu
        self.s_sig   = s_sig
        self.b_u     = b_u
        self.b_u_sig = b_u_sig
        self.alpha_u     = alpha_u
        self.alpha_u_sig = alpha_u_sig
        
        self.order = len(self.s_mu) #The order of the systematic functions
        
        #Magnification theory prediction, to be used alongside alpha params
        self.Dij   = self.get_Dij()        
        self.q     = np.concatenate([s_mu,  [b_u,     alpha_u]])
        self.q_sig = np.concatenate([s_sig, [b_u_sig, alpha_u_sig]])
        
        self.inv_C_Wz = np.linalg.inv(C_Wz)
        self.inv_C_q  = 1 / self.q_sig**2 #This assumes the param priors are uncorrelated
        
        self.Niter = Niter
        self.sysf  = self.setup_sys_functions()
        
        self.lnlk_tolerance = lnlk_tolerance
        
        
        
    def setup_sys_functions(self):
        
        z_min = np.min(self.z[self.zmsk])
        z_max = np.max(self.z[self.zmsk])
        
        x = -0.85 + 2 * (self.z[self.zmsk] - z_min)/(z_max - z_min) * 0.85
        l = np.zeros([self.order, x.size])
        
        l[0] = 1
        l[1] = x
        
        #Get polynomials recursively
        for n in range(1, self.order - 1):
            l[n + 1] = ( (2*n + 1) * x * l[n] - n * l[n - 1] ) / (n + 1)
        
        return l
        
        
    def get_Dij(self):
        
        Mpc_to_m = ccl.physical_constants.MPC_TO_METER
        c        = 2.99792458e8 * 1e-3 #m/s --> km/s

        cosmo = ccl.Cosmology(Omega_c = 0.26, Omega_b = 0.04, h = 0.7, sigma8 = 0.8, n_s = 0.96, matter_power_spectrum='halofit')
        cosmo.compute_sigma()
        cosmo.compute_nonlin_power()
        
        D  = np.zeros([self.z.size, self.z.size], dtype = np.float64)
        dz = self.z[1] - self.z[0]
        
        prefactor = 3 * (70/c)**2 * 0.3 #This is computing 3*(H0/c)^2 Omega_M
        
        self.wDM = []
        for i in tqdm(range(D.shape[0]), desc = 'Build Dij matrix'):
            
            z_min = self.z[i] - dz/2
            z_max = self.z[i] + dz/2
            
            z_bin = np.linspace(z_min*0.9, z_max*1.1, 100)
            dNdz  = np.where( (z_bin > z_min) & (z_bin < z_max), 1, 0) #This will be normalized later with ccl
            ell   = np.arange(1, 10_000).astype(int)
            
            d_A        = ccl.angular_diameter_distance(cosmo, 1/(1 + self.z[i]))
            theta_min  = 0.9 * self.R_min/d_A * 180/np.pi #in degrees
            theta_max  = 1.1 * self.R_max/d_A * 180/np.pi #in degrees
            theta      = np.geomspace(theta_min, theta_max, self.R_bins + 1)
            theta      = np.sqrt(theta[1:] * theta[:-1])

            gal_tracer = ccl.tracers.NumberCountsTracer(cosmo, dndz = (z_bin, dNdz), 
                                                        bias = (z_bin, np.ones_like(z_bin)), mag_bias = None, has_rsd = False,)
            Cells      = ccl.cls.angular_cl(cosmo, gal_tracer, gal_tracer, ell)
            correlator = ccl.correlation(cosmo, ell = ell, C_ell = Cells, theta = theta, type = 'NN', method = 'fftlog')

            
            weights    = theta**-1
            weights    = weights / np.trapz(weights, theta)
            wDM        = np.trapz(correlator * weights, theta)
            self.wDM.append(wDM)
            
            for j in range(i + 1, D.shape[0]): #We enforce j > i here by keeping j <= i to zero
                
                a_i     = 1/(1 + self.z[i])
                a_j     = 1/(1 + self.z[j])
                chi_i   = ccl.comoving_radial_distance(cosmo, a_i)
                chi_j   = ccl.comoving_radial_distance(cosmo, a_j)
                dchi_j  = chi_j - ccl.comoving_radial_distance(cosmo, 1/(1 + self.z[j - 1]) )
                D[i, j] = prefactor * wDM * (chi_i/a_i) * (chi_j - chi_i)/chi_j * dchi_j
        
        self.wDM = np.array(self.wDM)
        
        return D
    
    
    
    def get_m1_m2(self, Dij, nz):
        
        #When computing Dij we set j <= i to Dij = 0, so we can just do a
        #straight sum  below without worrying about anything.
        m1 = [np.sum(Dij[i, :] * nz,    axis = 0) for i in range(self.z.size)] #\Sum_j Dij * n(z_j)
        m2 = [np.sum(Dij[i, :] * nz[i], axis = 0) for i in range(self.z.size)] #\Sum_j Dij * n(z_i)
        
        return np.array(m1), np.array(m2)
        
    
    
    #Computed following Appendix A of 2012.08569
    def get_loglike(self, nz, verbose = False):
        
        s0 = self.q[:self.order] * 1
        old_lnlk = np.inf
        m1, m2   = self.get_m1_m2(self.Dij, nz)
        
        #Start iterations to find the maxlike point
        for iterations in range(self.Niter):
            
            
            meansys = np.exp( np.sum(self.sysf * s0[:, None], axis = 0) ) #Mean systematic function curve
            wtheory = self.wDM[self.zmsk] * nz[self.zmsk] * self.b_r * meansys #No magnification here
            
            #Compute the A matrix, which is derivative of prediction (or sys funct) w.r.t sys params
            A = np.zeros([len(self.Wz), self.order + 2], dtype = float)
            
            A[:, :self.order]    = self.sysf.T   * wtheory[:, None] #derivative of theory w.r.t sys func
            A[:, self.order]     = self.b_r      * np.sum(m1 * nz) #derivative w.r.t alpha_u
            A[:, self.order + 1] = self.alpha_r  * np.sum(m2 * nz) #derivative w.r.t b_u
            
            
            #Compute the constant vector, c. We don't use magnification params when
            #doing this. We linearlize only in sysfunc params
            c = self.Wz - wtheory + A[:, :self.order] @ s0
            
            
            #That's it, now compute the likelihood. This time also using magnification
            delta = c - A @ self.q
            cov   = np.einsum('ij,j,kj', A, self.q_sig**2, A) + self.C_Wz
            lnlk  = np.dot(delta, np.linalg.solve(cov, delta)) #Use solve for inverting the matrix. Y3 does it this way :P
            
            #B and d are just some names I make up for the vectors/matrices
            B = A.T @ self.inv_C_Wz @ A + np.diag(1/self.q_sig**2)
            d = A.T @ self.inv_C_Wz @ c + 1/self.q_sig**2 * self.q
            
        
            det   = np.linalg.slogdet(B); assert det[0] == 1, f"Determinant is non-positive definite. Sign {det[0]}."
            lnlk  += det[1] #Add determinant
            
            #Move onto next best guess for MAP
            q  = np.linalg.solve(B, d)
            s0 = q[:self.order]
            
            if verbose:
                print(f"On iteration {iterations}, and find solution q = {q} with lnlike = {lnlk}")
            
            #See if you converged. Break if you have.
            if np.abs(lnlk - old_lnlk) < self.lnlk_tolerance:
                break
            else:
                old_lnlk = lnlk * 1
            
        
        #If you have converged (or you gave up) then just compute typical chi2 = (w - wmodel)^T  Cinv (w - wmodel) to get log likelihood.
        wtheory = self.wDM[self.zmsk] * nz[self.zmsk] * self.b_r
        wtheory = wtheory  * np.exp( np.sum(self.sysf * s0[:, None], axis = 0) )
        wtheory = wtheory  + A[:, self.order]     * q[self.order] #Unknown bias
        wtheory = wtheory  + A[:, self.order + 1] * q[self.order + 1] #Unknown mag
        
        res     = self.Wz - wtheory
        chisq   = np.dot(res, np.linalg.solve(self.C_Wz, res)) * 0.5
            
        return lnlk, chisq
    
    
    
    def process(self, nz_array):
        
        N = nz_array.shape[1]
        
        pass
    
    
if __name__ == '__main__':
    
    WZ_data = np.load('/project/chihway/dhayaa/DECADE/Wz/20240613_fiducial_Wz_urmask.npy', allow_pickle = True)[()]

    z = np.linspace(0.11, 2.11, 41)
    z = (z[1:] + z[:-1])/2
    WZ   = WZ_data['w_ur'][0]
    dWZ  = WZ_data['dw_ur'][0]
    C_Wz = np.diag(dWZ**2)
    b_r  = WZ_data['b_z'][0]

    b_u  = 1
    b_u_sigma = 0.5
    alpha_u = 1 - 2
    alpha_u_sigma = 1
    
    var  = 0.15
    M    = 5

    s_mu  = np.zeros(M + 1)
    s_sig = (2*np.arange(M + 1) +1) / 0.85**2
    s_sig[0] *= np.log(b_u_sigma/b_u + 1)**2

    TEST = WZLikelihoodRunner(Wz = WZ, z = zbinsc, zmsk = zmsk, C_Wz = C_Wz, 
                              b_r = b_r, alpha_r = 0.1 * np.ones_like(b_r) - 2,
                              s_mu  = np.zeros(M + 1), s_sig = s_sig, 
                              b_u = b_u, b_u_sig = b_u_sigma,
                              alpha_u = alpha_u, alpha_u_sig = alpha_u_sigma,
                              R_min   = 1.5, R_max = 5, R_bins = 10,
                              lnlk_tolerance = 0.05, Niter = 100)
    
    RES = TEST.get_loglike(nz[3], verbose = True)