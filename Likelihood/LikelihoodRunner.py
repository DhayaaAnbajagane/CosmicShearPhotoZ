import numpy as np
from scipy import interpolate, stats
import joblib
import os, sys, glob, argparse, gc
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
        self.s_mu        = s_mu
        self.s_sig       = s_sig
        self.b_u         = b_u
        self.b_u_sig     = b_u_sig
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
        q        = self.q.copy()
        
        #Start iterations to find the maxlike point
        for iterations in range(self.Niter):
            
            
            meansys = np.exp( np.sum(self.sysf * s0[:, None], axis = 0) ) #Mean systematic function curve
            wtheory = self.wDM[self.zmsk] * nz[self.zmsk] * self.b_r * meansys #No magnification here
            
            #Compute the A matrix, which is derivative of prediction (or sys funct) w.r.t sys params
            A = np.zeros([len(self.Wz), self.order + 2], dtype = float)
            
            A[:, :self.order]    = self.sysf.T   * wtheory[:, None] #derivative of theory w.r.t sys func
            A[:, self.order]     = self.b_r      * np.sum(m1 * nz)  #derivative w.r.t alpha_u
            A[:, self.order + 1] = self.alpha_r  * np.sum(m2 * nz)  #derivative w.r.t b_u
            
            
            #Compute the constant vector, c. We don't use magnification params when
            #doing this. We linearlize only in sysfunc params
            c = self.Wz - wtheory + A[:, :self.order] @ s0
            
            
            #That's it, now compute the likelihood. This time also using magnification
            delta = c - A @ q
            cov   = np.einsum('ij,j,kj', A, self.q_sig**2, A) + self.C_Wz
            #lnlko = np.dot(delta, np.linalg.solve(cov, delta)) #Use solve for inverting the matrix. Y3 does it this way :P
            
            #B and d are just some names I make up for the vectors/matrices
            B = A.T @ self.inv_C_Wz @ A + np.diag(1/self.q_sig**2)
            d = A.T @ self.inv_C_Wz @ c + 1/self.q_sig**2 * self.q #Only use mean q for this evaluation
            
            lnlk  = -0.5 * np.dot(d, np.linalg.solve(B, d)) #Minus so its the negative log likelihood
            det   = np.linalg.slogdet(B); assert det[0] == 1, f"Determinant is non-positive definite. Sign {det[0]}."
            
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
        chisq   = np.dot(res, np.linalg.solve(self.C_Wz, res))
            
        return np.hstack([lnlk, chisq, q, wtheory])
    
    
    def process(self, nz_array):
         
        N      = nz_array.shape[0]
        ncpu   = joblib.cpu_count()
        Njobs  = np.max([ncpu, N//1_000_000])
        Nsplit = int(np.ceil(N / ncpu))
        
        path   = os.environ['TMPDIR'] + '/nz_array_TMP.npy'
        np.save(path, nz_array); del nz_array; gc.collect()
            
        def subfunc(i): 
            Min = i*Nsplit
            Max = (i+1)*Nsplit if i != Njobs - 1 else N
            iterator = tqdm(range(Min, Max)) if i == 0 else range(Min, Max)
            Nz  = np.load(path, mmap_mode = 'r')
            return i, [self.get_loglike(Nz[j]) for j in iterator]
        
        
        r = [0] * ncpu
        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(subfunc)(i) for i in range(Njobs)]
            outputs = joblib.Parallel(n_jobs = -1, verbose = 10, max_nbytes = None)(jobs)
            for o in outputs: r[o[0]] = o[1]
            r       = np.concatenate(r, axis = 0)
            
        os.remove(path)
        
        return r
    
    
    
if __name__ == '__main__':
    
    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--NzDir',  help = 'Directory of all the nz runs')
    my_parser.add_argument('--WzPath', help = 'Path to the Wz run we will use')
    my_parser.add_argument('--njobs',  action = 'store', type = int, default = -1, help = 'Number of parallel threads')
    
    my_parser.add_argument('--b_u',      type = float, default = 1.0, help = 'Bias of the unknown sample')
    my_parser.add_argument('--db_u',     type = float, default = 1.5, help = 'Uncertainty on bias of the unknown sample')
    my_parser.add_argument('--alpha_u',  type = float, default = 1.9, help = 'Magnification of the unknown sample')
    my_parser.add_argument('--dalpha_u', type = float, default = 1.0, help = 'Uncertainty on magnification of the unknown sample')
    
    my_parser.add_argument('--M',     type = int, default = 5, help = 'Order of the sys functions')
    my_parser.add_argument('--rms',   type = float, default = 0.15, help = 'Rms of the sys functions')
    
    my_parser.add_argument('--tol',   type = float, default = 0.05, help = 'Tolerance of the log likelihood linearization')
    my_parser.add_argument('--Niter', type = int, default = 100, help = 'Total number of iterations before converging on best fit')
    
    my_parser.add_argument('--Name',  type = str, default = None, help = 'Name to save the run under')
    
    args = vars(my_parser.parse_args())
    
    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')
    
    #Read out the WZ measurements
    WZ_data = np.load(args['WzPath'], allow_pickle = True)[()]

    # The params of the non-sys func marginalization
    b_u           = args['b_u']
    b_u_sigma     = args['db_u']
    alpha_u       = args['alpha_u'] - 2
    alpha_u_sigma = args['dalpha_u']
    
    # Now the sys function marginalization
    var   = args['rms']**2
    M     = args['M']
    s_mu  = np.zeros(M + 1)
    s_var = np.ones(M + 1)
    s_var[0]  *= np.log(b_u_sigma/b_u + 1)**2
    s_var[1:] *= var
    
    # Magnification coefficients (taking from section 3.3 of https://arxiv.org/pdf/2012.08569)
    
    z_RM = [0.25, 0.425, 0.575, 0.75, 0.9]
    #a_RM = [0.313, -1.515, -0.6628, 1.251, 0.9685]
    a_RM = [2.63, -1.04, 0.67, 4.50, 3.93]
    
    z_ML = [0.3, 0.475, 0.625, 0.775, 0.9, 1.0]
    a_ML = [2.43, 2.30, 3.75, 3.94, 3.56, 4.96]
    
    alpha_r = interpolate.interp1d(z_RM, a_RM, kind = 'cubic', fill_value = (a_RM[0], a_RM[-1]), bounds_error = False)
    
    
    #HARDCODED VALUES; MEASUREMENTS OF WZ
    R_min  = 1.5
    R_max  = 5.0
    R_bins = 10
    print(f"USING R_min = {R_min}, R_max = {R_max}, and R_bins = {R_bins}")
    
    #HARDCODED VALUES; MEASUREMENTS OF NZ
    min_z   = 0.01
    max_z   = 5
    delta_z = 0.05
    zbins   = np.arange(min_z,max_z+delta_z,delta_z)
    zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.
    zmsk    = np.zeros_like(zbinsc).astype(bool)
    zmsk[2:42] = True #Which parts of n(z) have Wz measurements to them.
    
    
    #Loop back to interpolate the z_bins and add the surface area factor (alpha = -2)
    #Can only evaluate this for redshift where we have boss galaxy data
    alpha_r = alpha_r(zbinsc[zmsk])
    
    print("ZBINS:", zbinsc[zmsk])
    print("ALPHA:", alpha_r)
    
    files = sorted(glob.glob(args['NzDir'] + '/nz_Samp*npy'))[:]
    nz    = np.concatenate([np.load(f) for f in files], axis = 0)
    res = []
    for i in range(4):
        
        WZ     = WZ_data['w_ur'][i]
        C_Wz   = WZ_data['Cw_ur'][i]
        b_r    = WZ_data['b_z'][i]
        
        #Hardcoded hartlap factor, for now....
        Njk   = 600
        Ndata = C_Wz.shape[0]
        hartlap  = (Njk - Ndata)/(Njk -1)
        dodelson = 1 / (1 + (Ndata - 3) * (Njk - Ndata - 2) / (Njk - Ndata - 1) / (Njk - Ndata - 4) )

        print(f"APPLYING HARTLAP {hartlap}, AND DODELSON {dodelson} TO BIN {i}. COMBINED {hartlap * dodelson}")
        
        C_Wz   = C_Wz / (hartlap * dodelson)

        #The Runner
        Runner = WZLikelihoodRunner(Wz  = WZ, z = zbinsc, zmsk = zmsk, C_Wz = C_Wz, 
                                    b_r = b_r, alpha_r = alpha_r,
                                    s_mu  = s_mu, s_sig = np.sqrt(s_var), 
                                    b_u = b_u, b_u_sig = b_u_sigma,
                                    alpha_u = alpha_u, alpha_u_sig = alpha_u_sigma,
                                    R_min   = R_min, R_max = R_max, R_bins = R_bins,
                                    lnlk_tolerance = args['tol'], Niter = args['Niter'])
    
        res.append(Runner.process(nz[:, i]))
        
    print("STARTING CONCAT")
    res = np.array(res)
    res = np.swapaxes(res, 0, 1)
    
    print("STARTING WRITE")
    Name = '' if args['Name'] is None else '_%s' % args['Name']
    np.save(args['NzDir'] + '/LnLikelihood%s.npy' % Name, res)
    
    print("FINISHED PROCESSING")

    