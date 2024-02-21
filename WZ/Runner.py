"""
Code for computing W(z) for DELVE (or DES)
"""

import os, sys
import numpy as np
import treecorr
import healpy as hp
import h5py
import fitsio
from astropy.cosmology import FlatwCDM
from tqdm import tqdm
import gc
import pyccl as ccl

class WZRunner:

    def __init__(self, unknown_cat, ref_cat, rand_cat, Mask, R_min, R_max, R_bins, z_min, z_max, z_bins, redshift_mask = False):

        self.unknown_cat = unknown_cat
        self.ref_cat     = ref_cat
        self.rand_cat    = rand_cat

        self.Mask   = Mask
        self.NSIDE  = hp.npix2nside(Mask.size)

        self.R_min  = R_min
        self.R_max  = R_max
        self.R_bins = R_bins

        self.z_min  = z_min
        self.z_max  = z_max
        self.z_bins = z_bins
        self.redshift_mask = redshift_mask


    def Wz(self, corr):

        assert len(corr) == 2, "Correlator must be [DD, DR]"

        DD   = corr[0].weight
        DR   = corr[1].weight * corr[0].tot/corr[1].tot
        xi   = DD/DR - 1
        ang  = corr[0].rnom
        wgt  = ang**-1
        wgt  = wgt/np.sum(wgt)
        w_ur = np.trapz(wgt * xi, x = ang)

        return np.array([w_ur])
    
    
    def WDM(self, z_min, z_max, theta_bins):
        
        cosmo = ccl.Cosmology(Omega_c = 0.26, Omega_b = 0.04, h = 0.7, sigma8 = 0.8, n_s = 0.96, matter_power_spectrum='halofit')
        cosmo.compute_sigma()
        cosmo.compute_nonlin_power()
        
        z_bin = np.linspace(0, 2, 400)
        dNdz  = np.where( (z_bin > z_min) & (z_bin < z_max), 1, 0) #This will be normalized later with ccl
        ell   = np.arange(1., 10_000.).astype(int)
        gal_tracer = ccl.tracers.NumberCountsTracer(cosmo, dndz = (z_bin, dNdz), 
                                                    bias = (z_bin, np.ones_like(z_bin)), mag_bias = None, has_rsd = False,)
        Cells      = ccl.cls.angular_cl(cosmo, gal_tracer, gal_tracer, ell)
        correlator = ccl.correlation(cosmo, ell = ell, C_ell = Cells, theta = theta_bins, type = 'NN', method = 'fftlog')
        
        return correlator
    

    def _define_patches(self):
        
        ra  = self.unknown_cat['ra']
        dec = self.unknown_cat['dec']
        m   = self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)]
        ra  = ra[m]
        dec = dec[m]
        
        del m
        
        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        os.system('rm %s' %  center_path)
        Nth    = int(len(ra)/10_000_000) #Select every Nth object such that we end up using 10 million to define patches
        if Nth < 1: Nth = 1
        small_cat = treecorr.Catalog(ra = ra[::Nth], dec = dec[::Nth], ra_units='deg',dec_units='deg', npatch = 100)
        small_cat.write_patch_centers(center_path)
        
        del ra, dec, small_cat
        gc.collect()

    
    def _get_unknown_cat(self, b):

        ra  = self.unknown_cat['ra']
        dec = self.unknown_cat['dec']
        w   = self.unknown_cat['w']

        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        m_u   = (self.unknown_cat['bin'] == (b + 1)) & self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)]
        cat_u = treecorr.Catalog(ra = ra[m_u], dec = dec[m_u], w  = w[m_u], ra_units = 'deg', dec_units = 'deg', patch_centers=center_path)
        
        del ra, dec, w, m_u
        gc.collect()
        
        return cat_u
    
    

    def _get_ref_cat(self, z_min, z_max):

        ra  = self.ref_cat['ra']
        dec = self.ref_cat['dec']
        w   = self.ref_cat['w']
        w   = np.ones_like(w)

        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        m_r   = (self.ref_cat['z'] > z_min) & (self.ref_cat['z'] < z_max) & (self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)])
        cat_r = treecorr.Catalog(ra = ra[m_r], dec = dec[m_r], w  = w[m_r], ra_units = 'deg', dec_units = 'deg', patch_centers=center_path)
        
        del ra, dec, w, m_r
        gc.collect()
        
        return cat_r
    


    def _get_rand_cat(self, z_min, z_max):

        ra  = self.rand_cat['ra']
        dec = self.rand_cat['dec']
        w   = self.rand_cat['w']
        w   = np.ones_like(w)

        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        m_r   = (self.rand_cat['z'] > z_min) & (self.rand_cat['z'] < z_max) & (self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)])
        cat_r = treecorr.Catalog(ra = ra[m_r], dec = dec[m_r], w  = w[m_r], ra_units = 'deg', dec_units = 'deg', patch_centers=center_path)
        
        del ra, dec, w, m_r
        gc.collect()
        
        return cat_r
    
    
    
    def _get_all_cats(self, b, z_min, z_max, delta_z):
        '''
        Function that doesn't use the overall mask,
        and instead uses a custom one for the exact catalog being used
        '''
        
        #Make a joint mask such that the catalogs overlap
        NSIDE = self.NSIDE
        u_ra  = self.unknown_cat['ra']
        u_dec = self.unknown_cat['dec']
        u_m   = self.unknown_cat['bin'] == (b + 1)
        u_ra  = u_ra[u_m]
        u_dec = u_dec[u_m]
        del u_m
        
        r_ra  = self.ref_cat['ra']
        r_dec = self.ref_cat['dec']
        r_m   = (self.ref_cat['z'] > z_min - delta_z) & (self.ref_cat['z'] < z_max + delta_z)
        r_ra  = r_ra[r_m]
        r_dec = r_dec[r_m]
        del r_m
        
        
        
        u_pix = hp.ang2pix(NSIDE, u_ra, u_dec, lonlat = True)
        r_pix = hp.ang2pix(NSIDE, r_ra, r_dec, lonlat = True)
        
        u_pix = np.bincount(u_pix, minlength = hp.nside2npix(NSIDE))
        r_pix = np.bincount(r_pix, minlength = hp.nside2npix(NSIDE))

        Mask  = (u_pix > 1) & (r_pix > 1); del u_pix, r_pix
        
        
        self.Mask = Mask
        print("Mask has been updated (f_area = %0.3f)" % np.average(self.Mask))
        
        self._define_patches()
        
        a = self._get_unknown_cat(b)
        b = self._get_ref_cat(z_min, z_max)
        c = self._get_rand_cat(z_min, z_max)
        
        del Mask, r_ra, r_dec, u_ra, u_dec
        gc.collect()
        
        return a, b, c
        


    def _generate_Wz_function(self, z_min, z_max, theta_bins):
        
        
        cosmo = ccl.Cosmology(Omega_c = 0.26, Omega_b = 0.04, h = 0.7, sigma8 = 0.8, n_s = 0.96, matter_power_spectrum='halofit')
        cosmo.compute_sigma()
        cosmo.compute_nonlin_power()
        
        z_bin = np.linspace(z_min*0.9, z_max*1.1, 100)
        dNdz  = np.where( (z_bin > z_min) & (z_bin < z_max), 1, 0) #This will be normalized later with ccl
        ell   = np.arange(1, 10_000).astype(int)
        gal_tracer = ccl.tracers.NumberCountsTracer(cosmo, dndz = (z_bin, dNdz), 
                                                    bias = (z_bin, np.ones_like(z_bin)), mag_bias = None, has_rsd = False,)
        Cells = ccl.cls.angular_cl(cosmo, gal_tracer, gal_tracer, ell)
        xi_mm = ccl.correlation(cosmo, ell = ell, C_ell = Cells, theta = theta_bins, type = 'NN', method = 'fftlog')
        
        wgt   = theta_bins**-1
        wgt   = wgt/np.trapz(wgt, x = theta_bins)
        W_DM  = np.trapz(wgt * xi_mm, x = theta_bins)
        
        
        del cosmo, gal_tracer, Cells, xi_mm
        gc.collect()
        
        def make_wz(corr):
            
            assert len(corr) == 4, "Correlator must be [DD_mcal, DR_mcal, DD_boss, DR_boss]"
            
            assert np.allclose(corr[0].rnom, corr[2].rnom), "The nominal bin centers are different"
            
            
            #First get w_ur
            DD   = corr[0].weight
            DR   = corr[1].weight * corr[0].tot/corr[1].tot
            xi   = DD/DR - 1
            ang  = corr[0].rnom
            wgt  = ang**-1
            wgt  = wgt/np.trapz(wgt, x = ang)
            w_ur = np.trapz(wgt * xi, x = ang)
            
            
            #Next get b(z)
            DD   = corr[2].weight
            DR   = corr[3].weight * corr[2].tot/corr[3].tot
            xi   = DD/DR - 1
            ang  = corr[2].rnom
            wgt  = ang**-1
            wgt  = wgt/np.trapz(wgt, x = ang)
            w_bb = np.trapz(wgt * xi, x = ang)
            b_z  = np.sqrt(w_bb/W_DM)
            
            #Finally estimate the Nz
            Nz   = w_ur / (b_z * W_DM)
            
            return np.array([Nz, w_ur, b_z, W_DM])
        
        
        return make_wz
    
    def process(self):

        cosmo = FlatwCDM(H0 = 70, Om0 = 0.3, w0 = -1)

        z = np.linspace(self.z_min, self.z_max, self.z_bins +1)
        
        N_z   = np.zeros([4, self.z_bins]) + np.NaN
        dN_z  = np.zeros([4, self.z_bins]) + np.NaN
        
        w_ur  = np.zeros([4, self.z_bins]) + np.NaN
        dw_ur = np.zeros([4, self.z_bins]) + np.NaN
        
        b_z   = np.zeros([4, self.z_bins]) + np.NaN
        db_z  = np.zeros([4, self.z_bins]) + np.NaN
        
        w_dm  = np.zeros([4, self.z_bins]) + np.NaN
        
        #Setup patches first for jackknifing
        if not self.redshift_mask: self._define_patches()
        
        with tqdm(total = (z.size - 1) * 4, desc = 'Building Wz') as pbar:
            for b_i in range(4):

                cat_u = self._get_unknown_cat(b_i)

                for z_i in range(z.size -1):
                    
                    if not self.redshift_mask:
                        cat_r = self._get_ref_cat(z[z_i], z[z_i + 1])
                        cat_n = self._get_rand_cat(z[z_i], z[z_i + 1])
                    
                    else:
                        cat_u, cat_r, cat_n = self._get_all_cats(b_i, z[z_i], z[z_i + 1], delta_z = 0.1)
                    
                    
                    if len(cat_r.ra) < 10: 
                        print("Only %d galaxies in combination b_i, z_i = (%d, %d)" % (len(cat_r.ra), b_i, z_i))
                        continue

                    
                    z_mid = (z[z_i] + z[z_i + 1])/2
                    physical_min, physical_max = 1.5, 5.0 #in Mpc
                    physical_min = 0.9*physical_min #simply enlarging the bottom and top bins a bit
                    physical_max = 1.1*physical_max 

                    theta_min = physical_min/cosmo.angular_diameter_distance(z_mid).value * 180/np.pi #in degrees
                    theta_max = physical_max/cosmo.angular_diameter_distance(z_mid).value * 180/np.pi #in degrees
                    
                    print(z_mid, theta_min* 60, theta_max * 60)

                    DD = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', nbins = self.R_bins, bin_slop = 0.01)
                    DR = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', nbins = self.R_bins, bin_slop = 0.01)
                    
                    DD.process(cat_u, cat_r)
                    DR.process(cat_u, cat_n)

                    #Get the BOSS correlations
                    DD_bb = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', nbins = self.R_bins, bin_slop = 0.01)
                    DR_bb = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', nbins = self.R_bins, bin_slop = 0.01)
                    DD_bb.process(cat_r)
                    DR_bb.process(cat_r, cat_n)

                    
                    #Now compute the Nz using the proper covariance estimate from TreeCorr.
                    correlators = [DD, DR, DD_bb, DR_bb]
                    calculator  = self._generate_Wz_function(z[z_i], z[z_i + 1], DD.rnom)
                    N_z_i, w_ur_i, b_z_i, w_dm_i = calculator(correlators)
                    Cov_i = treecorr.estimate_multi_cov(correlators, 'jackknife', func = calculator)

                    N_z[b_i, z_i]  = N_z_i
                    dN_z[b_i, z_i] = np.sqrt(Cov_i[0, 0])
                    
                    w_ur[b_i, z_i]  = w_ur_i
                    dw_ur[b_i, z_i] = np.sqrt(Cov_i[1, 1])
                    
                    b_z[b_i, z_i]   = b_z_i
                    db_z[b_i, z_i]  = np.sqrt(Cov_i[2, 2])
                    
                    w_dm[b_i, z_i]  = w_dm_i
                    
                    print(z_i, b_z_i, w_dm_i, b_z_i**2 * w_dm_i)
                        
                    
                    
                    del cat_r, cat_n
                    
                    if self.redshift_mask:
                        del cat_u
                    
                    del DD, DR, DD_bb, DR_bb, correlators
                    gc.collect()
                    
                    pbar.update(1)
                    

        DICT = {'N_z':N_z,
                'dN_z':dN_z,
                'w_ur':w_ur,
                'dw_ur':dw_ur,
                'b_z':b_z,
                'db_z':db_z,
                'w_dm':w_dm}

        return DICT
    

if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--DECADE',    action='store_true', dest = 'DECADE')
    my_parser.add_argument('--DES',       action='store_true', dest = 'DES')
    my_parser.add_argument('--NSIDE',     action='store', type = int, default = 256)
    my_parser.add_argument('--redshift_mask',  action='store_true', default = False)
    
    args  = vars(my_parser.parse_args())

    
    if args['DECADE']:

        with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20231212.hdf', 'r') as f:

            ra      = f['RA'][:]
            dec     = f['DEC'][:]
            w       = f['mcal_g_w'][:]

        with h5py.File('/project/chihway/data/decade/metacal_gold_combined_mask_20231212.hdf', 'r') as f:

            Tomobin = f['baseline_mcal_mask_noshear'][:]

        mask = Tomobin > 0
        ra   = ra[mask]
        dec  = dec[mask]
        w    = w[mask]
        bin  = Tomobin[mask]

        Cat  = {'ra' : ra, 'dec': dec, 'w' : w, 'bin' : bin}

        del bin, ra, dec, w, mask

            
    elif args['DES']:
        
        with h5py.File('/project/chihway/dhayaa/DES_Catalogs/DESY3_indexcat.h5') as f:
            mcal_selection = f['index/select'][:]

        with h5py.File('/project/chihway/dhayaa/DES_Catalogs/DESY3_metacal_v03-004.h5') as f:

            ra  = f['catalog/unsheared/ra'][:][mcal_selection]
            dec = f['catalog/unsheared/dec'][:][mcal_selection]  
            w   = f['catalog/unsheared/weight'][:][mcal_selection]

        with h5py.File('/project/chihway/dhayaa/DES_Catalogs/DESY3_sompz_v0.40.h5', 'r') as f:
    
            bin = f['catalog/sompz/unsheared/bhat'][:][mcal_selection]

        Cat  = {'ra' : ra, 'dec': dec, 'w' : w, 'bin' : bin}

        del bin, ra, dec, w, mcal_selection

    print("MADE LENSING CATALOG")

    #Load boss data
    boss_cat_file    = '/project/chihway/data/decade/BOSS_eBOSS.fits'
    boss_random_file = '/project/chihway/data/decade/BOSS_eBOSS_rnd.fits'
    boss_cat = fitsio.read(boss_cat_file) #get Z, RA, DEC from here
    boss_ran = fitsio.read(boss_random_file)

#     #Remove ELGs cause weights are weird
#     boss_cat = boss_cat[boss_cat['SAMPLE'] != 'ELG']
#     boss_ran = boss_ran[boss_ran['SAMPLE'] != 'ELG']


    Br_cat = {'ra' : boss_cat['RA'], 'dec' : boss_cat['DEC'], 'z' : boss_cat['Z'], 'w' : np.ones_like(boss_cat['WEIGHT_FKP'])}
    Bn_cat = {'ra' : boss_ran['RA'], 'dec' : boss_ran['DEC'], 'z' : boss_ran['Z'], 'w' : np.ones_like(boss_ran['WEIGHT_FKP'])}
    
    
    if not args['redshift_mask']:
        
        #Make a joint mask such that the catalogs overlap
        NSIDE = args['NSIDE']

        u_pix = hp.ang2pix(NSIDE, Cat['ra'],    Cat['dec'],    lonlat = True)
        r_pix = hp.ang2pix(NSIDE, Br_cat['ra'], Br_cat['dec'], lonlat = True)
        
        u_pix = np.bincount(u_pix, minlength = hp.nside2npix(NSIDE))
        r_pix = np.bincount(r_pix, minlength = hp.nside2npix(NSIDE))

        Mask  = (u_pix > 0) & (r_pix > 0); del u_pix, r_pix
        
        
    else:
        
        Mask = np.ones(hp.nside2npix(args['NSIDE']), dtype = int)

    
    
    print("MADE BOSS CATALOG")
    
    Runner = WZRunner(Cat, Br_cat, Bn_cat, Mask = Mask, redshift_mask = args['redshift_mask'],
                      R_min = 1.5, R_max = 5, R_bins = 20, z_min = 0.1, z_max = 1.1, z_bins = 40)
    result = Runner.process()
    
    
    np.save('/project/chihway/dhayaa/DECADE/Wz/20240213_Nz.npy', result)
