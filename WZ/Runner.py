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


class WZRunner:

    def __init__(self, unknown_cat, ref_cat, rand_cat, Mask, R_min, R_max, R_bins, z_min, z_max, z_bins):

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


    def Wz(self, corr):

        assert len(corr) == 2, "Correlator must be [DD, DR]"

        xi  = corr[0].xi/corr[1] - 1
        ang = corr[0].meanr
        w_ur = np.trapz(ang**-1 * xi, x = ang)

        return w_ur
    

    def _get_unknown_cat(self, bin):

        ra  = self.unknown_cat['ra']
        dec = self.unknown_cat['dec']
        w   = self.unknown_cat['w']

        m_u   = (self.unknown_cat['bin'] == bin) & self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)]
        cat_u = treecorr.Catalog(ra = ra[m_u], dec = dec[m_u], w  = w[m_u], ra_units = 'deg', dec_units = 'deg')
        
        return cat_u
    
    

    def _get_ref_cat(self, z_min, z_max):

        ra  = self.ref_cat['ra']
        dec = self.ref_cat['dec']
        w   = self.ref_cat['w']

        m_r   = (self.ref_cat['z'] > z_min) & (self.ref_cat['z'] < z_max) & (self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)])
        cat_r = treecorr.Catalog(ra = ra[m_r], dec = dec[m_r], w  = w[m_r], ra_units = 'deg', dec_units = 'deg')
        
        return cat_r
    


    def _get_rand_cat(self, z_min, z_max):

        ra  = self.rand_cat['ra']
        dec = self.rand_cat['dec']
        w   = self.rand_cat['w']

        m_r   = (self.rand_cat['z'] > z_min) & (self.rand_cat['z'] < z_max) & (self.Mask[hp.ang2pix(self.NSIDE, ra, dec, lonlat  = True)])
        cat_r = treecorr.Catalog(ra = ra[m_r], dec = dec[m_r], w  = w[m_r], ra_units = 'deg', dec_units = 'deg')
        
        return cat_r



    def process(self):

        cosmo = FlatwCDM(H0 = 70, Om0 = 0.3, w0 = -1)

        z = np.linspace(self.z_min, self.z_max, self.z_bins +1)
        
        N_z  = np.ones([4, self.z_bins])
        dN_z = np.ones([4, self.z_bins])

        for b_i in range(4):

            cat_u = self._get_unknown_cat(b_i)
            
            for z_i in range(z.size -1):

                cat_r = self._get_ref_cat(z[z_i], z[z_i + 1])
                cat_n = self._get_rand_cat(z[z_i], z[z_i + 1])

                z_mid = (z[z_i] + z[z_i + 1])/2
                physical_min, physical_max = 1.5, 5.0 #in Mpc
                physical_min = 0.9*physical_min #simply enlarging the bottom and top bins a bit
                physical_max = 1.1*physical_max 

                theta_min = physical_min/cosmo.angular_diameter_distance(z_mid) * 180/np.pi #in degrees
                theta_max = physical_max/cosmo.angular_diameter_distance(z_mid) * 180/np.pi #in degrees

                DD = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', bins = self.R_bins, bin_slop = 0.01)
                DR = treecorr.NNCorrelation(min_sep =  theta_min, max_sep = theta_max, sep_units = 'deg', bins = self.R_bins, bin_slop = 0.01)
                
                DD.process(cat_u, cat_r)
                DR.process(cat_u, cat_n)

                correlators = [DD, DR]
                w_ur        = self.Wz(correlators)
                w_ur_cov    = treecorr.estimate_multi_cov(correlators, 'jackknife', func = self.Wz)
                    

                N_z[b_i, z_i]  = w_ur
                dN_z[b_i, z_i] = np.sqrt(w_ur_cov)

        return N_z, dN_z
    

if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--DECADE',    action='store_true', dest = 'DECADE')
    my_parser.add_argument('--DES',       action='store_true', dest = 'DES')
    my_parser.add_argument('--NSIDE',     action='store', type = int, default = 256)

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


    #Load boss data
    boss_cat_file    = '/project/chihway/data/decade/BOSS_eBOSS.fits'
    boss_random_file = '/project/chihway/data/decade/BOSS_eBOSS_rnd.fits'
    boss_cat = fitsio.read(boss_cat_file) #get Z, RA, DEC from here
    boss_ran = fitsio.read(boss_random_file)

    #Remove ELGs cause weights are weird
    boss_cat = boss_cat[boss_cat['SAMPLE'] != 'ELG']
    boss_ran = boss_ran[boss_ran['SAMPLE'] != 'ELG']

    Br_cat = {'ra' : boss_cat['RA'], 'dec' : boss_cat['DEC'], 'z' : boss_cat['Z'], 'w' : boss_cat['WEIGHT_FKP']}
    Bn_cat = {'ra' : boss_ran['RA'], 'dec' : boss_ran['DEC'], 'z' : boss_ran['Z'], 'w' : boss_ran['WEIGHT_FKP']}

    #Make a joint mask such that the catalogs overlap
    NSIDE = args['NSIDE']
    B_pix = hp.pix2ang(NSIDE, Br_cat['ra'], Br_cat['dec'], lonlat = True)
    C_pix = hp.pix2ang(NSIDE, Cat['ra'],    Cat['dec'],    lonlat = True)

    joint = np.intersect1d(B_pix, C_pix) #Only select pixels that have galaxies in both catalogs
    Mask  = np.zeros(hp.nside2npix(NSIDE), dtype = bool)
    Mask[joint] = True

    Runner = WZRunner(Cat, Br_cat, Bn_cat, Mask, R_min = 1.5, R_max = 5, R_bins = 20, z_min = 0.1, z_max = 1.1, z_bins = 40)
    result = Runner.process()

    np.save('/project/chihway/dhayaa/DECADE/Wz/Nz.npy',  result[0])
    np.save('/project/chihway/dhayaa/DECADE/Wz/dNz.npy', result[1])
