import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import fitsio
import os, sys


def match_zcat_to_deepcat(zcat, z_col, ra_col, dec_col, source_name, deepcat):
    
    match_radius = 1
    # Match spectra to DELVE catalog
    spec_cd = SkyCoord(ra = zcat[ra_col].values  * u.deg, dec = zcat[dec_col].values  * u.deg)
    deep_cd = SkyCoord(ra = deepcat['RA'].values * u.deg, dec = deepcat['DEC'].values * u.deg)
    idx, d2d, d3d = deep_cd.match_to_catalog_sky(spec_cd)
    good_matches = d2d < match_radius * u.arcsec
    
    idx  = idx[good_matches]
    
    uniq_id, uniq_counts = np.unique(idx, return_counts = True)
    mask = np.isin(idx, uniq_id[uniq_counts == 1])
    
    idx  = idx[mask]
    good_matches[good_matches] = mask #Look at good entries of mask and additionally mask them as needed.
    
    print(f"IN {source_name}, FOUND {np.sum(~mask)} AMBIGUOUS MATCHES AND {np.sum(good_matches)} GOOD MATCHES") 
    
    
    # Add spectra to DELVE catalog
    output = deepcat.copy()
    output.loc[good_matches, 'Z']      = zcat.iloc[idx, zcat.columns.get_loc(z_col)].values
    output.loc[good_matches, 'SOURCE'] = source_name
    
    output.loc[good_matches, 'Z_%s' % source_name] = zcat.iloc[idx, zcat.columns.get_loc(z_col)].values
    
    return output


if __name__ == "__main__":
    

    specfile   = '/project/chihway/dhayaa/DECADE/Redshift_files/raw/BRPORTAL.E_6315_18670.csv'
    paufile    = '/project/chihway/dhayaa/DECADE/Redshift_files/raw/PAU_COSMOS_photoz_catalog_PZ_PIC_v0.4.csv.bz2'
    cosmosfile = '/project/chihway/dhayaa/DECADE/Redshift_files/raw/COSMOS2020_CLASSIC_R1_v2.1_p3.fits.gz'

    spec = pd.read_csv(specfile, low_memory = False); print("FINISHED LOADING SPECS")

    #C3R2 loaded separately due to format of available files
    c3r2_lines = open('/project/chihway/dhayaa/DECADE/Redshift_files/raw/c3r2_DR1+DR2_2019april11.txt').readlines()
    c3r2 = []
    for line in c3r2_lines[40:]:
        line = np.array(line.split(' '))
        line = line[line!='']
        line = line[:-1]
        ra = (360/24.)*(float(line[1]) + float(line[2])/60. + float(line[3])/3600.)
        dec = float(line[4]) + float(line[5])/60. + float(line[6])/3600.
        imag, zspec, qf = float(line[9]), float(line[10]), float(line[11])
        instr = line[12]
        c3r2.append([ra, dec, zspec])

    c3r2_lines = open('/project/chihway/dhayaa/DECADE/Redshift_files/raw/C3R2-DR3-18june2021.txt').readlines()
    for line in c3r2_lines[36:]:
        line = np.array(line.split(' '))
        line = line[line!='']
        line = line[:-1]
        ra = (360/24.)*(float(line[1]) + float(line[2])/60. + float(line[3])/3600.)
        dec = float(line[4]) + float(line[5])/60. + float(line[6])/3600.
        imag, zspec, qf = float(line[9]), float(line[10]), float(line[11])
        instr = line[12]
        c3r2.append([ra, dec, zspec])

    c3r2 = pd.DataFrame(c3r2, columns = ['ra', 'dec', 'Z']); print("FINISHED LOADING C3R2")

    
    #Cosmos also loaded separately
    cosmos = fitsio.read(cosmosfile, columns = ['FLAG_COMBINED', 'lp_type', 'lp_mask', 'ID', 'ez_z_phot',  'lp_zPDF', 'alpha_j2000', 'delta_j2000'])
    cosmos = cosmos[cosmos['FLAG_COMBINED'] == 0]
    #cosmos = cosmos[cosmos['lp_type'] == 0]
    #cosmos = cosmos[cosmos['lp_mask'] == 0]
    cosmos = pd.DataFrame(cosmos); print("FINISHED LOADING COSMOS")
    cosmos['ez_z_phot'] = cosmos['ez_z_phot'].values.byteswap().newbyteorder()
    
    #Finally load PAUS+COSMOS separately
    paus = pd.read_csv(paufile, compression = 'bz2', comment = '#', sep=",", na_values=r'\N'); print("FINISHED LOADING PAUS")
    
    #Now the deepfield catalog
    deep = pd.read_csv('/project/chihway/dhayaa/DECADE/Imsim_Inputs/deepfield_Y3_allfields.csv', index_col = 0); print("FINISHED LOADING Y3 DEEP")
    
    #################################################
    #Now do all the matches
    #################################################
    
    deep = match_zcat_to_deepcat(cosmos, "ez_z_phot", "ALPHA_J2000", "DELTA_J2000", "COSMOS2020", deep); print("FINISHED MATCHING COSMOS")
    deep = match_zcat_to_deepcat(paus, "photoz", "ra", "dec", "PAUS+COSMOS", deep); print("FINISHED MATCHING PAUS")
    for survey in ['VVDS', 'VIPERS', 'ZCOSMOS']:
        deep = match_zcat_to_deepcat(spec[spec['SOURCE'] == survey], "Z", "RA", "DEC", survey, deep); print("FINISHED MATCHING %s" % survey)
    deep = match_zcat_to_deepcat(c3r2, "Z", "ra", "dec", "C3R2", deep); print("FINISHED MATCHING C3R2")
    
    #Write out file
    deep.to_csv('/project/chihway/dhayaa/DECADE/Redshift_files/deepfields_raw_with_redshifts_20240723.csv.gz', index = False)
    
    
    #################################################
    #Now estimate the redshift bias
    #################################################
    
    Nbootstrap = 1000
    rng = np.random.default_rng(seed = 42)
    
    #I match all spec-zs to DF here, just for the median bias estimation.
    #This is okay because those "bad" spec-zs only have weird selections, not wrong specs.
    #So we can use still them to calibrate biases.
    deep = match_zcat_to_deepcat(spec[np.isin(spec.SOURCE.values, ['C3R2', 'ZCOSMOS', 'ZCOSMOS_DEEP'])], "Z", "RA", "DEC", "SPECZ", deep)
    deep = match_zcat_to_deepcat(c3r2, "Z", "ra", "dec", "C3R2", deep)
    
    i_bins = np.arange(17, 27, 0.5)
    i_bcen = (i_bins[1:] + i_bins[:-1])/2
    
    mask    = ((deep['SOURCE'].values != 'COSMOS2020') & 
               (deep['SOURCE'].values != 'PAUS+COSMOS') &
               (np.invert(deep['SOURCE'].isna()))
              )
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mag_i   = 30 - 2.5*np.log10(deep['BDF_FLUX_DERED_CALIB_I'].values)[mask]
    
    ##########################
    #Cosmos    
    ##########################
    
    delta_z = (deep['Z_COSMOS2020'].values - deep['Z'].values)/(1 + deep['Z'].values)
    delta_z = delta_z[mask]
    f_mask  = np.isfinite(delta_z)
    delta_z = delta_z[f_mask]
    
    bias    = np.zeros([Nbootstrap, i_bins.size - 1])
    for n in range(Nbootstrap):
        inds  = rng.choice(np.arange(delta_z.size), size = delta_z.size, replace = True)
        tmp_z = delta_z[inds]
        tmp_i = mag_i[f_mask][inds]
        for i in range(i_bins.size - 1):
            bias[n, i] = np.nanmedian(tmp_z[(tmp_i > i_bins[i]) & (tmp_i < i_bins[i+1])])
    np.savetxt('/project/chihway/dhayaa/DECADE/Redshift_files/median_bias_Cosmos.txt', 
               np.concatenate([i_bcen[:, None], bias.T], axis = 1))
    
    
    ##########################
    #Paus+Cosmos
    ##########################
    
    delta_z = (deep['Z_PAUS+COSMOS'].values - deep['Z'].values)/(1 + deep['Z'].values)
    delta_z = delta_z[mask]
    f_mask  = np.isfinite(delta_z)
    delta_z = delta_z[f_mask]
    
    bias    = np.zeros([Nbootstrap, i_bins.size - 1])
    for n in range(Nbootstrap):
        inds  = rng.choice(delta_z.size, size = delta_z.size, replace = True)
        tmp_z = delta_z[inds]
        tmp_i = mag_i[f_mask][inds]
        for i in range(i_bins.size - 1):
            bias[n, i] = np.nanmedian(tmp_z[(tmp_i > i_bins[i]) & (tmp_i < i_bins[i+1])])
    np.savetxt('/project/chihway/dhayaa/DECADE/Redshift_files/median_bias_PausCosmos.txt', 
               np.concatenate([i_bcen[:, None], bias.T], axis = 1))
    
    
    
