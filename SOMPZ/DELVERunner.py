import numpy as np, pandas as pd, h5py
from . import SOM
import argparse
import fitsio
import os
import joblib
import glob
from tqdm import tqdm

class TrainRunner:

    def __init__(self, seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path = None, Nth = 1):

        self.rng = np.random.default_rng(seed = seed)
        self.output_dir          = output_dir
        self.deep_catalog_path   = deep_catalog_path
        self.wide_catalog_path   = wide_catalog_path
        self.balrog_catalog_path = balrog_catalog_path #Needed if we subsample deep catalog based on Balrog
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.Nth = Nth
        
    def go(self):

        DEEP     = self.get_deep_fluxes(self.deep_catalog_path)
        DEEP_SOM = self.train(flux = DEEP[0][::self.Nth], flux_err = DEEP[1][::self.Nth])

        WIDE     = self.get_wide_fluxes(self.wide_catalog_path)
        inds     = self.rng.choice(len(WIDE[0]), np.min([len(WIDE[0]), 10_000_000]))
        WIDE_SOM = self.train(flux = WIDE[0][inds][::self.Nth], flux_err = WIDE[1][inds][::self.Nth])

        np.save(self.output_dir + '/DEEP_SOM.npy', DEEP_SOM)
        np.save(self.output_dir + '/WIDE_SOM.npy', WIDE_SOM)


    def get_deep_fluxes(self, path, balrog_path):

        #Deep field bands
        bands = 'ugrizJHKS'.upper()

        f = pd.read_csv(path)

        flux     = np.array([f['BDF_FLUX_DERED_CALIB_%s' % b].values for b in bands]).T
        flux_err = np.array([f['BDF_FLUX_ERR_DERED_CALIB_%s' % b].values for b in bands]).T
        ID       = f['ID'].values

        deep_was_detected = self.get_deep_mask(path, balrog_path)

        flux     = flux[deep_was_detected]
        flux_err = flux_err[deep_was_detected]
        ID       = ID[deep_was_detected]

        return flux, flux_err, ID
    
    def get_deep_mask(self, path, balrog_path):

        f  = pd.read_csv(path)
        ID = f['ID'].values

        balrog_gold = self.get_wl_sample_mask(balrog_path) & self.get_foreground_mask(balrog_path) & self.get_balrog_contam_mask(balrog_path)
        with h5py.File(balrog_path) as f:

            balrog_ID   = f['ID'][:]
            balrog_det  = f['detected'][:] == 1

        Mask = balrog_gold & balrog_det
        
        balrog_ID = np.unique(balrog_ID[Mask])
        
        deep_was_detected = np.isin(ID, balrog_ID)

        return deep_was_detected

    def get_wide_fluxes(self, path):

        Mask = self.get_wl_sample_mask(path) & self.get_foreground_mask(path) & self.get_balrog_contam_mask(path)
        with h5py.File(path) as f:

            #For Balrog, need both ID and tilename to get unique match
            ID = f['id'][:]
            tilename = f['tilename'][:]

            flux_r, flux_i, flux_z = f['mcal_flux_noshear'][:].T
            flux_err_r, flux_err_i, flux_err_z = f['mcal_flux_err_noshear'][:].T
            
            #These are the fluxes with all metacal cuts applied
            ID     = ID[Mask]
            flux_r = flux_r[Mask]
            flux_i = flux_i[Mask]
            flux_z = flux_z[Mask]
            flux_err_r = flux_err_r[Mask]
            flux_err_i = flux_err_i[Mask]
            flux_err_z = flux_err_z[Mask]

            flux     = np.hstack([flux_r, flux_i, flux_z]).T
            flux_err = np.hstack([flux_err_r, flux_err_i, flux_err_z]).T

        return flux, flux_err, ID, tilename
    

    def get_balrog_contam_mask(self, path):

        with h5py.File(path) as f:
            
            #Only select objects with no GOLD object within 1.5 arcsec
            if 'd_contam_arcsec' in f.keys():
                balrog_cont = f['d_contam_arcsec'][:] > 1.5 
            else:
                balrog_cont = np.ones_like(f['d_contam_arcsec'][:])

        return balrog_cont


    def get_wl_sample_mask(self, path, label = 'noshear'):

        with h5py.File(path) as f:
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
        
                flux_r, flux_i, flux_z = f[f'mcal_flux_{label}'][:].T

                mag_r = 30 - 2.5*np.log10(flux_r)
                mag_i = 30 - 2.5*np.log10(flux_i)
                mag_z = 30 - 2.5*np.log10(flux_z)

                mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                                (mag_r < 26)   & (mag_r > 15) & 
                                (mag_z < 26)   & (mag_z > 15) & 
                                (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                                (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

                SNR     = f[f'mcal_s2n_{label}'][:]
                T_ratio = f[f'mcal_T_ratio_{label}'][:]
                T       = f[f'mcal_T_{label}'][:]
                flags   = f['mcal_flags'][:]
                sg      = f['sg_bdf'][:] if 'sg_bdf' in f.keys() else np.ones_like(SNR)*99 #Need if/else because Balrog doesn't have sg_bdf
                g1, g2  = f[f'mcal_g_{label}'][:].T

                #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
                SNR_Mask   = (SNR > 10) & (SNR < 1000)
                Tratio_Mask= T_ratio > 0.5
                T_Mask     = T < 10
                Flag_Mask  = flags == 0
                SG_Mask = sg >= 4

                Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

                Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & SG_Mask

        return Mask
    

    def get_foreground_mask(self, path):

        with h5py.File(path) as f:
            FG_Mask = f['FLAGS_FOREGROUND'][:] == 0
            
            if 'ra' in f.keys():
                DR3_1_1_only = (f['ra'][:] < 180) & (f['dec'][:] > -25)
            elif 'RA' in f.keys():
                DR3_1_1_only = (f['RA'][:] < 180) & (f['DEC'][:] > -25)

        return FG_Mask & DR3_1_1_only


    def trainSOM(self, flux, flux_err):

        return SOM.TrainSOM(flux, flux_err)
    


class ClassifyRunner(TrainRunner):

    def initialize(self):

        DEEP = self.get_deep_fluxes(self.deep_catalog_path, self.balrog_catalog_path)
        WIDE = self.get_wide_fluxes(self.wide_catalog_path)
        BROG = self.get_wide_fluxes(self.balrog_catalog_path)

        for i, l in enumerate(['FLUX', 'FLUX_ERR', 'ID']):
            np.save(self.output_dir + '/DEEP_DATA_%s.npy' % l, DEEP[i])
            np.save(self.output_dir + '/WIDE_DATA_%s.npy' % l, WIDE[i])
            np.save(self.output_dir + '/BROG_DATA_%s.npy' % l, BROG[i])

        np.save(self.output_dir + '/BROG_DATA_TILENAME.npy', BROG[3])
    

    def classify(self, start, end, mode = 'DEEP'):

        SOM_weights = np.load(self.output_dir + '/%s_SOM.npy' % mode,           mmap_mode = 'r')[start:end]
        flux        = np.load(self.output_dir + '/%s_DATA_FLUX.npy' % mode,     mmap_mode = 'r')[start:end]
        flux_err    = np.load(self.output_dir + '/%s_DATA_FLUX_ERR.npy' % mode, mmap_mode = 'r')[start:end]

        Nproc = os.cpu_count()
        inds  = np.array_split(np.arange(start, end), Nproc)

        #Temp func to run joblib
        def _func_(i):
            cell_id_i = SOM.ClassifySOM(flux[inds[i], :], flux_err[inds[i], :], som_weights = SOM_weights)
            return i, cell_id_i

        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(_func_)(i) for i in range(Nproc)]
            outputs = joblib.Parallel(n_jobs = -1, verbose=10)(jobs)

            cell_id = np.zeros(flux.shape[0])
            for o in outputs: cell_id[inds[o[0]]] = o[1]

        return start, end, cell_id
    


    def classify_deep(self):

        #We do deep fields all at once, so let start/end be the full array
        CELL = self.classify(0, int(1e10), 'DEEP')
        np.save(self.output_dir + '/collated_deep_classifier.npy', CELL)

    
    def classify_wide(self, start, end):
        return self.classify(start, end, 'WIDE')

    def classify_balrog(self, start, end):
        return self.classify(start, end, 'BROG')


    def make_jobs(self, mode, folder_name = None):

        if folder_name is None:
            folder_name = mode.lower()

        #Make some dirs for storing the job files, log files, and output
        jobdir = self.output_dir + '/%s_jobs/' % mode; os.makedirs(jobdir)
        logdir = self.output_dir + '/%s_logs/' % mode; os.makedirs(logdir)
        outdir = self.output_dir + '/%s/' % folder_name; os.makedirs(outdir)
        

        text = """#!/bin/bash
#SBATCH --job-name %(NAME)s
#SBATCH --output=%(LOGDIR)s/%(NAME)s.log
#SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=2:00:00

source ${HOME}shear_bash_profile.sh

RUNNER = %(RUNNER_PATH)s

python -u ${RUNNER} --ClassifyRunner --%(MODE)s --start %(START)d --end %(END)d --outdir %(OUTDIR)s
        """


        args = {'LOGDIR' : logdir,
                'OUTDIR' : outdir,
                'RUNNER_PATH': __file__,
                'MODE': mode.upper()}

        Ngal = len(np.load(self.output_dir + '/%s_DATA_FLUX.npy' % mode.upper()))
        Ngal_per_job = 2_000_000

        i, N_i = 0, 0
        while N_i < Ngal:

            args['NAME']  = '%sclassify_%d_%d' % (mode.upper(), N_i, N_i + Ngal_per_job)
            args['START'] = N_i
            args['END']   = np.min([N_i + Ngal_per_job, Ngal])

            with open(jobdir + '/job_batch%03d.sh' % i, 'w') as f:

                f.write(text % args)

            i   += 1
            N_i += Ngal_per_job
            
        #Also make the script needed to submit all the other scripts.
        with open(jobdir + "/submit.sh", 'w') as f:
        
            text = """#!/bin/bash
for f in `ls %s`
do
    echo $f
    sbatch $f
done
            """ % (jobdir)
            f.write(text)
        

    def make_wide_jobs(self):
        self.make_jobs("WIDE", folder_name = "wide_classifier")

    def make_balrog_jobs(self):
        self.make_jobs("BALROG", folder_name = "balrog_classifier")



    def collate_jobs(self, folder_name):

        outdir = self.output_dir + '/%s/' % folder_name; os.makedirs(outdir)
        files  = sorted(glob.glob(outdir + "*.npy"))

        output = np.concatenate([np.load(f) for f in files])
        np.save(self.output_dir + '/collated_%s.npy' % folder_name, output)

    def collate_wide(self):
        self.collate_jobs(folder_name = "wide_classifier")

    def collate_balrog(self):
        self.collate_jobs(folder_name = "balrog_classifier")



class BinRunner(TrainRunner):    

    def make_bins(self, balrog_classified_df, z_bins = [0.0, 0.358, 0.631, 0.872, 2.0]):
        
        Mask = (self.get_wl_sample_mask(self.balrog_catalog_path) & 
                self.get_foreground_mask(self.balrog_catalog_path) & 
                self.get_balrog_contam_mask(self.balrog_catalog_path))

        with h5py.File(self.balrog_catalog_path, 'r') as f:
            
            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:][Mask]
            Balrog_df['Z']  = f['Z'][:][Mask]
            Balrog_df['id'] = f['id'][:][Mask]
            Balrog_df['tilename'] = f['tilename'][:][Mask]

#         assert len(Balrog_df) == len(balrog_classified_df), "Balrog dataframes %d != %d" %(len(Balrog_df), len(balrog_classified_df))

        #This treats "balrog_classified_df" as superior, meaning we only use Balrog galaxies that
        #were chosen to be classified. 
        Balrog_df = pd.merge(Balrog_df, balrog_classified_df, how = 'right', on = ['id', 'tilename', 'ID'])

        #For each wide cell, c_hat, find the distribution of deep galaxies in it
        SOMsize   = int(np.ceil(np.sqrt(Balrog_df['cell'].max())))
        Wide_bins = np.zeros(SOMsize * SOMsize, dtype = int).flatten()
        
        for i in tqdm(range(Wide_bins.size), desc = 'Assign chat to bins'):

            Balrog_in_this_cell = Balrog_df['cell'].values == i #Find all balrog injections in this cell

            if np.sum(Balrog_in_this_cell) == 0: #If this cell is empty
                Wide_bins[i] = -99
                continue

            else:
                redshift_in_this_cell = Balrog_df['Z'].values[Balrog_in_this_cell] #Deep/true Redshift of all gals in this cell
                bcounts_in_this_cell  = np.histogram(redshift_in_this_cell, z_bins)[0] #How many deep galaxy counts per z-bin
                Wide_bins[i]          = np.argmax(bcounts_in_this_cell) #Which bin is most populated by galaxies from this cell?

        
        return Wide_bins.reshape(SOMsize, SOMsize)

    
    def make_nz(self, z_bins, z_grid,
                balrog_classified_df, deep_classified_df, wide_classified_df):


        #-------------------------- READ OUT ALL DEEP QUANTITIES --------------------------
        f = pd.read_csv(self.deep_catalog_path).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        Deep_df = f[deep_was_detected]
        
        print(np.sum(deep_was_detected))

        #Check masking was ok. I'm paranoid about pandas masking sometimes
        assert len(f) == len(deep_was_detected), "Mask is not right size"
        assert len(Deep_df) == np.sum(deep_was_detected), "Masked df doesn't have right size"

        #Now check and merge with classifier
#         assert len(Deep_df) == len(deep_classified_df), "Balrog dataframes %d != %d" %(len(Deep_df), len(deep_classified_df))
        Deep_df = pd.merge(Deep_df, deep_classified_df[['cell', 'ID']], how = 'right', on = 'ID', suffixes = (None, '_classified'))


        #-------------------------- READ OUT ALL BALROG QUANTITIES --------------------------
        selection = self.get_wl_sample_mask(self.balrog_catalog_path) 
        purity    = self.get_balrog_contam_mask(self.balrog_catalog_path) & self.get_foreground_mask(self.balrog_catalog_path)
        dgamma = 2*0.01
        with h5py.File(self.balrog_catalog_path, 'r') as f:

            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:]
            Balrog_df['w']  = np.ones(len(Balrog_df)) #f['mcal_g_w'][:][Mask]
            R11       = (f['mcal_g_1p'][:][:, 0] - f['mcal_g_1m'][:][:, 0])/dgamma
            R22       = (f['mcal_g_2p'][:][:, 1] - f['mcal_g_2m'][:][:, 1])/dgamma
            Balrog_df['R']  = (R11 + R22)/2 #Average the two responses.
            Balrog_df['Z']  = f['Z'][:]

            Balrog_df['id']        = f['id'][:]
            Balrog_df['tilename']  = f['tilename'][:]
            Balrog_df['selection'] = selection & (f['detected'][:] == 1)
            Balrog_df['purity']    = purity

            Balrog_df = Balrog_df[Balrog_df['purity'] == True] #This needs to be happen at the very start
            Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'id', 'tilename', 'ID']], 
                                 how = 'left', on = ['id', 'tilename', 'ID'], suffixes = (None, '_classified'))

        
        #-------------------------- READ OUT ALL WIDE QUANTITIES --------------------------
        Mask = self.get_wl_sample_mask(self.wide_catalog_path) & self.get_foreground_mask(self.wide_catalog_path)
        with h5py.File(self.wide_catalog_path, 'r') as f:

            Wide_df = pd.DataFrame()
            Wide_df['id'] = f['id'][:][Mask]
            Wide_df['w']  = f['mcal_g_w'][:][Mask]
            R11 = (f['mcal_g_1p'][:][Mask, 0] - f['mcal_g_1m'][:][Mask, 0])/dgamma
            R22 = (f['mcal_g_2p'][:][Mask, 1] - f['mcal_g_2m'][:][Mask, 1])/dgamma
            Wide_df['R']  = (R11 + R22)/2 #Average the two responses.

#             assert len(Wide_df) == len(wide_classified_df), "Wide dataframes %d != %d" %(len(Wide_df), len(wide_classified_df))
            Wide_df = pd.merge(Wide_df, wide_classified_df[['cell', 'id']], how = 'right', on = 'id', suffixes = (None, '_classified'))
            
        
        #-------------------------- MAKE THE BINNING --------------------------

        Wide_bins     = self.make_bins(balrog_classified_df, z_bins)
        WIDESOMShape  = Wide_bins.shape
        Wide_bins     = Wide_bins.flatten()

        DEEPSOMsize   = int(np.ceil(np.sqrt(Deep_df['cell'].max())))
        
        print("DEEP SOM SIZE", DEEPSOMsize)
        print("WIDE SOM SIZE", WIDESOMShape)

        wide_weight   = Wide_df['w'].values * Wide_df['R'].values

        print("FRAC OF SAMPLE PER BIN")
        print(np.round(np.bincount(Wide_bins[Wide_df['cell'].values.astype(int)])/Wide_df['cell'].values.size, 2))
        
        #COMPUTE THE FIRST TERM: WIDE SOM OCCUPATION
        p_chat_given_shat_bhat = np.bincount(Wide_df['cell'].values.astype(int), weights = wide_weight, minlength = WIDESOMShape[0]**2)/np.sum(wide_weight)
        

        #COMPUTE THE SECOND TERM: REDSHIFT DIST. PER DEEP CELL
        Balrog_df_with_deep = pd.merge(Balrog_df, Deep_df[['ID',  'cell']], on = 'ID', suffixes = (None, '_deep'), how = 'left')
        Balrog_df_only_det  = Balrog_df_with_deep[Balrog_df_with_deep['selection'] == True]
        Balrog_total_injs   = Balrog_df[['ID', 'selection']].rename({'selection' : 'Ninj'}, axis = 1).groupby('ID').count()
        Balrog_total_dets   = Balrog_df[['ID', 'selection']].rename({'selection' : 'Ndet'}, axis = 1).groupby('ID').sum()
        Balrog_df_only_det  = pd.merge(Balrog_df_only_det, Balrog_total_injs, on = 'ID', how = 'left',)
        Balrog_df_only_det  = pd.merge(Balrog_df_only_det, Balrog_total_dets, on = 'ID', how = 'left',)
        weight              = (Balrog_df_only_det['w'].values * Balrog_df_only_det['R'].values)/Balrog_df_only_det['Ninj'].values
        p_z_given_c_bhat_shat  = np.zeros([len(z_bins) - 1, DEEPSOMsize * DEEPSOMsize, z_grid.size - 1])
        for b_i in tqdm(range(len(z_bins) - 1), desc = 'compute p_z_per_cell'):
            for c_i in np.arange(DEEPSOMsize * DEEPSOMsize):
                bhat_selection = Wide_bins[Balrog_df_only_det['cell'].values.astype(int)] == b_i
                c_selection    = Balrog_df_only_det['cell_deep'].values == c_i
                zs_selection   = np.invert(np.isnan(Balrog_df_only_det['Z'].values)) #Only select galaxies with redshifts this time alone
                selection      = bhat_selection & c_selection & zs_selection             

                p_z_given_c_bhat_shat[b_i, c_i, :] = np.histogram(Balrog_df_only_det['Z'].values[selection], z_grid, weights = weight[selection])[0]
                
                p_z_given_c_bhat_shat[b_i, c_i, :] /= np.sum(p_z_given_c_bhat_shat[b_i, c_i, :])
                


        #COMPUTE THE THIRD TERM: TRANSFER MATRIX
        p_c_chat_given_shat = np.zeros([Wide_bins.size, DEEPSOMsize**2])        
        p_chat_given_shat   = np.bincount(Balrog_df_only_det['cell'].values.astype(int), weights = weight, minlength = WIDESOMShape[0]*WIDESOMShape[1])
        for chat_i in tqdm(np.arange(Wide_bins.size), desc = 'compute transfer matrix'):
            wide_selection = Balrog_df_only_det['cell'].values.astype(int) == chat_i
            p_c_chat_given_shat[chat_i, :] = np.bincount(Balrog_df_only_det['cell_deep'].values.astype(int), weights = weight, minlength = DEEPSOMsize**2)

        p_c_given_chat_shat = p_c_chat_given_shat[:, None]/p_chat_given_shat


        #PUT IT TOGETHER TO GET THE n(z)
        p_z_given_bhat_shat = p_z_given_c_bhat_shat * p_c_given_chat_shat * p_chat_given_shat_bhat
        p_z_given_bhat_shat = np.sum(p_z_given_bhat_shat, axis = 0)

        return p_z_given_bhat_shat

    
    def postprocess_nz(self, z, list_of_nz):

        processed_nz = []
        for nz in list_of_nz:

            nz /= np.sum(nz)
            nz[np.argmin(np.abs(z - 3))] = np.sum(nz[z > 3])
            processed_nz.append(nz)

        return processed_nz


    

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--TrainRunner',    action = 'store_true', default = False, help = 'Run training module')
    my_parser.add_argument('--ClassifyRunner', action = 'store_true', default = False, help = 'Run classify module')
    my_parser.add_argument('--BinRunner',      action = 'store_true', default = False, help = 'Run binning/n(z) module')
    my_parser.add_argument('--DEEP',           action = 'store_true', default = False, help = 'Module operates on DEEP galaxies')
    my_parser.add_argument('--WIDE',           action = 'store_true', default = False, help = 'Module operates on WIDE galaxies')
    my_parser.add_argument('--BALROG',         action = 'store_true', default = False, help = 'Module operates on BALROG galaxies')

    my_parser.add_argument('--start',   action='store', type = int, default = 0,        help = 'first gal_index to process')
    my_parser.add_argument('--end',     action='store', type = int, default = int(1e9), help = 'last gal_index to process')

    args = vars(my_parser.parse_args())
    
    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')