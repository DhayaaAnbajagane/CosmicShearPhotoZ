from .SOM import TrainSOM, Classifier
from .Files import my_files
import numpy as np, pandas as pd, h5py, healpy as hp
from scipy import interpolate
import argparse
import os
import joblib
import glob
from tqdm import tqdm
import time


DEEP_COLS = ['ID', 'RA', 'DEC', 'KNN_CLASS', 'TILENAME',
             'FLAGS', 'FLAGSTR', 'FLAGSTR_NIR', 'FLAGS_NIR', 'MASK_FLAGS', 'MASK_FLAGS_NIR',
             'BDF_FLUX_DERED_CALIB_U', 'BDF_FLUX_DERED_CALIB_G', 'BDF_FLUX_DERED_CALIB_R',
             'BDF_FLUX_DERED_CALIB_I', 'BDF_FLUX_DERED_CALIB_Z', 'BDF_FLUX_DERED_CALIB_J', 
             'BDF_FLUX_DERED_CALIB_H', 'BDF_FLUX_DERED_CALIB_KS', 
             'BDF_FLUX_ERR_DERED_CALIB_U','BDF_FLUX_ERR_DERED_CALIB_G', 'BDF_FLUX_ERR_DERED_CALIB_R',
             'BDF_FLUX_ERR_DERED_CALIB_I', 'BDF_FLUX_ERR_DERED_CALIB_Z', 'BDF_FLUX_ERR_DERED_CALIB_J', 
             'BDF_FLUX_ERR_DERED_CALIB_H', 'BDF_FLUX_ERR_DERED_CALIB_KS', 
             ]

#For keeping track of how long steps take
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.5} seconds to run.")
        return result
    return wrapper



import tracemalloc

tracemalloc.start()
current, peak = tracemalloc.get_traced_memory()
def get_mem():
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")

    
class TrainRunner:

    def __init__(self, seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path = None):

        self.seed = seed
        self.output_dir          = output_dir
        self.deep_catalog_path   = deep_catalog_path
        self.wide_catalog_path   = wide_catalog_path
        self.balrog_catalog_path = balrog_catalog_path #Needed if we subsample deep catalog based on Balrog
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
    @timeit
    def go(self):

        self.train_deep()
        self.train_wide()
    
    
    @timeit
    def train_deep(self):
        
        rng      = np.random.default_rng(self.seed)
        DEEP     = self.get_deep_fluxes(self.deep_catalog_path, self.balrog_catalog_path)
        INDS     = rng.choice(len(DEEP[0]), size = len(DEEP[0]), replace = False) #Scramble during training
        DEEP_SOM = self.trainSOM(flux = DEEP[0][INDS], flux_err = DEEP[1][INDS], Ncells = 48)
        print("FINISHED TRAINING DEEP SOM")

        np.save(self.output_dir + '/DEEP_SOM_weights.npy', DEEP_SOM)
        
        
    @timeit
    def train_wide(self):
        
        rng      = np.random.default_rng(self.seed)
        WIDE     = self.get_wide_fluxes(self.wide_catalog_path)
        inds     = rng.choice(len(WIDE[0]), np.min([len(WIDE[0]), 2_000_000]))
        WIDE_SOM = self.trainSOM(flux = WIDE[0][inds], flux_err = WIDE[1][inds], Ncells = 32)
        print("FINISHED TRAINING WIDE SOM")

        np.save(self.output_dir + '/WIDE_SOM_weights.npy', WIDE_SOM)


    @timeit
    def get_deep_fluxes(self, path, balrog_path):

        #Deep field bands
        bands = [B.upper() for B in ['u', 'g', 'r', 'i', 'z', 'J', 'H', 'KS']]

        f = pd.read_csv(path, usecols = DEEP_COLS)

        flux     = np.array([f['BDF_FLUX_DERED_CALIB_%s' % b].values for b in bands]).T
        flux_err = np.array([f['BDF_FLUX_ERR_DERED_CALIB_%s' % b].values for b in bands]).T
        ID       = f['ID'].values
        tilename = f['TILENAME'].values

        deep_was_detected = self.get_deep_mask(path, balrog_path)
        deep_is_pure      = self.get_deep_sample_cuts(f)

        print("-----------------------")
        print("DEEP FIELD STATS")
        print("-----------------------")
        print("ORIGINAL: %d GALAXIES" % deep_was_detected.size)
        print("DETECTED: %d GALAXIES" % np.sum(deep_was_detected))
        print("PURE: %d GALAXIES" % np.sum(deep_is_pure))
        print("FINAL: %d GALAXIES" % np.sum(deep_was_detected & deep_is_pure))
        print("-----------------------\n\n")
        
        mask     = deep_was_detected & deep_is_pure
        flux     = flux[mask]
        flux_err = flux_err[mask]
        ID       = ID[mask]
        tilename = tilename[mask]

        return flux, flux_err, ID, tilename
    
    
    @timeit
    def get_deep_mask(self, path, balrog_path):

        f  = pd.read_csv(path, usecols = DEEP_COLS)
        ID = f['ID'].values
        
        #In Y3, some CCDs have bad chips. So we remove deepfield objects associated with those CCDs alone.
        BAD_CHIPS = ["SN-C3_C01", "SN-C3_C06", "SN-C3_C11", "SN-C3_C54", "SN-C3_C55", "SN-C3_C57", "SN-C3_C58", "SN-C3_C62",
             
                     "SN-X3_C10", "SN-X3_C12", "SN-X3_C15", "SN-X3_C19", "SN-X3_C29", "SN-X3_C46", 
                     "SN-X3_C47", "SN-X3_C49", "SN-X3_C52", "SN-X3_C60", "SN-X3_C62",

                     "SN-E2_C11", "SN-E2_C41", "SN-E2_C49",]
        
        deep_GOOD = np.invert(np.isin(f['TILENAME'].values, BAD_CHIPS))
        
        balrog_gold = (self.get_wl_sample_mask(balrog_path) & 
                       self.get_foreground_mask(balrog_path) & 
                       self.get_balrog_contam_mask(balrog_path))
        
        with h5py.File(balrog_path, 'r') as f:

            balrog_ID   = f['ID'][:]
            balrog_det  = f['detected'][:] == 1

        Mask = balrog_gold & balrog_det
        
        balrog_ID = np.unique(balrog_ID[Mask])
        
        deep_was_detected = np.isin(ID, balrog_ID)
        
        
        return deep_was_detected & deep_GOOD
    
    
    @timeit
    def get_deep_sample_cuts(self, deep_catalog):
        '''
        places color cuts on deep field catalog
        Credit: Alex Alarcon
        '''

        #Mask flagged regions -- not needed, saved deep catalog already has flag cuts in place
        mask  = deep_catalog.MASK_FLAGS_NIR==0
        mask &= deep_catalog.MASK_FLAGS==0
        mask &= deep_catalog.FLAGS_NIR==0
        mask &= deep_catalog.FLAGS==0
        
        #These two sometimes fail depending on the catalog and what format it writes
        #string into. Eitherway, this flag is equivalent to the ==0 flags above so
        #using those instead is better.
        #mask &= deep_catalog.FLAGSTR=="ok"
        #mask &= deep_catalog.FLAGSTR_NIR=="ok"
        
        
        deep_bands_ = ["U","G","R","I","Z","J","H","KS"]
        # remove crazy colors, defined as two 
        # consecutive colors (e.g u-g, g-r, r-i, etc) 
        # that have a value smaler than -1
        mags_d = np.zeros((len(deep_catalog),len(deep_bands_)))
        magerrs_d = np.zeros((len(deep_catalog),len(deep_bands_)))

        def flux2mag(flux):    
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return 30 - 2.5*np.log10(flux)
        
        for i,band in enumerate(deep_bands_):
            #print(i,band)
            mags_d[:,i] = flux2mag(deep_catalog['BDF_FLUX_DERED_CALIB_%s'%band].values)

        colors = np.zeros((len(deep_catalog),len(deep_bands_)-1))
        for i in range(len(deep_bands_)-1):
            colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

        normal_colors = np.all(colors > -1, axis=1)
        
        return mask & normal_colors

    
    @timeit
    def get_wide_fluxes(self, path, label = 'noshear'):

        #We check balrog_contam even in data, but the result is always TRUE if its data
        Mask = self.get_wl_sample_mask(path, label = label) & self.get_foreground_mask(path) & self.get_balrog_contam_mask(path)
        with h5py.File(path, 'r') as f:

            #For Balrog, need both ID and tilename to get unique match
            ID = f['id'][:]
            
            tilename = f['tilename'][:][Mask] if 'tilename' in f.keys() else None
            true_ra  = f['true_ra'][:][Mask]  if 'true_ra'  in f.keys() else None
            true_dec = f['true_dec'][:][Mask] if 'true_dec' in f.keys() else None

            #These are the fluxes with all metacal cuts applied
            ID       = ID[Mask]
            
            flux     = f[f'mcal_flux_{label}_dered_sfd98'][:][Mask]
            flux_err = f[f'mcal_flux_err_{label}_dered_sfd98'][:][Mask]

        return flux, flux_err, ID, tilename, true_ra, true_dec
    
    
    @timeit
    def get_balrog_contam_mask(self, path):

        with h5py.File(path, 'r') as f:
            
            #Only select objects with no GOLD object within 1.5 arcsec
            if 'd_contam_arcsec' in f.keys():
                balrog_cont = f['d_contam_arcsec'][:] > 1.5 
            else:
                balrog_cont = True
            
        return balrog_cont

    
    @timeit
    def get_wl_sample_mask(self, path, label = 'noshear'):

        with h5py.File(path, 'r') as f:
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
        
                flux_r, flux_i, flux_z = f[f'mcal_flux_{label}_dered_sfd98'][:].T
                
                mag_r = 30 - 2.5*np.log10(flux_r)
                mag_i = 30 - 2.5*np.log10(flux_i)
                mag_z = 30 - 2.5*np.log10(flux_z)

                mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                                (mag_r < 26)   & (mag_r > 15) & 
                                (mag_z < 26)   & (mag_z > 15) & 
                                (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                                (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

                del mag_i, mag_z
                
                SNR     = f[f'mcal_s2n_{label}'][:]
                T_ratio = f[f'mcal_T_ratio_{label}'][:]
                T       = f[f'mcal_T_{label}'][:]
                flags   = f['mcal_flags'][:]
                g1, g2  = f[f'mcal_g_{label}'][:].T

                #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
                Tratio_Mask= T_ratio > 0.5; del T_ratio
                Flag_Mask  = flags == 0; del flags
                SNR_Mask   = (SNR > 10) & (SNR < 1000)
                T_Mask     = T < 10
                
                Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

                del g1, g2, mag_r, T
                
                Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask

        return Mask
    
    
    @timeit
    def get_foreground_mask(self, path):

        Badcolor_map = hp.read_map('/project2/kadrlica/chinyi/DELVE_DR3_1_bad_colour_mask.fits', dtype = int)
        
        with h5py.File(path, 'r') as f:
            FG_mask = f['FLAGS_FOREGROUND'][:] == 0
            
            
            if 'RA' in f.keys():
                Region_mask = np.invert(f['DEC'][:] > np.where(f['RA'][:] < 225, 30 - (30 - 12)/(225 - 200) * (f['RA'][:] - 200), 12.))
                pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['RA'][:], f['DEC'][:], lonlat = True)
            else:
                Region_mask = np.invert(f['true_dec'][:] > np.where(f['true_ra'][:] < 225, 30 - (30 - 12)/(225 - 200) * (f['true_ra'][:] - 200), 12.))
                pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['true_ra'][:], f['true_dec'][:], lonlat = True)
                
            Color_mask = Badcolor_map[pix_assign] == 0; del pix_assign
        
        Mask = FG_mask & Region_mask & Color_mask; del Badcolor_map, FG_mask, Region_mask, Color_mask
        
        return Mask


    def trainSOM(self, flux, flux_err, Ncells):

        return TrainSOM(flux, flux_err, Ncells)
    


class ClassifyRunner(TrainRunner):

    def __init__(self, seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path, njobs = 10):
        
        self.njobs = njobs
        
        super().__init__(seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path)
        
    
    @timeit
    def initialize(self):

        DEEP = self.get_deep_fluxes(self.deep_catalog_path, self.balrog_catalog_path)
        WIDE = self.get_wide_fluxes(self.wide_catalog_path)
        BROG = self.get_wide_fluxes(self.balrog_catalog_path)

        for i, l in enumerate(['FLUX', 'FLUX_ERR', 'ID']):
            np.save(self.output_dir + '/DEEP_DATA_%s.npy' % l, DEEP[i])
            np.save(self.output_dir + '/WIDE_DATA_%s.npy' % l, WIDE[i])
            np.save(self.output_dir + '/BALROG_DATA_%s.npy' % l, BROG[i])

        np.save(self.output_dir + '/DEEP_DATA_TILENAME.npy', DEEP[3])
        
        np.save(self.output_dir + '/BALROG_DATA_TILENAME.npy', BROG[3])
        np.save(self.output_dir + '/BALROG_DATA_TRUE_RA.npy',  BROG[4])
        np.save(self.output_dir + '/BALROG_DATA_TRUE_DEC.npy', BROG[5])
    

    def classify(self, start, end, mode = 'DEEP'):

        if mode == 'BALROG':
            SOM_weights = np.load(self.output_dir + '/WIDE_SOM_weights.npy', mmap_mode = 'r')
        else:
            SOM_weights = np.load(self.output_dir + '/%s_SOM_weights.npy' % mode, mmap_mode = 'r')
        
        flux     = np.load(self.output_dir + '/%s_DATA_FLUX.npy' % mode,     mmap_mode = 'r')[start:end]
        flux_err = np.load(self.output_dir + '/%s_DATA_FLUX_ERR.npy' % mode, mmap_mode = 'r')[start:end]

        end   = np.min([end, flux.shape[0]])
        Nproc = np.max([os.cpu_count(), flux.shape[0]//1_000_000 + 1])
        inds  = np.array_split(np.arange(start, end), Nproc)

        print("RUNNING WITH NJOBS:", Nproc)
        som   = Classifier(som_weights = SOM_weights) #Build classifier first, as its faster
        
        #Temp func to run joblib
        def _func_(i):
            cell_id_i = som.classify(flux[inds[i], :], flux_err[inds[i], :])
            return i, cell_id_i

        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(_func_)(i) for i in range(Nproc)]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10,)(jobs)

            cell_id = np.zeros(flux.shape[0])
            for o in outputs: cell_id[inds[o[0]]] = o[1][0]

        return start, end, cell_id
    

    @timeit
    def classify_deep(self):

        #We do deep fields all at once, so let start/end be the full array
        CELL = self.classify(0, int(1e10), 'DEEP')
        np.save(self.output_dir + '/collated_deep_classifier.npy', CELL[-1])

        
    @timeit
    def classify_wide(self):
        CELL = self.classify(0, int(1e10), 'WIDE')
        np.save(self.output_dir + '/collated_wide_classifier.npy', CELL[-1])
        

    @timeit
    def classify_balrog(self):
        CELL = self.classify(0, int(1e10), 'BALROG')
        np.save(self.output_dir + '/collated_balrog_classifier.npy', CELL[-1])
    


class ClassifyAllMcalRunner(ClassifyRunner):
    
    @timeit
    def initialize(self, label = 'noshear'):

        WIDE = self.get_wide_fluxes(self.wide_catalog_path, label = label)

        for i, l in enumerate(['FLUX', 'FLUX_ERR', 'ID']):
            np.save(self.output_dir + f'/WIDE_DATA_{l}_{label}.npy', WIDE[i])
            

    def classify(self, start, end, label = 'noshear'):

        SOM_weights = np.load(self.output_dir + '/WIDE_SOM_weights.npy', mmap_mode = 'r')
        
        flux     = np.load(self.output_dir + f'/WIDE_DATA_FLUX_{label}.npy',     mmap_mode = 'r')[start:end]
        flux_err = np.load(self.output_dir + f'/WIDE_DATA_FLUX_ERR_{label}.npy', mmap_mode = 'r')[start:end]

        end   = np.min([end, flux.shape[0]])
        Nproc = np.max([os.cpu_count(), flux.shape[0]//1_000_000 + 1])
        inds  = np.array_split(np.arange(start, end), Nproc)

        print("RUNNING WITH NJOBS:", Nproc)
        som   = Classifier(som_weights = SOM_weights) #Build classifier first, as its faster
        
        #Temp func to run joblib
        def _func_(i):
            cell_id_i = som.classify(flux[inds[i], :], flux_err[inds[i], :])
            return i, cell_id_i

        with joblib.parallel_backend("loky"):
            jobs    = [joblib.delayed(_func_)(i) for i in range(Nproc)]
            outputs = joblib.Parallel(n_jobs = self.njobs, verbose=10,)(jobs)

            cell_id = np.zeros(flux.shape[0])
            for o in outputs: cell_id[inds[o[0]]] = o[1][0]

        return start, end, cell_id
    

    @timeit
    def classify_wide(self, label = 'noshear'):
        
        CELL = self.classify(0, int(1e10), label)
        np.save(self.output_dir + f'/collated_wide_{label}_classifier.npy', CELL[-1])
        
        
class BinRunner(ClassifyRunner):
    
    def __init__(self, seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path, 
                 tomo_redshift_catalog_path, redshift_catalog_path, njobs = 10):
        
        self.redshift_catalog_path = redshift_catalog_path
        self.tomo_redshift_catalog_path = tomo_redshift_catalog_path
        
        super().__init__(seed, output_dir, deep_catalog_path, wide_catalog_path, balrog_catalog_path, njobs)

    @timeit
    def get_shear_weights(self, S2N, T_over_Tpsf):
        
        path = os.path.dirname(__file__) + '/../grid_quantities_20240827.npy'
        res  = np.load(path)
        S    = res['SNR']
        T    = res['T_ratio']
        R    = (res['R11'] + res['R22'])/2 #Average over both components. No selection response, as in Y3
        w    = res['weight']

        # #The old version, using weights that had R_gamma and R_s of the grid.
        # weight_path = os.path.dirname(__file__) + '/../weights_20240209.npy'
        # X = np.load(weight_path, allow_pickle = True)[()]
        
        # S = X['s2n'].flatten()
        # T = X['T_over_Tpsf'].flatten()
        # R = X['R'].flatten()
        # w = X['w'].flatten()
        
        #Have checked that this what DESY3 uses.
        interp        = interpolate.NearestNDInterpolator((S, T), R * w,)
        shear_weights = interp( (S2N, T_over_Tpsf) )
        
        return shear_weights
        
        
    @timeit
    def make_bins_z_bins(self, balrog_classified_df,  z_bins = [0.0, 0.3767, 0.6343, 0.860, 2.0]):
        
        TMPDIR = os.environ['TMPDIR']
        
        Mask = (self.get_wl_sample_mask(self.balrog_catalog_path) & 
                self.get_foreground_mask(self.balrog_catalog_path) & 
                self.get_balrog_contam_mask(self.balrog_catalog_path))

        with h5py.File(self.balrog_catalog_path, 'r') as f:
            
            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:][Mask] #This is deepfield galaxy ID
            Balrog_df['id'] = f['id'][:][Mask]
            Balrog_df['tilename'] = f['tilename'][:][Mask]
            Balrog_df['true_ra']  = f['true_ra'][:][Mask].astype('float64')
            Balrog_df['true_dec'] = f['true_dec'][:][Mask].astype('float64')
            
        
        deep = pd.read_csv(self.deep_catalog_path, usecols = DEEP_COLS)
        Cuts = deep[self.get_deep_sample_cuts(deep)]
        
        print("DEEP", len(deep), "HAS LEN", len(Cuts))
        
        Balrog_df = Balrog_df[np.isin(Balrog_df['ID'], Cuts['ID'])]
        Balrog_df = pd.merge(Balrog_df, pd.read_csv(self.tomo_redshift_catalog_path)[['ID', 'Z']], on = "ID", how = 'left')
        
        #This treats "Balrog_df" as superior, all galaxies in left_df must be classified and available
        #The only difference is left DF only has Balrog galaxies from deep field objects that are "good"
        #whereas right DF didn't do this cut
        Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'true_ra', 'true_dec']], how = 'left', on = ['true_ra', 'true_dec'], validate = "1:1")
        
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
    
    
    @timeit
    def make_bins(self, balrog_classified_df, wide_classified_df, z_bins = [0.0, 0.3767, 0.6343, 0.860, 2.0]):
        
        print("NOT USING Z_BINS")
        
        TMPDIR = os.environ['TMPDIR']
        
        Mask = (self.get_wl_sample_mask(self.balrog_catalog_path) & 
                self.get_foreground_mask(self.balrog_catalog_path) & 
                self.get_balrog_contam_mask(self.balrog_catalog_path))

        p_chat = np.bincount(wide_classified_df['cell'].values.astype(int), minlength = 32**2)/len(wide_classified_df['cell'].values)
        
        with h5py.File(self.balrog_catalog_path, 'r') as f:
            
            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:][Mask] #This is deepfield galaxy ID
            Balrog_df['id'] = f['id'][:][Mask]
            Balrog_df['tilename'] = f['tilename'][:][Mask]
            Balrog_df['true_ra']  = f['true_ra'][:][Mask].astype('float64')
            Balrog_df['true_dec'] = f['true_dec'][:][Mask].astype('float64')
        
        
        deep = pd.read_csv(self.deep_catalog_path, usecols = DEEP_COLS)
        Cuts = deep[self.get_deep_sample_cuts(deep)]# = pd.read_csv('/project2/chihway/raulteixeira/data/deepfields_clean.csv.gz')
        
        print("DEEP", len(deep), "HAS LEN", len(Cuts))
        
        Balrog_df = Balrog_df[np.isin(Balrog_df['ID'], Cuts['ID'])]
        Balrog_df = pd.merge(Balrog_df, pd.read_csv(self.tomo_redshift_catalog_path)[['ID', 'Z']], on = "ID", how = 'left')
        
        #This treats "Balrog_df" as superior, all galaxies in left_df must be classified and available
        #The only difference is left DF only has Balrog galaxies from deep field objects that are "good"
        #whereas right DF didn't do this cut
        Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'true_ra', 'true_dec']], how = 'left', on = ['true_ra', 'true_dec'], validate = "1:1")
        
        #For each wide cell, c_hat, find the distribution of deep galaxies in it
        SOMsize   = int(np.ceil(np.sqrt(Balrog_df['cell'].max())))
        Wide_bins = np.zeros(SOMsize * SOMsize, dtype = int)
        median_z  = np.zeros(SOMsize * SOMsize, dtype = float)
        
        for i in tqdm(range(Wide_bins.size), desc = 'Compute mean-z per cell'):

            Balrog_in_this_cell   = Balrog_df['cell'].values == i #Find all balrog injections in this cell
            redshift_in_this_cell = Balrog_df['Z'].values[Balrog_in_this_cell] #Deep/true Redshift of all gals in this cell
            median_z[i]           = np.nanmedian(redshift_in_this_cell) #Which bin is most populated by galaxies from this cell?
            
            if np.sum(Balrog_in_this_cell) == 0: #If this cell is empty
                median_z[i] = 99.
                continue
            
        
        b_i = 0
        tot_prob = 0
        
        print("<z>:", median_z)
        inds = np.argsort(median_z)
        for i in tqdm(inds, desc = 'Assign cell to bins'):
            
            tot_prob += p_chat[i]
            
            if tot_prob <= 0.25:
                Wide_bins[i] = b_i
            else:
                print("FILLED UP BIN", b_i, "with max(z) = %0.4f" % median_z[i])
                b_i += 1
                tot_prob = 0
            
        return Wide_bins.reshape(SOMsize, SOMsize)

    
    @timeit
    def make_nz(self, z_bins, z_grid,
                balrog_classified_df, deep_classified_df, wide_classified_df):
        
        TMPDIR = os.environ['TMPDIR']
        

        MY_RESULT = {}
        
        #-------------------------- READ OUT ALL DEEP QUANTITIES --------------------------
        f = pd.read_csv(self.deep_catalog_path, usecols = DEEP_COLS).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        deep_sample_cuts  = self.get_deep_sample_cuts(f)
        Deep_df = f[deep_was_detected & deep_sample_cuts]
        
        #Check masking was ok. I'm paranoid about pandas masking sometimes
        assert len(f) == len(deep_was_detected & deep_sample_cuts), "Mask is not right size"
        assert len(Deep_df) == np.sum(deep_was_detected & deep_sample_cuts), "Masked df doesn't have right size"
        
        #Now check and merge with classifier
        assert len(Deep_df) == len(deep_classified_df), "Deep dataframes sizes %d != %d" %(len(Deep_df), len(deep_classified_df))
        Deep_df = pd.merge(Deep_df, deep_classified_df[['cell', 'ID']], how = 'left', on = 'ID', suffixes = (None, '_classified'))

        print("LOADED DEEP FIELD")
        
        
        #-------------------------- READ OUT ALL BALROG QUANTITIES --------------------------
        selection = self.get_wl_sample_mask(self.balrog_catalog_path)
        purity    = self.get_balrog_contam_mask(self.balrog_catalog_path) & self.get_foreground_mask(self.balrog_catalog_path)
        
        with h5py.File(self.balrog_catalog_path, 'r') as f:

            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:]
            Balrog_df['w']  = self.get_shear_weights(f['mcal_s2n_noshear'][:], f['mcal_T_ratio_noshear'][:])
            
            Balrog_df['id']        = f['id'][:]
            Balrog_df['tilename']  = f['tilename'][:]
            Balrog_df['selection'] = selection & (f['detected'][:] == 1)
            Balrog_df['purity']    = purity
            Balrog_df['true_ra']   = f['true_ra'][:]
            Balrog_df['true_dec']  = f['true_dec'][:]

            #Only keep Balrog objects that come from good DF objects
            #And only objects that have no contamination from real objects/sources
            Balrog_df = Balrog_df[np.isin(Balrog_df['ID'], Deep_df['ID'])]
            Balrog_df = Balrog_df[Balrog_df['purity'] == True]
            
            #Add other columns
            Balrog_df = pd.merge(Balrog_df, pd.read_csv(self.redshift_catalog_path)[['ID', 'Z']], on = "ID", how = 'left')
            Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'true_ra', 'true_dec']], 
                                 how = 'left', on = ['true_ra', 'true_dec'], suffixes = (None, '_classified'), 
                                 validate = "1:1")
        
        print("LOADED BROG FIELD")
        
        #-------------------------- READ OUT ALL WIDE QUANTITIES --------------------------
        
        if os.path.isfile(TMPDIR + '/wide_mask.npy'):
            Mask = np.load(TMPDIR + '/wide_mask.npy')
        else:
            Mask = self.get_wl_sample_mask(self.wide_catalog_path) & self.get_foreground_mask(self.wide_catalog_path)
            np.save(TMPDIR + '/wide_mask.npy', Mask)
            
        with h5py.File(self.wide_catalog_path, 'r') as f:

            Wide_df = pd.DataFrame()
            Wide_df['id'] = f['id'][:][Mask]
            Wide_df['w']  = self.get_shear_weights(f['mcal_s2n_noshear'][:][Mask], f['mcal_T_ratio_noshear'][:][Mask])

            assert len(Wide_df) == len(wide_classified_df), "Wide dataframes %d != %d" %(len(Wide_df), len(wide_classified_df))
            Wide_df = pd.merge(Wide_df, wide_classified_df[['cell', 'id']], 
                               how = 'left', on = 'id', suffixes = (None, '_classified'), validate = "1:1")
            
        print("LOADED WIDE FIELD")
        
        #-------------------------- MAKE THE BINNING --------------------------

        Wide_bins     = self.make_bins(balrog_classified_df, wide_classified_df, z_bins)
        WIDESOMShape  = Wide_bins.shape
        Wide_bins     = Wide_bins.flatten()

        MY_RESULT['Wide_bins'] = Wide_bins
        
        print("MADE BINS")
        
        DEEPSOMsize   = int(np.ceil(np.sqrt(np.max(Deep_df['cell'].values))))
        
        print("DEEP SOM SIZE", DEEPSOMsize)
        print("WIDE SOM SIZE", WIDESOMShape)

        wide_weight   = Wide_df['w'].values #This already has response in it, so it is weight = R*w
        
        print(wide_weight)

        print("FRAC OF SAMPLE PER BIN")
        print(np.round(np.bincount(Wide_bins[Wide_df['cell'].values.astype(int)])/Wide_df['cell'].values.size, 10))
        
        #COMPUTE THE FIRST TERM: WIDE SOM OCCUPATION
        p_chat_given_shat = np.bincount(Wide_df['cell'].values.astype(int), weights = wide_weight, 
                                        minlength = WIDESOMShape[0]**2)/np.sum(wide_weight)
        
        MY_RESULT['p_chat_given_shat'] = p_chat_given_shat
        
        MY_RESULT['p_c_given_shat'] = np.bincount(Deep_df['cell'].values.astype(int), minlength = DEEPSOMsize**2)/len(Deep_df)
        
        
        #COMPUTE THE SECOND TERM: REDSHIFT DIST. PER DEEP CELL
        Balrog_df_with_deep = pd.merge(Balrog_df, Deep_df[['ID',  'cell']], on = 'ID', suffixes = (None, '_deep'), how = 'left')
        Balrog_df_only_det  = Balrog_df_with_deep[Balrog_df_with_deep['selection'] == True]
        
        Balrog_total_injs   = Balrog_df[['ID', 'selection']].rename({'selection' : 'Ninj'}, axis = 1).groupby('ID').count()
        Balrog_total_dets   = Balrog_df[['ID', 'selection']].rename({'selection' : 'Ndet'}, axis = 1).groupby('ID').sum()
        
        Balrog_df_only_det  = pd.merge(Balrog_df_only_det, Balrog_total_injs, on = 'ID', how = 'left',)
        Balrog_df_only_det  = pd.merge(Balrog_df_only_det, Balrog_total_dets, on = 'ID', how = 'left',)
        
        weight              = Balrog_df_only_det['w'].values / Balrog_df_only_det['Ninj'].values #w already includes response
        
        p_z_given_c_bhat_shat  = np.zeros([len(z_bins) - 1, DEEPSOMsize * DEEPSOMsize, z_grid.size - 1])
        for b_i in tqdm(range(len(z_bins) - 1), desc = 'compute p_z_per_cell'):
            for c_i in np.arange(DEEPSOMsize * DEEPSOMsize):
                
                bhat_selection = Wide_bins[Balrog_df_only_det['cell'].values.astype(int)] == b_i
                c_selection    = Balrog_df_only_det['cell_deep'].values == c_i
                zs_selection   = np.invert(np.isnan(Balrog_df_only_det['Z'].values)) #Only select galaxies with redshifts this time alone
                selection      = bhat_selection & c_selection & zs_selection       

                p_z_given_c_bhat_shat[b_i, c_i, :] = np.histogram(Balrog_df_only_det['Z'].values[selection], z_grid, weights = weight[selection])[0]
                
                Norm = np.sum(p_z_given_c_bhat_shat[b_i, c_i, :])
                assert Norm >= 0, "Norm can't be negative"
                
                if Norm == 0: continue
                
                p_z_given_c_bhat_shat[b_i, c_i, :] /= np.sum(p_z_given_c_bhat_shat[b_i, c_i, :])
                
        MY_RESULT['p_z_given_c_bhat_shat'] = p_z_given_c_bhat_shat

        #COMPUTE THE THIRD TERM: TRANSFER MATRIX
        p_chat_c_given_shat = np.zeros([Wide_bins.size, DEEPSOMsize**2])       
        weight              = Balrog_df_only_det['w'].values / Balrog_df_only_det['Ninj'].values
        
        
        ####COMPUTE PART 1 OF THIRD TERM
        p_chat_given_shat   = np.bincount(Balrog_df_only_det['cell'].values.astype(int), 
                                          weights = weight, minlength = WIDESOMShape[0]*WIDESOMShape[1])
        
        p_chat_given_shat /= np.sum(p_chat_given_shat)
        MY_RESULT['p_chat_given_shat'] = p_chat_given_shat
        
        ####COMPUTE PART 2 OF THIRD TERM
        for chat_i in tqdm(np.arange(Wide_bins.size), desc = 'compute transfer matrix'):
            wide_selection = Balrog_df_only_det['cell'].values.astype(int) == chat_i
            
            p_chat_c_given_shat[chat_i, :] = np.bincount(Balrog_df_only_det['cell_deep'].values[wide_selection].astype(int), 
                                                         weights = weight[wide_selection], minlength = DEEPSOMsize**2)

        p_chat_c_given_shat /= np.sum(p_chat_c_given_shat)
        MY_RESULT['p_chat_c_given_shat'] = p_chat_c_given_shat
        p_c_given_chat_shat = p_chat_c_given_shat/p_chat_given_shat[:, None]
        
        MY_RESULT['p_c_given_chat_shat'] = p_c_given_chat_shat

        Ratio = MY_RESULT['p_c_given_shat']/np.sum(MY_RESULT['p_c_given_chat_shat'], axis = 0)
        
        #COMPUTE THE CHAT ASSIGNMENT PROBABILITY
        p_chat_given_bhat = np.array([Wide_bins == i for i in range(len(z_bins) - 1)])
        MY_RESULT['p_chat_given_bhat'] = p_chat_given_bhat

        p_z_given_bhat_shat = np.sum(p_z_given_c_bhat_shat * #Ratio[None, :, None] *
                                     np.sum(p_c_given_chat_shat * 
                                            (p_chat_given_shat[None, :] * p_chat_given_bhat)[:, :, None], 
                                             axis = 1)[:, :, None], 
                                     axis = 1)
        
        MY_RESULT['p_z_given_bhat_shat'] = p_z_given_bhat_shat

        MY_RESULT['z_grid'] = z_grid
        MY_RESULT['z_cen']  = (z_grid[1:]  + z_grid[:-1])/2
        
        return p_z_given_bhat_shat, MY_RESULT

    
    def postprocess_nz(self, z, list_of_nz):

        processed_nz = []
        for nz in list_of_nz:

            #Ramping, in order to set p(z = 0) --> 0
            nz *= np.where(z <= 0.055, nz * z/0.055, 1)
            
            #Normalize everything
            nz /= np.sum(nz)
            
            #Now do pileup of everything beyond z > 3.
            nz[np.argmin(np.abs(z - 3))] = np.sum(nz[z > 3])
            nz[z > 3] = 0
            
            #save
            processed_nz.append(nz)

        return processed_nz


    

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--TrainRunner',    action = 'store_true', default = False, help = 'Run training module')
    my_parser.add_argument('--ClassifyRunner', action = 'store_true', default = False, help = 'Run classify module')
    my_parser.add_argument('--AllMcalRunner',  action = 'store_true', default = False, help = 'Run classify module on 5 mcal')
    my_parser.add_argument('--BinRunner',      action = 'store_true', default = False, help = 'Run binning/n(z) module')
    my_parser.add_argument('--DEEP',           action = 'store_true', default = False, help = 'Module operates on DEEP galaxies')
    my_parser.add_argument('--WIDE',           action = 'store_true', default = False, help = 'Module operates on WIDE galaxies')
    my_parser.add_argument('--BALROG',         action = 'store_true', default = False, help = 'Module operates on BALROG galaxies')
    my_parser.add_argument('--MCAL_TYPE',      action = 'store',      type    = str,   help = 'The type of mcal version to classify')

    my_parser.add_argument('--njobs',   action = 'store', type = int, default = 15,       help = 'Number of parallel threads')
    my_parser.add_argument('--start',   action = 'store', type = int, default = 0,        help = 'first gal_index to process')
    my_parser.add_argument('--end',     action = 'store', type = int, default = int(1e9), help = 'last gal_index to process')

    args = vars(my_parser.parse_args())
    
    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')
    
    
    my_params = {'seed': 42,
                 'njobs' : args['njobs'],
                 'output_dir' : '/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/', 
                 }
    
    my_params = my_params | my_files
    
    
    if args['TrainRunner']:
        tmp = {k: v for k, v in my_params.items() if k not in ['njobs', 'redshift_catalog_path', 'tomo_redshift_catalog_path']}
        ONE = TrainRunner(**tmp)
        if args['DEEP']: ONE.train_deep()
        if args['WIDE']: ONE.train_wide()
        
    
    if args['ClassifyRunner']:
        tmp = {k: v for k, v in my_params.items() if k not in ['redshift_catalog_path', 'tomo_redshift_catalog_path']}
        TWO = ClassifyRunner(**tmp)
        TWO.initialize()
        if args['DEEP']: TWO.classify_deep()
        if args['BALROG']: TWO.classify_balrog()
        if args['WIDE']: TWO.classify_wide()
            
    
    if args['AllMcalRunner']:
        tmp = {k: v for k, v in my_params.items() if k not in ['redshift_catalog_path', 'tomo_redshift_catalog_path']}
        TWO = ClassifyAllMcalRunner(**tmp)
        TWO.initialize(label = args['MCAL_TYPE'])
        TWO.classify_wide(label = args['MCAL_TYPE'])
        
        
    if args['BinRunner']:
        tmp   = {k: v for k, v in my_params.items() if k not in ['njobs']}
        THREE = BinRunner(**tmp)
        
        bclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/BALROG_DATA_ID.npy'),
                                'true_ra'  : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_RA.npy'),
                                'true_dec' : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_DEC.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_balrog_classifier.npy')})
        
        dclass  = pd.DataFrame({'ID'       : np.load(my_params['output_dir'] + '/DEEP_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_deep_classifier.npy')})
        
        wclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/WIDE_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_wide_classifier.npy')})
        
        z_bins  = [0.0, 0.3639, 0.6143, 0.8558, 2.0]
        z_grid  = np.arange(0, 5.01, 0.05)
        nz, RES = THREE.make_nz(z_bins, z_grid, bclass, dclass, wclass)
        nz      = THREE.postprocess_nz( (z_grid[1:] + z_grid[:-1])/2, nz) #Include normalization and pile-up.
        
        np.save(os.path.join(my_params['output_dir'], 'TomoBinAssign.npy'), RES['Wide_bins'])
        np.save(os.path.join(my_params['output_dir'], 'n_of_z.npy'), nz)
        np.save(os.path.join(my_params['output_dir'], 'z_grid.npy'), (z_grid[1:] + z_grid[:-1])/2)
        np.save(os.path.join(my_params['output_dir'], 'IntermediateProducts.npy'), RES, allow_pickle = True)
        
            
        
        
        
