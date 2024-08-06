import sys
import numpy as np
import pandas as pd
import time, os, glob, pickle, joblib
import argparse
import h5py
from tqdm import tqdm
from scipy import stats, interpolate

from datetime import datetime
today = datetime.today()
today = today.strftime('%B%d')

from multiprocessing import Pool

from .SOM import Classifier
from .Files import my_files
from .DELVERunner import TrainRunner, BinRunner, timeit, DEEP_COLS

NSAMPLES_DEFAULT = 1e3
class ThreeSDirRunner(BinRunner):
    
    def __init__(self, Nsamples = NSAMPLES_DEFAULT, z_bin_edges = [0.0, 0.3639, 0.6143, 0.8558, 2.0], **kwargs):
        
        print(kwargs)
        
         ### This will produce Nsamples X 64 samples
        self.Nsamples = int(Nsamples)
        self.z_bin_edges = z_bin_edges

        super().__init__(**kwargs)
        
        self.rng = np.random.default_rng(seed = self.seed)
        
        self.rng_list = [np.random.default_rng(seed = self.rng.integers(2**60)) for i in range(os.cpu_count())]
    
    @timeit
    def get_balrog_catalog(self, balrog_classified_df):

        
        #-------------------------- READ OUT ALL DEEP QUANTITIES --------------------------
        f = pd.read_csv(self.deep_catalog_path, usecols = DEEP_COLS).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        deep_sample_cuts  = self.get_deep_sample_cuts(f)
        Deep_df = f[deep_was_detected & deep_sample_cuts]
        
        print("DEEP", len(Deep_df))
        
        
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
            
            Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'true_ra', 'true_dec']], 
                                 how = 'left', on = ['true_ra', 'true_dec'], suffixes = (None, '_classified'), 
                                 validate = "1:1")
            
            Balrog_df['cell_wide_unsheared'] = Balrog_df['cell'] #Rename so it works with Alex's code
            
            counts = pd.DataFrame()
            counts['ID'], counts['injection_counts']  = np.unique(f['ID'][:], return_counts = True)

            detect = pd.DataFrame()
            detect['ID'], detect['detect_counts'] = np.unique(f['ID'][:][selection & (f['detected'][:] == 1)], return_counts = True)


            Balrog_df = pd.merge(Balrog_df, counts, on = 'ID', how = 'left', validate = "m:1")
            Balrog_df = pd.merge(Balrog_df, detect, on = 'ID', how = 'left', validate = "m:1")

            Balrog_df['overlap_weight'] = 1/Balrog_df['injection_counts'] * Balrog_df['w']
            Balrog_df['true_id'] = Balrog_df['ID']
            
            #Use only Balrog objects that were actually detected
            Balrog_df = Balrog_df[Balrog_df['selection'] == True]

        return Balrog_df

    
    @timeit
    def get_deep_catalog(self,  deep_classified_df):

        f = pd.read_csv(self.deep_catalog_path).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        deep_sample_cuts  = self.get_deep_sample_cuts(f)
        Deep_df = f[deep_was_detected & deep_sample_cuts]
        
        #Check masking was ok. I'm paranoid about pandas masking sometimes
        assert len(f) == len(deep_was_detected & deep_sample_cuts), "Mask is not right size"
        assert len(Deep_df) == np.sum(deep_was_detected & deep_sample_cuts), "Masked df doesn't have right size"
        
        #Now check and merge with classifier
        Deep_df = pd.merge(Deep_df, deep_classified_df[['cell', 'ID']], how = 'right', on = 'ID', suffixes = (None, '_classified'))
        
        Deep_df['cell_deep'] = Deep_df['cell']
        Deep_df['true_id']   = Deep_df['ID']
        
        Z_df    = pd.read_csv(self.redshift_catalog_path)
        Deep_df = pd.merge(Deep_df, Z_df[['ID', 'Z', 'SOURCE']], on = "ID", how = 'left')

        return Deep_df
    
    
    @timeit
    def get_redshift_catalog(self,  deep_classified_df):
        
        df = self.get_deep_catalog(deep_classified_df)
        df = df[df['Z'] > 0]
        
        return df

    
    @timeit
    def make_3sdir_nz(self, balrog_classified_df, deep_classified_df, wide_classified_df):

        #####################################
        ### Load catalogs and essential matrices.
        #####################################

        ### Balrog files ###
        balrog_data = self.get_balrog_catalog(balrog_classified_df)
        spec_data   = self.get_deep_catalog(deep_classified_df)
        
        balrog_data['cell_wide_unsheared'] = balrog_data['cell_wide_unsheared'].astype(int)
        spec_data['cell_deep'] = spec_data['cell_deep'].astype(int)

        ## This computes the lensingXresponse weight for each galaxy, removing the Balrog injection rate.
        balrog_data['weight_response_shear'] = balrog_data['injection_counts']*balrog_data['overlap_weight']
        balrog_data = pd.merge(balrog_data, spec_data[['ID', 'cell_deep', 'Z']], on = 'ID', how = 'left')
        
        spec_data = pd.merge(spec_data, balrog_data[['ID','overlap_weight', 'injection_counts', 'cell_wide_unsheared']], on = 'ID', how = 'right')
        
        ## This computes the lensingXresponse weight for each galaxy, removing the Balrog injection rate.
        spec_data['weight_response_shear'] = spec_data['injection_counts']*spec_data['overlap_weight']
        
        ### Load dictionary containing which wide cells belong to which tomographic bin
        bins = self.make_bins(balrog_classified_df, wide_classified_df, self.z_bin_edges).flatten().astype(int) #Fixed bins for now
        tomo_bins_wide_modal_even = np.array([np.where(bins == i)[0] for i in range(4)])

        
        ### Load p(chat) with all weights included: Balrog, response, shear.
        chat  = balrog_data['cell_wide_unsheared'].values.astype(int)
        w     = balrog_data['overlap_weight'].values
        pchat = np.bincount(chat, weights = w, minlength = np.max(chat) + 1)/np.sum(w)
        
        WideSOMSize = 32
        DeepSOMSize = 48
        
        ### Load p(c|chat) with all weights included: Balrog, response, shear.
#         pc_chat = np.load(data_dir+'pcchat_som.npy')
        pc_chat = np.zeros([DeepSOMSize**2, WideSOMSize**2])
        pcchat = np.zeros_like(pc_chat)
        np.add.at(pcchat, 
                  (balrog_data.cell_deep.values.astype(int),balrog_data.cell_wide_unsheared.values.astype(int)),
                  balrog_data.overlap_weight.values)
        pc_chat_new = pcchat/np.sum(pcchat,axis=0)
        pc_chat = pc_chat_new

        
        ### Define the redshift binning. This is currently set by the sample variance.
        min_z   = 0.01
        max_z   = 5
        delta_z = 0.05
        zbins   = np.arange(min_z,max_z+delta_z,delta_z)
        zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.
        
        self.zbinsc = zbinsc
        self.zbins  = zbins


        ### Counts in the redshift sample (weighted by balrog, but not weighted by responseXlensing weights.)
        ### Including condition on tomographic bin.
        Nzc   = self.return_Nzc(spec_data)
        Nzc_0 = self.return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
        Nzc_1 = self.return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
        Nzc_2 = self.return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
        Nzc_3 = self.return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])
        
        #print("NZc", np.sum(Nzc), np.sum(Nzc_0), np.sum(Nzc_1), np.sum(Nzc_2), np.sum(Nzc_3))

        ### Counts in the deep sample (weighted by balrog, but not weighted by responseXlensing weights.)
        ### Including condition on tomographic bin.
        Nc = self.return_Nc(balrog_data)
        Nc_0 = self.return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
        Nc_1 = self.return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
        Nc_2 = self.return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
        Nc_3 = self.return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])
        
        #print("Nc", Nc, Nc_0, Nc_1, Nc_2, Nc_3)

        ### If after the bin condition there are no redshift counts in a deep cell, don't apply the bin condition in that deep cell.
        sel_0 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_0, axis=0)==0))
        sel_1 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_1, axis=0)==0))
        sel_2 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_2, axis=0)==0))
        sel_3 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_3, axis=0)==0))
        Nzc_0[:,sel_0] = Nzc[:,sel_0].copy()
        Nzc_1[:,sel_1] = Nzc[:,sel_1].copy()
        Nzc_2[:,sel_2] = Nzc[:,sel_2].copy()
        Nzc_3[:,sel_3] = Nzc[:,sel_3].copy()
        
        
        #print("NZc", np.sum(Nzc), np.sum(Nzc_0), np.sum(Nzc_1), np.sum(Nzc_2), np.sum(Nzc_3))
        #print("sel", np.sum(sel_0), np.sum(sel_1), np.sum(sel_2), np.sum(sel_3))

        ### Average responseXlensing in each deep cell and redshift bin. The responseXlensing of each galaxy is weighted by its balrog probability.
        ### Including condition on tomographic bin.

        Rzc = self.return_Rzc(spec_data)
        Rzc_0 = self.return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
        Rzc_1 = self.return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
        Rzc_2 = self.return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
        Rzc_3 = self.return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])


        ### Average responseXlensing in each deep cell in the REDSHIFT sample. The responseXlensing of each galaxy is weighted by its balrog probability.
        ### Including condition on tomographic bin.
        Rc_redshift = self.return_Rc(spec_data)
        Rc_0_redshift = self.return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
        Rc_1_redshift = self.return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
        Rc_2_redshift = self.return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
        Rc_3_redshift = self.return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

        ### Average responseXlensing in each deep cell in the DEEP sample. The responseXlensing of each galaxy is weighted by its balrog probability.
        ### Including condition on tomographic bin.
        Rc_deep = self.return_Rc(balrog_data)
        Rc_0_deep = self.return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
        Rc_1_deep = self.return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
        Rc_2_deep = self.return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
        Rc_3_deep = self.return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

        ### We do not need the balrog and redshift samples. We can delete them.
        del spec_data
        del balrog_data


        fraction_Nzt = self.return_bincondition_fraction_Nzt_redshiftsample(np.array([Nzc, Nzc_0, Nzc_1, Nzc_2, Nzc_3]))
        fraction_Nt_D = self.return_bincondition_fraction_Nt_deepsample(np.array([Nc, Nc_0, Nc_1, Nc_2, Nc_3]))
        self.bincond_combined = fraction_Nzt*fraction_Nt_D[:,None,:]
        
        #print("fraction_Nzt", fraction_Nzt)
        #print("fraction_Nt_D", fraction_Nt_D)
        #print("self.bincond_combined", self.bincond_combined)


        redshift_sample_Rzt = np.array([Rzc, Rzc_0, Rzc_1, Rzc_2, Rzc_3])
        redshift_sample_Rt = np.array([Rc_redshift, Rc_0_redshift, Rc_1_redshift, Rc_2_redshift, Rc_3_redshift])
        deep_sample_Rt = np.array([Rc_deep, Rc_0_deep, Rc_1_deep, Rc_2_deep, Rc_3_deep])
        self.Rt_combined = self.return_bincondition_weight_Rzt_combined(redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt)


        #####################################
        ### Load Sample Variance from theory. 
        ### Compute superphenotypes and N(T,c,Z) matrices.
        #####################################

        ### Load the sample variance theory ingredient. This estimates the ratio between Shot noise and sample variance.

        sv_th = np.load('/project/chihway/dhayaa/DECADE/Alex_NERSC_files/cosmos_sample_variance.npy')[0]
        sv_th = np.diagonal(sv_th)[:]
        sv_th = sv_th[:len(self.zbinsc)]
        assert sv_th.shape[0]==len(self.zbinsc)
        sv_th_new = np.load('/project/chihway/dhayaa/DECADE/Alex_NERSC_files/sample_variance.npy')
        sv_th_new_diag = np.array([np.diagonal(x) for x in sv_th_new])

        sv_th_new_final = np.linalg.pinv(np.sum(np.array([np.linalg.pinv(x) for x in sv_th_new]),axis=0))
        sv_th_new_final_diag = np.diagonal(sv_th_new_final)

        sv_th_new_diag = sv_th_new_diag[:,:len(self.zbinsc)]
        sv_th_new_final_diag = sv_th_new_final_diag[:len(self.zbinsc)]

        nts = Nc.copy()
        nzt = Nzc.copy()
        nz,nt = nzt.shape

        # Removing types that don't have galaxies
        self.maskt = (np.sum(nzt,axis=0)>0.)
        nts = nts[self.maskt]
        nzt = nzt[:,self.maskt]

        # What is the redshift of each type?
        # Computing the mean redshift per type
        self.zmeant = np.zeros(nzt.shape[1])
        for i in range(nzt.shape[1]):
            self.zmeant[i] = np.average(np.arange(len(self.zbinsc)),weights=nzt.T[i])
        self.zmeant = np.rint(self.zmeant)

        #sv_th_v2 = np.load('/global/cscratch1/sd/alexalar/desy3data/marco_sv_v2/sv_th_v2.npy')

        varn_th = 1 + np.sum(nzt,axis=1)*sv_th
        #varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_v2
        varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_new_final_diag


        ### Decide which phenotypes go to which superphenotype
        ########
        ### Choose number of superphenotypes
        self.nT = 6
        ########
        bins = {str(b):[] for b in range(self.nT)}
        j = 0 
        sumbin = 0
        nTs = np.zeros(len(self.zbinsc))
        for i in range(len(self.zbinsc)):
            sumbin += np.sum(nzt[:,((self.zmeant==i))],axis=1).sum()
            nTs[i] = np.sum(nzt[:,((self.zmeant==i))],axis=1).sum()
            if (sumbin <= np.sum(nzt)/(self.nT-1))|(j==self.nT-1):
                bins[str(j)].append(i)
                #continue
            else:
                j += 1
                bins[str(j)].append(i)
                sumbin = np.sum(nzt[:,((self.zmeant==i))],axis=1).sum()


        ### Compute p(T), p(z,T) for the superphenotypes
        nzTi = np.zeros((len(self.zbinsc),self.nT))
        nTi = np.zeros((self.nT))
        for i in range(self.nT):
            nzTi[:,i] = np.sum(self.make_nzT(nzt,1)[:,bins[str(i)]],axis=1)
            nTi[i] = np.sum(self.make_nT(nzt,nts,1)[bins[str(i)]])

        print ('Correlation metric = %.3f'%self.corr_metric(nzTi))


        #####################################
        ### Sampling.
        ### Prepare matrices for efficient sampling.
        #####################################

        ### Define p(c,chat|bhat)/[p(c|bhat)p(chat|bhat)] --  conditioned on tomographic bin
        fcchat = pcchat.T/pcchat.sum()

        fcchat_0 = fcchat[tomo_bins_wide_modal_even[0]]
        fcchat_1 = fcchat[tomo_bins_wide_modal_even[1]]
        fcchat_2 = fcchat[tomo_bins_wide_modal_even[2]]
        fcchat_3 = fcchat[tomo_bins_wide_modal_even[3]]
        
        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            fcchat_0 /= np.multiply.outer(np.sum(fcchat_0,axis=1), np.sum(fcchat_0,axis=0))
            fcchat_1 /= np.multiply.outer(np.sum(fcchat_1,axis=1), np.sum(fcchat_1,axis=0))
            fcchat_2 /= np.multiply.outer(np.sum(fcchat_2,axis=1), np.sum(fcchat_2,axis=0))
            fcchat_3 /= np.multiply.outer(np.sum(fcchat_3,axis=1), np.sum(fcchat_3,axis=0))

        fcchat_0[~np.isfinite(fcchat_0)] = 0
        fcchat_1[~np.isfinite(fcchat_1)] = 0
        fcchat_2[~np.isfinite(fcchat_2)] = 0
        fcchat_3[~np.isfinite(fcchat_3)] = 0

        ### Define p(chat|bhat) --  conditioned on tomographic bin
        fchat_0 = pchat[tomo_bins_wide_modal_even[0]]
        fchat_1 = pchat[tomo_bins_wide_modal_even[1]]
        fchat_2 = pchat[tomo_bins_wide_modal_even[2]]
        fchat_3 = pchat[tomo_bins_wide_modal_even[3]]

        z2Tmap = np.zeros((len(self.zmeant))).astype(int)
        for i in range(self.nT):
            z2Tmap[np.isin(self.zmeant.astype(int),bins[str(i)])] = i

        self.Fcchat_0 = fcchat_0*fchat_0[:,None]
        self.Fcchat_1 = fcchat_1*fchat_1[:,None]
        self.Fcchat_2 = fcchat_2*fchat_2[:,None]
        self.Fcchat_3 = fcchat_3*fchat_3[:,None]

        if False:
            print(save_h5)
            store = pd.HDFStore(save_h5)
            store['nzt'] = pd.DataFrame(nzt)
            store['nzTi'] = pd.DataFrame(nzTi)
            store['nTi'] = pd.Series(nTi)
            store['nts'] = pd.Series(nts)
            store['bincond_combined_0'] = pd.DataFrame(self.bincond_combined[:,:,self.maskt][0])
            store['bincond_combined_1'] = pd.DataFrame(self.bincond_combined[:,:,self.maskt][1])
            store['bincond_combined_2'] = pd.DataFrame(self.bincond_combined[:,:,self.maskt][2])
            store['bincond_combined_3'] = pd.DataFrame(self.bincond_combined[:,:,self.maskt][3])
            store['R_combined_0'] = pd.DataFrame(self.Rt_combined[:,:,self.maskt][0])
            store['R_combined_1'] = pd.DataFrame(self.Rt_combined[:,:,self.maskt][1])
            store['R_combined_2'] = pd.DataFrame(self.Rt_combined[:,:,self.maskt][2])
            store['R_combined_3'] = pd.DataFrame(self.Rt_combined[:,:,self.maskt][3])
            store['sv_th'] = pd.Series(sv_th)
            store['sv_th_deep'] = pd.Series(sv_th_new_final_diag)
            store['varn_th'] = pd.Series(varn_th)
            store['varn_th_deep'] = pd.Series(varn_th_deep_v2)
            store['fcchat_0'] = pd.DataFrame(fcchat_0[:,self.maskt])
            store['fcchat_1'] = pd.DataFrame(fcchat_1[:,self.maskt])
            store['fcchat_2'] = pd.DataFrame(fcchat_2[:,self.maskt])
            store['fcchat_3'] = pd.DataFrame(fcchat_3[:,self.maskt])
            store['fchat_0'] = pd.Series(fchat_0)
            store['fchat_1'] = pd.Series(fchat_1)
            store['fchat_2'] = pd.Series(fchat_2)
            store['fchat_3'] = pd.Series(fchat_3)
            store['z2Tmap'] = pd.Series(z2Tmap)
            store['maskt'] = pd.Series(self.maskt)
            
            store['tomo_bins_wide_modal_even'] = pd.DataFrame(tomo_bins_wide_modal_even)
            store['PCHAT'] = pd.Series(PCHAT)
            store['bins']  = pd.Series(bins)
            store.close()
        

        nt = sum(self.maskt)
        nz=len(self.zbinsc)
        N_Tcz_Rsample = np.zeros((self.nT,nt,nz))
        for i in range(self.nT):
            sel = z2Tmap==i
            N_Tcz_Rsample[i, sel] = nzt.T[sel]

        N_Tc_Dsample = np.zeros((self.nT,nt))
        for i in range(self.nT):
            sel = z2Tmap==i
            N_Tc_Dsample[i, sel] = nts[sel]


        self.alpha = 1e-300

        N_T_Rsample = np.sum(N_Tcz_Rsample, axis=(1,2))
        N_z_Rsample = np.sum(N_Tcz_Rsample, axis=(0,1))
        N_Tz_Rsample = np.sum(N_Tcz_Rsample, axis=(1))
        N_cz_Rsample = np.sum(N_Tcz_Rsample, axis=(0))

        N_T_Dsample = np.sum(N_Tc_Dsample, axis=(1))
        N_c_Dsample = np.sum(N_Tc_Dsample, axis=(0))

        lambda_z_step1 = varn_th_deep_v2.copy()
        lambda_z_step2 = varn_th.copy()
        lambda_mean = np.sum(lambda_z_step1*N_z_Rsample/N_z_Rsample.sum())
        lambda_mean_R = np.sum(lambda_z_step2*N_z_Rsample/N_z_Rsample.sum())
        lambda_T = np.array([np.sum(lambda_z_step2 * x/x.sum()) for x in N_Tz_Rsample])

        onecell = np.sum(N_cz_Rsample>0,axis=1) == 1
        N_cz_Rsample_onecell = (N_cz_Rsample/np.sum(N_cz_Rsample,axis=1)[:,None])[onecell]


        #This to get the function to parallelize properly
        self.nt = nt
        self.nz = nz
        self.z2Tmap = z2Tmap
        self.N_Tcz_Rsample = N_Tcz_Rsample
        self.N_T_Rsample = N_T_Rsample
        self.N_z_Rsample = N_z_Rsample
        self.N_Tz_Rsample = N_Tz_Rsample
        self.N_Tc_Dsample = N_Tc_Dsample
        self.N_cz_Rsample = N_cz_Rsample
        self.N_T_Dsample = N_T_Dsample
        self.N_c_Dsample = N_c_Dsample
        self.lambda_mean = lambda_mean
        self.lambda_mean_R = lambda_mean_R
        self.lambda_T = lambda_T
        self.onecell = onecell
        self.N_cz_Rsample_onecell = N_cz_Rsample_onecell

        Nproc = os.cpu_count()
        with Pool(Nproc) as p:
            
            res = list(tqdm(p.imap(self.aux_fun, range(Nproc)), total = Nproc))
            nz_samples_newmethod = np.concatenate(res, axis=0)
            
        sel = np.sum(np.isnan(nz_samples_newmethod),axis=(1,2))==0
        nz_samples_newmethod = nz_samples_newmethod[sel]

        return nz_samples_newmethod
        
    
    #####################################
    ### compute N(z,c) and N(c), R(z,c), R(c) 
    ### and bin conditionalization versions.
    #####################################

    def return_Nzc(self, df):
        """
        - This function returns the counts Nzc=N(z,c) in each bin z and cell c.
        - The input is a pandas Dataframe containing a redshift sample. 
        - The redshift sample must have redshift and deep cell assignment.
        - It computes the balrog probability defined as #detections/#injections 
        to weight the counts of each galaxy in N(z,c).
        """

        df = df[df['Z'] > 0]
        redshift_sample = df[['injection_counts','true_id','cell_deep', 'Z']].groupby('true_id').agg('mean').reset_index()
        unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
        redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
        redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
        zid = np.digitize(redshift_sample.Z.values, self.zbins)-1
        zid = np.clip(zid, 0, len(self.zbinsc)-1)
        redshift_sample['zid'] = zid
        redshift_sample_groupby = redshift_sample[['balrog_prob','zid','cell_deep']].groupby(['zid','cell_deep']).agg('sum')

        Nzc = np.zeros((len(self.zbins)-1, 48*48))

        for index, row in redshift_sample_groupby.iterrows():
            if (index[0]<0) | (index[0]>len(self.zbins)-1): continue
            Nzc[ int(index[0]) , int(index[1]) ] = row.balrog_prob
        return Nzc

    def return_Nc(self, df):
        """
        - This function returns the counts Nc=N(c) in each cell c.
        - The input is a pandas Dataframe containing a deep sample. 
        - The deep sample must have a deep cell assignment.
        - It computes the balrog probability defined as #detections/#injections 
        to weight the counts of each galaxy in N(c).
        """

        redshift_sample = df[['injection_counts','true_id','cell_deep']].groupby('true_id').agg('mean').reset_index()
        unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
        redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
        redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
        redshift_sample_groupby = redshift_sample[['balrog_prob','cell_deep']].groupby(['cell_deep']).agg('sum')

        Nc = np.zeros((48*48))
        for index, row in redshift_sample_groupby.iterrows():
            Nc[ int(index) ] = row.balrog_prob
        return Nc

    def return_Rzc(self, df):
        """
        - This function returns the average lensingXshear weight in each bin z and cell c, Rzc= <ResponseXshear>(z,c)
        - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
        """
        
        df = df[df['Z'] > 0]
        redshift_sample = df[['injection_counts','true_id','cell_deep', 'Z', 'weight_response_shear','overlap_weight']].groupby('true_id').agg('mean').reset_index()
        unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
        redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
        redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
        zid = np.digitize(redshift_sample.Z.values, self.zbins)-1
        zid = np.clip(zid, 0, len(self.zbinsc)-1)
        redshift_sample['zid'] = zid
        redshift_sample['weight_response_shear_balrogprob'] = redshift_sample['weight_response_shear']*redshift_sample['balrog_prob']


        redshift_sample_groupby = redshift_sample[['weight_response_shear_balrogprob', 'balrog_prob','zid','cell_deep']].groupby(['zid','cell_deep']).agg('sum')
        Rzc = np.zeros((len(self.zbins)-1,48*48))
        for index, row in redshift_sample_groupby.iterrows():
            if (index[0]<0)|(index[0]>len(self.zbins)-1): continue
            Rzc[ int(index[0]), int(index[1]) ] = row.weight_response_shear_balrogprob/row.balrog_prob
        return Rzc

    def return_Rc(self, df):
        """
        - This function returns the average lensingXshear weight in each cell c, Rc= <ResponseXshear>(c)
        - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
        """
        redshift_sample = df[['injection_counts','true_id','cell_deep', 'weight_response_shear','overlap_weight']].groupby('true_id').agg('mean').reset_index()
        unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
        redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
        redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
        redshift_sample['weight_response_shear_balrogprob'] = redshift_sample['weight_response_shear']*redshift_sample['balrog_prob']


        redshift_sample_groupby = redshift_sample[['weight_response_shear_balrogprob', 'balrog_prob','cell_deep']].groupby(['cell_deep']).agg('sum')
        Rc = np.zeros(48*48)
        for index, row in redshift_sample_groupby.iterrows():
            Rc[ int(index) ] = row.weight_response_shear_balrogprob/row.balrog_prob
        return Rc
    
    
    def return_bincondition_fraction_Nzt_redshiftsample(self, redshift_sample_Nzt):
        """This function returns the fraction of counts in Nzt with over 
        without bin condition, for each tomographic bin.
        """

        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pz_c_0 = redshift_sample_Nzt[0]/np.sum(redshift_sample_Nzt[0],axis=0)
            pz_c_1 = redshift_sample_Nzt[1]/np.sum(redshift_sample_Nzt[1],axis=0)
            pz_c_2 = redshift_sample_Nzt[2]/np.sum(redshift_sample_Nzt[2],axis=0)
            pz_c_3 = redshift_sample_Nzt[3]/np.sum(redshift_sample_Nzt[3],axis=0)
            pz_c_4 = redshift_sample_Nzt[4]/np.sum(redshift_sample_Nzt[4],axis=0)

        pz_c_0[~np.isfinite(pz_c_0)] = 0
        pz_c_1[~np.isfinite(pz_c_1)] = 0
        pz_c_2[~np.isfinite(pz_c_2)] = 0
        pz_c_3[~np.isfinite(pz_c_3)] = 0
        pz_c_4[~np.isfinite(pz_c_4)] = 0

        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            gzt_0 = pz_c_1/pz_c_0
            gzt_1 = pz_c_2/pz_c_0
            gzt_2 = pz_c_3/pz_c_0
            gzt_3 = pz_c_4/pz_c_0

        gzt_0[~np.isfinite(gzt_0)] = 0
        gzt_1[~np.isfinite(gzt_1)] = 0
        gzt_2[~np.isfinite(gzt_2)] = 0
        gzt_3[~np.isfinite(gzt_3)] = 0

        return np.array([gzt_0, gzt_1, gzt_2, gzt_3])


    def return_bincondition_fraction_Nt_deepsample(self, deep_sample_Nt):
        """This function returns the fraction of counts in Nt with over 
        without bin condition, for each tomographic bin. Deep sample.
        """    

        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            gt_0 = deep_sample_Nt[1]/deep_sample_Nt[0]
            gt_1 = deep_sample_Nt[2]/deep_sample_Nt[0]
            gt_2 = deep_sample_Nt[3]/deep_sample_Nt[0]
            gt_3 = deep_sample_Nt[4]/deep_sample_Nt[0]

        gt_0[~np.isfinite(gt_0)] = 0
        gt_1[~np.isfinite(gt_1)] = 0
        gt_2[~np.isfinite(gt_2)] = 0
        gt_3[~np.isfinite(gt_3)] = 0

        return np.array([gt_0, gt_1, gt_2, gt_3])
    
    
    def return_bincondition_weight_Rzt_combined(self, redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt):
        """This function returns the final average responseXshear weight in each deep cell and redshift bin: Rzc.
        Response weight = Response to shear of the balrog injection of a deep galaxy.
        Shear weight = Weight to optimize of signal to noise of some shear observable. 
        - final Rzt = <Rzt>r * <Rt>r / <Rt>d
        where: 
        - <Rzt>r: average weight in z,c in the redshift sample.
        - <Rt>r: average weight in c in the redshift sample.
        - <Rt>d: average weight in c in the deep sample.
        It basically rescales the weight in Rzt such that it matches the average weight according to the deep sample.
        """    

        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            Rzt_factor_0 = deep_sample_Rt[1]/redshift_sample_Rt[1]
            Rzt_factor_1 = deep_sample_Rt[2]/redshift_sample_Rt[2]
            Rzt_factor_2 = deep_sample_Rt[3]/redshift_sample_Rt[3]
            Rzt_factor_3 = deep_sample_Rt[4]/redshift_sample_Rt[4]

        Rzt_factor_0[~np.isfinite(Rzt_factor_0)] = 0
        Rzt_factor_1[~np.isfinite(Rzt_factor_1)] = 0
        Rzt_factor_2[~np.isfinite(Rzt_factor_2)] = 0
        Rzt_factor_3[~np.isfinite(Rzt_factor_3)] = 0

        Rzt_0_final = np.einsum('zt,t->zt', redshift_sample_Rzt[1], Rzt_factor_0)
        Rzt_1_final = np.einsum('zt,t->zt', redshift_sample_Rzt[2], Rzt_factor_1)
        Rzt_2_final = np.einsum('zt,t->zt', redshift_sample_Rzt[3], Rzt_factor_2)
        Rzt_3_final = np.einsum('zt,t->zt', redshift_sample_Rzt[4], Rzt_factor_3)
        
        return np.array([Rzt_0_final, Rzt_1_final, Rzt_2_final, Rzt_3_final])
    
    
    def make_nzT(self, nzti, njoin):
        
        zmeanti = np.zeros(nzti.shape[1])
        
        for i in range(nzti.shape[1]):
            try: zmeanti[i] = np.average(np.arange(len(self.zbinsc)),weights=nzti.T[i])
            except: zmeanti[i] = self.rng.integers(len(self.zbinsc))
        
        zmeanti = np.rint(self.zmeant)

        nzTi = np.zeros((len(self.zbinsc),int(len(self.zbinsc)/njoin)))
        for i in range(int(len(self.zbinsc)/njoin)):
            nzTi[:,i] = np.sum(nzti[:,((self.zmeant>=njoin*i)&(self.zmeant<njoin*i+njoin))],axis=1)

        return nzTi

    def make_nT(self, nzti, nti, njoin):
        zmeanti = np.zeros(nzti.shape[1])
        for i in range(nzti.shape[1]):
            try: zmeanti[i] = np.average(np.arange(len(self.zbinsc)),weights=nzti.T[i])
            except: zmeanti[i] = self.rng.integers(len(self.zbinsc))
        zmeanti = np.rint(self.zmeant)

        nTi = np.zeros(int(len(self.zbinsc)/njoin))
        for i in range(int(len(self.zbinsc)/njoin)):
            nTi[i] = np.sum(nti[((self.zmeant>=njoin*i)&(self.zmeant<njoin*i+njoin))])
        return nTi

    def corr_metric(self, pzT):
        pzT = pzT/pzT.sum()
        overlap = np.zeros((pzT.shape[1],pzT.shape[1]))
        for i in range(pzT.shape[1]):
            for j in range(pzT.shape[1]):
                overlap[i,j] = np.sum(pzT[:,i]*pzT[:,j])
        overlap = overlap/np.diagonal(overlap)[:,None]
        metric = np.linalg.det(overlap)**(float(pzT.shape[1])/float(len(self.zbinsc)))
        return metric
    
    
    def return_nzsamples_fromfzt(self, fzt_dummy):
        fzt = np.zeros((48**2, len(self.zbinsc))).T
        fzt[:,self.maskt] = fzt_dummy.T

        ### Multiply the f_{zc} by:
        ### - Rzt: the average weight (includes response and shear weight).
        ### - gzt: the fraction probability for each tomographic bin.
        ### to add the bin condition and the average response and shear weights.
        fzt_0 = fzt * self.bincond_combined[0] * self.Rt_combined[0]
        fzt_1 = fzt * self.bincond_combined[1] * self.Rt_combined[1]
        fzt_2 = fzt * self.bincond_combined[2] * self.Rt_combined[2]
        fzt_3 = fzt * self.bincond_combined[3] * self.Rt_combined[3]

        #Suppress error because in next step we set infinite terms to 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            fzt_0 /= np.sum(fzt_0)
            fzt_1 /= np.sum(fzt_1)
            fzt_2 /= np.sum(fzt_2)
            fzt_3 /= np.sum(fzt_3)

        fzt_0[~np.isfinite(fzt_0)] = 0
        fzt_1[~np.isfinite(fzt_1)] = 0
        fzt_2[~np.isfinite(fzt_2)] = 0
        fzt_3[~np.isfinite(fzt_3)] = 0

        ### SOMPZ: Equals Eq.2 in https://www.overleaf.com/project/5e8b5a7d3431a1000126471a
        nz_0 = np.einsum('zt,dt->z', fzt_0, self.Fcchat_0)
        nz_1 = np.einsum('zt,dt->z', fzt_1, self.Fcchat_1)
        nz_2 = np.einsum('zt,dt->z', fzt_2, self.Fcchat_2)
        nz_3 = np.einsum('zt,dt->z', fzt_3, self.Fcchat_3)

        nz_0 /= nz_0.sum()
        nz_1 /= nz_1.sum()
        nz_2 /= nz_2.sum()
        nz_3 /= nz_3.sum()

        nz_samples = np.array([nz_0, nz_1, nz_2, nz_3])
        return nz_samples
    
    
    def draw_3sdir_onlyR(self, rng):

        ### step1
        f_T = rng.dirichlet(self.N_T_Rsample/self.lambda_mean_R+self.alpha)

        ### step2
        f_z_T = np.array([rng.dirichlet(x/self.lambda_T[i]+self.alpha) for i,x in enumerate(self.N_Tz_Rsample)])

        ### step3
        f_cz_Rsample = rng.dirichlet(self.N_cz_Rsample.reshape(np.prod(self.N_cz_Rsample.shape))+self.alpha).reshape(self.N_cz_Rsample.shape)
        f_cz = np.zeros( (self.nt, self.nz) )
        for k in range(self.N_Tcz_Rsample.shape[0]):
            sel = self.z2Tmap==k
            dummy = f_cz_Rsample[sel] 

            #Suppress error because in next step we set infinite terms to 0
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                dummy = dummy/np.sum(dummy,axis=0)

            dummy[~np.isfinite(dummy)] = 0 #DHAYAA: I changed this from np.isnan to np.isfinite
            f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]

        return f_cz


    def draw_3sdir_newmethod(self, rng):
        ### step1
        f_T = rng.dirichlet(self.N_T_Dsample/self.lambda_mean+self.alpha)

        ### step2
        f_cT = np.zeros(self.nt)
        for k in range(self.nT):
            sel = self.z2Tmap==k
            f_cT[sel] = rng.dirichlet(self.N_Tc_Dsample[k,sel]+self.alpha) * f_T[k]

        ### step3
        f_cz = self.draw_3sdir_onlyR(rng)
        f_z_c = f_cz/np.sum(f_cz,axis=1)[:,None]
        f_z_c[self.onecell] = self.N_cz_Rsample_onecell

        ### compute f_{zc}
        f_cz = f_z_c * f_cT[:,None]
        return f_cz



    def aux_fun(self, i):
        
        #t0 = time.time()
        nz_samples_newmethod = np.zeros((self.Nsamples,4, len(self.zbinsc)))

        #t0 = time.time()
        for i_sample in range(self.Nsamples):
            f_zt = self.draw_3sdir_newmethod(self.rng_list[i])
            nz_samples_newmethod[i_sample] = self.return_nzsamples_fromfzt(f_zt)

        #t1 = time.time()
        #print(t1-t0)
        return nz_samples_newmethod  
    
    
    def draw_3sdir_step3_p_zT_onlyR(self,):

        ### step1
        f_T = self.rng.dirichlet(self.N_T_Dsample/self.lambda_mean+self.alpha)

        ### step2
        f_z_T = np.array([self.rng.dirichlet(x/self.lambda_T[i]+self.alpha) for i,x in enumerate(self.N_Tz_Rsample)])

        ### step3
        f_cz_Rsample = self.rng.dirichlet(self.N_cz_Rsample.reshape(np.prod(self.N_cz_Rsample.shape))+self.alpha).reshape(self.N_cz_Rsample.shape)
        f_cz = np.zeros( (self.nt,self.nz) )
        for k in range(self.N_Tcz_Rsample.shape[0]):
            sel = self.z2Tmap==k
            dummy = f_cz_Rsample[sel] 
            
            #Suppress error because in next step we set infinite terms to 0
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                dummy = dummy/np.sum(dummy,axis=0)
            
            dummy[np.isnan(dummy)] = 0
            f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]

        return f_cz

    def aux_fun_2(self, i):
        
        #t0 = time.time()
        nz_samples_newmethod = np.zeros((self.Nsamples,4, len(self.zbinsc)))

        t0 = time.time()
        for i_sample in range(self.Nsamples):
            f_zt = self.draw_3sdir_step3_p_zT_onlyR()
            nz_samples_newmethod[i_sample] = self.return_nzsamples_fromfzt(f_zt)

        return nz_samples_newmethod 
    
    
    def postprocess_nz(self, z, list_of_nz):

        processed_nz = []
        for nz in list_of_nz:

            #Ramping, in order to set p(z = 0) --> 0
            nz *= np.where(z[None, :] <= 0.055, nz * z[None,:]/0.055, 1)
            
            #Normalize everything
            nz /= np.sum(nz, axis = -1)[..., None]
            
            #Now do pileup of everything beyond z > 3.
            nz[:, np.argmin(np.abs(z - 3))] = np.sum(nz[:, z > 3])
            
            #save
            processed_nz.append(nz)

        return processed_nz


class ZPOffsetRunner(TrainRunner):
    
    
    def __init__(self, seed, Nsamples, sigma_ZP, output_dir, deep_catalog_path, balrog_catalog_path, njobs = 10):
        
        self.njobs    = njobs
        self.Nsamples = Nsamples
        self.sigma_ZP = sigma_ZP
        self.seed     = seed
        
        self.samples  = self.build_LH()
        
        super().__init__(seed, output_dir, deep_catalog_path, None, balrog_catalog_path)
        
    
    def build_LH(self):
        
        assert len(self.sigma_ZP) == 8, "Current code assumes DESY3, which has 8 bands in deep fields"
        
        sampler = stats.qmc.LatinHypercube(d = len(self.sigma_ZP) * 3, seed = self.seed) #Times x3 is because we need 3 different fields
        samples = sampler.random(n = self.Nsamples)
        samples = stats.norm(scale = np.tile(self.sigma_ZP, 3) ).ppf(samples)
        
        return samples
        
        
        
    @timeit
    def go(self):
        
        #Init so all the data is written out to disk properly first
        self.initialize()
        
        os.makedirs(self.output_dir + '/ZP/', exist_ok = True)
        for i in range(self.Nsamples):
            
            CELL = self.classify(i)
            np.save(self.output_dir + '/ZP/collated_deep_classifier_Samp%d.npy' % i, CELL[-1])
            
        
        
    @timeit
    def initialize(self):

        DEEP = self.get_deep_fluxes(self.deep_catalog_path, self.balrog_catalog_path)
        
        for i, l in enumerate(['FLUX', 'FLUX_ERR', 'ID', 'TILENAME']):
            np.save(self.output_dir + '/DEEP_DATA_%s.npy' % l, DEEP[i])
            
        #Build a separate file which shows which field the galaxy is in
        #COSMOS == 0, C3 == 1, E2 == 2, X3 == 3
        
        tile_tag = np.zeros(DEEP[-1].shape[0], dtype = int) - 99
        for i in range(DEEP[-1].size):
            
            if DEEP[-1][i][:-4] == 'COSMOS': tile_tag[i] = 0
            elif DEEP[-1][i][:-4] == 'SN-C3': tile_tag[i] = 1
            elif DEEP[-1][i][:-4] == 'SN-E2': tile_tag[i] = 2
            elif DEEP[-1][i][:-4] == 'SN-X3': tile_tag[i] = 3
                
        np.save(self.output_dir + '/DEEP_DATA_TILETAG.npy', tile_tag)
    
    
    def offset_photometry(self, ind, flux, flux_err, tiletag):
        
        offsets = self.samples[ind].reshape(3, 8)
        
        #Only offset the SN field photometry. The COSMOS field is fixed in place.
        #This is because we are only concerned with the relative differences between
        #the four fields. So one of them can have fixed photometry
        flux     *= np.where(tiletag[:, None] > 0, 10**( -0.4*offsets[tiletag - 1, :] ), 1)
        flux_err *= np.where(tiletag[:, None] > 0, 10**( -0.4*offsets[tiletag - 1, :] ), 1)
        
        return flux, flux_err
    

    def classify(self, ind):

        mode = 'DEEP'
        SOM_weights = np.load(self.output_dir + '/%s_SOM_weights.npy'   % mode)[:]
        flux        = np.load(self.output_dir + '/%s_DATA_FLUX.npy'     % mode)[:]
        flux_err    = np.load(self.output_dir + '/%s_DATA_FLUX_ERR.npy' % mode)[:]
        tiletag     = np.load(self.output_dir + '/%s_DATA_TILETAG.npy'  % mode)[:]

        flux, flux_err = self.offset_photometry(ind, flux, flux_err, tiletag)
        
        start = 0
        end   = flux.shape[0]
        
        Nproc = np.max([os.cpu_count(), flux.shape[0]//1_000_000 + 1])
        inds  = np.array_split(np.arange(start, end), Nproc)
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

        return ind, cell_id
    

    @timeit
    def classify_deep(self):

        #We do deep fields all at once, so let start/end be the full array
        CELL = self.classify(0, int(1e10), 'DEEP')
        np.save(self.output_dir + '/collated_deep_classifier.npy', CELL[-1])
        

class ThreeSDirRedbiasRunner(ThreeSDirRunner):
    
    def __init__(self, bias_function, Nsamples = NSAMPLES_DEFAULT, z_bin_edges = [0.0, 0.3639, 0.6143, 0.8558, 2.0], **kwargs):
        
        self.bias_function = bias_function
        
        super().__init__(Nsamples, z_bin_edges, **kwargs)
        
        
    @timeit
    def get_deep_catalog(self,  deep_classified_df):

        Deep_df = super().get_deep_catalog(deep_classified_df)
        Deep_df = self.bias_function(Deep_df)

        return Deep_df
    

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--ZPOffsetRunner',  action = 'store_true', default = False, help = 'Setup ZP uncertainty module')
    my_parser.add_argument('--ZPUncertRunner',  action = 'store_true', default = False, help = 'Run ZP uncertainty module')
    my_parser.add_argument('--ThreeSDirRunner', action = 'store_true', default = False, help = 'Run 3sDir module')
    my_parser.add_argument('--ThreeSDirRedbiasRunner', action = 'store_true', default = False, help = 'Run 3sDir module with z-bias')
    my_parser.add_argument('--FinalRunner',     action = 'store_true', default = False, help = 'Run 3sDir module with all uncertainties')
    
    my_parser.add_argument('--Summarize',  action = 'store_true', default = False, help = 'Compress samples')
    
    my_parser.add_argument('--Nsamples', action = 'store', type = int, default = 5,       help = 'Number of ZP samples to make')
    my_parser.add_argument('--njobs',    action = 'store', type = int, default = 15,      help = 'Number of parallel threads')

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
                 'Nsamples' : args['Nsamples'],
                 'sigma_ZP' : np.array([0.055, 0.005, 0.005, 0.005, 0.005, 0.008, 0.008, 0.008])
                }
    
    my_params = my_params | my_files
    
    
    #Redshift distrbution is hardcoded in
    zbins  = np.arange(0.01, 5.05, 0.05)
    zbinsc = zbins[:-1] + (zbins[1] - zbins[0])/2.
    
    if args['ThreeSDirRunner']:
        tmp = {k: v for k, v in my_params.items() if k not in ['njobs', 'sigma_ZP', 'Nsamples']}
        
        bclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/BALROG_DATA_ID.npy'),
                                'true_ra'  : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_RA.npy'),
                                'true_dec' : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_DEC.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_balrog_classifier.npy')})
        
        wclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/WIDE_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_wide_classifier.npy')})
        
        dclass  = pd.DataFrame({'ID'       : np.load(my_params['output_dir'] + '/DEEP_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_deep_classifier.npy')})
        
        
        ONE = ThreeSDirRunner(**tmp)
        ONE.go()

        
    if args['ZPOffsetRunner']:
        tmp = {k: v for k, v in my_params.items() if k not in ['wide_catalog_path', 'redshift_catalog_path', 'tomo_redshift_catalog_path']}
        ONE = ZPOffsetRunner(**tmp)
        ONE.go()
        
        
    if args['ZPUncertRunner']:
        
        tmp = {k: v for k, v in my_params.items() if k not in ['njobs', 'sigma_ZP', 'Nsamples']}
        ONE = ThreeSDirRunner(**tmp)
        
        bclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/BALROG_DATA_ID.npy'),
                                'true_ra'  : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_RA.npy'),
                                'true_dec' : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_DEC.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_balrog_classifier.npy')})
        
        wclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/WIDE_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_wide_classifier.npy')})
        
        for i in range(args['Nsamples']):
            dclass  = pd.DataFrame({'ID'       : np.load(my_params['output_dir'] + '/DEEP_DATA_ID.npy'),
                                    'cell'     : np.load(my_params['output_dir'] + '/ZP/collated_deep_classifier_Samp%d.npy' % i)})
            
        
            n_of_z = ONE.make_3sdir_nz(bclass, dclass, wclass)
            n_of_z = ONE.postprocess_nz(zbinsc, n_of_z)
                
            
            np.save(my_params['output_dir'] + '/ZP/nz_Samp%d.npy' % i, n_of_z)
            
            
    if args['ThreeSDirRedbiasRunner'] | args['FinalRunner']:
        
        
        
        
        bclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/BALROG_DATA_ID.npy'),
                                'true_ra'  : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_RA.npy'),
                                'true_dec' : np.load(my_params['output_dir'] + '/BALROG_DATA_TRUE_DEC.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_balrog_classifier.npy')})
        
        wclass  = pd.DataFrame({'id'       : np.load(my_params['output_dir'] + '/WIDE_DATA_ID.npy'),
                                'cell'     : np.load(my_params['output_dir'] + '/collated_wide_classifier.npy')})
        
        if args['FinalRunner']:
            os.makedirs(my_params['output_dir'] + '/Final/', exist_ok = True)
        elif args['ThreeSDirRedbiasRunner']:
            os.makedirs(my_params['output_dir'] + '/ZB/', exist_ok = True)
            
        
        for i in range(args['Nsamples']):
            
            tmp = {k: v for k, v in my_params.items() if k not in ['njobs', 'sigma_ZP', 'Nsamples']}
            
            def bias_function(data):
                
                SRC = data['SOURCE'].values
                Z   = data['Z'].values
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    Mi  = 30 - 2.5*np.log10(data['BDF_FLUX_DERED_CALIB_I'].values)
                
                seed  = np.random.default_rng(my_params['seed']).integers(0, 2**30, args['Nsamples'])[i]
                sigma = np.random.default_rng(seed).normal(loc = 0, scale = 1, size = 2)
                
                #Do Cosmos-only fist
                bfile = np.loadtxt('/project/chihway/dhayaa/DECADE/Redshift_files/median_bias_Cosmos.txt')
                bfile = np.stack([bfile[:, 0], np.median(bfile[:, 1:], axis = 1)], axis = 1)
                bfile = bfile[np.isfinite(bfile[:, 1])]
                bias  = interpolate.CubicSpline(bfile[:, 0], bfile[:, 1], extrapolate = False); Mmin = bfile[0, 0]; Mmax = bfile[-1, 0];
                Z     = np.where( (SRC == 'COSMOS2020')  & (Mi > Mmin) & (Mi < Mmax), Z + (1 + Z) * bias(Mi) * sigma[0], Z)
                
                #Do Cosmos + PAUS next
                bfile = np.loadtxt('/project/chihway/dhayaa/DECADE/Redshift_files/median_bias_PausCosmos.txt')
                bfile = np.stack([bfile[:, 0], np.median(bfile[:, 1:], axis = 1)], axis = 1)
                bfile = bfile[np.isfinite(bfile[:, 1])]
                bias  = interpolate.CubicSpline(bfile[:, 0], bfile[:, 1], extrapolate = False); Mmin = bfile[0, 0]; Mmax = bfile[-1, 0];
                Z     = np.where( (SRC == 'PAUS+COSMOS') & (Mi > Mmin) & (Mi < Mmax), Z + (1 + Z) * bias(Mi) * sigma[1], Z)
                
                #Assign back to the deepfields file
                data['Z'] = Z
                
                return data
            
            tmp['bias_function'] = bias_function #Supply function as argument to class initialization dict
            ONE = ThreeSDirRedbiasRunner(**tmp) #Initialize class
            
            if args['FinalRunner']:
                
                dclass  = pd.DataFrame({'ID'       : np.load(my_params['output_dir'] + '/DEEP_DATA_ID.npy'),
                                        'cell'     : np.load(my_params['output_dir'] + '/ZP/collated_deep_classifier_Samp%d.npy' % i)})
                n_of_z = ONE.make_3sdir_nz(bclass, dclass, wclass)
                n_of_z = ONE.postprocess_nz(zbinsc, n_of_z)

                np.save(my_params['output_dir'] + '/Final/nz_Samp%d.npy' % i, n_of_z)
            
                
            elif args['ThreeSDirRedbiasRunner']:

                dclass  = pd.DataFrame({'ID'       : np.load(my_params['output_dir'] + '/DEEP_DATA_ID.npy'),
                                        'cell'     : np.load(my_params['output_dir'] + '/collated_deep_classifier.npy')})
                
                n_of_z = ONE.make_3sdir_nz(bclass, dclass, wclass)
                n_of_z = ONE.postprocess_nz(zbinsc, n_of_z)

                np.save(my_params['output_dir'] + '/ZB/nz_Samp%d.npy' % i, n_of_z)
        
        
    if args['Summarize']:
        
        from scipy import stats
        
        print("SUMMARY USES 5SIGMA AS THRESHOLD WITH 8 DEGREES OF FREEDOM. "
              "CHANGE SRC CODE IF YOU DON'T WANT THIS")
        
        path = my_params['output_dir'] + '/Summary'
        os.makedirs(path, exist_ok = True)
        
        for Mode in ['ZP', 'ZB', 'Final']:
            
            print('-----------------------')
            print("IN MODE", Mode)
            
            if os.path.isdir(my_params['output_dir'] + '/%s/' % Mode):

                print("FOUND FILES! LOADING THEM NOW...")
                files = sorted(glob.glob(my_params['output_dir'] + '/%s/nz_*.npy' % Mode))
                NZ    = np.concatenate([np.load(f) for f in files])
                print(f"USING {len(files)} FILES. TOTAL OF {NZ.shape[0]} SAMPLES")

                np.save(path + '/mean_nz_%s.npy' % Mode, np.mean(NZ, axis = 0))

                mean_z = np.trapz(NZ * zbinsc[None, None, :], zbinsc, axis = -1)/np.trapz(NZ, zbinsc, axis = -1)
                summ   = np.stack([np.mean(mean_z, axis = 0), np.std(mean_z, axis = 0)])
                np.savetxt(path + '/nz_priors_%s.txt' % Mode, summ)
                
                Likelihood_file = my_params['output_dir'] + '/%s/LnLikelihood_Fiducial.npy' % Mode
                if os.path.isfile(Likelihood_file):
                    
                    print("FOUND CALCULATED LIKELIHOOD! LOADING IT NOW...")
                    Likelihood = np.load(Likelihood_file)
                    max_like   = np.min(Likelihood[..., 0], axis = 0)
                    good       = np.abs(Likelihood[..., 0] - max_like) < stats.chi2.ppf(stats.norm.cdf(5), 8)/2
                    
                    print(f"SELECT {np.sum(good, axis = 0)} SAMPLES PER BIN [1 2 3 4]")
                    mean = np.array([np.mean(NZ[good[:, i], i, :], axis = 0) for i in range(4)])
                    np.save(path + '/mean_nz_combined_%s.npy' % Mode, mean)

                    mean_z = np.trapz(NZ * zbinsc[None, None, :], zbinsc, axis = -1)/np.trapz(NZ, zbinsc, axis = -1)
                    summ   = np.stack([[np.mean(mean_z[good[:, i], i], axis = 0) for i in range(4)], 
                                       [np.std(mean_z[good[:, i], i], axis = 0) for i in range(4)]])
                    np.savetxt(path + '/nz_priors_combined_%s.txt' % Mode, summ)
                    
                print("FINISHED", Mode, "\n")
                print('-----------------------')
                    
                    
            
            
            
            
            
    
