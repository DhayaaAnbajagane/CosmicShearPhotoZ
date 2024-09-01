#!/bin/bash
#SBATCH --job-name LikelihoodRun
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=60:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/Likelihood/log_%x
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


RUNNER=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/Likelihood/LikelihoodRunner.py


python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240831/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240831_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 5 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial_all
         

python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240831/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240831_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 6.0 --alpha_u 1.9 --dalpha_u 4 \
                  --M 5 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighUncert_balpha_u_all
                  
                  
python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240831/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240831_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 5 --rms 0.6 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighUncert_Sys_all
                  
                  
python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240831/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240831_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 10 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighOrder_Sys_all
                  
