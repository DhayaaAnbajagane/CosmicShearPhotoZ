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


python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240726_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 5 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial
         

python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240726_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 3.0 --alpha_u 1.9 --dalpha_u 2 \
                  --M 5 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighUncert_balpha_u
                  
                  
python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240726_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 5 --rms 0.3 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighUncert_Sys
                  
                  
python -u $RUNNER --NzDir /project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/Final/ \
                  --WzPath /project/chihway/dhayaa/DECADE/Wz/20240726_fiducial_Wz_600patch_urmask_CovMat.npy\
                  --njobs 16 \
                  --b_u 1 --db_u 1.5 --alpha_u 1.9 --dalpha_u 1 \
                  --M 8 --rms 0.15 --tol 0.05 --Niter 30 \
                  --Name Fiducial_HighOrder_Sys
