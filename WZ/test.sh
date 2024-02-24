#!/bin/bash
#SBATCH --job-name Wz_test
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=30:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/WZ/test.log
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END


RUNNER=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/WZ/WZRunner.py

#for NSIDE in 32 128 512 1024
#do 
#    python -u $RUNNER --DECADE --NSIDE ${NSIDE} --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE${NSIDE}.npy
#done

#for C in 0 1 2 4 8
#do
#    python -u $RUNNER --DECADE --NSIDE 128 --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE128_CountT${C}.npy --Count_threshold ${C}
#done


#python -u $RUNNER --DECADE --NSIDE 128 --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE128_MaskUnknownOnly.npy --OnlyUnknownMask

#for delta_z in 0.025 0.05 0.1 0.2
#do
#    python -u $RUNNER --DECADE --NSIDE 128 --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE128_RedshiftMask_delta${delta_z}.npy --redshift_mask --redshift_mask_deltaz ${delta_z}
#done

python -u $RUNNER --DECADE --NSIDE 128 --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE128_MaskRefOnly.npy --OnlyRefMask
