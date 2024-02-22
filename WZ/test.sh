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

for NSIDE in 32 128 512 1024
do 
    python -u $RUNNER --DECADE --NSIDE ${NSIDE} --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE${NSIDE}.npy
done
