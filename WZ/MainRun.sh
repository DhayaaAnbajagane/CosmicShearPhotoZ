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


python -u $RUNNER --DECADE --NSIDE 128 --OutPath /project/chihway/dhayaa/DECADE/Wz/20240221_Nz_NSIDE128_MainRun.py
