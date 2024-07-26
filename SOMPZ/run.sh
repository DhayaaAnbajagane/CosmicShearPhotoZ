#!/bin/bash
#SBATCH --job-name SOMPZ_RUN
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=60:00:00
#SBATCH --output=/home/dhayaa/Desktop/DECADE/CosmicShearPhotoZ/SOMPZ/log_run
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway2-0696,midway2-0694


BASE=/home/dhayaa/Desktop/DECADE/
RUNNER=CosmicShearPhotoZ.SOMPZ.DELVERunner
RUNNER2=CosmicShearPhotoZ.SOMPZ.ThreeSDirRunner


#PYTHONPATH=$BASE python -m $RUNNER --TrainRunner --WIDE --DEEP
#PYTHONPATH=$BASE python -m $RUNNER --ClassifyRunner --BALROG --WIDE --DEEP
#PYTHONPATH=$BASE python -m $RUNNER --BinRunner

PYTHONPATH=$BASE python -m $RUNNER2 --ZPOffsetRunner --Nsamples 100
PYTHONPATH=$BASE python -m $RUNNER2 --ZPUncertRunner --Nsamples 100

PYTHONPATH=$BASE python -m $RUNNER2 --ThreeSDirRedbiasRunner --Nsamples 100
PYTHONPATH=$BASE python -m $RUNNER2 --FinalRunner --Nsamples 100
