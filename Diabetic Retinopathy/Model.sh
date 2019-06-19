#!/bin/sh
#SBATCH --job-name=DiaRet_VAE
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH -t 01:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=t_im1996@live.nl

cd ~/Tim

source activate twoVAE

echo "Starting"
python main_VAE.py --batch-size 64 --epochs 100 --arch fcn
python main_VAE.py --batch-size 64 --epochs 100 --arch convlin

echo "Done"
