#!/bin/bash
#SBATCH -p gpuA
#SBATCH --job-name=mogp_airquality
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mogp_out.txt
#SBATCH --error=mogp_err.txt

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
module load miniforge/24.11.2
source /scratch/$USER/venvs/mogp-gpytorch/bin/activate

# Prevent thread oversubscription — GPyTorch manages its own parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# -----------------------------------------------------------------------
# Create the venv and install deps on first run (comment out after):
# -----------------------------------------------------------------------
# python -m venv /scratch/$USER/venvs/mogp-gpytorch
# source /scratch/$USER/venvs/mogp-gpytorch/bin/activate
# pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# pip install gpytorch==1.12 linear_operator==0.5.0
# pip install numpy pandas scikit-learn matplotlib seaborn scipy requests

cd /net/scratch/$USER/modeling-air-quality-mogp
mkdir -p outputs

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "GPyTorch: $(python -c 'import gpytorch; print(gpytorch.__version__)')"
echo ""

echo "=== 02: Independent GPs ==="
python 02_independent_gps.py

echo "=== 03: ICM vs LCM ==="
python 03_icm_vs_lcm.py

echo "=== 04: Low-Data Regime ==="
python 04_low_data_regime.py

echo "=== 05: Pareto BO ==="
python 05_pareto_optimization.py

echo "=== 06: NN Comparison ==="
python 06_nn_comparison.py

echo "=== Job finished: $(date) ==="
