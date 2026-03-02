conda create -n kai0 python=3.11 -y
conda activate kai0

# 建议先升级 pip
python -m pip install -U pip

# 安装依赖（JAX CUDA 轮子链接一并给上）
GIT_LFS_SKIP_SMUDGE=1 pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装当前仓库
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
