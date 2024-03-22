export LD_LIBRARY_PATH=/opt/openmpi/lib/:$LD_LIBRARY_PATH
export PATH=/opt/openmpi/bin/:$PATH
export OPAL_PREFIX=
export PYTHONPATH="${PYTHONPATH}:./text_autoencoder"
export PYTHONPATH="${PYTHONPATH}:text_autoencoder"
# download data
mkdir ./data-bin

sudo apt install libopenmpi-dev
python -m pip install --upgrade pip

pip install -r requirements.txt

# '1.9.1+cu111' torch
pip install -U torchmetrics[image]
pip install mpi4py 
pip install gdown av click einops Pillow tensorboardX imageio imageio-ffmpeg
pip install hydra-core --upgrade
pip install timm==0.4.12

# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
pip install --upgrade torch torchvision
conda install mpi4py
cd neural_diffusion
pip install -e .
cd ..

pip install gpustat torchinfo fairseq==0.10.0
pip install sentencepiece

pip install opencv-python

python --version
echo "setup done"