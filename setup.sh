# Create Conda Enviornment 
conda create --yes --name bot python=3.10

# Activate the Enviornment
source activate bot

# Installing Pytorch with Cuda 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installing Nvidia Cuda-toolkit
# conda install --yes -c nvidia cuda-toolkit

# Installing Requirements file
pip install -r requirements.txt

# Flash Attension Installation
# MAX_JOBS=4 pip install flash-attn --no-build-isolation 