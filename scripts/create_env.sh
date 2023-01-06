conda create -n pylantern_env python=3.10 -y
conda activate pylantern_env
pip3 install torch==1.12.1 torchvision==0.13.1
pip3 install -r requirements.txt
pip3 install ./libs/matches