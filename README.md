# Skill-Enhanced Reinforcement Learning Acceleration from Heterogeneous Demonstrations
#### [[Project Website]](https://clvrai.github.io/spirl/) [[Paper]](https://arxiv.org/abs/2010.11944)

【Install】
pip3 install -r requirements.txt
pip3 install -e .
pip3 install "Cython<3"
pip3 install setuptools==65.5.0 wheel==0.38.4 "pip<24.1" 
pip install "gym==0.21.0" "h5py>=3.7" "numpy>=1.21" "tqdm"
pip install "d4rl==1.1"
pip uninstall -y mujoco dm-control mujoco-py glfw
pip install "d4rl==1.1"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install tensorboardX wandb cv2
pip install opencv-python
pip3 install scikit-image
pip install "wandb<=0.12.21"
pip install "numpy<1.24"
pip install mujoco_py
pip3 install matplotlib tensorboardX funcsigs
pip install "mujoco==2.1.5"
pip install "dm-control==0.0.403778684" # XML问题
pip install mpi4py
conda install -c conda-forge libstdcxx-ng=12 libgcc-ng=12 -y
conda install -c conda-forge gxx_linux-64=12 -y
【Skill】
./scripts/run_skill_with_d4rl.sh kitchen kitchen-partial-v0 kitchen_partial
nohup ./scripts/run_skill_with_d4rl.sh kitchen kitchen_microwave_kettle_light_slider-v0 kitchen_partial &
【RL】
cd spirl/spirl
conda activate serla
export EXP_DIR=${EXP_DIR:-"$PWD/exp"}
export DATA_DIR=${DATA_DIR:-"$PWD/data_d4rl_general"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nohup python -m spirl.rl.train --path configs/hrl/kitchen/spirl_cl_partial --prefix kitchen_partial > train.log &

## Citation
```
@misc{zhang2024skillenhancedreinforcementlearningacceleration,
      title={Skill-Enhanced Reinforcement Learning Acceleration from Demonstrations}, 
      author={Hanping Zhang and Yuhong Guo},
      year={2024},
      eprint={2412.06207},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06207}, 
}
```
