# Skill-Enhanced Reinforcement Learning Acceleration from Heterogeneous Demonstrations

## Skill Training

./scripts/run_skill_with_d4rl.sh kitchen kitchen-partial-v0 kitchen_partial

nohup ./scripts/run_skill_with_d4rl.sh kitchen kitchen_microwave_kettle_light_slider-v0 kitchen_partial &

# Downstream

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
