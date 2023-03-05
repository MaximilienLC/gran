python3 -m gran -m hydra/launcher=submitit_slurm +launcher=beluga env=acrobot,cart_pole,mountain_car,mountain_car_continuous,lunar_lander,lunar_lander_continuous agent.gen_transfer=none,fit,mem,fit+mem agent.run_num_steps=0; 
python3 -m gran -m hydra/launcher=submitit_slurm +launcher=beluga env=acrobot,cart_pole,mountain_car,mountain_car_continuous,lunar_lander,lunar_lander_continuous agent.gen_transfer=none,fit,mem,fit+mem agent.run_num_steps=0 stage=test

python3 -m gran -m hydra/launcher=submitit_slurm +launcher=beluga env=acrobot,cart_pole,mountain_car,mountain_car_continuous,lunar_lander,lunar_lander_continuous agent.gen_transfer=env,env+fit,env+mem,env+fit+mem agent.run_num_steps=50;
python3 -m gran -m hydra/launcher=submitit_slurm +launcher=beluga env=acrobot,cart_pole,mountain_car,mountain_car_continuous,lunar_lander,lunar_lander_continuous agent.gen_transfer=env,env+fit,env+mem,env+fit+mem agent.run_num_steps=50 stage=test