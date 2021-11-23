#!/bin/bash

# Script to reproduce results

# DEFAULT
for ((i=10;i<13;i+=1))
do 
	python3 td3.py \
	--env "InvertedPendulumBulletEnv-v0" \
	--seed=$i \
	--max_timesteps=200000 \
    --save_model
done

for ((i=1;i<3;i+=1))
do 
	python3 td3.py \
	--env "HalfCheetahBulletEnv-v0" \
	--seed=$i \
	--max_timesteps=1000000 \
    --save_model
done

#############################################

# TEST: more noise
for ((i=10;i<13;i+=1))
do 
	python3 td3.py \
	--env "InvertedPendulumBulletEnv-v0" \
	--seed=$i \
	--max_timesteps=200000 \
	--policy_noise=1 \
	--noise_clip=1 \
    --save_model
done

for ((i=1;i<4;i+=1))
do 
	python3 td3.py \
	--env "HalfCheetahBulletEnv-v0" \
	--seed=$i \
	--max_timesteps=1000000 \
	--policy_noise=1 \
	--noise_clip=1 \
    --save_model
done