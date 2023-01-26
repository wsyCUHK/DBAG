# DBAG

## Introduction
This is the implementations related to our paper 

S. Wang, S. Bi and Y. -J. A. Zhang, " <a href="https://ieeexplore.ieee.org/document/10025689">Edge Video Analytics with Adaptive Information Gathering: A Deep Reinforcement Learning Approach</a>," Accepted by  IEEE Transactions on Wireless Communications.

This is also a updated Transformer Model for DRL in communications. Please also refer to the previous work <a href="https://github.com/wsyCUHK/SACCT">Deep Reinforcement Learning With Communication Transformer for Adaptive Live Streaming in Wireless Edge Networks</a> in JSAC.

## Usage
You may try the following command:
python DBAG.py --clr=0.0025 --alr=0.00025 --nstep=3 --gamma=0.6  --device=cuda:0 --replay_size=1024 --drop=0.5  --num_steps=5000 --seed=1 --embedding_size=128 --layers=1 --num_header=4 --target_update_interval=1 --history_length=10 --filename=your_name --automatic_entropy_tuning=True --eta=0.6 --start_steps=2000 --user=3 --maxep=300 --dataset=SelfDriving

## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* Torch 1.8.1
* CUDA 11.1


## About Authors
Shuoyao Wang, sywang[AT]szu[DOT]edu[DOT]cn :Shuoyao Wang received the B.Eng. degree (with first class Hons.) and the Ph.D degree in information engineering from The Chinese University of Hong Kong, Hong Kong, in 2013 and 2018, respectively. From 2018 to 2020, he was an senior researcher with the Department of Risk Management, Tencent, Shenzhen, China. Since 2020, he has been with the College of Electronic and Information Engineering, Shenzhen University, Shenzhen, China, where he is currently an Assistant Professor. His research interests include optimization theory, operational research, and machine learning in Multimedia Processing, Smart Grid, and Communications. See more details in the <a href="https://wsycuhk.github.io/">personal webpage</a>.

This is a co-work with Suzhi Bi and Yingjun Angela Zhang.

## Citation
If the implementation helps, you might cite our paper with the following format:

@ARTICLE{10025689,
  author={Wang, Shuoyao and Bi, Suzhi and Zhang, Ying-Jun Angela},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Edge Video Analytics with Adaptive Information Gathering: A Deep Reinforcement Learning Approach}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2023.3237202}}
