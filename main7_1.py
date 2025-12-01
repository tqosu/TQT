#!/usr/bin/python3.6

import torch
from torch.utils.tensorboard import SummaryWriter
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
from eval import evaluate4
from eval import evaluate5
import numpy as np
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# comment out seed to train the model
seed = 1538574472
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help='two options: train or predict')
parser.add_argument('--dataset', default="breakfast", help='three dataset: breakfast, 50salads, gtea')
parser.add_argument('--split', default='1')
# parser.add_argument('--bz_stages', default='/margin_map_both10')
# parser.add_argument('--pre_bz_stages', default='/margin_map_both9')
parser.add_argument('--time_data', default='')
parser.add_argument('--pre_time_data', default='')
parser.add_argument('--num_epochs', type=int,  default=50)
parser.add_argument('--set', default='train')
parser.add_argument('--pre_num_epochs', type=int,  default=-1)
parser.add_argument('--save_f', type=int,  default=1)
parser.add_argument('--bz', type=int,  default=8)
parser.add_argument('--start_epochs', type=int,  default=0)
# 23 behaves different
parser.add_argument('--gen_type', type=int,  default=21)
parser.add_argument('--q_nhead', default=1, type=int)
parser.add_argument('--q_num_layers', default=1, type=int)
parser.add_argument('--q_dropout', default=0.1, type=float)
parser.add_argument('--decoder',  default='v1', type=str)
parser.add_argument('--tst_split', default='', type=str)
parser.add_argument(
    '--bg_idx', 
    nargs='*',  # "*" means zero or more arguments are expected
    type=int,   # Ensure the input is treated as integers
    default=[],  # Default to an empty list if no arguments are provided
    help='edit score'
)
parser.add_argument('--num_codes', default=-1, type=int)
parser.add_argument('--iter', default=-1, type=int)
parser.add_argument('--stop', default=-1, type=int) # load best model for generating label and stop before training
# parser.add_argument('--iter', default='train', type=str)
parser.add_argument('--baseline', default=0, type=int)

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
lr = 0.0005
bz = args.bz
num_epochs = args.num_epochs
args.device=device

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "gtea":
    args.epsilon_l = 0.005
    args.epsilon = 0.02
    args.delta = 0.02
    args.num_samples_frames = 20
    args.bz = 25

if args.dataset == "50salads":
    args.epsilon_l = 0.002
    args.epsilon = 0.05
    args.delta = 0.5
    args.num_samples_frames = 70
    args.bz = 10
    sample_rate = 2

if args.dataset == "breakfast":
    args.epsilon_l = 0.005
    args.epsilon = 0.03
    args.delta =  0.03
    args.num_samples_frames = 20
    args.bz = 12 # this larger batch size is only for label generation


vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+f"/splits/{args.tst_split}.split"+args.split+".bundle"
vid_list_file_all = "./data/"+args.dataset+"/splits/all_files.txt"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"
args.gt_path=gt_path
mapping_file = "./data/"+args.dataset+"/mapping.txt"

if args.set=='all':
    vid_list_file=vid_list_file_all

# Use time data to distinguish output folders in different training
# time_data = '2020-10-15_08-52-26' # turn on this line in evaluation
# time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# bz_stages = args.bz_stages
if args.time_data=='':
    args.time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
bz_stages = '/margin_map_both' + args.time_data
load_model_dir="./models/"+args.dataset + '/margin_map_both' + args.pre_time_data +"/split_"+args.split

model_dir0 = "./models/"+args.dataset + bz_stages
model_dir = model_dir0 + "/split_"+args.split
# model_dir2 = "./models/"+args.dataset + '/margin_map_both2022-07-18_13-50-12'  + "_split_"+args.split
# model_dir2 = "./models/"+args.dataset + '/margin_map_both'+args.time_data0  + "_split_"+args.split
pseudo_path = model_dir+'/pseudo_labels_dir3/'
results_dir = "./results/"+args.dataset + bz_stages + "_split_"+args.split
 

if not os.path.exists(model_dir0):
    os.makedirs(model_dir0)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(pseudo_path):
    os.makedirs(pseudo_path)

print("{} dataset {} in split {} for single stamp supervision".format(args.action, args.dataset, args.split))
print('batch size is {}, number of stages is {}, sample rate is {}\n'.format(args.bz, num_stages, sample_rate))

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
writer = SummaryWriter()
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes,args)

batch_gen_gt = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate,args)
batch_gen_gt.read_data(vid_list_file)

# if args.gen_type in [21,22,23]:
# if args.baseline in [0,1,2,3,4]:
trainer.predict23(batch_gen_gt,load_model_dir, pseudo_path, features_path, vid_list_file, args.pre_num_epochs, actions_dict, device, sample_rate)
# else:
#     trainer.predict21(batch_gen_gt,load_model_dir, pseudo_path, features_path, vid_list_file, args.pre_num_epochs, actions_dict, device, sample_rate)
evaluate5(args.dataset, args.split, model_dir,pseudo_path,args.pre_num_epochs)

if args.stop == 1:
    exit()
# exit()

# if args.action == "train":

batch_gen = BatchGenerator(num_classes, actions_dict, pseudo_path, features_path, sample_rate,args)
batch_gen_gt = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate,args)
batch_gen.read_data(vid_list_file)
batch_gen_gt.read_data(vid_list_file)
if args.tst_split!='':
    batch_gen_eva = BatchGenerator(
        num_classes, actions_dict, gt_path, features_path, sample_rate, args
    )
    batch_gen_eva.read_data(vid_list_file_tst)
    trainer.batch_gen_eva = batch_gen_eva

# Train the model
trainer.train3(load_model_dir,  args.pre_num_epochs, device)
# here bz is still 8

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(model_dir+'/app.log'),
        logging.StreamHandler()
    ]
)


logging.info(args)

action_map = {
    'train15_1': trainer.train15_1,
    # 'train15_2': trainer.train15_2,
    # 'train15_3': trainer.train15_3,
    # 'train15_4': trainer.train15_4,
    # 'train15_ShiftCenter_b1': trainer.train15_ShiftCenter_b1,
    # 'train15_ShiftCenter_b4_xr': trainer.train15_ShiftCenter_b4_xr,
    }
action_map[args.action](model_dir, batch_gen, batch_gen_gt, writer, num_epochs=num_epochs, \
    batch_size=bz, learning_rate=lr, device=device, start_epochs=args.start_epochs,save_f=args.save_f)
# trainer.train5(load_model_dir, model_dir, batch_gen, writer, args.pre_num_epochs, num_epochs, batch_size=args.bz, learning_rate=lr, device=device)

trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
# Read output files and measure metrics (F1@10, 25, 50, Edit, Acc)
evaluate4(args.dataset, args.split, model_dir,results_dir+'/',num_epochs) 
