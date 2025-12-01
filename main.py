# copy from 8/27
#!/usr/bin/python3.6

import torch
# from torch.utils.tensorboard import SummaryWriter
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
from eval import evaluate4
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
parser.add_argument('--time_data', default='')
parser.add_argument('--pre_time_data', default='')
parser.add_argument('--pre_num_epochs', type=int,  default=-1)
parser.add_argument('--num_epochs', type=int,  default=50)
parser.add_argument('--save_f', type=int,  default=1)
parser.add_argument('--start_epochs', type=int,  default=0)
parser.add_argument('--bz', type=int,  default=8)
parser.add_argument('--tstype',  default='random')
parser.add_argument('--tsnum',  default='0')
parser.add_argument('--epoch_step', type=int,  default=1)
parser.add_argument('--tst_split', default='test', type=str)
parser.add_argument('--channel_masking_rate', default=0, type=float)
parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str)
parser.add_argument('--snum_epochs', default=0, type=int)
parser.add_argument('--pretrain_weights', default='', type=str)
parser.add_argument('--q_nhead', default=1, type=int)
parser.add_argument('--q_num_layers', default=1, type=int)
parser.add_argument('--q_dropout', default=0.1, type=float)
parser.add_argument('--gen_type', default=0, type=int)
parser.add_argument('--decoder',  default='v1', type=str)
parser.add_argument('--iter',  default=-1, type=int)
parser.add_argument('--num_codes', default=-1, type=int)

args = parser.parse_args()
args.device=device

if args.dataset == 'gtea':
    args.bg_idx = [10]
else:
    args.bg_idx = []

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = args.bz
lr = 0.0005
num_epochs = args.num_epochs

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

args.sample_rate = sample_rate

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+f"/splits/{args.tst_split}.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"
args.vid_list_file = vid_list_file
args.vid_list_file_tst = vid_list_file_tst
args.features_path = features_path
args.gt_path=gt_path

mapping_file = "./data/"+args.dataset+"/mapping.txt"

# Use time data to distinguish output folders in different training
# time_data = '2020-10-15_08-52-26' # turn on this line in evaluation
# time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# time_data = args.time_data
if args.time_data=='':
    args.time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
bz_stages = '/margin_map_both' + args.time_data
# model_dir = "./models/"+args.dataset + bz_stages + "_split_"+args.split
model_dir0 = "./models/"+args.dataset + bz_stages 
model_dir = model_dir0 + "/split_"+args.split
pre_model_dir = "./models/"+args.dataset + '/margin_map_both' + args.pre_time_data + "/split_"+args.split
results_dir = "./results/"+args.dataset + bz_stages + "_split_"+args.split

if not os.path.exists(model_dir0):
    os.makedirs(model_dir0)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

args.tspath='./'+args.tstype+'/'+args.tsnum+'/'

results_dir = "./results/"+args.dataset + bz_stages + "_split_"+args.split

if args.dataset == "gtea":
    args.epsilon_l = 0.005
    args.epsilon = 0.02
    args.delta = 0.02
    args.num_samples_frames = 20

if args.dataset == "50salads":
    args.epsilon_l = 0.002
    args.epsilon = 0.05
    args.delta = 0.5
    args.num_samples_frames = 70
    sample_rate = 2

if args.dataset == "breakfast":
    args.epsilon_l = 0.005
    args.epsilon = 0.03
    args.delta =  0.03
    args.num_samples_frames = 20

print("{} dataset {} in split {} for single stamp supervision".format(args.action, args.dataset, args.split))
print('batch size is {}, number of stages is {}, sample rate is {}\n'.format(bz, num_stages, sample_rate))

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

args.actions_dict = actions_dict
num_classes = len(actions_dict)
# writer = SummaryWriter()
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes,args)


# Define the mapping from action string to trainer method
action_map = {
    # evaluate model direct predict
    "train_boundary_eval3": trainer.train_boundary_eval3,
    "train_boundary_eval4": trainer.train_boundary_eval4,
    # pretraining
    'train': trainer.train,
    # 'train2_1': trainer.train2_1,
    'train2_2': trainer.train2_2,
    # 'train2_3': trainer.train2_3,
    # 'train2_4': trainer.train2_4,
    # 'train2_5': trainer.train2_5,
    # 'predict24': trainer.predict24,
    
}

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(model_dir+'/app.log'),
        logging.StreamHandler()
    ]
)


logging.info(args)

if args.action in action_map:
    # Shared batch generator setup
    batch_gen = BatchGenerator(
        num_classes, actions_dict, gt_path, features_path, sample_rate, args
    )
    batch_gen.read_data(vid_list_file)
    trainer.bench_gen = batch_gen
    # print(args.tst_split)
    if args.tst_split!='':
        batch_gen_eva = BatchGenerator(
            num_classes, actions_dict, gt_path, features_path, sample_rate, args
        )
        batch_gen_eva.read_data(vid_list_file_tst)
        # trainer.bench_gen_eva = batch_gen_eva # 无情typo
        trainer.batch_gen_eva = batch_gen_eva # 修好的第一个bug
        
    model_path = None
    # Call the appropriate training function
    if args.pretrain_weights != '' or args.pre_time_data != '':
        if args.pre_num_epochs == -1:
            # 加载上一轮训练的特征提取器模型
            # if args.tst_split == 'train':
            #     model_path = pre_model_dir + "/test_" + str(args.iter - 1) + "_fea.model"
            #     trainer.model.load_state_dict(torch.load(model_path))
            # else:
            #     model_path = pre_model_dir + "/train_" + str(args.iter) + "_acc.model"
            #     trainer.model.load_state_dict(torch.load(model_path))
            model_path = pre_model_dir + "/test_" + str(args.iter - 1) + "_fea.model"
            trainer.model.load_state_dict(torch.load(model_path))
        else:
            try:
                # 优先尝试加载指定 epoch 的模型
                model_path = pre_model_dir + "/epoch-" + str(args.pre_num_epochs) + ".model"
                trainer.model.load_state_dict(torch.load(model_path))
            except Exception:
                # 如果失败，就加载用户指定的路径
                model_path = args.pretrain_weights
                trainer.model.load_state_dict(torch.load(model_path))
        print("Loaded pretrain weights from {}".format(model_path))
    
    action_map[args.action](
        model_dir, batch_gen, None,
        num_epochs=num_epochs,
        batch_size=bz,
        learning_rate=lr,
        device=device,
    )
else:
    raise ValueError(f"Unknown action: {args.action}")