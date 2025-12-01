#!/usr/bin/python3.6
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import os,random, csv
import pandas as pd
from eval import edit_score,f_score,levenstein
from decoder import TemporalMaskFormer_V1, TemporalMaskFormer_V2, TemporalMaskFormer_V3, TemporalMaskFormer_V4
from collections import defaultdict

def remove_duplicates(nums, bg_idx):
    if nums.shape[0]==0:
        return []
    result=[]
    # print(bg_idx)
    nums = [item for index, item in enumerate(nums) 
                   if item not in bg_idx]
    for i in range(len(nums)):
        # if nums[i] in bg_idx:continue
        # print(nums[i], end =' ')
        if len(result)==0:
            result = [nums[i]]
        elif nums[i] != result[-1]:    
            result.append(nums[i])
    return result


def insert_and_sort_csv(filename, new_row, header=['Dataset','Epoch', 'Split', 'F1@10', 'F1@25', 'F1@50', 'Edit', 'Acc']):
    """
    Inserts a new row into a CSV file. If the file doesn't exist, it creates it with the given header.
    Then sorts the CSV by the 'Epoch' column and saves it.

    Args:
        filename (str): The path to the CSV file.
        new_row (list): The row to insert (should match the header order).
        header (list): The CSV header (default as specified).
    """
    if not os.path.exists(filename):
        # print("case 1")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(new_row)
    else:
        # print("case 2")
        df = pd.read_csv(filename)
        df.loc[len(df)] = new_row
        # df = df.sort_values(by='Epoch')
        df.to_csv(filename, index=False)


# 2025/06/26 create for train_boundary_eval4
def insert_and_sort_csv2(filename, new_row, header=['Dataset', 'Iteration', 'Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50', 'Edit', 'Acc', 'FEA']):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(new_row)
    else:
        df = pd.read_csv(filename)

        # Step 1: Add missing column if needed
        for col in header:
            if col not in df.columns:
                print(f"[INFO] Adding missing column '{col}' with default value")
                df[col] = -1 if col == "Iteration" else 0

        # Step 3: Reorder columns
        df = df[header]

        # Step 4: Validate shape
        if len(new_row) != len(header):
            raise ValueError(f"Mismatch between new_row ({len(new_row)} fields) and header ({len(header)}).")

        # Step 5: Append new row and save
        df.loc[len(df)] = new_row
        df = df.sort_values(by=["Iteration", "Split", "Epoch"])
        df.to_csv(filename, index=False)


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, args):
        super(MultiStageModel, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])
        self.decoder = TemporalMaskFormer_V1(num_queries=num_classes, d_model=num_f_maps, nhead=args.q_nhead, num_layers=args.q_num_layers, dim_feedforward=num_f_maps, dropout=args.q_dropout)

    def forward(self, x, mask):
        
        middle_out, out = self.tower_stage(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return middle_out, outputs


class MultiStageModel_V2(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, args):
        super(MultiStageModel_V2, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])

        action_map = {
            'v2': TemporalMaskFormer_V2,
            'v3': TemporalMaskFormer_V3,
            'v4': TemporalMaskFormer_V4
        }
        self.decoder = action_map[args.decoder](num_queries=num_classes, d_model=num_f_maps, nhead=args.q_nhead, num_layers=args.q_num_layers, dim_feedforward=num_f_maps, dropout=args.q_dropout)
        
    def forward(self, x, mask): 
        features, out = self.tower_stage(x, mask)  # shape: (B, C, T)
        features_stack = features.unsqueeze(0)
        outputs = out.unsqueeze(0)                   # (1, B, C, T)

        for s in self.single_stages:
            features, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            features_stack = torch.cat((features_stack, features.unsqueeze(0)), dim=0)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return features_stack, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes,args):
        if args.decoder == 'v1':
            self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes,args)
        else:
            self.model = MultiStageModel_V2(num_blocks, num_layers, num_f_maps, dim, num_classes,args)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.args=args
        self.gt={}


    def confidence_loss(self, pred, confidence_mask, device):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss
    

    def get_unsupervised_losses(self, count, outp1, actual_labels, activity_labels):
        vid_ids = []
        f1 = []
        t1 = []
        label_info = []
        
        feature_activity = []
        maxpool_features = []

        bsize = outp1.shape[0] 


        for j in range(bsize): 
            vidlen = count[j]
            
            sel_frames_current = torch.linspace(0, vidlen, self.args.num_samples_frames,  dtype=int)
            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end   = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])
            
            idx = torch.tensor(idx).type(torch.long).to(outp1.device)
            idx = torch.clamp(idx, 0, vidlen - 1)


            # Sampling of second set of frames from surroundings epsilon
            # vlow = 1   # To prevent value 0 in variable low
            vlow = int(np.ceil(self.args.epsilon_l * vidlen.item()))
            vhigh = int(np.ceil(self.args.epsilon * vidlen.item()))

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow, 
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(outp1.device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
           
            # Now adding all frames togather 
            f1.append(outp1[j].permute(1,0)[idx, :])
            f1.append(outp1[j].permute(1,0)[previdx, :])

            if activity_labels is not None: 
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None
           

            label_info.append(actual_labels[j][idx])
            label_info.append(actual_labels[j][previdx])
            
            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(previdx))
            
            idx = idx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            previdx = previdx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            
            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(previdx.detach().cpu().numpy().tolist())
            
            maxpool_features.append(torch.max(outp1[j,:,:vidlen], dim=-1)[0])

        # Gathering all features togather  
        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)
        f1 = torch.cat(f1, dim=0)
        label_info = torch.cat(label_info, dim=0).cpu().numpy()
    
        if feature_activity is not None:
            feature_activity = np.array(feature_activity)
        
        # print(f1.norm(dim=1).shape)
        # return 0
        f1=f1/f1.norm(dim=1)[:, None]
        # print(f1.shape)
        # print(f1[0])
        # print(torch.norm(f1[0]))
        # return 0
        sim_f1 = (f1 @ f1.data.T)
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # print(torch.max(sim_f1))
        # print("##########")
        # return 0
        f11 = torch.exp(sim_f1 / 0.1)
        # print(torch.min(f11))
        # print(torch.max(f11))
        # return 0
        if feature_activity is None:
            pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                          (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                          (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                  (label_info[:, None] == label_info[None, :])).type(torch.float32).to(outp1.device)
            pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) &\
                                                           (label_info[:, None] == label_info[None, :]))
        else: 
            pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                          (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                          (label_info[:, None] == label_info[None, :]))

            negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                  (label_info[:, None] == label_info[None, :])).type(torch.float32).to(outp1.device)
          
        I = torch.eye(len(pos_weight_mat)).to(outp1.device)
        
        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(outp1.device) - I
        not_same_activity = 1 - pos_weight_mat - I - negative_samples_minus
        countpos = torch.sum(pos_weight_mat)
        # print(torch.sum(pos_weight_mat))


        if countpos == 0:
            print("Feature level contrast no positive is found")
            feature_contrast_loss = 0
       
        else: 
            featsim_pos = pos_weight_mat * f11
            max_val = torch.max(not_same_activity * f11, dim=1, keepdim=True)[0]
            acc = torch.sum(featsim_pos > max_val) / countpos
            featsim_negsum = torch.sum(not_same_activity * f11, dim=1)
            # print(featsim_negsum.shape)
            # print(featsim_negsum)
            # print("34333######")
            
            simprob = (featsim_pos / (featsim_negsum + featsim_pos)) + not_same_activity + I + negative_samples_minus
            # print(torch.min(simprob))
            
            feature_contrast_loss = -torch.sum(torch.log(simprob)) / countpos

        return feature_contrast_loss,acc


    def train(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence, count, item = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device) 
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                if epoch < start_epochs:
                    batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                else:
                    batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())
                batch_boundary = batch_boundary.to(device)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            # writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct)/total))
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f,   acc = %f"%(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total)
                print(print_string)
                fp.write(print_string+'\n')


    
    def train3(self, model_dir,  num_epochs, device):
        self.model.to(device)
        # self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(num_epochs) + ".model"))
        if self.args.pre_num_epochs != -1:
            print("load weights from {}".format(model_dir + "/epoch-" + str(epoch) + ".model"))
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        else:
            print("load weights from {}".format(model_dir + "/" + "train" + "_" + str(self.args.iter) + "_acc.model"))
            self.model.load_state_dict(torch.load(model_dir + "/" + "train" + "_" + str(self.args.iter) + "_acc.model"))

    def train15(self, save_dir, batch_gen, batch_gen_gt, writer, num_epochs, batch_size, learning_rate, device,start_epochs,save_f):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        min_epoch_loss=99999
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            # m_f1=self.get_mf1(batch_gen_gt,batch_size,device)
            batch_gen_gt.reset2(batch_gen.list_of_examples)
            while batch_gen.has_next():
                batch_input, batch_boundary, mask, batch_confidence ,_,__= batch_gen.next_batch(batch_size)
                batch_input_gt, batch_target, mask_gt, batch_confidence_gt ,_,__= batch_gen_gt.next_batch(batch_size)

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                batch_boundary = batch_boundary.to(device)
                # if gen_type==0:
                #     batch_boundary2 = batch_gen.get_boundary(batch_size, middle_pred.detach())
                # elif gen_type==20:
                #     batch_boundary2 = batch_gen.get_boundary20(batch_size, middle_pred.detach(),m_f1)
                # elif gen_type==21:
                #     batch_boundary2 = batch_gen.get_boundary21(batch_size, middle_pred.detach(),m_f1)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            batch_gen_gt.reset2(batch_gen.list_of_examples)
            if (epoch+1)%save_f==0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if (epoch+1+5)>num_epochs:
                t_epoch_loss = epoch_loss / len(batch_gen.list_of_examples)
                if min_epoch_loss>t_epoch_loss:
                    min_epoch_loss=t_epoch_loss
                    torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(num_epochs + 1) + ".model")
                    torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(num_epochs + 1) + ".opt")
                    with open(save_dir + "/results_file.txt", "a+") as fp:
                        print_string = "+[epoch %d]: epoch loss = %f,   acc = %f"%(num_epochs + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                                    float(correct)/total)
                        print(print_string)
                        fp.write(print_string+'\n')


            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f,   acc = %f"%(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total)
                print(print_string)
                fp.write(print_string+'\n')
    
    def get_consistency_losses(self,  single_frame_lst, outp1, actual_labels, predictions):
        vid_ids = []
        f1 = []
        p1 = []
        label_info = []

        bsize = outp1.shape[0] 
        # print(outp1.shape)
        for j in range(bsize):
            idx=single_frame_lst[j]
            # print(idx)
            f1.append(outp1[j].permute(1,0)[idx, :])
            p1.append(predictions[j].permute(1,0)[idx, :])
            label_info.append(actual_labels[j][idx])
            vid_ids.extend([j] * len(idx))
            # break
        vid_ids = torch.tensor(vid_ids).numpy()
        f1 = torch.cat(f1, dim=0)
        p1 = torch.cat(p1, dim=0)
        # print(f1.shape,p1.shape)
        # return
        label_info = torch.cat(label_info, dim=0).cpu().numpy()

        f1=f1/f1.norm(dim=1)[:, None]
        sim_f1 = (f1 @ f1.data.T)
        # f11 =(sim_f1+1)/2.0
        f11 = torch.exp(sim_f1)

        p1=p1/p1.norm(dim=1)[:, None]
        sim_p1 = (p1 @ p1.data.T)
        p11=(sim_p1+1)/2.0

        # print(f11.shape)
        # print(p11.shape)

        fp=-f11*torch.log(p11)
        # print(fp.shape)
        # print(torch.min(p1),torch.max(p1))
        # print(torch.min(p11),torch.max(p11))
        # print(torch.min(f11),torch.max(f11))
        # print(p11[0])
        # print(f11[0])
        # return
        


        pos_weight_mat = torch.tensor(label_info[:, None] == label_info[None, :]).type(torch.float32).to(outp1.device)
        # I = torch.eye(len(pos_weight_mat)).to(outp1.device)

        # pos_weight_mat = (pos_weight_mat).type(torch.float32).to(outp1.device) - I
        # not_same_activity = 1 - pos_weight_mat - I
        countpos = torch.sum(pos_weight_mat)
        # print(pos_weight_mat.device)
        # print(fp.device)
        featsim_pos = pos_weight_mat * fp
        # print(torch.min(featsim_pos),torch.max(featsim_pos))
        # max_val = torch.max(not_same_activity * f11, dim=1, keepdim=True)[0]
        # acc = torch.sum(featsim_pos > max_val) / countpos
        # featsim_negsum = torch.sum(not_same_activity * f11, dim=1)

        # simprob = (featsim_pos / (featsim_negsum + featsim_pos)) + not_same_activity + I

        consistency_loss = torch.sum(featsim_pos) / countpos
        # print(feature_contrast_loss)
        return consistency_loss

    def train_eva(self):
        self.model.eval()
        results=[]
        for vid in self.batch_gen_eva.list_of_examples:
            results.append(self.process_file(vid))
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0
        
        for result in results:
            correct += result[0]
            total += result[1]
            edit += result[2]
   
            tp += result[3]
            fp += result[4]
            fn += result[5]

        ans=[]
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
        
            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            ans.append(f1)
        
        ans.append((1.0*edit)/len(results))
        ans.append(100*float(correct)/total)

        self.model.train()
        return np.array(ans)

    # evalute directly on prediction performance
    def train_boundary_eval3(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        args=self.args
        self.model.eval()
        self.model.to(device)
        header=['Dataset','Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50','Edit', 'Acc','FEA']
        best_fea = 0
        best_acc = 0
        file_name =f'{save_dir}/{args.tst_split}_eval.csv'
        # epoch_set, best_acc, best_fea = process_csv(file_name)
        # print('location 1')

        for epoch in range(args.snum_epochs, num_epochs, args.epoch_step):
            source = save_dir + "/epoch-" + str(epoch+1) + ".model"
            self.model.load_state_dict(torch.load(source))
            row = [args.dataset,args.split,epoch+1] 
            ans=self.train_eva()
            row.extend(ans)
            row.append(sum(ans)/len(ans))
            
            
            insert_and_sort_csv(file_name, row, header=header)
            if row[-1]>best_fea:
                # source = save_dir + "/epoch-" + str(epoch+1) + ".model"
                target = save_dir + f"/{args.tst_split}_fea.model"
                shutil.copy(source,target)
                best_fea=row[-1]
            if row[-1]>best_acc:
                target = save_dir + f"/{args.tst_split}_acc.model"
                shutil.copy(source,target)
                best_acc=row[-1]
            print(row)

    # 2025/06/26 base on train_boundary_eval3 + iteration onto csv
    def train_boundary_eval4(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        args=self.args
        self.model.eval()
        self.model.to(device)
        if args.iter == -1:
            exit("train_boundary_eval4 only for iteration")
            
        header=['Dataset', 'Iteration', 'Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50','Edit', 'Acc','FEA']
        best_fea = 0
        best_acc = 0
        file_name =f'{save_dir}/{args.tst_split}_eval.csv'
        # epoch_set, best_acc, best_fea = process_csv(file_name)
        # print('location 1')

        for epoch in range(args.snum_epochs, num_epochs, args.epoch_step):
            source = save_dir + "/epoch-" + str(epoch+1) + ".model"
            self.model.load_state_dict(torch.load(source))

            row = [args.dataset, args.iter, args.split, epoch+1]
            ans = self.train_eva()  # len(ans) = 5
            row.extend(ans)
            row.append(sum(ans)/len(ans))  # FEA

            assert len(row) == 10  # match new header
            insert_and_sort_csv2(file_name, row)

            if row[-1]>best_fea:
                # source = save_dir + "/epoch-" + str(epoch+1) + ".model"
                    target = save_dir + f"/{args.tst_split}_{args.iter}_fea.model"
                    shutil.copy(source,target)
                    best_fea=row[-1]
            if row[-1]>best_acc:
                    target = save_dir + f"/{args.tst_split}_{args.iter}_acc.model"
                    shutil.copy(source,target)
                    best_acc=row[-1]
            print(row)
    

    def train4(self, model_dir, save_dir, batch_gen, writer, num_epochs0,num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        print('load model from {}'.format(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence, count, item = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                # if epoch < start_epochs:
                #     batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                # else:
                #     batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())
                batch_boundary = batch_gen.get_all(batch_size, batch_input.size(-1))
                batch_boundary = batch_boundary.to(device)
                # print("NICE?")

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                            # float(correct)/total))
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f,   acc = %f"%(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total)
                print(print_string)
                fp.write(print_string+'\n')

    # unsupervised training
    def train5(self, model_dir,save_dir, batch_gen, writer, num_epochs0,num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        print('load model from {}'.format(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_epochs = 0
        print('start epoch of unsupervision is:', start_epochs)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence, count, item = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device), 
                # print(count)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)
                # print(item)
                if self.args.dataset == 'breakfast':
                    activity_labels = np.array([name.split('_')[-1] for name in item])
                elif self.args.dataset == '50salads':
                    activity_labels = None 
                elif self.args.dataset == 'gtea':
                    activity_labels = None 
                # print(activity_labels)
                # return 
                # print(middle_pred.shape)
                # print(mask.shape)
                # print(batch_target.shape)

                
                loss,acc=self.get_unsupervised_losses(count, middle_pred, batch_target, activity_labels)

                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # _, predicted = torch.max(predictions[-1].data, 1)
                # correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                # total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            # writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct)/total))
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f,   \
                    acc = %f"%(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               acc)
                print(print_string)
                fp.write(print_string+'\n')
    
    def train6(self, model_dir,  num_epochs, device):
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(num_epochs) + ".model"))

    def train7(self, model_dir, save_dir, batch_gen, writer, num_epochs0,num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        print('load model from {}'.format(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(num_epochs0) + ".model"))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence, count, item = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                if epoch < start_epochs:
                    batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                else:
                    batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())
                # batch_boundary = batch_gen.get_all(batch_size, batch_input.size(-1))
                batch_boundary = batch_boundary.to(device)
                # print("NICE?")

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                            # float(correct)/total))
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f,   acc = %f"%(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total)
                print(print_string)
                fp.write(print_string+'\n')

    def get_single_frame_feature(self,  single_frame_lst, outp1,actual_labels, label_info,f1):
        bsize = outp1.shape[0] 
        # print(outp1.shape)
        for j in range(bsize):
            idx=single_frame_lst[j]
            # print(idx)
            f1.append(outp1[j].permute(1,0)[idx, :])
            label_info.append(actual_labels[j][idx])

    def get_mf1(self,batch_gen,batch_size,device):
        with torch.no_grad():
            label_info,f1=[],[]
            # print('A')
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence ,_,__= batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                # print('B')
                middle_pred, predictions = self.model(batch_input, mask)
                batch_boundary, single_frame_lst = batch_gen.get_single_random2(batch_size, batch_input.size(-1))
                self.get_single_frame_feature(single_frame_lst, middle_pred, batch_target, label_info,f1)
            #     print(batch_gen.index/len(batch_gen.list_of_examples),end=' ')
            # print()
            #     print(len(label_info),len(f1))
                # break
            # print(len(label_info),len(f1))
            batch_gen.reset()
            f1 = torch.cat(f1, dim=0)
            f1_mean=torch.mean(f1,dim=0,keepdim=True)
            # print(f1.shape)
            label_info = torch.cat(label_info, dim=0)#.cpu().numpy()
            # print(f1.shape,label_info.shape)
            # print(label_info)
            # m_f1=torch.ones(batch_gen.num_classes,64)
            m_f1=[]
            for i in range(batch_gen.num_classes):
                idx=(label_info==i)
                idx_f1=f1[idx]
                if len(idx_f1)!=0:
                    m_idx_f1=torch.mean(idx_f1,dim=0,keepdim=True)
                else:
                    m_idx_f1=f1_mean
                m_f1.append(m_idx_f1)
                    # m_f1[].append(m_idx_f1)
                # print(i,len(idx_f1))
            # print(batch_gen.num_classes)
            m_f1 = torch.cat(m_f1, dim=0)
            # print(m_f1.shape)
        return m_f1


    def get_q_losses(self,  single_frame_lst, outp1, actual_labels):
        # print("# get_q_losses: outp1.shape", outp1.shape, "actual_labels.shape", actual_labels.shape)
        vid_ids = []
        f1 = []
        t1 = []
        label_info = []
        
        feature_activity = []
        maxpool_features = []

        bsize = outp1.shape[0] 
        for j in range(bsize):
            idx=single_frame_lst[j]
            f1.append(outp1[j].permute(1,0)[idx, :])
            label_info.append(actual_labels[j][idx])
            vid_ids.extend([j] * len(idx))

        vid_ids = torch.tensor(vid_ids).numpy()
        
        f1 = torch.cat(f1, dim=0)
        batch_target = torch.cat(label_info, dim=0)
        memory = f1.unsqueeze(0)
        mask_logits, class_logits = self.model.decoder(memory)

        mask_logits = mask_logits.squeeze(0).T
        # print(mask_logits.shape, batch_target.shape)
        # exit()

        loss = self.ce(mask_logits, batch_target)
        return loss
        # print("# location 1")
        # torch.Size([229, 64]) (229,)
        # print(f1.shape,label_info.shape)
        # print(memory.shape, mask_logits.shape, class_logits.shape)
        # # print(label_info)
        # exit()
        # return feature_contrast_loss

    def get_q_losses2(self,  single_frame_lst, features, actual_labels):
        vid_ids = []
        
        t1 = []
        label_info = []
        
        feature_activity = []
        maxpool_features = []

        bsize = features.shape[0] 
        for j in range(bsize):
            idx=single_frame_lst[j]
            # f1.append(features[j].permute(1,0)[idx, :])
            label_info.append(actual_labels[j][idx])
            vid_ids.extend([j] * len(idx))
        f1 = []
        for i in range(features.shape[0]):
            f2 = []
            for j in range(bsize):
                idx=single_frame_lst[j]
                f2.append(features[i][j].permute(1,0)[idx, :])
            f2=torch.cat(f2, dim=0).unsqueeze(0)
            f1.append(f2) 
        memory = torch.cat(f1, dim=0).unsqueeze(1)

        # print(f1.shape, label_info.shape)
        # exit()

        vid_ids = torch.tensor(vid_ids).numpy()
        

        batch_target = torch.cat(label_info, dim=0)
        # print(f1.shape, batch_target.shape)
        # exit()
        # memory = f1.unsqueeze(0)
        mask_logits, class_logits = self.model.decoder(memory)

        mask_logits = mask_logits.squeeze(0).T
        # print(mask_logits.shape, batch_target.shape)
        # exit()

        loss = self.ce(mask_logits, batch_target)
        return loss
        # print("# location 1")
        # torch.Size([229, 64]) (229,)
        # print(f1.shape,label_info.shape)
        # print(memory.shape, mask_logits.shape, class_logits.shape)
        # # print(label_info)
        # exit()
        # return feature_contrast_loss

    # Tue Jun 24 20:09:34 PM PDT 2025
    # based on get_q_losses2, to adapt to train2_5, in order to support multiple layer loss
    def get_q_losses3(self,  single_frame_lst, features, actual_labels):
        vid_ids = []
        label_info = []
        f1 = []

        bsize = features.shape[0] 
        for j in range(bsize):
            idx = single_frame_lst[j]
            label_info.append(actual_labels[j][idx])
            vid_ids.extend([j] * len(idx))

        for i in range(features.shape[0]):  # for each decoder layer
            f2 = []
            for j in range(bsize):
                idx = single_frame_lst[j]
                f2.append(features[i][j].permute(1,0)[idx, :])  # (len(idx), C)
            f2 = torch.cat(f2, dim=0).unsqueeze(0)  # (1, len_total, C)
            f1.append(f2)

        memory = torch.cat(f1, dim=0).unsqueeze(1)  # → (L, 1, T_total, C)

        vid_ids = torch.tensor(vid_ids).numpy()
        batch_target = torch.cat(label_info, dim=0)  # shape: (T_total,)

        # Decoder now returns list of outputs
        mask_logits_list, _ = self.model.decoder(memory)  # (L, B, Q, T)

        loss = 0
        layer_weights = [1.0 for _ in mask_logits_list]
        for logits, w in zip(mask_logits_list, layer_weights):
            logits = logits.squeeze(0).T  # → [T, Q]
            loss += w * self.ce(logits, batch_target)

        return loss

    def train15_1(self, save_dir, batch_gen, batch_gen_gt, writer, num_epochs, batch_size, \
        learning_rate, device,start_epochs,save_f):
        args=self.args
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        header=['Dataset', 'Iteration', 'Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50','Edit', 'Acc','FEA']
     


        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_loss_q = 0
            epoch_loss_c = 0
            correct = 0
            correct2 = 0

            total = 0
            batch_gen_gt.reset2(batch_gen.list_of_examples)
            while batch_gen.has_next():
                batch_input, batch_boundary, mask, batch_confidence ,_,__= batch_gen.next_batch(batch_size)
                batch_input_gt, batch_target, mask_gt, batch_confidence_gt ,_,__= batch_gen_gt.next_batch(batch_size)

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                # if epoch < args.start_epochs:
                batch_boundary1,single_frame_lst = batch_gen.get_single_random2(batch_size, batch_input.size(-1))
                # else:
                #     batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())

                batch_boundary = batch_boundary.to(device)

                loss_b = 0
                for p in predictions:
                    loss_b += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss_b += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss_b += 0.075 * self.confidence_loss(p, batch_confidence, device)

                loss_q=self.get_q_losses(single_frame_lst, middle_pred, batch_target)

                with torch.no_grad():
                    memory = middle_pred.permute(0, 2, 1)
                    mask_logits, class_logits = self.model.decoder(memory)

                loss = loss_b + loss_q #    + loss_c
                epoch_loss_q += loss_q.item()
                # epoch_loss_c += loss_c.item()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()

                _, predicted = torch.max(mask_logits, 1)
                correct2 += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()

                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            batch_gen_gt.reset2(batch_gen.list_of_examples)
            if (epoch+1)%save_f==0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")


            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f, epoch loss q = %f, epoch loss c = %f, acc = %f, acc2 = %f"%(epoch + 1, 
                epoch_loss / len(batch_gen.list_of_examples), epoch_loss_q / len(batch_gen.list_of_examples), epoch_loss_c / len(batch_gen.list_of_examples),
                    float(correct)/total, float(correct2)/total)
                print(print_string)
                fp.write(print_string+'\n')

            if args.gen_type!=0:
                row = [args.dataset,args.iter,args.split,epoch+1] 
                ans=self.train_eva()
                row.extend(ans)
                row.append(sum(ans)/len(ans))
                
                file_name =f'{save_dir}/{args.tst_split}_eval.csv'
                insert_and_sort_csv(file_name, row, header=header)
                print(row)
    
   
  
    # Mon Jun 23 04:36:44 PM PDT 2025
    def train2_2(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        args=self.args
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        args=self.args
        # edit for iteration 
        # header=['Dataset','Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50','Edit', 'Acc','FEA']
        header=['Dataset','Iteration','Split', 'Epoch', 'F1@10', 'F1@25', 'F1@50','Edit', 'Acc','FEA']
        for epoch in range(args.snum_epochs, num_epochs):
            epoch_loss = 0
            epoch_loss_q = 0
            epoch_loss_c = 0
            correct = 0
            correct2 = 0

            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence ,_,__= batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                batch_boundary,single_frame_lst = batch_gen.get_single_random2(batch_size, batch_input.size(-1))
                batch_boundary = batch_boundary.to(device)

                loss_b = 0
                for p in predictions:
                    loss_b += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss_b += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    loss_b += 0.075 * self.confidence_loss(p, batch_confidence, device)

                loss_q=self.get_q_losses(single_frame_lst, middle_pred, batch_target)

              
                with torch.no_grad():
                    memory = middle_pred.permute(0, 2, 1)
                    mask_logits, class_logits = self.model.decoder(memory)

                loss = loss_b + loss_q
                epoch_loss_q += loss_q.item()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()

                _, predicted = torch.max(mask_logits, 1)
                correct2 += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()

                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            if (epoch+1)%args.save_f==0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            with open(save_dir + "/results_file.txt", "a+") as fp:
                print_string = "[epoch %d]: epoch loss = %f, epoch loss q = %f, acc = %f, acc2 = %f"%(epoch + 1, 
                epoch_loss / len(batch_gen.list_of_examples), epoch_loss_q / len(batch_gen.list_of_examples), 
                    float(correct)/total, float(correct2)/total)
                print(print_string)
                fp.write(print_string+'\n')
            if args.gen_type!=0:
                ## edit for iter
                # row = [args.dataset,args.split,epoch+1] 
                row = [args.dataset,args.iter,args.split,epoch+1] 
                ans=self.train_eva()
                row.extend(ans)
                row.append(sum(ans)/len(ans))
                
                file_name =f'{save_dir}/{args.tst_split}_eval.csv'
                ## edit for iter
                # insert_and_sort_csv(file_name, row, header=header)
                insert_and_sort_csv2(file_name, row, header=header)
                print(row)


    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

    def predict2(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        os.system('cp ' + self.args.gt_path + '{}_annotation_all.npy'.format(self.args.dataset)+\
             " "  + results_dir)
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                with open(results_dir+'/'+f_name+'.txt', "w") as fp:
                    fp.write("\n".join(recognition))
                    fp.write("\n")
                # f_ptr = open(results_dir + "/" + f_name, "w")
                # f_ptr.write("### Frame level recognition: ###\n")
                # f_ptr.write(' '.join(recognition))
                # f_ptr.close()
    
    
    def predict3_1(self,vid,pred):
        single_idx = self.random_index[vid]
        vid_gt = self.gt[vid]
        features = pred[0]
        features=features/features.norm(dim=1)[:, None]
        # print(features.shape)
        # return 
        boundary_target = np.ones(vid_gt.shape) * (-100)
        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
        left_bound = [0]
        # self.get_boundary9_2(vid_gt)


        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            for t in range(start + 1, end):
                center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                center_right = torch.mean(features[:, t:end], dim=1)
                score_left,score_right=self.get_boundary9_1(features, center_left,center_right,start,end,t)

                left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)
            # print(left_score)
            # return 
            cur_bound = torch.argmax(left_score) + start + 1
            left_bound.append(cur_bound.item())
            # print(left_bound)
            # return 

        # Backward to find action boundaries
        right_bound = [vid_gt.shape[0]]
        for i in range(len(single_idx) - 1, 0, -1):
            start = single_idx[i - 1]
            end = single_idx[i] + 1
            right_score = torch.zeros(end - start - 1, dtype=torch.float)
            for t in range(end - 1, start, -1):
                center_left = torch.mean(features[:, start:t], dim=1)
                center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                score_left,score_right=self.get_boundary9_1(features, center_left,center_right,start,end,t)

                right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

            cur_bound = torch.argmax(right_score) + start + 1
            right_bound.append(cur_bound.item())
        

        # Average two action boundaries for same segment and generate pseudo labels
        left_bound = left_bound[1:]
        right_bound = right_bound[1:]
        num_bound = len(left_bound)
        for i in range(num_bound):
            temp_left = left_bound[i]
            temp_right = right_bound[num_bound - i - 1]
            middle_bound = int((temp_left + temp_right)/2)
            boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
            boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]


        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target

   
    def predict4(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        os.system('cp ' + self.args.gt_path + '{}_annotation_all.npy'.format(self.args.dataset)+\
             " "  + results_dir)
        self.random_index = np.load(self.args.gt_path + self.args.dataset + "_annotation_all.npy", allow_pickle=True).item()
        # print(random_index.shape)
        # self.action
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                single_idx = self.random_index[vid]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                file_ptr = open(self.args.gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                classes = np.zeros(len(content))
                for i in range(len(classes)):
                    classes[i] = actions_dict[content[i]]
                classes = classes[::sample_rate]
                self.gt[vid] = classes

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                middle_pred, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                # _, predicted = torch.max(predictions[-1].data, 1)
                # predicted = predicted.squeeze()
                predicted=self.predict3_1(vid,middle_pred)
                # print(predicted.shape)
                # print(predicted2.shape)
                # print(middle_pred.shape)
                # return 
                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                with open(results_dir+'/'+f_name+'.txt', "w") as fp:
                    fp.write("\n".join(recognition))
                    fp.write("\n")
    

     # get_boundary21
    def predict22_1(self, vid, pred, m_f1):
        single_idx = self.random_index[vid]
        vid_gt = self.gt[vid]
        
        features = pred[0].transpose(1,0)
        # features=features/features.norm(dim=1,keepdim=True)
        # m_f1=m_f1/m_f1.norm(dim=1,keepdim=True)

        boundary_target = np.ones(vid_gt.shape) * (-100)
        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
        # left_bound = [0]


        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            
            label_s=int(vid_gt[start])
            label_e=int(vid_gt[end-1])

            f2=features[start:end]
            m_f2=m_f1[[label_s,label_e]]
            
            f2=features[start:end].unsqueeze(0)
            m_f2=m_f1[[label_s,label_e]].unsqueeze(1)
            # print(f2.shape)
            # print()
            score2=-torch.norm(f2-m_f2, dim=2)
            pred2 = torch.argmax(score2, dim=0)
            
            guess=torch.zeros_like(pred2)
            # print(pred.shape)
            for t in range(1, end-start):
                # print(t)
                if pred2[t]==pred2[t-1]:continue
                guess[0:t]=0
                guess[t:end-start]=1
                # print(guess)
                correct=(guess==pred2).sum()
                # print(t,correct)
                # print(correct)
                left_score[t-1] = correct
            cur_bound = torch.argmax(left_score)+start + 1
            # boundary_target[start:cur_bound] = 0
            # boundary_target[cur_bound:end+1] = 1
            boundary_target[start:cur_bound] = label_s
            boundary_target[cur_bound:end+1] = label_e


        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target

    
    
        # get_boundary21
    def predict21_1(self, vid, pred, random_index, gt, m_f1):
        vid_gt = gt[vid]
        single_idx = random_index[vid]
        features = pred[0].transpose(1,0)
        print(features.shape)
        exit()
        features=features/features.norm(dim=1,keepdim=True)
        m_f1=m_f1/m_f1.norm(dim=1,keepdim=True)

        boundary_target = np.ones(vid_gt.shape) * (-100)
        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
        # left_bound = [0]


        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            
            label_s=int(vid_gt[start])
            label_e=int(vid_gt[end-1])

            f2=features[start:end]
            m_f2=m_f1[[label_s,label_e]]
            # print(f2.shape)
            # print(m_f2.shape)

            score2=f2@m_f2.transpose(1,0)
            # print(score2.shape)
            # print(score2)
            pred2 = torch.argmax(score2, dim=1)
            # print(pred2)
            # print(vid_gt[start:end])

            guess=torch.zeros_like(pred2)
            # print(pred.shape)
            for t in range(1, end-start):
                # print(t)
                if pred2[t]==pred2[t-1]:continue
                guess[0:t]=0
                guess[t:end-start]=1
                # print(guess)
                correct=(guess==pred2).sum()
                # print(t,correct)
                # print(correct)
                left_score[t-1] = correct
            cur_bound = torch.argmax(left_score)+start + 1
            # boundary_target[start:cur_bound] = 0
            # boundary_target[cur_bound:end+1] = 1
            boundary_target[start:cur_bound] = label_s
            boundary_target[cur_bound:end+1] = label_e

        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target
    
    # 基于关键帧，两类标签分割，枚举所有切点，选择预测一致性最高的分割点
    def predict23_1(self, vid, pred, random_index, gt):
        single_idx = random_index[vid]
        vid_gt = gt[vid]
        # print(pred.shape)
        # exit()
        if pred.dim() == 3:
            memory = pred.permute(0, 2, 1)
        else:
            memory = pred.permute(0, 1, 3, 2)

        mask_logits, class_logits = self.model.decoder(memory)

        if self.args.decoder == 'v3' or self.args.decoder == 'v4':
            mask_logits = mask_logits[-1]
        # features = mask_logits.squeeze(0)

        # print(mask_logits.shape, class_logits.shape)
        # exit()


        boundary_target = np.ones(vid_gt.shape) * (-100)
        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label

        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            
            label_s=int(vid_gt[start])
            label_e=int(vid_gt[end-1])

            # f2=features[start:end]
           
            score2 =  mask_logits[0, [label_s, label_e], start:end]
            # print(start,end)
            # print(score2.shape)
            # exit()
            pred2 = torch.argmax(score2, dim=0)
            # print(pred2)
            
            guess=torch.zeros_like(pred2)
            # print(pred.shape)
            for t in range(1, end-start):
                # print(t)
                if pred2[t]==pred2[t-1]:continue
                guess[0:t]=0
                guess[t:end-start]=1
                # print(guess)
                correct=(guess==pred2).sum()
                # print(t,correct)
                # print(correct)
                left_score[t-1] = correct
            cur_bound = torch.argmax(left_score)+start + 1
            # print(cur_bound)
            # exit()
            boundary_target[start:cur_bound] = label_s
            boundary_target[cur_bound:end+1] = label_e


        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target
    
    # 基于关键帧，两类标签分割，直接比较两个类别 logit 得分，逐帧赋值
    def predict23_2(self, vid, pred, random_index, gt):
        args = self.args
        single_idx = random_index[vid]
        vid_gt = gt[vid]
        # print(pred.shape)
        # exit()
        if pred.dim() == 3:
            memory = pred.permute(0, 2, 1)
        else:
            memory = pred.permute(0, 1, 3, 2)

        mask_logits, class_logits = self.model.decoder(memory)

        if self.args.decoder == 'v3' or self.args.decoder == 'v4':
            mask_logits = mask_logits[-1]
        # features = mask_logits.squeeze(0)

        # print(mask_logits.shape, class_logits.shape)
        # exit()


        boundary_target = np.ones(vid_gt.shape) * (-100)
        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label

        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            
            label_s=int(vid_gt[start])
            label_e=int(vid_gt[end-1])

            # 得到得分矩阵，2 行分别是 label_s, label_e
            score2 = mask_logits[0, [label_s, label_e], start:end]  # [2, N]

            # 对这2行做 argmax，得到是label_s还是label_e
            which = torch.argmax(score2, dim=0)  # shape: [N], 每个值为 0 (label_s) 或 1 (label_e)

            # 通过 index 把 0/1 变成真实的 label_s / label_e
            boundary_target[start:end] = torch.tensor([label_s, label_e], device=score2.device)[which].cpu().numpy()

        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target
    
    # 基于整段预测，直接对 decoder mask_logits 做 argmax 推断
    def predict23_3(self, vid, pred, random_index, gt):
        args = self.args
        single_idx = random_index[vid]
        vid_gt = gt[vid]
        if pred.dim() == 3:
            memory = pred.permute(0, 2, 1)
        else:
            memory = pred.permute(0, 1, 3, 2)

        mask_logits, class_logits = self.model.decoder(memory)

        if self.args.decoder == 'v3' or self.args.decoder == 'v4':
            mask_logits = mask_logits[-1]

        ret1=torch.argmax(mask_logits[0, :, :], dim=0)
        return ret1
        # print(ret1.shape, vid_gt.shape)
  

    # Sat Jul  5 01:28:37 PM PDT 2025
    # baseline 2021 基于特征变化，无需分类头，通过最小化前后段方差估计边界位置
    def predict23_4(self, vid, pred, random_index, gt):
        # This function is to generate pseudo labels

        single_idx = random_index[vid]
        vid_gt = gt[vid]
        boundary_target = np.ones(vid_gt.shape) * (-100)

        features = pred[0]

        # print(features.shape, gt[vid].shape)
        # exit()

        boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
        left_bound = [0]

        # Forward to find action boundaries
        for i in range(len(single_idx) - 1):
            start = single_idx[i]
            end = single_idx[i + 1] + 1
            left_score = torch.zeros(end - start - 1, dtype=torch.float)
            for t in range(start + 1, end):
                center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                score_left = torch.mean(torch.norm(diff_left, dim=0))

                center_right = torch.mean(features[:, t:end], dim=1)
                diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                score_right = torch.mean(torch.norm(diff_right, dim=0))

                left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

            cur_bound = torch.argmin(left_score) + start + 1
            left_bound.append(cur_bound.item())

        # Backward to find action boundaries
        right_bound = [vid_gt.shape[0]]
        for i in range(len(single_idx) - 1, 0, -1):
            start = single_idx[i - 1]
            end = single_idx[i] + 1
            right_score = torch.zeros(end - start - 1, dtype=torch.float)
            for t in range(end - 1, start, -1):
                center_left = torch.mean(features[:, start:t], dim=1)
                diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                score_left = torch.mean(torch.norm(diff_left, dim=0))

                center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                score_right = torch.mean(torch.norm(diff_right, dim=0))

                right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

            cur_bound = torch.argmin(right_score) + start + 1
            right_bound.append(cur_bound.item())

        # Average two action boundaries for same segment and generate pseudo labels
        left_bound = left_bound[1:]
        right_bound = right_bound[1:]
        num_bound = len(left_bound)
        for i in range(num_bound):
            temp_left = left_bound[i]
            temp_right = right_bound[num_bound - i - 1]
            middle_bound = int((temp_left + temp_right)/2)
            boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
            boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]

        boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
        return boundary_target


    def compute_class_prototypes(features_path, device):
        # Step 1: Collect features per class for prototype
        print("Building class prototypes from GT...")
        class_feats = {}  # key: class_id -> list of [C]
        for cid in actions_dict.values():
            class_feats[cid] = []

        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')[:, ::sample_rate]  # [C, T]
            
            with open(self.args.gt_path + vid, 'r') as f:
                content = f.read().split('\n')[:-1]
            gt_labels = np.array([actions_dict[l] for l in content])[::sample_rate]  # [T]

            assert features.shape[1] == len(gt_labels), f"Mismatch in {vid}: {features.shape[1]} vs {len(gt_labels)}"

            for t in range(len(gt_labels)):
                class_feats[gt_labels[t]].append(features[:, t])  # 1 vector

        # Average to get prototype
        prototypes = []
        for cid in range(len(actions_dict)):
            if class_feats[cid]:
                feats = np.stack(class_feats[cid], axis=0)  # [N, C]
                prototypes.append(np.mean(feats, axis=0))
            else:
                prototypes.append(np.zeros_like(features[:, 0]))  # zero if no data

        self.class_prototypes = torch.tensor(np.stack(prototypes, axis=0), dtype=torch.float).to(device)
        print("Prototype built with shape:", self.class_prototypes.shape)
        return prototypes

    # ZZHC Wed Jul  9 07:19:36 PM PDT 2025
    # baseline 5 只使用average Q
    def predict23_5(self, vid, pred, random_index, gt):
        vid_gt = gt[vid]
        features = pred[0]  # [C, T]
        C, T = features.shape
        prototypes = self.class_prototypes.to(features.device)  # [num_classes, C]

        # [T, C]
        features_t = features.permute(1, 0)  # [T, C]

        # 计算每帧与所有类别 prototype 的欧氏距离：[T, num_classes]
        dists = torch.cdist(features_t.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)  # [T, num_classes]

        # 每帧分配最近类别
        pred_labels = torch.argmin(dists, dim=1).cpu().numpy()  # [T]

        return pred_labels

    def predict24(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        args = self.args
        actions_dict = args.actions_dict
        vid_list_file = args.vid_list_file_tst
        print(vid_list_file)
        
        features_path = args.features_path
        sample_rate = args.sample_rate

        self.model.eval()
        self.model.to(device)
        results_dir=save_dir + '/p24/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with torch.no_grad():
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            # if args.baseline == 5:
            #     prototype_Q = compute_class_prototypes(features_path, self.args.gt_path device)
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                file_ptr = open(self.args.gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                classes = np.zeros(len(content))
                for i in range(len(classes)):
                    classes[i] = actions_dict[content[i]]
                classes = classes[::sample_rate]
                self.gt[vid] = classes

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                middle_pred, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                predicted = torch.argmax(predictions[-1,0, :, :], dim=0)
                # print(self.gt[vid].shape, predicted.shape, predictions.shape )
                # exit()

                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                txt_path = results_dir+'/'+f_name+'.txt'
                print(txt_path)
                with open(txt_path, "w") as fp:
                    fp.write("\n".join(recognition))
                    fp.write("\n")

    def predict23(self, batch_gen_gt, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        args = self.args
        self.model.eval()
        self.model.to(device)
        os.system('cp ' + self.args.gt_path + '{}_annotation_all.npy'.format(self.args.dataset)+ \
             " "  + results_dir)
        self.random_index = np.load(self.args.gt_path + self.args.dataset + "_annotation_all.npy", allow_pickle=True).item()

        if args.baseline == 21:
            m_f1=self.get_mf1(batch_gen_gt,1,device)

        action_map = {
            # TQT
            1: self.predict23_1,
            # Table 4, Timestamp Query, no optimization
            2: self.predict23_2,
            # Table 4, Class Query
            3: self.predict23_3,
            4: self.predict23_4,
            5: self.predict23_5,
            21:self.predict21_1,
        }
        # torch.Size([11, 64])
        # print(m_f1.shape)
        # exit()

        with torch.no_grad():
            self.model.to(device)
            if self.args.pre_num_epochs != -1:
                self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            else:
                self.model.load_state_dict(torch.load(model_dir + "/" + "train" + "_" + str(self.args.iter) + "_acc.model"))

            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            # if args.baseline == 5:
            #     prototype_Q = compute_class_prototypes(features_path, self.args.gt_path device)
            for vid in list_of_vids:
                # print(vid)
                single_idx = self.random_index[vid]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                file_ptr = open(self.args.gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                classes = np.zeros(len(content))
                for i in range(len(classes)):
                    classes[i] = actions_dict[content[i]]
                classes = classes[::sample_rate]
                self.gt[vid] = classes

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                middle_pred, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                # print(middle_pred.shape)
                # exit()
                if args.baseline in [1, 2, 3, 4, 5]:
                    predicted=action_map[args.baseline](vid,middle_pred, self.random_index, self.gt)  # self.predict23_2(vid,middle_pred, self.random_index, self.gt)
                elif args.baseline == 21:
                    predicted=action_map[args.baseline](vid,middle_pred, self.random_index, self.gt, m_f1)
                #     predicted=action_map[args.baseline](vid, predictions[-1], m_f1)

                # print(predicted.shape)
                # exit()

                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                with open(results_dir+'/'+f_name+'.txt', "w") as fp:
                    fp.write("\n".join(recognition))
                    fp.write("\n")
                    
    def process_file(self, vid):
        device= self.args.device

        input_x = torch.tensor(self.batch_gen_eva.features[vid], dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(device)
        if self.args.num_codes==-1:
            middle_preds, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
        else:
            middle_preds, predictions, _, __ = self.model(input_x, torch.ones(input_x.size(), device=device))
        _, predicted = torch.max(predictions[-1].data, 1)

        predicted = predicted.squeeze().cpu().numpy()
        if self.args.gen_type==23:
            
            predicted = self.predict23_1(vid,middle_preds, self.batch_gen_eva.random_index, self.batch_gen_eva.gt)

        gt=self.batch_gen_eva.gt[vid]

        

        correct=sum(predicted==gt)
        total=gt.shape[0]

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        
        edit = 0
        gt_content=gt
        recog_content=predicted

        bg_idx=self.args.bg_idx
        p_trans=remove_duplicates(predicted, bg_idx)
        g_trans=remove_duplicates(gt, bg_idx)

        edit += levenstein(g_trans, p_trans, norm=True)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s],bg_idx)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        # print(correct0,correct,total)
        # exit()
        return [correct,total,edit,tp,fp,fn]
