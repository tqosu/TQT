#!/usr/bin/python3.6

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate,args):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}
        self.features = {}
        self.args=args

        dataset_name = gt_path.split('/')[2]
        self.random_index = np.load(gt_path + dataset_name + "_annotation_all.npy", allow_pickle=True).item()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def reset2(self,list_of_examples2):
        self.index = 0
        self.list_of_examples=list_of_examples2

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        self.generate_confidence_mask()

    def generate_confidence_mask(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            self.gt[vid] = classes
            num_frames = classes.shape[0]

            random_idx = self.random_index[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_idx) - 1):
                left_mask[int(classes[random_idx[j]]), random_idx[j]:random_idx[j + 1]] = 1
                right_mask[int(classes[random_idx[j + 1]]), random_idx[j]:random_idx[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            self.features[vid] = features[:, ::self.sample_rate]

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        for vid in batch:
            # features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            # batch_input.append(features[:, ::self.sample_rate])
            batch_input.append(self.features[vid])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask =  torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        vlen =torch.zeros(len(batch_input), dtype=torch.int)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            vlen[i]=np.shape(batch_target[i])[0]
        return batch_input_tensor, batch_target_tensor, mask, batch_confidence, vlen, batch

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def get_single_random2(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        single_frame_lst=[]
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            single_frame_lst.append(frame_idx_tensor)
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor,single_frame_lst

    def get_single_random3(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        single_frame_lst=[]
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            single_frame2=[]
            for i in range(1,len(single_frame)):
                idx1=random.randint(single_frame[i-1], single_frame[i])
                single_frame2.append(idx1)
            single_frame=single_frame2
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            single_frame_lst.append(frame_idx_tensor)
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor,single_frame_lst

    def get_all(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            # single_frame = self.random_index[vid]
            gt = self.gt[vid]
            # frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            len_gt_tensor=gt_tensor.shape[0]
            # print(gt_tensor.shape)
            # print(boundary_target_tensor[b].shape)
            # return 
            boundary_target_tensor[b, :len_gt_tensor] = gt_tensor

        return boundary_target_tensor

    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
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
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor

    # global distance
    def get_boundary22(self, batch_size, pred, m_f1):
        # This function is to generate pseudo labels
        
        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)
        # m_f1=m_f1/m_f1.norm(dim=1,keepdim=True)
        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b].transpose(1,0)
            # features (T,64)
            # features.norm(dim=0) T

            # print(features.norm(dim=1,keepdim=True).shape)
            # print(features.shape)
            # print(m_f1.shape)
            # features=features/features.norm(dim=1,keepdim=True)
            # features=features.norm(dim=1,keepdim=True)
            # return 

            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                
                label_s=int(vid_gt[start])
                label_e=int(vid_gt[end-1])

                f2=features[start:end].unsqueeze(0)
                m_f2=m_f1[[label_s,label_e]].unsqueeze(1)
                # print(f2.shape)
                # print(m_f2.shape)
                score2=-torch.norm(f2-m_f2, dim=2)
                # print(score2.shape)
                # print(score2)
                pred2 = torch.argmax(score2, dim=0)
                # print(pred2)

                # return 

                # score2=f2@m_f2.transpose(1,0)
                # # print(score2.shape)
                # # print(score2)
                # pred2 = torch.argmax(score2, dim=1)
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
                # guess[0:cur_bound+1]=0
                # guess[cur_bound+1,end]=0
                # cur_bound = cur_bound +start + 1
                # print(pred2)
                # print(vid_gt[start:end])
                # print(boundary_target[start:end])
                # print()
            

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor

    def get_boundary17_1(self, center_p, center_n, features):
        sim_f1 = (center_p.reshape(1,-1)@features)
        sim_f1 = torch.exp(sim_f1)

        sim_f2 = (center_n.reshape(1,-1)@features)
        sim_f2 = torch.exp(sim_f2)

        con_p = (sim_f1)/(sim_f1+sim_f2)

        return con_p,torch.log(con_p)

    # based on cosine similarity
    # based on norm
    def get_boundary17(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            # features (64,T)
            # features.norm(dim=0) T
            features_n=features/features.norm(dim=0)[None,:]

            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left  = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    center_left_n= center_left/torch.norm(center_left)

                    center_right = torch.mean(features[:, t:end], dim=1)
                    center_right_n= center_right/torch.norm(center_right)

                    con_p,con_l=self.get_boundary17_1(center_left_n,center_right_n,features_n[:, start:t])
                    score_left = torch.mean(con_p)

                    con_p,con_l=self.get_boundary17_1(center_right_n,center_left_n,features_n[:, t:end])
                    score_right = torch.mean(con_p)

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmax(left_score) + start + 1

                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    center_left_n= center_left/torch.norm(center_left)
                    
                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    center_right_n= center_right/torch.norm(center_right)

                    con_p,con_l=self.get_boundary17_1(center_left_n,center_right_n,features_n[:, start:t])
                    score_left = torch.mean(con_p)

                    con_p,con_l=self.get_boundary17_1(center_right_n,center_left_n,features_n[:, t:end])
                    score_right = torch.mean(con_p)
                
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
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor

    # contrastive loss
    def get_boundary18(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            # features (64,T)
            # features.norm(dim=0) T
            features_n=features/features.norm(dim=0)[None,:]

            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left  = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    center_left_n= center_left/torch.norm(center_left)

                    center_right = torch.mean(features[:, t:end], dim=1)
                    center_right_n= center_right/torch.norm(center_right)

                    con_p,con_l=self.get_boundary17_1(center_left_n,center_right_n,features_n[:, start:t])
                    score_left = torch.mean(con_l)

                    con_p,con_l=self.get_boundary17_1(center_right_n,center_left_n,features_n[:, t:end])
                    score_right = torch.mean(con_l)

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmax(left_score) + start + 1

                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    center_left_n= center_left/torch.norm(center_left)
                    
                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    center_right_n= center_right/torch.norm(center_right)

                    con_p,con_l=self.get_boundary17_1(center_left_n,center_right_n,features_n[:, start:t])
                    score_left = torch.mean(con_l)

                    con_p,con_l=self.get_boundary17_1(center_right_n,center_left_n,features_n[:, t:end])
                    score_right = torch.mean(con_l)
                
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
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor
    
    # contrastive loss based on prototype
    def get_boundary21(self, batch_size, pred, m_f1):
        # This function is to generate pseudo labels
        
        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)
        m_f1=m_f1/m_f1.norm(dim=1,keepdim=True)
        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b].transpose(1,0)
            # features (T,64)
            # features.norm(dim=0) T

            # print(features.norm(dim=1,keepdim=True).shape)
            # print(features.shape)
            # print(m_f1.shape)
            features=features/features.norm(dim=1,keepdim=True)
            # features=features.norm(dim=1,keepdim=True)
            # return 

            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label

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
                # guess[0:cur_bound+1]=0
                # guess[cur_bound+1,end]=0
                # cur_bound = cur_bound +start + 1
                # print(pred2)
                # print(vid_gt[start:end])
                # print(boundary_target[start:end])
                # print()
            

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor

    def get_boundary11_4(self, batch_size,  pred2, results_dir,actions_dict,sample_rate):

        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, max_frames = pred2.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]

            # predicted=np.array(pred[b].cpu())[:vid_gt.shape[0]]
            predicted=np.array(pred2[b].cpu())[:vid_gt.shape[0]]
     
            print(vid)
            # correct = (predicted == vid_gt).sum().item()
            # print(correct/vid_gt.shape[0])
                
            recognition = []
            print(list(actions_dict.values()))
            for i in range(len(predicted)):
                index = list(actions_dict.values()).index(predicted[i].item())
                
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
    
            # s='{:.3f}_{:.3f}_{:.3f}'.format(correct3/total,correct1/total,correct2/total)+'_'
            f_name = vid.split('/')[-1].split('.')[0]
            # f_name = s+vid.split('/')[-1].split('.')[0]
            with open(results_dir+'/'+f_name+'.txt', "w") as fp:
                fp.write("\n".join(recognition))
                fp.write("\n")

    def get_boundary2(self, count, outp1, actual_labels, activity_labels):
        # This function is to generate pseudo labels
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
        
        # part2
        batch_size=outp1.shape[0] 
        pred=outp1
        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
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
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor
    # ZZHC for SHIFT CENTER
    def get_center_frames_from_gt(self, gt: np.ndarray):

        centers = []
        prev_label = gt[0]
        start = 0
        for i in range(1, len(gt)):
            if gt[i] != prev_label:
                center = (start + i - 1) // 2
                centers.append(center)
                start = i
                prev_label = gt[i]
        # 最后一个 segment
        center = (start + len(gt) - 1) // 2
        centers.append(center)
        return centers