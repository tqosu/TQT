***

## Data Setup

1. Download the feature and label data following the instructions in the TimestampActionSeg repository:  
   [https://github.com/ZheLi2020/TimestampActionSeg](https://github.com/ZheLi2020/TimestampActionSeg).
2. Place the downloaded data under the `data/` subfolder of this project (for example, `data/gtea`, `data/breakfast`, `data/50salads`).

***

## Step 1: Pretraining

Run the following command to pretrain the backbone model (example for GTEA, split 1):

```bash
python main.py \
  --action train2_2 \
  --num_epochs 60 \
  --dataset gtea \
  --split 1 \
  --time_data 2025-06-27_14-00-00-A \
  --gen_type 23
```

- `time_data` specifies the folder where model checkpoints and logs will be saved.  
- `gen_type` > 0 enables evaluation after each epoch.  
- In practice, 50~70 pretraining epochs are typically sufficient for Breakfast, GTEA, and 50Salads.

***

## Step 2: Iterative Training with Pseudo Labels

### Paper Settings
The following epochs were used in the paper for each dataset:

| Dataset    | Iteration | Step 2.1 (Pseudo-label Training) | Step 2.2 (Query Refinement) |
|------------|-----------|----------------------------------|-----------------------------|
| Breakfast | 1         | 50 epochs                       | 10 epochs  |
| GTEA      | 1         | 50 epochs                       | 20 epochs  |
| 50salads  | 1         | 50 epochs                       | 20 epochs  |
| 50salads  | 2         | 20 epochs                       | 20 epochs  |

### 2.1 Pseudo-label training (50 epochs)

```bash
python main7_1.py \
  --action train15_1 \
  --pre_time_data 2025-06-27_14-00-00-A \
  --num_epochs 50 \
  --split 1 \
  --dataset gtea \
  --tst_split test \
  --time_data 2025-06-27_14-00-00-B \
  --iter 0 \
  --baseline 1
```

- Loads the pretrained checkpoint from `pre_time_data`.  
- Generates pseudo labels using the query module and trains for 50 epochs.  
- Saves the new checkpoint under `time_data`, selecting the best epoch based on the validation set.  
- `baseline 1` corresponds to the TQT setting described in the paper.

### 2.2 Query refinement

```bash
python main.py \
  --action train2_2 \
  --num_epochs 5 \
  --dataset gtea \
  --split 1 \
  --gen_type 23 \
  --time_data 2025-06-27_14-00-00-A \
  --pretrain_weights models/gtea/margin_map_both2025-06-27_14-00-00-B/split_1/epoch-49.model \
  --iter 1
```

- Refines the query module using the checkpoint obtained in Step 2.1 (`pretrain_weights`).  
- Runs a short training (e.g., 5 epochs) to update the queries and improve pseudo labels.

### 2.3 Final pseudo-label generation

Steps 2.1 and 2.2 can be repeated a few times as needed (e.g., per the paper settings above).  
To generate final pseudo labels without additional training:

```bash
python main7_1.py \
  --action train15_1 \
  --pre_time_data 2025-06-27_14-00-00-A \
  --split 1 \
  --dataset gtea \
  --tst_split test \
  --time_data 2025-06-27_14-00-00-C1 \
  --iter 0 \
  --baseline 1 \
  --stop 1
```

- Setting `--stop 1` generates pseudo labels only and skips training.

***

## Step 3: Training ASFormer with Pseudo Labels

Once the final pseudo labels are generated, they can be used to train ASFormer, following the protocol used in previous work:  
[https://github.com/ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer).