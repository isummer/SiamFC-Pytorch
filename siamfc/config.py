"""
Configuration for training SiamFC and tracking evaluation
Written by Heng Fan
"""


class Config:
    def __init__(self):
        # parameters for training
        self.from_range = 100
        self.num_pairs = 53200
        self.val_ratio = 0.1
        self.num_epochs = 50
        self.batch_size = 8
        self.exemplar_size = 127
        self.instance_size = 255
        self.sub_mean = 0

        self.train_batch_size = 32
        self.val_batch_size = 8
        self.train_num_workers = 12  # number of threads to load data when training
        self.val_num_workers = 8

        self.total_stride = 4 # 8
        self.rPos = 8 # 16
        self.rNeg = 0
        self.label_weight_method = "balanced"

        self.radius = 8 # 16
        self.train_response_sz = 31 # 15
        self.response_sz = 33 # 17

        self.lr = 1e-2               # learning rate of SGD
        self.momentum = 0.9          # momentum of SGD
        self.weight_decay = 5e-4     # weight decay of optimizator
        self.step_size = 1           # step size of LR_Schedular
        self.gamma = 0.8685          # decay rate of LR_Schedular

        # parameters for tracking (SiamFC-3s by default)
        self.num_scale = 3
        self.scale_step = 1.0816 # 1.0375
        self.scale_penalty = 0.97 # 0.9745
        self.scale_LR = 0.59
        self.response_UP = 8 # 16
        self.windowing = "cosine"
        self.w_influence = 0.25 # 0.176

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 33 # 17

config = Config()
