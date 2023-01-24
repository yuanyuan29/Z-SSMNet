# This code is adapted from https://github.com/MrGiovanni/ModelsGenesis/blob/master/competition/config.py. 
# The original code is licensed under the attached LICENSE (https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/ssl/LICENSE).

import os
import shutil

class Config:
    DATA_DIR = "/workdir/SSL/generated_cubes"
    nb_epoch = 1000
    patience = 20
    lr = 1e-1
    train_fold=[0,1,2,3,4]
    valid_fold=[5]
    
    adc_max = 3000.0
    adc_min = 0.0
    def __init__(self, 
                 note="",
                 data_augmentation=True,
                 input_rows=64, 
                 input_cols=64,
                 input_deps=16,
                 batch_size=24,
                 weights=None,
                 nb_class=3,
                 nonlinear_rate=0.9,
                 paint_rate=0.9,
                 outpaint_rate=0.8,
                 rotation_rate=0.0,
                 flip_rate=0.4,
                 local_rate=0.5,
                 verbose=1,
                 scale=12,
                ):
        self.exp_name = "ssl_mnet_zonal"
        self.data_augmentation = data_augmentation
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.nonlinear_rate = nonlinear_rate
        self.paint_rate = paint_rate
        self.outpaint_rate = outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.rotation_rate = rotation_rate
        self.flip_rate = flip_rate
        self.local_rate = local_rate
        self.nb_class = nb_class
        self.scale = scale
        self.weights = weights

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
