#Path: /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/config.py
import os
import os.path as osp
import sys
import numpy as np
import datetime
import inspect

class Config:
    
    ## dataset
    dataset = 'COCO' # 'COCO', 'PoseTrack', 'MPII'
    testset = 'test' # train, test, val (there is no validation set for MPII)

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dump_dir = osp.join(output_dir, 'model_dump', dataset)
    #modified
    model_pretrained_dir = osp.join(output_dir, 'model_dump', dataset)
    #modified end
    predict_inputmodel_file = None 

    

    vis_dir = osp.join(output_dir, 'vis', dataset)
    log_dir = osp.join(output_dir, 'log', dataset)
    result_dir = osp.join(output_dir, 'result', dataset)
 
    ## model setting
    backbone = 'resnet50' # 'resnet50', 'resnet101', 'resnet152'
    init_model = osp.join(data_dir, 'imagenet_weights', 'resnet_v1_' + backbone[6:] + '.ckpt')
    
    ## input, output
    input_shape = (384, 288) # (256,192), (384,288)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    if output_shape[0] == 64:
        input_sigma = 7.0
    elif output_shape[0] == 96:
        input_sigma = 9.0
    pixel_means = np.array([[[123.68, 116.78, 103.94]]])

    ## training config
    lr_dec_epoch = [90, 120]
    end_epoch = 140
    lr = 5e-4
    lr_dec_factor = 10
    optimizer = 'adam'
    weight_decay = 1e-5
    bn_train = True
    batch_size = 16 #32
    scale_factor = 0.3
    rotation_factor = 40

    ## testing config
    flip_test = True
    oks_nms_thr = 0.9
    test_batch_size = 32

    ## others
    multi_thread_enable = True
    num_thread = 10
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    display = 1
    #modified
    loss_to_config_period = 2#50
    #modified end
    
    ## helper functions
    def get_lr(self, epoch):
        for e in self.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.lr_dec_epoch[-1]:
            i = self.lr_dec_epoch.index(e)
            return self.lr / (self.lr_dec_factor ** i)
        else:
            return self.lr / (self.lr_dec_factor ** len(self.lr_dec_epoch))
    
    def normalize_input(self, img):
        return img - self.pixel_means

    def denormalize_input(self, img):
        return img + self.pixel_means

    #modified: CUDA_VISIBLE_DEVICES not supported
    def set_args(self, gpu_ids, predictPath, continue_train=False):
        #self.gpu_ids = gpu_ids
        #self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        self.predict_inputmodel_file = predictPath
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        #print('>>> Using /gpu:{}'.format(self.gpu_ids))
    #modified end

    def set_modeldir_for_test(self, model_dir):
        self.model_dump_dir = model_dir
         # Save the config to file
    
    #modified
    def saveconfig(self, outputdir, args=None, params_train=None):
        print("Saving configs to folder: ",outputdir)
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attr = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        
        with open(os.path.join(outputdir, 'model_config.txt'), 'w') as f:
            f.write(' '.join(["%s = %s \n" % (k,v) for (k,v) in attr])) 
            #f.write(' '.join(["%s = %s \n" % (k,v) for k,v in vars(self).items()])) 

        with open(os.path.join(outputdir, 'config.txt'), 'w') as f:
            if args is not None: #prediction
                f.write("Model folder: %s"%args.modelfolder + os.linesep)
                f.write("Model epoch: %s"%args.test_epoch + os.linesep)
                f.write("Input file: %s"%args.inputs + os.linesep)
                f.write("Corresponding image folder: %s"%args.imagefolder + os.linesep)
            else: #training
                f.write("Input file: %s"%params_train[0] + os.linesep)
                f.write("Continue train: %s"%params_train[1] + os.linesep)


        print("Saving configs done.")
    #modified end

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'lib'))
from tfflat.utils import add_pypath, make_dir
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
#make_dir(cfg.model_dump_dir)
make_dir(cfg.vis_dir)
make_dir(cfg.log_dir)
make_dir(cfg.result_dir)

from dataset import dbcfg
cfg.num_kps = dbcfg.num_kps
cfg.kps_names = dbcfg.kps_names
cfg.kps_lines = dbcfg.kps_lines
cfg.kps_symmetry = dbcfg.kps_symmetry
cfg.kps_sigmas = dbcfg.kps_sigmas
cfg.ignore_kps = dbcfg.ignore_kps
cfg.train_img_path = dbcfg.train_img_path
cfg.vis_keypoints = dbcfg.vis_keypoints



