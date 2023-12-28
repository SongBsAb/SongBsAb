
import pickle
import numpy as np
import torch

# from model.ResNet import ResNet
# from model.AutoSpeech.models.model import Network
# from model.AutoSpeech.data_objects.params_data import *
import sys
sys.path.append('../../')
from model.ResNet import ResNet
from models.model import Network
from data_objects.params_data import *

MODEL_LAYERS = 8 # same for iden and veri
MODEL_INIT_CHANNELS = 128 # same for iden and veri

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
text_arch = "Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), \
    ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), \
        ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], \
            normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), \
                ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), \
                    ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], \
                        reduce_concat=range(2, 6))" # the best arch reported in the paper
genotype = eval(text_arch)

class AutoSpeechScratch(ResNet):

    def __init__(self, encoder_ckpt, model_file=None, mean_file=None, device='cpu', 
                transform_layer=None, transform_param=None, threshold=None,
                feat_mean_path=None, feat_std_path=None):
        
        torch.nn.Module.__init__(self)

        checkpoint = torch.load(encoder_ckpt)
        # print(checkpoint.keys())
        # for k, v in checkpoint.items():
        #     if isinstance(v, dict):
        #         print(k, v.keys())
        #         print()
        #     else:
        #         print(k, v)
        print(checkpoint['genotype'])
        # torch.save({'state_dict': checkpoint['state_dict'], 'genotype': checkpoint['genotype']}, '../../autospeech_sctrach_iden_ckpt')
        # torch.save({'state_dict': checkpoint['state_dict']}, '../../autospeech_sctrach_iden_ckpt')
        torch.save({'state_dict': checkpoint['state_dict']}, '../../model_file/AutoSpeech/autospeech_sctrach_veri_ckpt')
        # print(checkpoint['state_dict'].keys())
        # print(checkpoint['genotype'])
        MODEL_NUM_CLASSES = checkpoint['state_dict']['classifier.weight'].shape[0]
        # genotype = checkpoint['genotype']
        self.encoder = Network(MODEL_INIT_CHANNELS, MODEL_NUM_CLASSES, MODEL_LAYERS, genotype)
        # load checkpoint
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.encoder.to(device).eval()
        assert self.encoder.training == False

        if mean_file is None:
            self.emb_mean = 0
        else:
            with open(mean_file, 'rb') as reader:
                self.emb_mean = pickle.load(reader).squeeze(0).to(device) # (emb_dim, )
        
        self.device = device

        if model_file is not None:
            self.model_file = model_file
            model_info = np.loadtxt(self.model_file, dtype=str)
            if len(model_info.shape) == 1:
                model_info = model_info[np.newaxis, :] # for SV
            self.num_spks = model_info.shape[0]
            self.spk_ids = list(model_info[:, 0])
            self.identity_locations = list(model_info[:, 1])
            
            self.z_norm_means = (model_info[:, 2]).astype(
                np.float32)  # float32, make consistency
            self.z_norm_stds = (model_info[:, 3]).astype(
                np.float32)  # float32, make consistency
            self.z_norm_means = torch.tensor(self.z_norm_means, device=self.device)
            self.z_norm_stds = torch.tensor(self.z_norm_stds, device=self.device)

            self.enroll_embs = None
            for index, path in enumerate(self.identity_locations):
                emb = torch.load(path, map_location=self.device).unsqueeze(0)
                if index == 0:
                    self.enroll_embs = emb
                else:
                    self.enroll_embs = torch.cat([self.enroll_embs, emb], dim=0)
            self.enroll_embs = self.enroll_embs.squeeze(1)
        
        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: infty

        assert transform_layer is None and transform_param is None

        # feature mean and std
        if feat_mean_path is not None:
            # feat_mean = np.load(feat_mean_path)
            feat_mean = torch.load(feat_mean_path, map_location=self.device)
            assert feat_mean.shape[0] == n_fft // 2 + 1
            self.feat_mean = torch.tensor(feat_mean, dtype=torch.float).to(self.device)
        else:
            self.feat_mean = 0
        if feat_std_path is not None:
            # feat_std = np.load(feat_std_path)
            feat_std = torch.load(feat_std_path, map_location=self.device)
            assert feat_std.shape[0] == n_fft // 2 + 1
            self.feat_std = torch.tensor(feat_std, dtype=torch.float).to(self.device)
        else:
            self.feat_std = 1

import torch
# import os
import numpy as np
# import torchaudio
import pickle

# from model.xvector_PLDA_helper import xvector_PLDA_helper
# from model.ivector_PLDA_helper import ivector_PLDA_helper

# from model.ECAPA_CSINE import ECAPA_CSINE
# from model.AudioNet_CSINE import AudioNet_CSINE
# from model.XV_CSINE import XV_CSINE
# from model.SincNet import SincNet
# from model.ResNet18 import ResNet
# from model.AutoSpeechScratch import AutoSpeechScratch
# from AutoSpeechScratch import AutoSpeechScratch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):

    #Step 1: set up system helper
    # if args.system_type == 'iv':
    #     helper = ivector_PLDA_helper(args.gmm, args.extractor, 
    #             args.plda, args.mean, args.transform, device=device)
    # elif args.system_type == 'xv':
    #     helper = xvector_PLDA_helper(args.extractor, args.plda, args.mean, args.transform, device=device)
    if False:
        pass
    elif args.system_type == 'ecapa_csine':
        model = ECAPA_CSINE(args.extractor, device=device)
    elif args.system_type == 'audionet_csine':
        model = AudioNet_CSINE(args.extractor, device=device)
    elif args.system_type == 'xv_csine':
        model = XV_CSINE(args.extractor, device=device)
    elif args.system_type == 'sinc_csine':
        model = SincNet(args.extractor, args.cfg_file, device=device)
    # elif args.system_type == 'resnet18_iden':
    elif 'resnet' in args.system_type:
        MODEL_NAME = args.system_type.split('_')[0]
        model = ResNet(MODEL_NAME, args.extractor, device=device, feat_mean_path=args.feat_mean,
                            feat_std_path=args.feat_std)
    elif 'autospeech_scratch' in args.system_type:
        model = AutoSpeechScratch(args.extractor, device=device, feat_mean_path=args.feat_mean,
                            feat_std_path=args.feat_std)
    else:
        raise NotImplementedError('Unsupported System Type')
    

    # # with open('vox1-dev.pickle', 'rb') as reader:
    # with open('../../vox1-dev.pickle', 'rb') as reader:
    #     print('begin loading data')
    #     loader = pickle.load(reader)
    #     print('load data done')

    # emb_mean = None
    # with torch.no_grad():
    #     for index, (origin, _, file_name, lens) in enumerate(loader):

    #         # if index != 12287:
    #         #     continue
            
    #         try:
    #             origin = origin.to(device)
    #             x = origin

    #             # emb = model.embedding(x).detach().cpu() # (1, dim)
    #             emb = model.embedding(x).detach().cpu() # (1, dim)

    #             if index == 0:
    #                 emb_mean = emb
                
    #             else:
    #                 emb_mean.data += emb
                
    #             print(index, file_name[0], emb.shape)
    #         except (RuntimeError):
    #             print(index, file_name[0], 'Error, go to next')
    #             continue
        
    #     emb_mean.data /= len(loader)

    #     with open('{}-emb-mean.pickle'.format(args.system_type), 'wb') as writer:
    #         pickle.dump(emb_mean, writer, -1)
    #         print('save emb mean file')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(dest='system_type') # either iv (ivector-PLDA) or xv (xvector-PLDA)

    # iv_parser = subparser.add_parser("iv")
    # iv_parser.add_argument('-plda', default='./iv_system/plda.txt')
    # iv_parser.add_argument('-mean', default='./iv_system/mean.vec')
    # iv_parser.add_argument('-transform', default='./iv_system/transform.txt')
    # iv_parser.add_argument('-extractor', default='./iv_system/final_ie.txt')
    # iv_parser.add_argument('-gmm', default='./iv_system/final_ubm.txt') 

    # xv_parser = subparser.add_parser("xv")
    # xv_parser.add_argument('-plda', default='./xv_system/plda.txt')
    # xv_parser.add_argument('-mean', default='./xv_system/mean.vec')
    # xv_parser.add_argument('-transform', default='./xv_system/transform.txt')
    # xv_parser.add_argument('-extractor', default='./xv_system/xvecTDNN_origin.ckpt')

    xv_c_parser = subparser.add_parser("xv_csine")
    xv_c_parser.add_argument('-extractor', 
                default='./train-test-csv-vox1-dev-xvector/CKPT+2021-07-30+03-30-22+00/embedding_model.ckpt')
    
    ecapa_c_parser = subparser.add_parser("ecapa_csine")
    ecapa_c_parser.add_argument('-extractor', 
                default='./train-test-csv-vox1-dev/CKPT+2021-07-29+07-27-52+00/embedding_model.ckpt')
    
    audionet_c_parser = subparser.add_parser("audionet_csine")
    audionet_c_parser.add_argument('-extractor', 
                default='./audionet-c.ckpt')
    
    sinc_c_parser = subparser.add_parser("sinc_csine")
    sinc_c_parser.add_argument('-extractor', 
                default='./model_file/SincNet/TIMIT-model_raw.pkl')
    sinc_c_parser.add_argument('-cfg_file',
                default='./model_file/SincNet/SincNet_TIMIT.cfg')
    
    resnet18_iden_parser = subparser.add_parser("resnet18_iden")
    resnet18_iden_parser.add_argument('-extractor',
                default='./model_file/AutoSpeech/autoSpeech-resnet18-iden.pth')
    resnet18_iden_parser.add_argument('-feat_mean',
                default='./model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    resnet18_iden_parser.add_argument('-feat_std',
                default='./model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34
    
    resnet18_veri_parser = subparser.add_parser("resnet18_veri")
    resnet18_veri_parser.add_argument('-extractor',
                default='./model_file/AutoSpeech/autoSpeech-resnet18-veri.pth')
    resnet18_veri_parser.add_argument('-feat_mean',
                default='./model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    resnet18_veri_parser.add_argument('-feat_std',
                default='./model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34
    
    resnet34_iden_parser = subparser.add_parser("resnet34_iden")
    resnet34_iden_parser.add_argument('-extractor',
                default='./model_file/AutoSpeech/autoSpeech-resnet34-iden.pth')
    resnet34_iden_parser.add_argument('-feat_mean',
                default='./model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    resnet34_iden_parser.add_argument('-feat_std',
                default='./model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34
    
    resnet34_veri_parser = subparser.add_parser("resnet34_veri")
    resnet34_veri_parser.add_argument('-extractor',
                default='./model_file/AutoSpeech/autoSpeech-resnet34-veri.pth')
    resnet34_veri_parser.add_argument('-feat_mean',
                default='./model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    resnet34_veri_parser.add_argument('-feat_std',
                default='./model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34
    
    autospeech_scratch_iden_parser = subparser.add_parser("autospeech_scratch_iden")
    autospeech_scratch_iden_parser.add_argument('-extractor',
                default='../../model_file/AutoSpeech/autoSpeech-scratch-iden.pth')
    autospeech_scratch_iden_parser.add_argument('-feat_mean',
                default='../../model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    autospeech_scratch_iden_parser.add_argument('-feat_std',
                default='../../model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34
    
    autospeech_scratch_veri_parser = subparser.add_parser("autospeech_scratch_veri")
    autospeech_scratch_veri_parser.add_argument('-extractor',
                default='../../model_file/AutoSpeech/autoSpeech-scratch-veri.pth')
    autospeech_scratch_veri_parser.add_argument('-feat_mean',
                default='../../model_file/AutoSpeech/resNet18-feat-mean') # common for 18 and 34
    autospeech_scratch_veri_parser.add_argument('-feat_std',
                default='../../model_file/AutoSpeech/resNet18-feat-std') # common for 18 and 34

    args = parser.parse_args()
    main(args)
