import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
from src.utils import IID_losses,miro,loss
from copy import deepcopy
import torch.nn.functional as F
import clip
from src.utils.utils import *
from sklearn.cluster import KMeans
logger = logging.getLogger(__name__)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(cfg,optimizer, iter_num, max_iter, gamma=10, power=0.75):
    if (cfg.SETTING.DATASET =='office-home'):
        # print(1)
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    else :
        decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()

    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders

###############---Fuzzy---################################
def norm_fea(fea):    
    fea = torch.cat((fea, torch.ones(fea.size(0), 1)), 1)
    fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t()   
    return fea

def clu_mem(fea, cen, cfg):  
    #if cfg.distance == 'cosine': #print(cen.shape)
     #  fea = norm_fea(fea).numpy()
    #else:
     #   fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t() 
    fea = norm_fea(fea).numpy()
    #fea = all_fea print(fea.shape)
    dist_c = cdist(fea, cen, cfg.distance)
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(cen.shape[0], axis=1)
    #mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c*dda)))).numpy() 
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy()     
        
    return mem_ship

def fuz_mem(fea, cen, cfg):
    
    mem_ship = clu_mem(fea.detach().cpu(), cen, cfg)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()            
         
    return mem_ship, tar_rule

def cal_output(output, member, cfg):    
    outputs = torch.zeros([output[0].shape[0], cfg.class_num]).cuda()
    for i in range(len(output)):                       
        outputs += member[:,i].reshape(output[0].shape[0],1)*output[i]               			
    return outputs
    
def cal_output_sel(output, member, tar_rule, cfg):    
    outputs = torch.zeros([output[0].shape[0], cfg.class_num]).cuda()
    for i in range(cfg.rule_num): #rule_num
        for j in range(output[0].shape[0]):
            outputs[j,:] = outputs[j,:] + member[j,tar_rule[j,i]]*output[tar_rule[j,i]][j,:]              			
    return outputs

def srcnet_output(inputs, netF, netB, netC, cen, cfg):
    
    feas = netB(netF(inputs))
    mem_ship = clu_mem(feas.detach().cpu(), cen, cfg)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()               
    outputs = netC(feas) 
        
    return mem_ship, tar_rule, outputs

###############################################
def cal_acc(loader, netF, netB, netC, cen, cfg, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, cfg)
            outputs = cal_output(outputs, mem_ship, cfg) 
            
            #outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc_multi(loader, netF, netB_list, netC_list, netG_list, cen_list, cfg, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            weights = torch.ones(inputs.shape[0], len(cfg.MODEL.ARCHs)-1)
            outputs = torch.zeros(len(cfg.MODEL.ARCHs)-1, inputs.shape[0], cfg.class_num)
            outputs_w = torch.zeros(inputs.shape[0], cfg.class_num)
        
        
            for i in range(len(cfg.MODEL.ARCHs)-1):
                mem_ship, tar_rule, outputs_rule = srcnet_output(inputs, netF, netB_list[i], netC_list[i], cen_list[i], cfg)
                outputs[i] = cal_output(outputs_rule, mem_ship, cfg)
                weights[:, i] = netG_list[i](netB_list[i](netF(inputs))).squeeze()   
            
            z = torch.sum(weights, dim=1)
            z = z + 1e-16

            weights = torch.transpose(torch.transpose(weights,0,1)/z,0,1)
            outputs = torch.transpose(outputs, 0, 1)

            z_ = torch.sum(weights, dim=0)
        
            z_2 = torch.sum(weights)
            z_ = z_/z_2
    
            for i in range(inputs.shape[0]):
                outputs_w[i] = torch.matmul(torch.transpose(outputs[i],0,1), weights[i])

            outputs = outputs_w
            #outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
#######################################################################################
def cal_acc_multi_oda(loader, netF, netB_list, netC_list, netG_list, cen_list, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            weights = torch.ones(inputs.shape[0], len(cfg.MODEL.ARCHs)-1)
            outputs = torch.zeros(len(cfg.MODEL.ARCHs)-1, inputs.shape[0], cfg.class_num)
            outputs_w = torch.zeros(inputs.shape[0], cfg.class_num)
        
        
            for i in range(len(cfg.MODEL.ARCHs)-1):
                mem_ship, tar_rule, outputs_rule = srcnet_output(inputs, netF, netB_list[i], netC_list[i], cen_list[i], cfg)
                outputs[i] = cal_output(outputs_rule, mem_ship, cfg)
                weights[:, i] = netG_list[i](netB_list[i](netF(inputs))).squeeze()   
            
            z = torch.sum(weights, dim=1)
            z = z + 1e-16

            weights = torch.transpose(torch.transpose(weights,0,1)/z,0,1)
            outputs = torch.transpose(outputs, 0, 1)

            z_ = torch.sum(weights, dim=0)
        
            z_2 = torch.sum(weights)
            z_ = z_/z_2
    
            for i in range(inputs.shape[0]):
                outputs_w[i] = torch.matmul(torch.transpose(outputs[i],0,1), weights[i])

            outputs = outputs_w
            
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + cfg.epsilon), dim=1) / np.log(cfg.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = cfg.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()
    
    hm = 2 * (np.mean(acc[:-1]) * unknown_acc)/(np.mean(acc[:-1]) + unknown_acc)

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc, hm
    # return np.mean(acc), np.mean(acc[:-1])

#####################################################################################
def train_target(cfg):
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)
    ##heterogeneou networks

    ## set base network
    if cfg.SETTING.DATASET == 'office-home':
        #office-home [A-next50-512, C-next101-256, P-net152-1024, R-net101-768]
        if cfg.SETTING.T == 0:
            task = [1,2,3,0]
        elif cfg.SETTING.T == 1:
            task = [0,2,3,1]
        elif cfg.SETTING.T == 2:
            task = [0,1,3,2]
        elif cfg.SETTING.T == 3:
            task = [0,1,2,3]
        cfg.MODEL.ARCHs = ['resnext50', 'resnext101', 'resnet152', 'resnet101']
        cfg.bottlenecks = [512, 256, 1024, 768]
    if cfg.SETTING.DATASET == 'office':
    #office31: [A-next101-512, D-next50-256, W-net101-768]
        if cfg.SETTING.T == 0:
            task = [1,2,0]
        elif cfg.SETTING.T == 1:
            task = [0,2,1]
        elif cfg.SETTING.T == 2:
            task = [0,1,2]
        cfg.MODEL.ARCHs = ['resnext101', 'resnext50', 'resnet101']
        cfg.bottlenecks = [512, 256, 768]   
    if cfg.SETTING.DATASET == 'domainnet126':
        #domainnet [C-next101-768, P-net152-256, R-net152-512, S-net101-1024]
        if cfg.SETTING.T == 0:
            task = [1,2,3,0]
        elif cfg.SETTING.T == 1:
            task = [0,2,3,1]
        elif cfg.SETTING.T == 2:
            task = [0,1,3,2]
        elif cfg.SETTING.T == 3:
            task = [0,1,2,3]
        cfg.MODEL.ARCHs = ['resnext101', 'resnet152', 'resnet152', 'resnet101']
        cfg.bottlenecks = [768, 256, 512, 1024]
    
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    model = get_coop(cfg.LCFD.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.LCFD.N_CTX, cfg.LCFD.CTX_INIT)
    #netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netB_list = [network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottlenecks[task[i]]).cuda() for i in range(len(cfg.MODEL.ARCHs)-1)]
    
    #netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC_list = [network.feat_classifier_fuz(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottlenecks[task[i]], rule_num = cfg.rule_num).cuda() for i in range(len(cfg.MODEL.ARCHs)-1)]
    
    w = 2*torch.rand(len(cfg.MODEL.ARCHs)-1)-1
    print(w)
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(cfg.MODEL.ARCHs)-1)] 
    
    output_bb = './bb_target/'
    
    cen_list = [np.load(output_bb + cfg.DA + '/' + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T][0].upper() +'/' + cfg.domain[task[i]][0].upper() + '_fuz_cen_'+str(cfg.rule_num)+'.npy') for i in range(len(cfg.MODEL.ARCHs)-1)]  
    
    
    cfg.pre_tra_dir_src = output_bb + cfg.DA + '/' + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T][0].upper()
    cfg.modelpath = cfg.pre_tra_dir_src + '/source_F.pt' 
    print(cfg.modelpath)  
    netF.load_state_dict(torch.load(cfg.modelpath))
    netF.eval()
    
    for i in range(len(cfg.MODEL.ARCHs)-1): 
        cfg.modelpath = cfg.pre_tra_dir_src + '/source_' + cfg.domain[task[i]][0].upper() + '_B.pt' 
        print(cfg.modelpath)  
        netB_list[i].load_state_dict(torch.load(cfg.modelpath))
        netB_list[i].eval()
        
        cfg.modelpath = cfg.pre_tra_dir_src + '/source_' + cfg.domain[task[i]][0].upper() + '_C.pt' 
        print(cfg.modelpath)  
        netC_list[i].load_state_dict(torch.load(cfg.modelpath))
        netC_list[i].eval()
        
        netG_list[i].eval()
           
    target_logits = torch.ones(cfg.TEST.BATCH_SIZE,cfg.class_num) ####
    im_re_o = miro.MIRO(target_logits.shape).cuda()
    del target_logits
    
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    param_group = []
    param_group_ib = []

    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
             param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for i in range(len(cfg.MODEL.ARCHs)-1):
        for k, v in netB_list[i].named_parameters():
            if cfg.OPTIM.LR_DECAY2 > 0:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
            else:
                v.requires_grad = False
        for k, v in netC_list[i].named_parameters():
            if cfg.OPTIM.LR_DECAY1 > 0:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
            else:
                v.requires_grad = False
        for k, v in netG_list[i].named_parameters():
            param_group += [{'params':v, 'lr':cfg.OPTIM.LR}]
    
    for k, v in im_re_o.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY3}]

    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY3}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    classnames = cfg.classname
    model.reset_classnames(classnames, cfg.LCFD.ARCH)
    start = True
    epoch = 0 
    
    acc_init = 0
    
    while iter_num < max_iter:
        try:
            inputs_test, labels, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, labels, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(cfg,optimizer, iter_num=iter_num, max_iter=max_iter)
        
        ############################################
        ######  learn from multiple sources  #######
        ############################################
        
        #outputs_test_new = torch.zeros(len(cfg.MODEL.ARCHs)-1, inputs_test.shape[0], cfg.class_num)
        weights_test_new = torch.ones(inputs_test.shape[0], len(cfg.MODEL.ARCHs)-1)
        #outputs_test_new_w = torch.zeros(inputs_test.shape[0], cfg.class_num)
        
        #####################################################
        outputs_test = torch.zeros(len(cfg.MODEL.ARCHs)-1, inputs_test.shape[0], cfg.class_num)
        outputs_test_w = torch.zeros(inputs_test.shape[0], cfg.class_num)
        
        output_clip = torch.zeros(len(cfg.MODEL.ARCHs)-1, inputs_test.shape[0], cfg.class_num)
        output_clip_w = torch.zeros(inputs_test.shape[0], cfg.class_num)
        
        
        for i in range(len(cfg.MODEL.ARCHs)-1):
        
            with torch.no_grad():
                #outputs_test_new = netC(netB(netF(inputs_test))).detach()
                mem_ship, tar_rule, outputs_test_new = srcnet_output(inputs_test, netF, netB_list[i], netC_list[i], cen_list[i], cfg)
                outputs_test_new = cal_output(outputs_test_new, mem_ship, cfg).detach() 
                #outputs_test_new -> output_clip
                
                #features_test = netB_list[i](netF(inputs_test))
                #weights_test = netG_list[i](features_test)
                #weights_test_new[:, i] = weights_test.squeeze() 
                weights_test_new[:, i] = netG_list[i](netB_list[i](netF(inputs_test))).squeeze()   
            netF.eval()
            netB_list[i].eval()
            netC_list[i].eval()
            netG_list[i].eval()
            model.train()
            im_re_o.train()
            
            output_clip[i],_ = test_time_adapt_eval(inputs_test, labels, model, optimizer_ib, optim_state, cfg, outputs_test_new, im_re_o)
            output_clip[i] = output_clip[i].detach().cuda().float()
            
            
            ################################################
            ## both output_clip and outputs_test are needed for loss ######
        
            #output_clip,_ = test_time_adapt_eval(inputs_test, labels, model, optimizer_ib, optim_state, cfg, outputs_test_new[i], im_re_o)
            #output_clip = output_clip.detach().cuda().float()
            #### combine clip outputs #####
            
            
            #output_clip_sm = nn.Softmax(dim=1)(output_clip)
            netF.train()
            netB_list[i].train()
            netC_list[i].train()
            netG_list[i].train()
            model.eval()
            im_re_o.eval()
            
                       
            mem_ship, tar_rule, outputs_test_rule = srcnet_output(inputs_test, netF, netB_list[i], netC_list[i], cen_list[i], cfg)
            outputs_test[i] = cal_output(outputs_test_rule, mem_ship, cfg) 
            
            #outputs_test = netC(netB(netF(inputs_test)))
        
        ############ calculate weighted predictions #################
        z = torch.sum(weights_test_new, dim=1)
        z = z + 1e-16

        weights_test_new = torch.transpose(torch.transpose(weights_test_new,0,1)/z,0,1)
        output_clip = torch.transpose(output_clip, 0, 1)
        outputs_test = torch.transpose(outputs_test, 0, 1)

        z_ = torch.sum(weights_test_new, dim=0)
        
        z_2 = torch.sum(weights_test_new)
        z_ = z_/z_2
    
        for i in range(inputs_test.shape[0]):
            outputs_test_w[i] = torch.matmul(torch.transpose(outputs_test[i],0,1), weights_test_new[i])
            output_clip_w[i] = torch.matmul(torch.transpose(output_clip[i],0,1), weights_test_new[i])
        
        
        output_clip = output_clip_w
        outputs_test = outputs_test_w
        
        
        ##### for uda ###########################################################
        ####  else  ########################################################
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        output_clip_sm = nn.Softmax(dim=1)(output_clip)
        #### default sce
        if (cfg.LCFD.LOSS_FUNC=="l1"):
            loss_l1 = torch.nn.L1Loss(reduction='mean')
            classifier_loss = loss_l1(softmax_out, output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="l2"):
            loss_l2 = torch.nn.MSELoss(reduction='mean')
            classifier_loss = loss_l2(softmax_out,output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="iid"):
            classifier_loss = IID_losses.IID_loss(softmax_out,output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="kl"):
            classifier_loss = F.kl_div(softmax_out.log(),output_clip_sm, reduction='sum')
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="sce"): #######
            _, pred = torch.max(output_clip, 1)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= cfg.LCFD.CLS_PAR
        else :
            classifier_loss = torch.tensor(0.0).cuda()


        #'''
        if cfg.LCFD.ENT:
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if cfg.LCFD.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
                entropy_loss -= cfg.LCFD.GENT_PAR*gentropy_loss
            im_loss = entropy_loss
            classifier_loss += im_loss
            #classifier_loss = im_loss

        #'''
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if start:
                all_output_clip = output_clip.float().cpu()
                all_label = labels.float()
                start = False
            else:
                all_output_clip = torch.cat((all_output_clip, output_clip.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            start = True
            epoch = epoch + 1
            _, clip_predict = torch.max(all_output_clip, 1)
            accuracy = torch.sum(torch.squeeze(clip_predict).float() == all_label).item() / float(all_label.size()[0])
            accuracy = accuracy*100
            log_str ='CLIP_Accuracy = {:.2f}%'.format(accuracy)
            logging.info(log_str)
            netF.eval()
            for i in range(len(cfg.MODEL.ARCHs)-1):
                netB_list[i].eval()
                netC_list[i].eval()
                netG_list[i].eval()
            im_re_o.eval()
            model.eval()

            if cfg.SETTING.DATASET=='VISDA-C':
                acc_t_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_t_te) + '\n' + acc_list
            else:
                #acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                acc_t_te, _ = cal_acc_multi(dset_loaders['test'], netF, netB_list, netC_list, netG_list, cen_list, cfg, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_t_te)
                if cfg.DA == 'oda':
                    acc_os1, acc_os2, acc_unknown, hm = cal_acc_multi_oda(dset_loaders['test'], netF, netB_list, netC_list, netG_list, cen_list, cfg)#(loader, netF, netB, netC, cen, cfg)
                    #log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, hm)
                    log_str = 'S2T: {}, Iter:{}/{};  Accuracy = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%'.format(cfg.domain[task[i]], iter_num, max_iter, acc_os2, acc_os1, acc_unknown, hm)                   
                    logging.info(log_str)
                    acc_t_te = hm

            logging.info(log_str)
            
            if acc_t_te >= acc_init:
                acc_init = acc_t_te
                
                torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
                for i in range(len(cfg.MODEL.ARCHs)-1):
                    torch.save(netB_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_B_" + cfg.savename + ".pt"))
                    torch.save(netC_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_C_" + cfg.savename + ".pt"))
                    torch.save(netG_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_W_" + cfg.savename + ".pt"))
        
            
            netF.train()
            for i in range(len(cfg.MODEL.ARCHs)-1):
                netB_list[i].train()
                netC_list[i].train()
                netG_list[i].train()

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        for i in range(len(cfg.MODEL.ARCHs)-1):
            torch.save(netB_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_B_" + cfg.savename + ".pt"))
            torch.save(netC_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_C_" + cfg.savename + ".pt"))
            torch.save(netG_list[i].state_dict(), osp.join(cfg.output_dir, "target_" + cfg.domain[task[i]][0].upper() + "_W_" + cfg.savename + ".pt"))
        
    return netF, netB_list, netC_list, netG_list

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def test_time_tuning(model, inputs, optimizer, cfg,target_output,im_re_o):

    target_output = target_output.cuda()

    for j in range(cfg.LCFD.TTA_STEPS):
        with torch.cuda.amp.autocast():

            output_logits,_ = model(inputs)             
            if(output_logits.shape[0]!=cfg.TEST.BATCH_SIZE):
                padding_f=torch.zeros([cfg.TEST.BATCH_SIZE-output_logits.shape[0],output_logits.shape[1]],dtype=torch.float).cuda()
                output_logits = torch.cat((output_logits, padding_f.float()), 0)
                target_output =  torch.cat((target_output, padding_f.float()), 0)

            im_loss_o, Delta = im_re_o.update(output_logits,target_output)
            Delta = 1.0/(Delta+1e-5) #softplus(x) log(1+e^x) x->pre_f
            Delta = nn.Softmax(dim=1)(Delta)
            output_logits_sm = nn.Softmax(dim=1)(output_logits)
            output = Delta*output_logits_sm
            iic_loss = IID_losses.IID_loss(output, output_logits_sm)
            loss = 0.5*(iic_loss - 0.0003*im_loss_o) #0.0003
            ################ abs loss ###############################
            #loss = 0.5*iic_loss #(- 0.0003*im_loss_o) #
            if(inputs.shape[0]!=cfg.TEST.BATCH_SIZE):
                output = output[:inputs.shape[0]]
                target_output = target_output[:inputs.shape[0]]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return output,loss


def test_time_adapt_eval(input,target, model, optimizer, optim_state, cfg, target_output, im_re_o):
    if cfg.LCFD.TTA_STEPS > 0:
        with torch.no_grad():
            model.train()
            im_re_o.train()
    optimizer.load_state_dict(optim_state)
    output,loss_ib = test_time_tuning(model, input, optimizer, cfg,target_output,im_re_o)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            im_re_o.eval()
            output,_ = model(input)
    output = output.cpu()
    return output,loss_ib


def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:#cfg.name_file #cfg.LCFD.NAME_FILE
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.LCFD.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts
