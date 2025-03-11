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
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.utils.utils import *
from sklearn.cluster import KMeans
from fcmeans import FCM

logger = logging.getLogger(__name__)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
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
    txt_src = open(cfg.s_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()

    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if cfg.SOURCE.TRTE == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)

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

##################--open-set---###################################################
def cal_acc_oda(loader, netF, netB, netC, cen, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            #outputs = netC(netB(netF(inputs)))
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, cfg)
            outputs = cal_output(outputs, mem_ship, cfg) 
            
            
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

##############################################################################
def rule_loss(output, labels, src_rule_num, cfg):
    loss = torch.tensor(0.0).cuda()
    for i in range(src_rule_num):
        loss += CrossEntropyLabelSmooth(num_classes=cfg.class_num, epsilon=cfg.SOURCE.EPSILON)(output[i], labels)
   
    return loss

def cal_loss(inputs, labels, netF, netB, netC, cen, src_rule_num, cfg):
    mem_ship, tar_rule, output = srcnet_output(inputs, netF, netB, netC, cen, cfg)                                                                    			
    loss = rule_loss(output, labels, src_rule_num,cfg)
    output = cal_output(output, mem_ship, cfg)
    loss += CrossEntropyLabelSmooth(num_classes=cfg.class_num, epsilon=cfg.SOURCE.EPSILON)(output, labels)
        
    return loss

def cal_fuz_cen(loader, netF, netB, cfg, ini=0, fuz_cen=None):
    start_test = True # loader = dset_loaders['source_te']
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            feas = netB(netF(inputs))

            if start_test:
                all_fea = feas.float().cpu()
                #all_idx = idx.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                #all_idx = torch.cat((all_idx, idx.float()), 0)

    '''
    if cfg.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy() 
    '''
    
    #all_fea = np.random.rand(101, 256)*5
    #all_fea = torch.from_numpy(np.random.rand(101, 256)*5)
    #all_fea[0:50,:] = all_fea[0:50,:]-np.random.rand(50, 256)*2
    
    if ini==0:
        #if cfg.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy() 
        #print(all_fea.shape)
        
        fcm = FCM(n_clusters=cfg.rule_num)
        #fcm = FCM(n_clusters=5)
        fcm.fit(all_fea)
        fcm_centers = fcm.centers
        fcm_labels = fcm.predict(all_fea)
        
        K = cfg.rule_num
        aff = np.eye(K)[fcm_labels] #label vector sample size classes  .int()
        
        cls_count = np.eye(K)[fcm_labels].sum(axis=0) #cluster number 
        labelset = np.where(cls_count>0)
        labelset = labelset[0]
        
        fcm_centers = update_cen_src(aff, all_fea, labelset, K, fcm_centers, cfg)
        
    else:
        
        mem_ship = clu_mem(all_fea.detach().cpu(), fuz_cen, cfg)
        #mem_ship = clu_mem(all_fea, fuz_cen, cfg) fuz_cen = fcm_centers
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy() 
        for round in range(1):
            aff = np.power(mem_ship,2)#mem_ship #########################
            fcm_centers = aff.transpose().dot(all_fea)
            fcm_centers = fcm_centers / (1e-8 + aff.sum(axis=0)[:,None])   
    
    return fcm_centers

def update_cen_src(aff, all_fea, labelset, K, fcm_centers, cfg):
    
    initc0 = aff.transpose().dot(all_fea)
    initc0 = initc0 / (1e-8 + aff.sum(axis=0)[:,None])
    
    for i in range(len(initc0)):
        if i not in labelset:
            #print(i)
            initc0[i] = fcm_centers[i]
        
    #dist_c = cdist(all_fea, initc0[labelset], cfg.distance)#
    dist_c = cdist(all_fea, initc0, cfg.distance)#
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    #dda=dist_a.repeat(len(labelset), axis=1)
    dda=dist_a.repeat(len(initc0), axis=1)
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 
    #np.savetxt(args.output_dir_src+"/"+args.name_src1+"_mem_ship.csv", mem_ship, fmt='%.4f', delimiter=',')
    #nn.Softmax(dim=1)().numpy(torch.from_numpy())

    for round in range(1):
        aff = np.power(mem_ship,2)#mem_ship #########################
        initc1 = aff.transpose().dot(all_fea)
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])   
    
    return initc1

####################################################
def train_source(cfg):
    #cfg.output_dir_src = './source_fuz/' + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S][0].upper()
    #output_src = './source_fuz/'
    output_src = './source_fuz_abs/'
    cfg.output_dir_src = output_src + cfg.DA + '/' + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S][0].upper()  
    #cfg.output_dir_src = './source_fuz_abs/' + cfg.DA + '/' + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S][0].upper()
    
    
    if not osp.exists(cfg.output_dir_src):
        os.system('mkdir -p ' + cfg.output_dir_src)
    if not osp.exists(cfg.output_dir_src):
        os.mkdir(cfg.output_dir_src)

    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier_fuz(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck, rule_num = cfg.rule_num).cuda()

    param_group = []
    learning_rate = cfg.OPTIM.LR
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        if iter_num == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            cen = cal_fuz_cen(dset_loaders['source_te'], netF, netB, cfg, ini=0, fuz_cen=None)
            #print(cen.shape)
            netF.train()
            netB.train()
            netC.train()
        elif iter_num % interval_iter == 0 and iter_num > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            cen = cal_fuz_cen(dset_loaders['source_te'], netF, netB, cfg, ini=1, fuz_cen=cen)
            netF.train()
            netB.train()
            netC.train()
        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        src_rule_num = cfg.rule_num #if select rules change this value
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        classifier_loss = cal_loss(inputs_source, labels_source, netF, netB, netC, cen, src_rule_num, cfg)                       
        
        #outputs_source = netC(netB(netF(inputs_source)))
        #classifier_loss = CrossEntropyLabelSmooth(num_classes=cfg.class_num, epsilon=cfg.SOURCE.EPSILON)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, cen, cfg, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, cen, cfg, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name_src, iter_num, max_iter, acc_s_te)
            logging.info(log_str)
            

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                
                best_cen = cen

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(cfg.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(cfg.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(cfg.output_dir_src, "source_C.pt"))
    
    np.save(cfg.output_dir_src+"/"+"fuz_cen_" +str(cfg.rule_num)+".npy", best_cen)
    
    
    test_target(cfg)
    return netF, netB, netC

def test_target(cfg):
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    #netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier_fuz(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck, rule_num = cfg.rule_num).cuda()

    
    cfg.modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(cfg.modelpath))
    cfg.modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(cfg.modelpath))
    cfg.modelpath = cfg.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(cfg.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    
    cen = np.load(cfg.output_dir_src+"/"+"fuz_cen_" +str(cfg.rule_num)+".npy")

    for i in range(len(cfg.domain)):
        if i == cfg.SETTING.S:
            continue
        cfg.SETTING.T = i
        cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
        cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
        cfg.name = cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper()
        dset_loaders = data_load(cfg)
        if cfg.DA == 'oda':
            acc_os1, acc_os2, acc_unknown, hm = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc_os2, acc_os1, acc_unknown, hm)
        else:
            if cfg.SETTING.DATASET=='VISDA-C':
                acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, cen, cfg, True)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc) + '\n' + acc_list
            else:
                acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, cen, cfg, False)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc)

        logging.info(log_str)
