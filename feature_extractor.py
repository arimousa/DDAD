import torch
import torch.nn as nn
import tqdm
from tqdm import tqdm
from forward_process import *
from dataset import *
from dataset import *
import timm
from torch import Tensor, nn
from typing import Callable, List, Tuple, Union
from unet import *
from omegaconf import OmegaConf
from sample import *
from visualize import *
from resnet import *
from de_resnet import de_wide_resnet50_2
import torchvision.transforms as T


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"



def build_model(config):
    #model = SimpleUnet()
    unet = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=config.data.imput_channel)
    return unet



def patchify(features, return_spatial_info=False):
    """Convert a tensor into a tensor of respective patches.
    Args:
        x: [torch.Tensor, bs x c x w x h]
    Returns:
        x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
        patchsize]
    """
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features

def loss_fucntion1(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    # l1 = torch.nn.L1Loss()
    loss1 = 0
    for item in range(len(a)):
        # if item == 0:
        #     continue
        # ap = (patchify(a[item])).contiguous()
        # bp = (patchify(b[item])).contiguous()
        loss1 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))
        # loss1 += torch.mean(l1(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))

    return loss1 

def loss_fucntion2(a, b, c, d):
    cos_loss = torch.nn.CosineSimilarity()
    # l1 = torch.nn.L1Loss()
    loss1 = 0
    loss2 = 0
    # loss3 = 0

    for item in range(len(a)):

        ap = (patchify(a[item])).contiguous()
        bp = (patchify(b[item])).contiguous()
        cp = (patchify(c[item])).contiguous() 
        dp = (patchify(d[item])).contiguous() 

        loss1 += torch.mean(1-cos_loss(ap.view(a[item].shape[0],-1), bp.view(b[item].shape[0],-1)))
        loss2 += torch.mean(1-cos_loss(cp.view(c[item].shape[0],-1), dp.view(d[item].shape[0],-1)))

    return (loss1 + loss2)



def tune_feature_extractor(constants_dict, unet, config):

    
    
    # t_encoder, bn = resnet18(pretrained=True)
    # t_encoder, bn = wide_resnet50_2(pretrained=True)
    t_encoder, bn = wide_resnet101_2(pretrained=True)
    # s_encoder, bn2 = wide_resnet50_2(pretrained=False)


    t_encoder.to(config.model.device)  
    bn.to(config.model.device)
    # s_encoder.to(config.model.device)
    # bn2.to(config.model.device)
    # decoder = de_wide_resnet50_2(pretrained=False)
    # decoder = decoder.to(config.model.device)


    if config.data.name == 'MVTec' or config.data.name == 'BTAD' or config.data.name == 'MTD' or config.data.name == 'VisA_pytorch':
        train_dataset = MVTecDataset(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=True,
        )   
    else:
        trainloader, testloader = load_data(dataset_name='cifar10')
    if config.model.fine_tune:      
        # checkpoint_e = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_200'))
        # decoder.load_state_dict(checkpoint_e) 
        # checkpoint_bn = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'bn_200'))
        # bn.load_state_dict(checkpoint_bn) 
        unet.eval()
        t_encoder.train()
        # s_encoder.train()
        # decoder.train()
        bn.train()
        
        for param in t_encoder.parameters():
            param.requires_grad = True
        # for param in s_encoder.parameters():
        #     param.requires_grad = True

        # for param in decoder.parameters():
        #     param.requires_grad = True
        # for param in bn.parameters():
        #     param.requires_grad = True
        
        transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

        # optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()),lr=.4e-4, betas=(0.5,0.999)) #config.model.learning_rate        
        optimizer = torch.optim.Adam(t_encoder.parameters(),lr= 1e-4) #config.model.learning_rate        
        
        for epoch in range(0):
            gama = torch.rand(1).item()
            gama = (gama)
            gama = torch.round(torch.tensor(gama), decimals=2)
            
            # print('gama: ', gama)
            # gama= 0.05
            for step, batch in enumerate(trainloader):
                # data = batch[0].to(config.model.device)   
                half_batch_size = batch[0].shape[0]//2
                data = batch[0][:half_batch_size].to(config.model.device)  
                data2 = batch[0][half_batch_size:].to(config.model.device)     
                # data = (data + data2) /2              

                # if torch.randint(1, 3, (1,)).item() ==1:
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64).to(config.model.device)
                # noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                at = compute_alpha(constants_dict['betas'], test_trajectoy_steps.long(),config)
                noisy_image = at.sqrt() * data2 + (1- at).sqrt() * torch.randn_like(data2).to('cuda')
                seq = range(0 , config.model.test_trajectoy_steps, config.model.skip)
                
                # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, constants_dict, noisy_image, seq, unet,  constants_dict['betas'], H_funcs, data2, gama = .00,sigma_0 = 0.1, cls_fn=None, classes=None, early_stop=False)


                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config, eta2=3, eta3=0, constants_dict=constants_dict, eraly_stop = False)
                # reconstructed, rec_x0 = generalized_steps(noisy_image, seq, unet, constants_dict['betas'], config)
                data_reconstructed = reconstructed[-1].to(config.model.device)
                
                # visualalize_distance(data2, data, data_reconstructed)

                data_reconstructed = transform(data_reconstructed)
                r_inputs = t_encoder(data_reconstructed)
                # r_outputs = decoder(bn(r_inputs))
                
                # data_reconstructed_n = transform(reconstructed[-4].to(config.model.device))
                # data_reconstructed_n = t_encoder(data_reconstructed_n)

                # r_inputs_n = rec_x0[-4].to(config.model.device)
                # r_inputs_n = transform(r_inputs_n)
                # r_inputs_n = t_encoder(r_inputs_n)
                



                data = transform(data)
                inputs = t_encoder(data)
                # outputs = decoder(bn(inputs))

                # visualalize_distance(noisy_image, data_reconstructed2)

                # loss = loss_fucntion2(inputs, r_inputs, r_inputs_n, data_reconstructed_n) #, rn_outputs, r2_outputs
                loss = loss_fucntion1(r_inputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if step == 5:
                #     break
            # if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            # torch.save(t_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'encoder_{epoch+1}'))

        if config.data.category:
            torch.save(t_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'encoder_00'))
            torch.save(bn.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'bn_250'))
        # else:
        #     torch.save(decoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
    else:
        if config.data.category:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'encoder_00'))
            
            # checkpoint_bn = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'bn_200'))
            
        # else:
        #     checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
        t_encoder.load_state_dict(checkpoint)  
        # bn.load_state_dict(checkpoint_bn)
    return t_encoder, t_encoder, bn





def extract_features(feature_extractor, x, config, out_indices=['layer1','layer2','layer3']):
    with torch.no_grad():
        feature_extractor.eval()
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2))
        ])
        x = reverse_transforms(x)
            
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.features_only = True
        activations = []
        if config.model.backbone in ['deit_base_distilled_patch16_384']:
            input_size = 384
            x = feature_extractor.patch_embed(x)
            cls_token = feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = feature_extractor.pos_drop(x + feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = feature_extractor.blocks[i](x)
            x = feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            embeddings = x.reshape(N, C, input_size // 16, input_size // 16)
        elif config.model.backbone in['cait_m48_448']:
            input_size = 448
            x = feature_extractor.patch_embed(x)
            x = x + feature_extractor.pos_embed
            x = feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            embeddings = x.reshape(N, C, input_size // 16, input_size // 16)
        else: #elif config.model.backbone in ['resnet18','wide_resnet101_2','wide_resnet50_2']:
            for name, module in feature_extractor.named_children():
                x = module(x)
                # print('name : ', name)
                if name in out_indices:
                    activations.append(x)
            embeddings = activations[0]
            for feature in activations[1:]:
                layer_embedding = feature
                layer_embedding = F.interpolate(layer_embedding, size=int(embeddings.shape[-2]), mode='bilinear', align_corners=False)
                embeddings = torch.cat((embeddings,layer_embedding),1)
        return embeddings

                


