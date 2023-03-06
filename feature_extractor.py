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
from model import *
from omegaconf import OmegaConf
from sample import *
from visualize import *
from resnet import *
from de_resnet import de_wide_resnet50_2



def build_model(config):
    #model = SimpleUnet()
    model = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=config.data.imput_channel)
    return model

def fake_real_dataset(config, constants_dict):
    if config.data.name == 'MVTec':
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
    R_F_dataset=[]
    print("Start generating fake real dataset")
    for step, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
        print('step: ',step)
        image = batch[0]
        image = image.to(config.model.device)
        model = build_model(config)
        if config.data.category:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'5000')) # config.model.checkpoint_name 300+50
        else:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), '100'))
        model.load_state_dict(checkpoint)    
        model.to(config.model.device)
        model.eval()
        generate_time_steps = torch.Tensor([config.model.generate_time_steps]).type(torch.int64)
        noise = get_noise(image,config) 
        # noise = forward_diffusion_sample(image, generate_time_steps, constants_dict, config)[0]
        seq = range(0, config.model.generate_time_steps, config.model.skip_generation)
        # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
        # reconstructed,_ =  efficient_generalized_steps(config, noise, seq, model,  constants_dict['betas'], H_funcs, image, cls_fn=None, classes=None) 
        reconstructed,_ = generalized_steps(noise, seq, model, constants_dict['betas'], config, eta=config.model.eta)
        generated_image = reconstructed[-1]
        generated_image = generated_image.to(config.model.device)
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.1),
                ])
        generated_image = transform(generated_image)

        for fake, real in zip(generated_image, image):
            fake_label = torch.Tensor([1,0]).type(torch.float32).to(config.model.device)
            real_label = torch.Tensor([0,1]).type(torch.float32).to(config.model.device)
            R_F_dataset.append((fake.type(torch.float32), fake_label))
            R_F_dataset.append((real.type(torch.float32), real_label))
            # break
            # if R_F_dataset.__len__() == 40:
            #     return R_F_dataset
        
        if step == 0:
            return R_F_dataset
    return R_F_dataset


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


def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss1 = 0
    loss2 = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss1 += torch.mean(1-cos_loss((a[item]),
                                      (b[item])))
        # loss2 += torch.mean(1-cos_loss(patchify(c[item]),
        #                               patchify(d[item])))
    # print('loss1: ',loss1, '    loss2: ',loss2)
    return loss1 #+ loss2

def tune_feature_extractor(constants_dict, model, config):

    
    
    t_encoder, bn = wide_resnet50_2(pretrained=True)
    # s_encoder, bn = wide_resnet50_2(pretrained=False)
    # t_encoder, tbn = resnet18(pretrained=True)
    # s_encoder, sbn = resnet18(pretrained=False)
    t_encoder.to(config.model.device)  
    # s_encoder.to(config.model.device) 
    # tbn.to(config.model.device)
    # sbn.to(config.model.device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(config.model.device)

    # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
    # s_encoder.load_state_dict(checkpoint)  

    # if config.data.name == 'MVTec':
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
    if config.model.fine_tune:      
        # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_2_200'))
        # s_encoder.load_state_dict(checkpoint) 
        s_encoder.train()
        for param in t_encoder.parameters():
            param.requires_grad = False

        for param in s_encoder.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(s_encoder.parameters(), lr=0.001) #config.model.learning_rate
        # for epoch in range(20):
        #     for step, batch in enumerate(trainloader):
        #         data = batch[0]
        #         data = data.to(config.model.device)
        #         TF = t_encoder(data)
        #         SF = s_encoder(data)
        #         loss = loss_fucntion(TF, SF)
        #         optimizer.zero_grad()
        #         # loss.requires_grad = True
        #         loss.backward()
        #         optimizer.step()
        #         # loss_list.append(loss.item())

        #     print(f"Epoch {epoch} | Loss: {loss.item()}")

        # torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_b_100'))

        
        
        for epoch in range(400):
            gama = torch.rand(1).item()
            gama = gama/2 + 0.1
            print('gama: ', gama)
            for step, batch in enumerate(trainloader):
                data = batch[0]
                data = data.to(config.model.device)
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64).to(config.model.device)
                noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                seq = range(0 , config.model.test_trajectoy_steps, config.model.skip)
                
                # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .0, cls_fn=None, classes=None, early_stop=False) 
                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama= gama, constants_dict=constants_dict, eraly_stop = False)
                data_reconstructed = reconstructed[-1].to(config.model.device)
                # visualalize_distance(data, data_reconstructed)
                transform = transforms.Compose([
                    # transforms.CenterCrop(224), 
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                data = transform(data)
                data_reconstructed = transform(data_reconstructed)
                inputs = s_encoder(data)
                outputs = decoder(bn(inputs))
                # SFD = s_encoder(data)
                # TFR = t_encoder(data_reconstructed)
                # SFD2 = s_encoder(data)
                # TFR2 = t_encoder(data)
                # loss = loss_fucntion(SFD, TFR, SFD2, TFR2)
                loss = loss_fucntion(inputs, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch == 100:
                torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.outputs.category,'feature_100'))
            elif epoch == 200:
                torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_200'))
            elif epoch == 300:
                torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_300'))
        if config.data.category:
            torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_400'))
        else:
            torch.save(s_encoder.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
    else:
        if config.data.category:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature_300'))
        else:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
        s_encoder.load_state_dict(checkpoint)  
    return s_encoder, t_encoder





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

                


