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




def tune_feature_extractor(constants_dict, unet, config):

    
    
    feature_extractor =  timm.create_model(
                        config.model.backbone,
                        pretrained=True,
                        # num_classes=1,
                    )
    feature_extractor.to(config.model.device)    

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
        feature_extractor.train()
        
        # if config.model.backbone in ['resnet18', 'wide_resnet50_2', 'wide_resnet101_2']:
        #     for param in feature_extractor.parameters():
        #         param.requires_grad = False
        #     for param in feature_extractor.layer1.parameters():
        #         param.requires_grad = True

        # for param in feature_extractor.parameters():
        #     print('param grad: ',param.requires_grad) 

        optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.001) #config.model.learning_rate
        criterion = nn.MSELoss() #nn.CrossEntropyLoss()
        for epoch in range(0):
            for step, batch in enumerate(trainloader):
                data = batch[0]
                data = data.to(config.model.device)
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps2+(epoch * 100)]).type(torch.int64).to(config.model.device)
                noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                seq = range(0 , config.model.test_trajectoy_steps2+(epoch * 100), config.model.skip2)
                # print('seq : ',seq)
                # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)

                # cr = 0.
                H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .0, cls_fn=None, classes=None, early_stop=False)
                # reconstructed, rec_x0 = generalized_steps(data, seq, model, constants_dict['betas'], config, eta = 1.0)
                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config, gama= 0.01, constants_dict=constants_dict, eraly_stop = False)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .0, cls_fn=None, classes=None) 
                data_reconstructed = reconstructed[-1].to(config.model.device)

                if config.model.backbone == 'deit_base_distilled_patch16_384':
                    transform = transforms.Compose([
                    transforms.CenterCrop(224), 
                    transforms.Resize((384,384)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif config.model.backbone == 'cait_m48_448':
                    transform = transforms.Compose([
                        transforms.CenterCrop(224), 
                        transforms.Resize((448,448)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.CenterCrop(224), 
                        transforms.Lambda(lambda t: (t + 1) / (2)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])  
                
                visualalize_distance(data, data_reconstructed, transform(data), transform(data_reconstructed))


                data_reconstructed = transform(data_reconstructed)
                data = transform(data)
                outputs_features = extract_features(feature_extractor=feature_extractor, x=data_reconstructed, config=config, out_indices='layer1')   # #feature_extractor((data_reconstructed)) #
                targets_features = extract_features(feature_extractor=feature_extractor, x=data, config=config, out_indices='layer1')   # #feature_extractor((data)) #
                optimizer.zero_grad()

                # cosine_distance = 1 - F.cosine_similarity(patchify(outputs_features) , patchify(targets_features), dim=1).to(config.model.device).unsqueeze(1)
                loss = 100* torch.mean(1 - F.cosine_similarity((outputs_features) , (targets_features), dim=1).to(config.model.device).unsqueeze(1))
                # loss= torch.max(loss)
                
                # loss = criterion(outputs_features, targets_features)
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
                # if step == 2:
                #     break
            print(f"Epoch {epoch} | Loss: {loss.item()}")
        if config.data.category:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
        else:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
    else:
        if config.data.category:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
        else:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
        feature_extractor.load_state_dict(checkpoint)  
    return feature_extractor 





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

                