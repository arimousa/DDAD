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




def build_model(config):
    #model = SimpleUnet()
    model = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=config.data.imput_channel)
    return model

def fake_real_dataset(config, constants_dict):
    train_dataset = Dataset(
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
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'1000')) # config.model.checkpoint_name 300+50
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






def tune_feature_extractor(constants_dict, model, config):

    
    
    feature_extractor =  timm.create_model(
                        config.model.backbone,
                        pretrained=True,
                        # num_classes=2,
                    )
    feature_extractor.to(config.model.device)    



    train_dataset = Dataset(
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

        optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.001) #config.model.learning_rate
        criterion = nn.MSELoss() #nn.CrossEntropyLoss()
        for epoch in range(1):
            for step, batch in enumerate(trainloader):
                data = batch[0]
                data = data.to(config.model.device)
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64).to(config.model.device)
                noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                k = 0
                seq = range(0 , config.model.test_trajectoy_steps+200, config.model.skip2)
                # print('seq : ',seq)
                # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)

                # cr = 0.0
                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama= 0.2, eraly_stop = False)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .6, cls_fn=None, classes=None) 
                data_reconstructed = reconstructed[-1].to(config.model.device)

                # r = 5 + epoch
                # for i in range(r):
                #     noisy_image = forward_ti_steps(test_trajectoy_steps, 1 * config.model.skip, data_reconstructed, data, constants_dict['betas'], config)
                #     # print('gama : ',gama)
                #     reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0.2, eraly_stop = True)
                #     data_reconstructed = reconstructed[-1]

                # seq = range(0, config.model.test_trajectoy_steps2, config.model.skip2)
                # noisy_image = forward_ti_steps(test_trajectoy_steps, 1 * config.model.skip, data_reconstructed, data, constants_dict['betas'], config)
                # reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0, eraly_stop = False)
                # data_reconstructed = reconstructed[-1].to(config.model.device)

                # loss = torch.max(feature_distance(data_reconstructed, data, feature_extractor, constants_dict, config))
                if config.model.backbone == 'deit_base_distilled_patch16_384':
                    transform = transforms.Compose([
                    transforms.Resize((384,384)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif config.model.backbone == 'cait_m48_448':
                    transform = transforms.Compose([
                    transforms.Resize((448,448)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])

                data_reconstructed = transform(data_reconstructed)
                data = transform(data)
                outputs_features = feature_extractor(data_reconstructed) # extract_features(feature_extractor=feature_extractor, x=data_reconstructed, config=config, out_indices='fc') 
                targets_features = feature_extractor(data) # extract_features(feature_extractor=feature_extractor, x=data, config=config, out_indices='fc') 
                optimizer.zero_grad()

                # cosine_distance = 1 - F.cosine_similarity(patchify(outputs_features) , patchify(targets_features), dim=1).to(config.model.device).unsqueeze(1)
                # loss = 1 - F.cosine_similarity(outputs_features , targets_features, dim=1).to(config.model.device).unsqueeze(1)
                # loss= torch.max(loss)
                
                loss = criterion(outputs_features, targets_features)
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
                if step > 10:
                    break
            print(f"Epoch {epoch} | Loss: {loss.item()}")
        if config.data.category:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
        else:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))
    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
        feature_extractor.load_state_dict(checkpoint)  
    return feature_extractor 




    # if config.model.fine_tune:
    #     R_F_dataset = fake_real_dataset(config, constants_dict)
    #     R_F_dataloader = torch.utils.data.DataLoader(R_F_dataset, batch_size=config.data.batch_size, shuffle=True) 
    #     feature_extractor.train()
    #     optimizer = torch.optim.SGD(feature_extractor.parameters(), lr=0.001, momentum=0.9) #config.model.learning_rate
    #     criterion =  nn.CrossEntropyLoss()
    #     print("Start training feature extractor")
    #     for step, batch in enumerate(R_F_dataloader):
    #         image = batch[0]
    #         label = batch[1]
    #         plt.figure(figsize=(11,11))
    #         plt.axis('off')
    #         plt.subplot(1, 1, 1)
    #         plt.imshow(show_tensor_image(image))
    #         plt.title(label[0])
    #         plt.savefig('results/F_or_R{}.png'.format(step))
    #         plt.close()
    #         if config.model.backbone == 'deit_base_distilled_patch16_384':
    #             transform = transforms.Compose([
    #             transforms.Resize((384,384)),
    #             ])
    #             image = transform(image)
    #         elif config.model.backbone == 'cait_m48_448':
    #             transform = transforms.Compose([
    #             transforms.Resize((448,448)),
    #             ])
    #             image = transform(image)
    #         output = feature_extractor(image)
    #         # if epoch ==49:
    #         #     for l, o in zip(label, output):
    #         #         print('output : ' , o , 'label : ' , l,'\n')
    #         loss = criterion(output, label)
    #         loss.requires_grad = True
    #         optimizer.zero_grad()

    #         loss.backward()
    #         optimizer.step()
    #     if config.data.category:
    #         torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
    #     else:
    #         torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))

    # return feature_extractor




def feature_distance(output, target,feature_extractor, constants_dict, config):

    # output = ((output - output.min())/ (output.max() - output.min())) 
    # target = ((target - target.min())/ (target.max() - target.min())) 
    

   # print('output : ', output.max(), output.min())

    outputs_features = extract_features(feature_extractor=feature_extractor, x=output, config=config) 
    targets_features = extract_features(feature_extractor=feature_extractor, x=target, config=config) 

    # p_id = patchify(outputs_features) - patchify(targets_features)

    # cosine_distance = 1 - F.cosine_similarity(patchify(outputs_features) , patchify(targets_features), dim=1).to(config.model.device).unsqueeze(1)
    cosine_distance = 1 - F.cosine_similarity(outputs_features , targets_features, dim=1).to(config.model.device).unsqueeze(1)

    # euclidian_distance = torch.sqrt(torch.sum((outputs_features - targets_features)**2, dim=1).unsqueeze(1))
    # euclidian_distance = torch.sqrt(torch.sum((patchify(outputs_features) - patchify(targets_features))**2, dim=1).unsqueeze(1))
    # L1d = torch.sqrt(torch.sum((outputs_features - targets_features), dim=1).unsqueeze(1))
    # euclidian_distance = torch.cdist(outputs_features, targets_features, p=2)
    # print('euclidian_distance : ', euclidian_distance.shape)
    distance_map = F.interpolate(cosine_distance , size = int(config.data.image_size), mode="bilinear")


    return distance_map



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
            input_size: 448
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

                