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
        image = batch[0]
        image = image.to(config.model.device)
        model = build_model(config)
        if config.data.category:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'200')) # config.model.checkpoint_name 300+50
        else:
            checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), '200'))
        model.load_state_dict(checkpoint)    
        model.to(config.model.device)
        model.eval()
        generate_time_steps = torch.Tensor([config.model.generate_time_steps]).type(torch.int64)
        noise = get_noise(image,config) 
        # noise = forward_diffusion_sample(image, generate_time_steps, constants_dict, config)[0]
        seq = range(0, config.model.generate_time_steps, 50 * config.model.skip)
        H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
        reconstructed,_ =  efficient_generalized_steps(config, noise, seq, model,  constants_dict['betas'], H_funcs, image, cls_fn=None, classes=None) #generalized_steps(noise, seq, model, constants_dict['betas'], config, eta=config.model.eta)
        generated_image = reconstructed[-1]
        generated_image = generated_image.to(config.model.device)

        for fake, real in zip(generated_image, image):
            fake_label = torch.Tensor([1,0]).type(torch.float32).to(config.model.device)
            real_label = torch.Tensor([0,1]).type(torch.float32).to(config.model.device)
            R_F_dataset.append((fake.type(torch.float32), fake_label))
            R_F_dataset.append((real.type(torch.float32), real_label))
        # break
    return R_F_dataset






def tune_feature_extractor(constants_dict, config):
    
    feature_extractor =  timm.create_model(
                        config.model.backbone,
                        pretrained=True,
                        num_classes=2,
                    )
    feature_extractor.to(config.model.device)                
    if config.model.fine_tune:
        R_F_dataset = fake_real_dataset(config, constants_dict)
        R_F_dataloader = torch.utils.data.DataLoader(R_F_dataset, batch_size=config.data.batch_size, shuffle=True) 
        feature_extractor.train()
        optimizer = torch.optim.SGD(feature_extractor.parameters(), lr=0.001, momentum=0.9) #config.model.learning_rate
        criterion =  nn.CrossEntropyLoss()
        print("Start training feature extractor")
        for epoch in tqdm(range(50)):
            for step, batch in enumerate(R_F_dataloader):
                image = batch[0]
                label = batch[1]
                if epoch == 49:
                    plt.figure(figsize=(11,11))
                    plt.axis('off')
                    plt.subplot(1, 1, 1)
                    plt.imshow(show_tensor_image(image))
                    plt.title(label[0])
                    plt.savefig('results/F_or_R{}_{}.png'.format(epoch,step))
                    plt.close()
                output = feature_extractor(image)
                # if epoch ==49:
                #     for l, o in zip(label, output):
                #         print('output : ' , o , 'label : ' , l,'\n')
                loss = criterion(output, label)
                loss.requires_grad = True
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 50, loss.item()))
        if config.data.category:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))
        else:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), 'feature'))

    return feature_extractor





def extract_features(feature_extractor, x, out_indices, config):
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
        for name, module in feature_extractor.named_children():
            x = module(x)
            if name in [ 'layer1', 'layer2', 'layer3']:
                activations.append(x)
        embeddings = activations[0]
        for feature in activations[1:]:
            layer_embedding = feature
            layer_embedding = F.interpolate(layer_embedding, size=int(embeddings.shape[-2]), mode='bilinear', align_corners=False)
            embeddings = torch.cat((embeddings,layer_embedding),1)
        return embeddings

                