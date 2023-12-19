import logging
import torch
from dataset import *
from dataset import *
from unet import *
from visualize import *
from resnet import *
import torchvision.transforms as T
from reconstruction import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def loss_fucntion(a, b, c, d, config):
    cos_loss = torch.nn.CosineSimilarity()
    loss1 = 0
    loss2 = 0
    loss3 = 0
    for item in range(len(a)):
        loss1 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1))) 
        loss2 += torch.mean(1-cos_loss(b[item].view(b[item].shape[0],-1),c[item].view(c[item].shape[0],-1))) * config.model.DLlambda
        loss3 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),d[item].view(d[item].shape[0],-1))) * config.model.DLlambda
    loss = loss1+loss2+loss3
    return loss



def domain_adaptation(unet, config, fine_tune):
    if config.model.feature_extractor == 'wide_resnet101_2':
        feature_extractor = wide_resnet101_2(pretrained=True)
        frozen_feature_extractor = wide_resnet101_2(pretrained=True)
    elif config.model.feature_extractor == 'wide_resnet50_2':
        feature_extractor = wide_resnet50_2(pretrained=True)
        frozen_feature_extractor = wide_resnet50_2(pretrained=True)
    elif config.model.feature_extractor == 'resnet50': 
        feature_extractor = resnet50(pretrained=True)
        frozen_feature_extractor = resnet50(pretrained=True)
    else:
        logging.warning("Feature extractor is not correctly selected, Default: wide_resnet101_2")
        feature_extractor = wide_resnet101_2(pretrained=True)
        frozen_feature_extractor = wide_resnet101_2(pretrained=True)

    feature_extractor.to(config.model.device)  
    frozen_feature_extractor.to(config.model.device)

    frozen_feature_extractor.eval()

    feature_extractor = torch.nn.DataParallel(feature_extractor)
    frozen_feature_extractor = torch.nn.DataParallel(frozen_feature_extractor)


    train_dataset = Dataset_maker(
        root= config.data.data_dir,
        category= config.data.category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.DA_batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )   

    if fine_tune:      
        unet.eval()
        feature_extractor.train()


        transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        optimizer = torch.optim.AdamW(feature_extractor.parameters(),lr= 1e-4)
        torch.save(frozen_feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'feat0'))
        reconstruction = Reconstruction(unet, config)
        for epoch in range(config.model.DA_epochs):
            for step, batch in enumerate(trainloader):
                half_batch_size = batch[0].shape[0]//2
                target = batch[0][:half_batch_size].to(config.model.device)  
                input = batch[0][half_batch_size:].to(config.model.device)   
                
                x0 = reconstruction(input, target, config.model.w_DA)[-1].to(config.model.device)
                x0 = transform(x0)
                target = transform(target)

                reconst_fe = feature_extractor(x0)
                target_fe = feature_extractor(target)

                target_frozen_fe = frozen_feature_extractor(target)
                reconst_frozen_fe = frozen_feature_extractor(x0)
                
                

                loss = loss_fucntion(reconst_fe, target_fe, target_frozen_fe,reconst_frozen_fe, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} | Loss: {loss.item()}")
            # if (epoch+1) % 5 == 0:
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'feat{epoch+1}'))
    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'feat{config.model.DA_chp}'))#{config.model.DA_chp}            
        feature_extractor.load_state_dict(checkpoint)  
    return feature_extractor