import logging
import torch
from forward_process import *
from dataset import *
from dataset import *
from unet import *
from sample import *
from visualize import *
from resnet import *
import torchvision.transforms as T


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))
    return loss



def Domain_adaptation(unet, config, fine_tune):
    if config.model.feature_extractor == 'wide_resnet101_2':
        feature_extractor = wide_resnet101_2(pretrained=True)
    elif config.model.feature_extractor == 'wide_resnet50_2':
        feature_extractor = wide_resnet50_2(pretrained=True)
    elif config.model.feature_extractor == 'resnet50': 
        feature_extractor = resnet50(pretrained=True)
    else:
        logging.warning("Feature extractor is not correctly selected, Default: wide_resnet101_2")
        feature_extractor = wide_resnet101_2(pretrained=True)

    feature_extractor.to(config.model.device)  


    train_dataset = Dataset_maker(
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

    if fine_tune:      
        unet.eval()
        feature_extractor.train()
        
        for param in feature_extractor.parameters():
            param.requires_grad = True

        transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        optimizer = torch.optim.Adam(feature_extractor.parameters(),lr= 1e-4)      
        
        for epoch in range(config.model.DA_epochs):
            for step, batch in enumerate(trainloader):
                half_batch_size = batch[0].shape[0]//2
                target = batch[0][:half_batch_size].to(config.model.device)  
                input = batch[0][half_batch_size:].to(config.model.device)     
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps_DA]).type(torch.int64).to(config.model.device)
                at = compute_alpha(test_trajectoy_steps.long(),config)
                noisy_image = at.sqrt() * input + (1- at).sqrt() * torch.randn_like(input).to('cuda')
                seq = range(0 , config.model.test_trajectoy_steps_DA, config.model.skip_DA)

                reconstructed = Reconstruction(target, noisy_image, seq, unet, config, config.model.w)
                data_reconstructed = reconstructed[-1].to(config.model.device)


                data_reconstructed = transform(data_reconstructed)
                reconst_fe = feature_extractor(data_reconstructed)

                target = transform(target)
                target_fe = feature_extractor(target)

                loss = loss_fucntion(reconst_fe, target_fe)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} | Loss: {loss.item()}")
        torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))

    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'feature'))            
        feature_extractor.load_state_dict(checkpoint)  
    return feature_extractor