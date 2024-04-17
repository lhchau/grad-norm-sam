import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime

import os
import argparse
import yaml
import wandb
import pprint
import shutil
from torch.utils.tensorboard import SummaryWriter

from accelerated_sam.models import *
from accelerated_sam.utils import *
from accelerated_sam.dataloader import *
from accelerated_sam.scheduler import *
from accelerated_sam.optimizer import *
from accelerated_sam.utils.pyhessian import get_eigen_hessian_plot


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)
    pprint.pprint(cfg)
seed = cfg['trainer'].get('seed', 42)
initialize(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0

EPOCHS = cfg['trainer']['epochs'] 

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += ('_' + current_time + '_resume')

framework_name = cfg['logging']['framework_name']
if framework_name == 'tensorboard':
    writer = SummaryWriter(os.path.join('runs', logging_name))
elif framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=cfg['wandb']['name'])

logging_dict = {}
################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(**cfg['dataloader'])
try:
    num_classes = len(classes)
except:
    num_classes = classes

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(
    **cfg['model'],
    num_classes=num_classes
)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(cfg['trainer']['load_path'])
net.load_state_dict(checkpoint['net'])

start_epoch = checkpoint['epoch']
EPOCHS = EPOCHS - start_epoch

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]["model_name"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss().to(device)
sch = cfg['trainer'].get('sch', None)
optimizer = get_optimizer(
    net, 
    **cfg['optimizer']
)
scheduler = get_scheduler(
    optimizer, 
    **cfg['scheduler']
)

for i in range(start_epoch):
    scheduler.step()
################################
#### 3.b Training 
################################
if __name__ == "__main__":
    try: 
        for epoch in range(start_epoch, start_epoch+EPOCHS):
            print('\nEpoch: %d' % epoch)
            loop_one_epoch(
                dataloader=train_dataloader,
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                logging_dict=logging_dict,
                epoch=epoch,
                loop_type='train',
                logging_name=logging_name
            )
            best_acc = loop_one_epoch(
                dataloader=val_dataloader,
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                logging_dict=logging_dict,
                epoch=epoch,
                loop_type='val',
                logging_name=logging_name,
                best_acc=best_acc
            )
            scheduler.step()
            
            if framework_name == 'tensorboard':
                for key, value in logging_dict.items():
                    if not isinstance(key, str):
                        writer.add_scalar(key[0], value[0], global_step=key[1] + epoch*value[1])
                    else:
                        writer.add_scalar(key, value, global_step=epoch)
            elif framework_name == 'wandb':
                for key, value in logging_dict.items():
                    if not isinstance(key, str):
                        logging_dict[key[0]] = value[0]
                        del logging_dict[key]
                wandb.log(**logging_dict)
        
        logging_dict = {}
        loop_one_epoch(
            dataloader=test_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name
        )
        
        if framework_name == 'tensorboard':
            for key, value in logging_dict.items():
                if not isinstance(key, str):
                    writer.add_scalar(key[0], value[0], global_step=key[1] + epoch*value[1])
                else:
                    writer.add_scalar(key, value, global_step=epoch)
        elif framework_name == 'wandb':
            for key, value in logging_dict.items():
                if not isinstance(key, str):
                    logging_dict[key[0]] = value[0]
                    del logging_dict[key]
            wandb.log(**logging_dict)
            
        get_eigen_hessian_plot(
            name=logging_name, 
            net=net,
            criterion=criterion,
            dataloader=train_dataloader
        )
        
    except KeyboardInterrupt as e:
        save_dir = os.path.join('checkpoint', logging_name)
        logging_dir = os.path.join('runs', logging_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        print(f"Error: {e}")