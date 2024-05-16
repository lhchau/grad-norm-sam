import torch.optim as optim

from .sam import SAM
from .usam import USAM
from .sama import SAMA
from .samaccer import SAMACCER
from .samyan import SAMYAN
from .fsam import FSAM
from .fzsam import FZSAM
from .fznsam import FZNSAM
from .grsam import GRSAM
from .mgrsam import MGRSAM
from .varsam import VARSAM
from .varsam1 import VARSAM1
from .varsam2 import VARSAM2
from .fsam2 import FSAM2
from .extrasam import EXTRASAM
from .wsam import WSAM
from .chausam import CHAUSAM
from .chausam2 import CHAUSAM2
from .samasc import SAMASC
from .csam import CSAM
from .clipsam import CLIPSAM


def get_optimizer(
    net,
    base_opt_name=None,
    opt_name='sam',
    opt_hyperparameter={}):
    if base_opt_name is None:
        base_optimizer = optim.SGD
    elif base_opt_name == 'adam':
        base_optimizer = optim.Adam
    if opt_name == 'sam':
        return SAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'usam':
        return USAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'fsam':
        return FSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'sama':
        return SAMA(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'samaccer':
        return SAMACCER(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'samyan':
        return SAMYAN(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'fzsam':
        return FZSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'grsam':
        return GRSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'mgrsam':
        return MGRSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'fznsam':
        return FZNSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'varsam':
        return VARSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'varsam1':
        return VARSAM1(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'varsam2':
        return VARSAM2(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'fsam2':
        return FSAM2(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'extrasam':
        return EXTRASAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'wsam':
        return WSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'chausam':
        return CHAUSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'chausam2':
        return CHAUSAM2(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'samasc':
        return SAMASC(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'csam':
        return CSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    elif opt_name == 'clipsam':
        return CLIPSAM(
            net.parameters(), 
            base_optimizer, 
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")