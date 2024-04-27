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
    else:
        raise ValueError("Invalid optimizer!!!")