from src.val.BERT import val_BERT
from src.val.GPT2 import val_GPT2

def get_val(model_name, state_dict_full, logger):
    if model_name == 'BERT':
        return val_BERT(state_dict_full, logger)
    elif model_name == 'GPT2':
        return val_GPT2(state_dict_full, logger)
    else:
        return False