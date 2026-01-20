from src.val.Bert import val_Bert

def get_val(state_dict_full, cut_layers,bottleneck_config, logger):
    val_Bert(state_dict_full, cut_layers , bottleneck_config, logger)
    return True
