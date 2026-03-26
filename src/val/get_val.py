from src.val.GPT2 import val_GPT2
from src.val.Llama import val_Llama
from src.val.Bert import val_Bert


def get_val(model_name, data_name, state_dict_full, logger):
    """
    Chạy validation và trả về full state dict nếu thành công, None nếu lỗi.
    Server dùng full state dict này để lưu vào GPT2.pt cho round tiếp theo.
    """
    try:
        if model_name == 'GPT2':
            return val_GPT2(model_name, data_name, state_dict_full, logger)
        elif model_name == 'Llama':
            return val_Llama(model_name, data_name, state_dict_full, logger)
        elif model_name == 'Bert':
            return val_Bert(model_name, data_name, state_dict_full, logger)
        else:
            logger.log_warning(f"Unknown model_name '{model_name}' for validation.")
            return None
    except Exception as e:
        logger.log_error(f"Validation failed with exception: {e}")
        return None