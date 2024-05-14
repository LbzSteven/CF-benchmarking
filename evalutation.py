from utils.model_util import get_AE_dict

AE_dict = get_AE_dict('FCN_AE', '../models/AE/GunPoint', 1, 150)
for keys, values in AE_dict.items():
    print(keys)
    # print(values)