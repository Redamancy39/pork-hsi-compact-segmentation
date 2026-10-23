import numpy as np
def decode_label(a, cfg):
    # Return original class IDs and derived binary view. Unknowns remain -1.
    original = np.full(a.shape[:2], -1, dtype=np.int16)
    if a.ndim == 2:
        for i in range(13): original[a == i] = i
        encoding = 'class_id'
    elif a.ndim == 3 and a.shape[-1] == 3:
        for k,v in cfg['original_id_to_rgb'].items(): original[np.all(a == v, axis=-1)] = int(k)
        encoding = 'exact_RGB_palette'
    else:
        raise ValueError(f'Unsupported label layout {a.shape}; no guessed argmax/channel mapping')
    binary = np.full(original.shape,-1,dtype=np.int8)
    binary[np.isin(original,cfg['derived_foreground'])] = 1
    binary[np.isin(original,cfg['derived_background'])] = 0
    return original, binary, encoding
