### ScanNet Benchmark constants ###
VALID_CLASS_IDS_OR42 = (
    0, 
    1, 
    2, 
    3, 
    4, 
    5, 
    6, 
    7, 
    8, 
    9, 
    10, 
    11, 
    12, 
    13, 
    14, 
    15, 
    16, 
    17, 
    18, 
    19, 
    20, 
    21, 
    22, 
    23, 
    24, 
    25, 
    26, 
    27, 
    28, 
    29, 
    30, 
    31, 
    32, 
    33, 
    34, 
    35, 
    36, 
    37, 
    38, 
    # 39, 
    # 40, 
    # 41, 
)

CLASS_LABELS_OR42 = (
    'curtain', 
    'bike', 
    'washing_machine', 
    'table', 
    'dishwasher', 
    'bowl', 
    'bookshelf', 
    'sofa', 
    'speaker', 
    'trash_bin', 
    'piano', 
    'guitar', 
    'pillow', 
    'jar', 
    'bed', 
    'bottle', 
    'clock', 
    'chair', 
    'computer_keyboard', 
    'monitor', 
    'bathtub', 
    'stove', 
    'microwave', 
    'file_cabinet', 
    'flowerpot', 
    'cap', 
    'window', 
    'ceiling_lamp', 
    'telephone', 
    'printer', 
    'basket', 
    'faucet', 
    'bag', 
    'laptop', 
    'lamp', 
    'can', 
    'bench', 
    'door', 
    'cabinet', 
    # 'wall', 
    # 'floor', 
    # 'ceiling'
    # 'unlabelled', 
)

CLASS_COLOR_MAP_OR42 = {
    0: (174, 199, 232), 
    1: (152, 223, 138), 
    2: (31, 119, 180), 
    3: (255, 187, 120), 
    4: (214, 39, 40), 
    5: (197, 176, 213), 
    6: (148, 103, 189), 
    7: (196, 156, 148), 
    8: (23, 190, 207), 
    9: (178, 76, 76), 
    10: (247, 182, 210), 
    11: (66, 188, 102), 
    12: (219, 219, 141), 
    13: (140, 57, 197), 
    14: (202, 185, 52), 
    15: (51, 176, 203), 
    16: (200, 54, 131), 
    17: (92, 193, 61), 
    18: (78, 71, 183), 
    19: (172, 114, 82), 
    20: (91, 163, 138), 
    21: (153, 98, 156), 
    22: (140, 153, 101), 
    23: (158, 218, 229), 
    24: (100, 125, 154), 
    25: (178, 127, 135), 
    26: (120, 185, 128), 
    27: (190, 153, 153), 
    28: (44, 160, 44), 
    29: (112, 128, 144), 
    30: (96, 207, 209), 
    31: (227, 119, 194), 
    32: (213, 92, 176), 
    33: (94, 106, 211), 
    34: (146, 111, 194), 
    35: (82, 84, 163), 
    36: (100, 85, 144), 
    37: (0, 0, 230), 
    38: (119, 11, 32), 
    39: (102, 102, 156), 
    40: (0, 0, 0), 
    41: (0, 80, 100), 
    255: (255, 255, 255), 
}
### For instance segmentation the non-object categories ###
# VALID_PANOPTIC_IDS = (1, 3)

CLASS_LABELS_PANOPTIC = ("wall", "floor", "ceiling")