import numpy as np

def color_map(dataset):
    if dataset == 'SECOND_SCD':
        cmap = np.zeros((7, 3), dtype=np.uint8)
        cmap[0] = np.array([255, 255, 255])
        cmap[1] = np.array([0, 0, 255])
        cmap[2] = np.array([128, 128, 128])
        cmap[3] = np.array([0, 128, 0])
        cmap[4] = np.array([0, 255, 0])
        cmap[5] = np.array([128, 0, 0])
        cmap[6] = np.array([255, 0, 0])
    else:
        raise NotImplementedError

    return cmap


def Color2Index(dataset, ColorLabel):
    if dataset == 'SECOND_SCD':
        num_classes = 7
        ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0],
                       [255, 0, 0]]
        CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
    else:
        raise NotImplementedError

    colormap2label = np.zeros(256 ** 3)
    for i, cm in enumerate(ST_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)

    return IndexMap



