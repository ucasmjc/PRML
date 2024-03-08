from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

#ex:111.95060214943143  117.3518117503383  126.65842765412415
#std:75.79144205482484  75.87104329766099  78.56573392604658
@DATASETS.register_module()
class carDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'car',),
        palette=[[120, 120, 120], [6, 230, 230]],
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
@DATASETS.register_module()
class kaggleDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'car',),
        palette=[[120, 120, 120], [6, 230, 230]],
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

