# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .ocr_head import OCRHead
from .pid_head import PIDHead
from .psp_head import PSPHead
from .stdc_head import STDCHead
from .pp_head import ppHead
from .light import LightHead
from .custom_head import testHead,bdHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead',  'OCRHead',
 'STDCHead',  'PIDHead','ppHead','LightHead','testHead','bdHead'
]
