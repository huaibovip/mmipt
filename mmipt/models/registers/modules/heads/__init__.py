# Copyright (c) MMIPT. All rights reserved.
from .base_registration_head import BaseRegistrationHead
from .affine_registration_head import AffineRegistrationHead
from .bspline_registration_head import BSplineRegistrationHead
from .deformable_registration_head import DeformableRegistrationHead
from .dual_registration_head import DualRegistrationHead

__ALL__ = [
    'BaseRegistrationHead',
    'BSplineRegistrationHead',
    'AffineRegistrationHead',
    'DeformableRegistrationHead',
    'DualRegistrationHead',
]
