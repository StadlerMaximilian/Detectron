# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""VGG_CNN_M_1024 from https://arxiv.org/abs/1405.3531."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

from caffe2.python import cnn
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

from detectron.core.config import cfg
from detectron.ops.collect_and_distribute_fpn_rpn_proposals \
    import CollectAndDistributeFpnRpnProposalsOp
from detectron.ops.generate_proposal_labels import GenerateProposalLabelsOp
from detectron.ops.generate_proposals import GenerateProposalsOp
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.utils.c2 as c2_utils
from detectron.modeling.detector import _get_lr_change_ratio, DetectionModelHelper

logger = logging.getLogger(__name__)


class PerceptualGanModelHelper(DetectionModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.generator_params = [] # Param on this list are part of Generator network
        self.discriminator_params_adv = [] # Param on this list are part of adversarial branch of discriminator
        self.discriminator_params_per = [] # Param on this list are part of the perceptual branch of discriminator
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = cfg.NUM_GPUS * 4
        self.prev_use_cudnn = self.use_cudnn
        self.gn_params = []  # Param on this list are GroupNorm parameters

    def TrainableParams(self, gpu_id=-1):
        """Get the blob names for all trainable parameters, possibly filtered by
        GPU id.
        """
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]


def add_generator_base(model, blob_in, dim_in, num_blocks):
    """ adds the residual blocks of the generator upon the backbone-specific base """
    """ 3x3conv -- BN -- ReLU -- 3x3conv -- elt-sum with residual """
    for n in range(num_blocks-1):
        blob_in = add_generator_base_block(
            model,
            'gen_res{}'.format(n+1),
            blob_in,
            dim_in,
            inplace_sum=True
        )
    blob_in = add_generator_base_block(model,
                                       'gen_res{}'.format(num_blocks),
                                       blob_in,
                                       dim_in,
                                       inplace_sum=False)
    return blob_in, dim_in


def add_generator_base_block(model,
                             prefix,
                             blob_in,
                             dim_in,
                             inplace_sum=True):
    """ adds one residual block of the generator """
    # prefix = gen_res<stage>, e.g. gen_res1, as generator does not uses several substages
    tr = generator_transformation(model, blob_in, dim_in, dim_in, prefix)
    sc = generator_shortcut(model, blob_in, dim_in, dim_in, prefix)

    if inplace_sum:
        blob_out = model.net.Sum([tr, sc], tr)
    else:
        blob_out = model.net.Sum([tr, sc], prefix + '_sum')
    return blob_out


def generator_transformation(model,
                             blob_in,
                             dim_in,
                             dim_out,
                             prefix,
                             group=1):
    """ represents the convolutional transforation within one residual block"""

    blob = model.ConvAffine(blob_in,
                            prefix + '_branch2a',
                            dim_in,
                            dim_in,
                            kernel=3,
                            stride=1,
                            pad=1,
                            group=group,
                            inplace=True
                            )

    blob = model.Relu(blob, blob)

    blob_out = model.Conv(blob,
                          prefix + '_branch2b',
                          dim_in,
                          dim_out,
                          3,
                          pad=1,
                          stride=1,
                          group=group,
                          inplace=True
                          )

    return blob_out


def generator_shortcut(model, blob_in, dim_in, dim_out, prefix, stride=1, bn=False):
    """
    defines the shortcut withing one residual block in the generator
    if bn=True: for a pre-trained network that used BN, an AffineChannel op replaces BN
    during fine-tuning.
    """

    if dim_in == dim_out:
        return blob_in
    elif bn:
        return model.Conv(blob_in, prefix + '_branch1', dim_in, dim_out, kernel=1, stride=stride)

    blob = model.Conv(blob_in, prefix + '_branch1', dim_in, dim_out, kernel=1,stride=stride, no_bias=1)
    return model.AffineChannel(blob, prefix + '_branch1_bn', dim=dim_out)