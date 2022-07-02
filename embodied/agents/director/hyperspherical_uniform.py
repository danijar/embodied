# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""The Hyperspherical Uniform distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class HypersphericalUniform(tfd.Distribution):
    """Hyperspherical Uniform distribution with `dim` parameter.

    #### Mathematical Details

    """

    def __init__(self, dim, dtype=tf.float32, validate_args=False, allow_nan_stats=True,
                 name="HypersphericalUniform"):
        """Initialize a batch of Hyperspherical Uniform distributions.

        Args:
          dim: Integer tensor, dimensionality of the distribution(s). Must
            be `dim > 0`.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
            (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
            result is undefined. When `False`, an exception is raised if one or
            more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to tf created by this class.

        Raises:
          InvalidArgumentError: if `dim > 0` and `validate_args=False`.
        """
        parameters = locals()
        with tf.name_scope(name, values=[dim]):
            with tf.control_dependencies([tf.assert_positive(dim),
                                           tf.assert_integer(dim),
                                           tf.assert_scalar(dim)] if validate_args else []):
                self._dim = dim

            super(HypersphericalUniform, self).__init__(
                dtype=dtype,
                reparameterization_type=tfd.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                graph_parents=[],
                name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return {}

    @property
    def dim(self):
        """Dimensionality of the distribution(s)."""
        return self._dim

    def _batch_shape_tensor(self):
        return tf.constant([self._dim + 1], dtype=tf.int32)

    def _batch_shape(self):
        return tf.TensorShape(self._dim + 1)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return []

    def _sample_n(self, n, seed=None):
        return tf.nn.l2_normalize(tf.random.normal(shape=tf.concat(([n], [self._dim + 1]), 0),
                                                             dtype=self.dtype, seed=seed), axis=-1)

    def _log_prob(self, x):
        return - tf.ones(shape=tf.shape(x)[:-1], dtype=self.dtype) * self.__log_surface_area()

    def _prob(self, x):
        return tf.exp(self._log_prob(x))

    def _entropy(self):
        return self.__log_surface_area()

    def __log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - tf.math.lgamma(
            tf.cast((self._dim + 1) / 2, dtype=self.dtype))
