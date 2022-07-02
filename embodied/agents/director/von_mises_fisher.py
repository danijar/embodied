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
"""The von-Mises-Fisher distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import scipy.special
from tensorflow_probability import distributions as tfd

from .hyperspherical_uniform import HypersphericalUniform

__all__ = [
    "VonMisesFisher",
]


class VonMisesFisher(tfd.Distribution):
    """The von-Mises-Fisher distribution with location `loc` and `scale` parameters.
    #### Mathematical details

    The probability density function (pdf) is,

    ```none
    pdf(x; mu, k) = exp(k mu^T x) / Z
    Z = (k ** (m / 2 - 1)) / ((2pi ** m / 2) * besseli(m / 2 - 1, k))
    ```
    where `loc = mu` is the mean, `scale = k` is the concentration, `m` is the dimensionality, and, `Z`
    is the normalization constant.

    See https://en.wikipedia.org/wiki/Von_Mises-Fisher distribution for more details on the
    Von Mises-Fiser distribution.

    """

    def __init__(self, loc, scale, validate_args=False, allow_nan_stats=True, name="von-Mises-Fisher"):
        """Construct von-Mises-Fisher distributions with mean and concentration `loc` and `scale`.

        Args:
          loc: Floating point tensor; the mean of the distribution(s).
          scale: Floating point tensor; the concentration of the distribution(s).
            Must contain only non-negative values.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to tf created by this class.

        Raises:
          TypeError: if `loc` and `scale` have different `dtype`.
        """
        parameters = locals()
        with tf.control_dependencies([tf.assert_positive(scale),
                                       tf.assert_near(tf.norm(loc, axis=-1), 1, atol=1e-7)]
                                      if validate_args else []):
            self._loc = tf.identity(loc, name="loc")
            self._scale = tf.identity(scale, name="scale")

        super(VonMisesFisher, self).__init__(
            dtype=self._scale.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._loc, self._scale],
            name=name)

        self.__m = tf.cast(self._loc.shape[-1], tf.int32)
        self.__mf = tf.cast(self.__m, dtype=self.dtype)
        self.__e1 = tf.one_hot([0], self.__m, dtype=self.dtype)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(zip(("loc", "scale"), ([tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                                            tf.convert_to_tensor(sample_shape[:-1].concatenate([1]),
                                                                  dtype=tf.int32)])))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for concentration."""
        return self._scale

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.shape(self._loc),
            tf.shape(self._scale))

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self._loc.get_shape(),
            self._scale.get_shape())

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])  # tensor_shape.scalar()

    def _sample_n(self, n, seed=None):
        shape = tf.concat([[n], self.batch_shape_tensor()], 0)
        w = tf.cond(tf.equal(self.__m, 3),
                                  lambda: self.__sample_w3(n, seed),
                                  lambda: self.__sample_w_rej(n, seed))

        v = tf.nn.l2_normalize(tf.transpose(
            tf.transpose(tf.random.normal(shape, dtype=self.dtype, seed=seed))[1:]), axis=-1)

        x = tf.concat((w, tf.sqrt(1 - w ** 2) * v), axis=-1)
        z = self.__householder_rotation(x)

        return z

    def __sample_w3(self, n, seed):
        shape = tf.concat(([n], self.batch_shape_tensor()[:-1], [1]), 0)
        u = tf.random.uniform(shape, dtype=self.dtype, seed=seed)
        self.__w = 1 + tf.reduce_logsumexp([tf.math.log(u), tf.math.log(1 - u) - 2 * self.scale], axis=0) / self.scale
        return self.__w

    def __sample_w_rej(self, n, seed):
        c = tf.sqrt((4 * (self.scale ** 2)) + (self.__mf - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__mf - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__mf - 1) / (4 * self.scale)
        s = tf.minimum(tf.maximum(0., self.scale - 10), 1.)
        b = b_app * s + b_true * (1 - s)

        a = (self.__mf - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__mf - 1) * tf.math.log(self.__mf - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, n, seed)
        return self.__w

    def __while_loop(self, b, a, d, n, seed):
        def __cond(w, e, bool_mask, b, a, d):
            return tf.reduce_any(bool_mask)

        def __body(w_, e_, bool_mask, b, a, d):
            e = tf.cast(tfd.Beta((self.__mf - 1) / 2, (self.__mf - 1) / 2).sample(
                shape, seed=seed), dtype=self.dtype)

            u = tf.random.uniform(shape, dtype=self.dtype, seed=seed)

            w = (1 - (1 + b) * e) / (1 - (1 - b) * e)
            t = (2 * a * b) / (1 - (1 - b) * e)

            accept = tf.greater(((self.__mf - 1) * tf.math.log(t) - t + d), tf.math.log(u))
            reject = tf.math.logical_not(accept)

            w_ = tf.where(tf.math.logical_and(bool_mask, accept), w, w_)
            e_ = tf.where(tf.math.logical_and(bool_mask, accept), e, e_)
            bool_mask = tf.where(tf.math.logical_and(bool_mask, accept), reject, bool_mask)

            return w_, e_, bool_mask, b, a, d

        shape = tf.concat([[n], self.batch_shape_tensor()[:-1], [1]], 0)
        b, a, d = [tf.tile(tf.expand_dims(e, axis=0), [n] + [1] * len(e.shape)) for e in (b, a, d)]

        w, e, bool_mask, b, a, d = tf.while_loop(__cond, __body,
                                                               [tf.zeros_like(b, dtype=self.dtype),
                                                                tf.zeros_like(b, dtype=self.dtype),
                                                                tf.ones_like(b, tf.bool),
                                                                b, a, d])

        return e, w

    def __householder_rotation(self, x):
        u = tf.nn.l2_normalize(self.__e1 - self._loc, axis=-1)
        z = x - 2 * tf.reduce_sum(x * u, axis=-1, keepdims=True) * u
        return z

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return tf.exp(self._log_prob(x))

    def _log_unnormalized_prob(self, x):
        with tf.control_dependencies(
                [tf.assert_near(tf.norm(x, axis=-1), 1, atol=1e-3)] if self.validate_args else []):
            output = self.scale * tf.reduce_sum(self._loc * x, axis=-1, keepdims=True)

        return tf.reshape(output, tf.convert_to_tensor(tf.shape(output)[:-1]))

    def _log_normalization(self):
        output = -((self.__mf / 2 - 1) * tf.math.log(self.scale) - (self.__mf / 2) * math.log(2 * math.pi) - (
                    self.scale + tf.math.log(ive(self.__mf / 2 - 1, self.scale))))

        return tf.reshape(output, tf.convert_to_tensor(tf.shape(output)[:-1]))

    def _entropy(self):
        return - tf.reshape(self.scale * ive(self.__mf / 2, self.scale) / ive((self.__mf / 2) - 1, self.scale),
                                   tf.convert_to_tensor(tf.shape(self.scale)[:-1])) + self._log_normalization()

    def _mean(self):
        return self._loc * (ive(self.__mf / 2, self.scale) / ive(self.__mf / 2 - 1, self.scale))

    def _mode(self):
        return self._mean()


@tfd.RegisterKL(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu, name=None):
    with tf.control_dependencies([tf.assert_equal(vmf.loc.shape[-1] - 1, hyu.dim)]):
        with tf.name_scope(name, "_kl_vmf_uniform", [vmf.scale]):
            return - vmf.entropy() + hyu.entropy()


@tf.custom_gradient
def ive(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = tf.reshape(tf.py_function(
        lambda v, z: np.select(condlist=[v == 0, v == 1],
                               choicelist=[scipy.special.i0e(z, dtype=np.float32),
                                           scipy.special.i1e(z, dtype=np.float32)],
                               default=scipy.special.ive(v, z, dtype=np.float32)), [v, z], np.float32),
        tf.convert_to_tensor(tf.shape(z), dtype=tf.int32))

    def grad(dy):
        return None, dy * (ive(v - 1, z) - ive(v, z) * (v + z) / z)

    return output, grad
