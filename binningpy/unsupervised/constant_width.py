from typing import (
    List,
    Optional,
    Sequence
)
from ..base import BinningBase
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

class ConstantWidthBinning(BinningBase):
    """等宽分箱.

    Public:
        bin_nbr (int): 分箱的个数
        confined (bool): 是否以训练数据的上下限位上下限
        copy (bool): 是否复制输入

    Protected:
        _bins (List[float]): 分箱的间隔位置组成的list,间隔点属于其左边的区间
        _data_min (float): 训练数据的最小值
        _data_max (float): 训练数据的最大值
        _step (float): 间隔大小
    """

    def __init__(self, bin_nbr:int=4,confined:bool=True, copy:bool=True)->None:
        self.bin_nbr = bin_nbr
        self.confined = confined
        self.copy = copy

    def _reset(self)->None:
        """Reset internal data-dependent state of the binning, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, '_bins'):
            del self._bins
            del self._data_min
            del self._data_max
            del self._step

    def fit(self, X:Sequence[float], y=None)->None:
        """[summary]

        Args:
            X ([type]): [description]
            y ([type], optional): Defaults to None. [description]

        Returns:
            [type]: [description]
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X:Sequence[float], y=None)->None:
        """

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : Passthrough for ``Pipeline`` compatibility.
        """

        X = check_array(X, copy=self.copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """连续数据变换为离散值.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        """逆变换.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X -= self.min_
        X /= self.scale_
        return X
