from typing import (
    List,
    Optional,
    Sequence
)
from ..base import BinningBase
from sklearn.utils import check_array
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted
)


class ConstantWidthBinning(BinningBase):
    """等宽分箱.

    Public:
        bin_nbr (int): 分箱的个数
        confined (bool): 是否以训练数据的上下限位上下限
        copy (bool): 是否复制输入

    Protected:
        _bins (List[float]): 分箱的间隔位置组成的list,相邻两点左开右闭
        _data_min (float): 训练数据的最小值
        _data_max (float): 训练数据的最大值
        _n_samples_seen (int): 训练的X样本大小
        _data_range (float): 最大值与最小值间的距离
        _step (float): 每步间隔大小
    """

    def __init__(self, bin_nbr: int=4, confined: bool=True, copy: bool=True)->None:
        if not isinstance(bin_nbr, int)
            raise AttributeError("bin number must be int")
        if confined and bin_nbr <= 0:
            raise AttributeError("bin number must > 0 when confined is True")
        if not confined and bin_nbr <= 2:
            raise AttributeError("bin number must > 2 when confined is False")

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
            del self._n_samples_seen
            del self._data_range
            del self._step

    def fit(self, X, y=None)->None:
        """[summary]

        Args:
            X (Sequence[float]): 待训练的连续型参数
            y (Optional[Sequence[float]]): Defaults to None. 没用

        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None)->None:
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
        if not hasattr(self, '_n_samples_seen'):
            self._n_samples_seen = X.shape[0]
        # Next steps
        else:
            data_min = np.minimum(self._data_min, data_min)
            data_max = np.maximum(self._data_max, data_max)
            self._n_samples_seen += X.shape[0]

        data_range = data_max - data_min
        self._data_min = data_min
        self._data_max = data_max
        self._data_range = data_range

        if self.confined:
            self._step = self._data_range / self.bin_nbr
            self._bins = np.zeros((self._step.shape[0], self.bin_nbr))
            res = []
            for i in range(self.bin_nbr):
                r = data_min + _step * i
                res.append(r)
            res.append(data_max)
            bins = np.array(res)
            # 最左和最右都扩大1%以囊括最小最大值
            bins[0] = bins[0] * 0.99 if bins[0] > 0 else bins[0] - 0.01
            bins[-1] = bins[-1] * 1.01
            self._bins = bins.T
        else:
            self._step = self._data_range / (self.bin_nbr - 2)
            self._bins = np.zeros((self._step.shape[0], (self.bin_nbr - 2)))
            res = []
            for i in range(self.bin_nbr - 2):
                r = data_min + _step * i
                res.append(r)
            res.append(data_max)
            bins = np.array(res)
            self._bins = bins.T
        return self

    def _transform_item(self, x, features_line)->int:
        for i in range(len(self._bins[features_line]) - 1):
            if self.confined:
                if self._bins[i] <= x < self._bins[i + 1]:
                    return i
                else:
                    continue
            else:
                if self._bins[i] <= x < self._bins[i + 1]:
                    return i + 1
                else:
                    continue
        else:
            if self.confined:
                raise AttributeError(f"{x} not in range")
            else:
                if x >= self._bins[-1]:
                    return len(self._bins) + 1
                if x < self._bins[0]:
                    return 0

    def _transform(self, X, features_line):
        for i,value in enumerate(X):
            X[i] = self._transform_item(value,features_line)
        return X

    def transform(self, X):
        """连续数据变换为离散值.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, '_bins')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        result = []
        for features_line, x in enumerate(X.T):
            result.append(self._transform(x, features_line))
        return np.array(result).T

    def inverse_transform(self, X):
        """逆变换.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_is_fitted(self, '_bins')

        X=check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X -= self.min_
        X /= self.scale_
        return X
