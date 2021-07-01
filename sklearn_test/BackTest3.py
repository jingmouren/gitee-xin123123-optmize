from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xarray as xr

ds = xr.Dataset(
    {
        "foo": (("x", "y"), np.random.rand(4,3))
    },
    coords={
        "x": [10,20,30,40], "letters": ("x", list("abba"))
    }
)

print(ds)


