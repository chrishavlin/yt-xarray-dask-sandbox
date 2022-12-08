import numpy as np
import pandas as pd
import xarray as xr
from typing import Union

def get_chunk_info(xr_dask_obj: Union[xr.Dataset, xr.DataArray], return_df: bool=True) -> Union[list, pd.DataFrame]:

    # xr_dask_obj.chunks may be dict (for Dataset) or a tuple (for DataArray)
    # so need to regularize a bit:
    if isinstance(xr_dask_obj, xr.Dataset):
        dim_ordering = tuple(xr_dask_obj.dims.keys())  # e.g., x, y, z
        coords_in_order = dim_ordering  # e.g., x,y,z
    elif isinstance(xr_dask_obj, xr.DataArray):
        dim_ordering = (0, 1, 2)
        coords_in_order = xr_dask_obj.dims # e.g., x, y, z
    else:
        raise TypeError(f"type(xr_dask_obj)={type(xr_dask_obj)}, "
                        f"expected xr Dataset or DataArray")

    chunk_info = []
    start_index = np.array([0, 0, 0])
    end_index = np.array([0, 0, 0])
    chunk_size = np.array([0, 0, 0])
    le = np.array([0., 0., 0.])  # left edge of each chunk
    re = np.array([0., 0., 0.])  # right edge of each chunk

    ichunk = 0

    for d0_chunks in xr_dask_obj.chunks[dim_ordering[0]]:
        end_index[0] = start_index[0] + d0_chunks
        le[0] = xr_dask_obj.coords[coords_in_order[0]][start_index[0]]
        re[0] = xr_dask_obj.coords[coords_in_order[0]][end_index[0] - 1]

        start_index[1] = 0
        for d1_chunks in xr_dask_obj.chunks[dim_ordering[1]]:
            end_index[1] = start_index[1] + d1_chunks
            le[1] = xr_dask_obj.coords[coords_in_order[1]][start_index[1]]
            re[1] = xr_dask_obj.coords[coords_in_order[1]][end_index[1] - 1]

            start_index[2] = 0
            for d2_chunks in xr_dask_obj.chunks[dim_ordering[2]]:
                end_index[2] = start_index[2] + d2_chunks
                le[2] = xr_dask_obj.coords[coords_in_order[2]][start_index[2]]
                re[2] = xr_dask_obj.coords[coords_in_order[2]][end_index[2] - 1]

                chunk_info.append({'si': start_index.copy(),
                                   'ei': end_index.copy(),
                                   'size': np.array(
                                       [d0_chunks, d1_chunks, d2_chunks]),
                                   'le': le.copy(),
                                   're': re.copy(),
                                   'chunk_number': ichunk})

                start_index[2] += d2_chunks
                ichunk += 1
            start_index[1] += d1_chunks
        start_index[0] += d0_chunks

    if return_df:
        return pd.DataFrame(chunk_info).set_index('chunk_number')
    return chunk_info
