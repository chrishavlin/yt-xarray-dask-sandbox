import numpy as np
import pandas as pd
import xarray as xr
from typing import Union


def _get_single_coord_chunk_info(x, si: int, ei: int, nodes: bool = True):
    # x: the full in-memory 1d coordinate array of interest
    # si, ei : start, end indices for the current chunk
    # nodes: if True, data is node-centered. currently, this is what is supported...
    x_chunk = x[si:ei].values
    dx = x_chunk[1:] - x_chunk[0:-1]
    unique_dx = np.unique(dx)
    if len(unique_dx) > 1:
        if np.allclose(unique_dx, unique_dx[0]) is False:
            print(unique_dx)
            raise NotImplementedError("Variable grid widths need some work. Can vary "
                                      "by chunk, but not within chunk...")
    elif len(unique_dx) != 1:
        print(unique_dx)
        raise RuntimeError("Something went wrong with calculating the grid width.")

    unique_dx = unique_dx[0]

    if nodes is False:
        raise NotImplementedError("This only works for node-centered data right now")

    dx2 = unique_dx/2.
    le = x_chunk[0] - dx2
    re = x_chunk[-1] + dx2
    return le, re, unique_dx

class ChunkWalker:
    def __init__(self):

        # global state, calculated after recursion
        self.global_le = None
        self.global_re = None
        self.global_size = None
        self.df: pd.DataFrame = None

        self._initialize_walk()  # initialize the recursion state variables

    def _initialize_walk(self):
        # chunk state during recursion: these all change during the recursion!
        self.chunk_info = []
        self.start_index = np.array([0, 0, 0])
        self.end_index = np.array([0, 0, 0])
        self.le = np.array([0., 0., 0.])  # left edge of each chunk
        self.re = np.array([0., 0., 0.])  # right edge of each chunk
        self.ichunk = 0
        self.cwidth = np.array([0., 0., 0.])  # cell width
        self.csize = np.array([0, 0, 0])  # chunk size
        self.c_dim_i = 0  # current dimension index
        self.ndims = 0 # total dimensions

    def _find_global_stats(self):
        if self.df is None:
            raise RuntimeError("No df, run `walk_the_chunks`")

        self.global_le = np.column_stack(self.df['le'].to_numpy()).min(axis=1)
        self.global_re = np.column_stack(self.df['re'].to_numpy()).max(axis=1)
        self.global_size = np.column_stack(self.df['size'].to_numpy()).sum(axis=1)

    def walk_the_chunks(self, xr_dask_obj: Union[xr.Dataset, xr.DataArray]):

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

        # ndim check here maybe
        self._initialize_walk()
        self.ndims = len(coords_in_order)
        self._recursive_chonker(xr_dask_obj, dim_ordering, coords_in_order)
        self.df = pd.DataFrame(self.chunk_info).set_index('chunk_number')
        self._find_global_stats()

    def _store_chunk(self):
        # stores the current state of the chunk info
        self.chunk_info.append({'si': self.start_index.copy(),
                           'ei': self.end_index.copy(),
                           'size': self.csize.copy(),
                           'le': self.le.copy(),
                           're': self.re.copy(),
                           'cell_widths': self.cwidth.copy(),
                           'chunk_number': self.ichunk})

        self.ichunk += 1

    def _recursive_chonker(self,
                           xr_dask_obj,
                           dim_ordering,
                           coords_in_order):

        # this recursively steps through the chunks of an xarray-dask object,
        # recording the left/right edge of each chunk and more (see _store_chunk
        # for all of the info stored on each chunk).

        for di_chunks in xr_dask_obj.chunks[dim_ordering[self.c_dim_i]]:
            self.end_index[self.c_dim_i] = self.start_index[self.c_dim_i] + di_chunks
            self.csize[self.c_dim_i] = di_chunks
            lei, rei, cwidi= _get_single_coord_chunk_info(
                xr_dask_obj.coords[coords_in_order[self.c_dim_i]],
                self.start_index[self.c_dim_i],
                self.end_index[self.c_dim_i])

            self.le[self.c_dim_i] = lei
            self.re[self.c_dim_i] = rei
            self.cwidth[self.c_dim_i] = cwidi

            if self.ndims == self.c_dim_i + 1:
                # we are at the bottom
                self._store_chunk()
            else:
                # we must go deeper
                self.c_dim_i += 1
                self.start_index[self.c_dim_i] = 0  # zero out
                self._recursive_chonker(xr_dask_obj, dim_ordering, coords_in_order)
                self.c_dim_i -= 1

            self.start_index[self.c_dim_i] += di_chunks
