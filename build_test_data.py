from daxryt import data
import numpy as np
from dask.distributed import LocalCluster

if __name__ == '__main__':

    c = LocalCluster(n_workers=6, threads_per_worker=1)

    data.build_test_data(
        n_grids_xyz=(5, 5, 5),
        grid_wids_xyz=np.array([1., 1., 1.]),
        cells_per_grid_xyz=(50, 61, 54),
        output_dir="data",
        clear_output_dir=True,
    )

    c.close()
