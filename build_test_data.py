from daxryt import data_handling
import numpy as np
from dask.distributed import Client
import dask

if __name__ == '__main__':

    with dask.config.set({"distributed.scheduler.worker-saturation": "inf"}):
        c = Client(n_workers=6, threads_per_worker=1)

        data_handling.build_test_data(
            n_grids_xyz=(5, 5, 5),
            grid_wids_xyz=np.array([1., 1., 1.]),
            cells_per_grid_xyz=(50, 61, 54),
            output_dir="data",
            clear_output_dir=True,
            dask_client=c,
        )

        c.close()
