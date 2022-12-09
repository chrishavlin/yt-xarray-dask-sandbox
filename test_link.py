from daxryt import data_handling, chunk_inspector
import numpy as np
from dask.distributed import Client
import xarray as xr
import dask 
import os
import yt

if __name__ == "__main__":
    dask.config.set({"distributed.scheduler.worker-saturation": "inf"})
    c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)
    ds_xr = xr.open_mfdataset("data/*.nc")

    def _read_data(handle):
        def _reader(grid, field_name):
            ftype, fname = field_name
            si = grid.get_global_startindex()
            ei = si + grid.ActiveDimensions
            var = getattr(handle, fname)  # variables are top-level attrs of xr datasets        
            data = var[si[0] : ei[0], si[1] : ei[1], si[2] : ei[2]]
            return data.values
    
        return _reader
    
    
    reader = _read_data(ds_xr)
    
    fields = list(ds_xr.data_vars)
    global_shape = ds_xr.data_vars[fields[0]].shape
    global_left = np.array([ds_xr.coords[cn].min() for cn in ds_xr.dims.keys()])
    global_right = np.array([ds_xr.coords[cn].max() for cn in ds_xr.dims.keys()])
    bbox = np.column_stack([global_left, global_right])
    
    
    # decompose the grid for using the xarray chunking 
    chunk_info = chunk_inspector.get_chunk_info(ds_xr)
    chunk_info = chunk_info.rename(columns={'le': 'left_edge', 're': 'right_edge', 'size': 'dimensions'}) 
    grid_data = chunk_info[['left_edge', 'right_edge', 'dimensions', 'si']].to_dict('records')
    starting_indices = []
    n_grid_chunks = len(grid_data)  # (nprocs in load_uniform_grid)
    
    # add on the reader to each grid
    for gid in range(n_grid_chunks):
        grid_data[gid]['level'] = 0
        for field in fields:
            grid_data[gid][field] = reader
        starting_indices.append(grid_data[gid].pop('si'))
            
    ds = yt.load_amr_grids(grid_data, global_shape, bbox=bbox, geometry= ('cartesian', tuple(ds_xr.dims.keys())))
    ad = ds.all_data()
    # this is probably erroring due to the yt get global start index
    # start_index = (self.LeftEdge - self.ds.domain_left_edge) / self.dds
    # self.start_index = np.rint(start_index).astype("int64").ravel()
    # rounding error leading to si/ei that aren't quite right
    # gs = ad[("stream", "gauss")]

    for gid, g in enumerate(ds.index.grids):
        g.start_index = starting_indices[gid]
    gs = ad[("stream", "gauss")]
    print(gs.min())
