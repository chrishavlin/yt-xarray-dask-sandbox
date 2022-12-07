import dask 
import xarray as xr 
import numpy as np 

# build a parallel netcdf dataset. each .nc file is a different part of the grid. 
# grid is uniform (though varies in x, y, z), with no refinement (yt level 0 only)
# uses dask to build sample data on each grid, write to nc

def build_write_chunk(chunk_bbox, chunk_sizes, chunk_linear_index: int):
    xcoords = np.linspace(chunk_bbox[0][0], chunk_bbox[0][1], chunk_sizes[0])
    ycoords = np.linspace(chunk_bbox[1][0], chunk_bbox[1][1], chunk_sizes[1])
    zcoords = np.linspace(chunk_bbox[2][0], chunk_bbox[2][1], chunk_sizes[2])

    x1 = xr.Dataset(
        {
        "temperature": (("x", "y", "z"), 20 * np.random.rand(*chunk_sizes)),
        },
        coords={"x": xcoords, "y": ycoords, "z": zcoords},
    )
    x1.to_netcdf(path=f"data/chunk_{chunk_linear_index}.nc")


n_grids = 100
grid_wids = np.array([1., 1., 1.])
cells_per_grid = np.array([65, 70, 80])
cell_sizes = grid_wids / cells_per_grid 
hwidth = cell_sizes / 2


write_grids = []
for igrid in range(n_grids):
    left_edge = grid_wids * igrid + hwidth 
    right_edge = left_edge + grid_wids - cell_sizes
    this_grid = np.column_stack([left_edge, right_edge])
    #build_write_chunk(this_grid, cells_per_grid, igrid)
    write_grids.append(dask.delayed(build_write_chunk)(this_grid, cells_per_grid, igrid))

dask.compute(*write_grids)    
