import dask
import xarray as xr
import numpy as np
import os
import shutil

# !!!!! note, NEED TO IMPORT the following TO AVOID SEGFAULT!!!! 
# I think cause if we dont, the different dask processes 
# will import. and import order when both h5py and netCDF4 
# are around can result in weird bugs. Not entirely sure if this is 
# user error or a bug. May also be version dependent as I 
# dont see it on my desktop, only in a fresh environment
# on my laptop.
import h5py
import netCDF4

# build a parallel netcdf dataset. each .nc file is a different part of the grid.
# grid is uniform (though varies in x, y, z), with no refinement (yt level 0 only)
# uses dask to build sample data on each grid, write to nc

def _build_write_chunk(chunk_bbox, chunk_sizes, outfile):

    xcoords = np.linspace(chunk_bbox[0][0], chunk_bbox[0][1], chunk_sizes[0])
    ycoords = np.linspace(chunk_bbox[1][0], chunk_bbox[1][1], chunk_sizes[1])
    zcoords = np.linspace(chunk_bbox[2][0], chunk_bbox[2][1], chunk_sizes[2])

    x_vals, y_vals, z_vals = np.meshgrid(xcoords, ycoords, zcoords, indexing="ij")
    x1 = xr.Dataset(
        {
            "temperature": (
            ("x", "y", "z"), 20 * np.random.rand(*chunk_sizes)),
            "xvals": (("x", "y", "z"), x_vals,),
            "yvals": (("x", "y", "z"), y_vals,),
            "zvals": (("x", "y", "z"), z_vals,),

        },
        coords={"x": xcoords, "y": ycoords, "z": zcoords},
    )
    x1.to_netcdf(path=outfile)

def build_test_data(n_grids_xyz=None,
                    grid_wids_xyz=None, 
                    cells_per_grid_xyz=None, 
                    output_dir=None, 
                    clear_output_dir=False, 
                    use_dask = False):

    if n_grids_xyz is None:
        n_grids_xyz = (5, 5, 5)

    if grid_wids_xyz is None:
        grid_wids_xyz = np.array([1., 1., 1.])
    else:
        grid_wids_xyz = np.asarray(grid_wids_xyz)

    if cells_per_grid_xyz is None:
        cells_per_grid_xyz = np.array([65, 70, 80])
    else:
        cells_per_grid_xyz = np.asarray(cells_per_grid_xyz)

    if os.path.isdir(output_dir):
        if clear_output_dir:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)

    cell_sizes = grid_wids_xyz / cells_per_grid_xyz
    hwidth = cell_sizes / 2

    igrid_index = 0
    write_grids = []
    for igrid_x in range(n_grids_xyz[0]):
        for igrid_y in range(n_grids_xyz[1]):
            for igrid_z in range(n_grids_xyz[2]):
                igrid = np.array([igrid_x, igrid_y, igrid_z])
                left_edge = grid_wids_xyz * igrid + hwidth
                right_edge = left_edge + grid_wids_xyz - cell_sizes
                this_grid = np.column_stack([left_edge, right_edge])

#                if igrid_index == 0:
#                    # ug, actuall, need this for a jupyter notebook. wtfffffff.
#                    new_fi = os.path.join(output_dir, "_temp_.nc")
#                    _build_write_chunk(this_grid, np.array([2,2,2]), new_fi)

                new_file = os.path.join(output_dir,
                                        f"chunk_{igrid_index}.nc")
                if use_dask is False:
                    _build_write_chunk(this_grid, cells_per_grid_xyz, new_file)
                else:
                    write_grids.append(
                        dask.delayed(_build_write_chunk)(this_grid, cells_per_grid_xyz,
                                                    new_file))
                igrid_index += 1


    if use_dask:
        results = dask.compute(*write_grids)


