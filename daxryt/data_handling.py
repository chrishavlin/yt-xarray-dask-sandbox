import dask
import xarray as xr
import numpy as np
import os
import shutil
from dask.distributed import Client

# build a parallel netcdf dataset. each .nc file is a different part of the grid.
# grid is uniform (though varies in x, y, z), with no refinement (yt level 0 only)
# uses dask to build sample data on each grid, write to nc
# note, might be able to simplify this with xrarray save_mfdataset, but
# its nice to have a general solution too...

def _build_write_chunk(chunk_bbox, chunk_sizes, outfile):

    xcoords = np.linspace(chunk_bbox[0][0], chunk_bbox[0][1], chunk_sizes[0])
    ycoords = np.linspace(chunk_bbox[1][0], chunk_bbox[1][1], chunk_sizes[1])
    zcoords = np.linspace(chunk_bbox[2][0], chunk_bbox[2][1], chunk_sizes[2])

    x_vals, y_vals, z_vals = np.meshgrid(xcoords, ycoords, zcoords, indexing="ij")

    center = [3., 3., 3.]
    dist = np.sqrt((x_vals - center[0])**2 +
                   (y_vals - center[1])**2 +
                   (z_vals - center[2])**2)
    gauss = np.exp(-(dist/1.)**2) + np.random.rand(*chunk_sizes)*0.1
    coord_tup = ("x", "y", "z")
    x1 = xr.Dataset(
        {
            "temperature": (coord_tup, 20 * np.random.rand(*chunk_sizes)),
            "gauss": (coord_tup, gauss),
            "xvals": (coord_tup, x_vals,),
            "yvals": (coord_tup, y_vals,),
            "zvals": (coord_tup, z_vals,),

        },
        coords={"x": xcoords, "y": ycoords, "z": zcoords},
    )
    x1.to_netcdf(path=outfile)

def build_test_data(n_grids_xyz=None,
                    grid_wids_xyz=None, 
                    cells_per_grid_xyz=None, 
                    output_dir=None, 
                    clear_output_dir=False, 
                    dask_client: Client = None):

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
    fwrites = []  # future writes
    filelist = []
    for igrid_x in range(n_grids_xyz[0]):
        for igrid_y in range(n_grids_xyz[1]):
            for igrid_z in range(n_grids_xyz[2]):
                igrid = np.array([igrid_x, igrid_y, igrid_z])
                left_edge = grid_wids_xyz * igrid + hwidth
                right_edge = left_edge + grid_wids_xyz - cell_sizes
                this_grid = np.column_stack([left_edge, right_edge])

                new_file = os.path.join(output_dir,
                                        f"chunk_{igrid_index}.nc")
                filelist.append(new_file)
                if dask_client is None:
                    _build_write_chunk(this_grid, cells_per_grid_xyz, new_file)
                else:
                    fwrites.append(dask_client.submit(_build_write_chunk,
                                       this_grid,
                                       cells_per_grid_xyz,
                                        new_file
                                       ))
                igrid_index += 1

    if len(fwrites) > 0:
        _ = dask_client.gather(fwrites)
        # make sure all the files are there
        missing_files = [fi for fi in filelist if os.path.isfile(fi) is False]
        if len(missing_files) > 0:
            raise RuntimeError(f"Some files were not written: {missing_files}")

    print("Finished data construction.")


