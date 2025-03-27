import numpy as np
import pickle
from process_aero_data import *
from visualize_wing import *
import sys
import argparse

def main():
    # Parse COmmand Line Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--data_type', type=str, required=True, help='aero data type')
    parser.add_argument('--num_spline_points', type=int, required=True, help='number of sample points along spline')
    args = parser.parse_args()

    with open(args.input_file, 'rb') as handle:
        dataframes = pickle.load(handle)

    ndata = 0

    for case_key, case_dict in dataframes.items():
        for slice_key, df in case_dict.items():
            ndata += 1

    # Set spline sample points, dimensionality, number of field variables, and number of cross-sections
    ns = args.num_spline_points # Spline samples per top and bottom
    if args.data_type == '2D':
        ndim = 3 # Keep trivial z-direction data
        nvar = 4 # Keep trivial z-direction data
        nsection = 1
    elif args.data_type == '3D':
        ndim = 3
        nvar = 4
        nsection = 9
    else:
        sys.exit('Unsupported Data Type (Must be 2D airfoil or 3D wing)')

    idata = 0
    index_lookup = [None] * ndata

    geom_array = np.zeros((ndata, nsection, 2*ns, ndim))
    field_array = np.zeros((ndata, nsection, 2*ns, nvar))

    idata = 0
    index_lookup = [None] * ndata

    # Loop over cases, slices, and cross-sections
    for i, (case_key, case_dict) in enumerate(dataframes.items()):
        for j, (slice_key, df_list) in enumerate(case_dict.items()):
            print(case_key, slice_key)
            for i, df in enumerate(df_list):
                # Extract geometry and field data
                x, y, z, cp, vx, vy, vz = df['CoordinateX'].values, df['CoordinateY'].values, df['CoordinateZ'].values, \
                                        df['CP'].values, df['VX'].values, df['VY'].values,  df['VX'].values

                # Shift indexing so that index 0 corredponds to most leading edge point
                xy = np.column_stack((x, y))
                idx = np.argmin(np.sqrt(np.sum((xy-np.array([[0., 0.]]))**2, axis=1)))

                x, y , z = np.concatenate((x[idx:], x[:idx])), np.concatenate((y[idx:], y[:idx])), np.concatenate((z[idx:], z[:idx]))
                cp, vx, vy , vz = np.concatenate((cp[idx:], cp[:idx])), np.concatenate((vx[idx:], vx[:idx])),\
                                np.concatenate((vy[idx:], vy[:idx])), np.concatenate((vz[idx:], vz[:idx]))
                
                '''
                # Fit raw B-Spline to identify trailing edge tip location
                #xspline, yspline, s, xtip, ytip, idx_tip = fit_raw_Bspline(x, y)

                # Rotate and scale raw data so leading edge is at (0,0) and xtip, ytip is at (1,0)
                #x, y, xtip, ytip = rotate_scale(x, y, xtip, ytip)
                '''

                # set rear tip index to closest point to (1., 0.)
                xy = np.column_stack((x, y))
                idx_tip = np.argmin(np.sqrt(np.sum((xy-np.array([[1., 0.]]))**2, axis=1)))

                # Combine field varible data
                fvars = np.concatenate((cp.reshape(-1,1), vx.reshape(-1,1), vy.reshape(-1,1), vz.reshape(-1,1)), axis=1)

                # Combine geometric data
                xvars = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis=1)

                # Separate data (geometric and fluid) intp top and bottom components
                xb, xt, vb, vt, idx_tip = separate_top_bottom(xvars, fvars, idx_tip)

                # Fix any duplicate points
                xb[:,:2] = fix_duplicate_points(xb[:,:2])
                xt[:,:2] = fix_duplicate_points(xt[:,:2])

                # Fit B-Spline to bottom processed coordinate/field data
                xsb, vsb, sb = spline_field_fit(xb[:,:2], vb, ns)

                # Fit B-Spline to top processed coordinate/field data
                xst, vst, st = spline_field_fit(xt[:,:2], vt, ns)

                # Recombine spline fit with z coordinates
                xsb = np.concatenate((xsb, np.ones((xsb.shape[0], 1))*z[0]), axis=1)
                xst = np.concatenate((xst, np.ones((xst.shape[0], 1))*z[0]), axis=1)

                # Combine top and bottom spline fits into unified geometric/field representation
                xs = np.concatenate((xsb, xst), axis=0)
                vs = np.concatenate((vsb, vst), axis=0)

                # Store spline-based geometric representation in global data array
                geom_array[idata, i, ...] = xs
                field_array[idata, i, ...] = vs

            # increment data index
            index_lookup[idata] = case_key+'_'+slice_key
            idata += 1

    plot_3D_Wing_Splines(geom_array, index_lookup, args.data_type)
    plot_3D_Field_Splines(geom_array, field_array, index_lookup, args.data_type)

    with open(args.output_dir+'index_lookup.txt', 'wb') as fp:
        pickle.dump(index_lookup, fp)

    np.save(args.output_dir+'geometry_array.npy', geom_array)
    np.save(args.output_dir+'field_array.npy', field_array)

    print(geom_array.shape)
    print(field_array.shape)

if __name__ == '__main__':
    main()


            
