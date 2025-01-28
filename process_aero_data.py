import numpy as np
import glob
from reorder import reorder_coords
from scipy.interpolate import splprep, splev, BSpline, spalde, RBFInterpolator
import sys
from scipy.spatial.distance import pdist, squareform

def closest_point(x, y, xtarget, ytarget):
    '''
    Identify closest (Euclidean) point in x, y coordinate list to a single point (xtarget, ytarget)
    '''
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy_target = np.ones(xy.shape) @ np.diag([xtarget, ytarget])

    d = np.sqrt(np.sum((xy - xy_target)**2, axis=1))

    return np.argmin(d)

def two_closest_points(x, y, xtarget, ytarget):
    '''
    Identify two closest (Euclidean) point in x, y coordinate list to a single point (xtarget, ytarget)
    '''
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy_target = np.ones(xy.shape) @ np.diag([xtarget, ytarget])
    
    d = np.sqrt(np.sum((xy - xy_target)**2, axis=1))

    idx = np.argsort(d)[:2]

    xy1 = xy[idx[0],:]
    xy2 = xy[idx[1],:]

    d1 = d[idx[0]]
    d2 = d[idx[1]]

    return xy1, xy2, d1, d2, idx[0], idx[1]

def fit_raw_Bspline(x, y):
    ''' Fit parameterized B-Spline to curve characterized by list of points (x,y) and find point of maximum curvature '''
    tck, u = splprep([x, y], s=0, k=3)

    # Assumption: s=0.7 encompasses point of maximum curvature
    
    u = np.linspace(0., 1.0, 1000)
    new_points = splev(u, tck, der=0)
    xspline, yspline = new_points[0], new_points[1]

    '''
    d0, d1 = splev(u, tck, der=3)
    idxs = np.argsort(np.abs(d1))[::-1]
    imax = idxs[0]
    '''
    # Take closest point to (1.0, 0.0) as tip. Note this assumes certain scaling features/etc of raw geometry
    imax = closest_point(xspline, yspline, 1.0, 0.0)
    
    xtip, ytip = xspline[imax], yspline[imax]

    # Fit entire wing surface
    s = np.linspace(0., 1.0, 1000)
    new_points = splev(s, tck)
    xspline, yspline = new_points[0], new_points[1]

    # Get coordinate index closest to B-spline tip
    imin = closest_point(x, y, xtip, ytip)
    xtip, ytip = x[imin], y[imin]
    idx_tip = imin

    return xspline, yspline, s, xtip, ytip, idx_tip

def rotate_scale(x, y, xtip, ytip):
    '''
    Rotate and scale airfoil coordinate points such that leading edge is located at (0,0) and
    trailing tip is located at (1,0)
    '''
     # Rotate
    t = np.arctan(ytip/xtip)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy = xy @ R
    x, y = xy[:,0], xy[:,1]
    xytip = np.array([[xtip, ytip]]) @ R
    xtip, ytip = xytip[0,0], xytip[0,1]

    # Scale volume equally along x and y axes
    V = np.array([[1/xtip, 0],[0, 1/xtip]])
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    xy = xy @ V
    x, y = xy[:,0], xy[:,1]
    xytip = np.array([[xtip, ytip]]) @ V
    xtip, ytip = xytip[0,0], xytip[0,1]

    return x, y, xtip, ytip

def fit_processed_Bspline(x, y, ns=int(1e3)):
    ''' 
    Fit B-Spline to processed data defined by list of x, y coordinate points. 
    Evenly sample ns points along parameterized spline curve
    '''
    tck, _ = splprep([x, y], s=0, k=3)

    # Fit entire wing surface
    s = np.linspace(0., 1.0, ns)
    new_points = splev(s, tck)
    xspline, yspline = new_points[0], new_points[1]

    return xspline, yspline, s, tck

def separate_top_bottom(xvars, fvars, idx_tip):
    '''
    Separate data (geometric and/or fluid) into top and bottom components
    '''

    xb, vb = xvars[:idx_tip,:], fvars[:idx_tip,:]
    xt, vt = xvars[idx_tip:,:], fvars[idx_tip:,:]

    idx_tip -= 1

    return xb, xt, vb, vt, idx_tip

def fix_duplicate_points(xy):
    # Check for duplicate points in raw xy data and replace duplicate points by avergaing of NN points
    distances = squareform(pdist(xy))
    np.fill_diagonal(distances, np.inf)
    close_points = np.where(distances < 1e-10)
    
    # To-Do: Handle case with multiple duplicates
    if len(close_points[0]) > 0:
        #print(f"Found {len(close_points[0])//2} pairs of nearly duplicate points")
        idx0, idx1 = close_points[0]
        if (idx1==xy.shape[0]-1):
            xy[idx1,:] = np.mean(xy[[idx1-1,idx1],:],axis=0)
        else:
            xy[idx1,:] = np.mean(xy[[idx1-1,idx1+1],:],axis=0)
        #print(xy[idx0,:])
        #print(xy[idx1-1:idx1+2,:])
        #print(np.mean(xy[[idx1-1,idx1+1],:],axis=0))
    
    return xy

def spline_field_fit(xy, fvar, ns=int(1e3)):
    '''
    Fit field variables along airfoil surface (generally one would call this either on the bottom or the top surface)
    '''

    # Fit B-Spline to bottom processed coordinate data
    xs, ys, s, tck = fit_processed_Bspline(xy[:,0], xy[:,1], ns)
    xys = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)

    # Use RBF interpolation to interpolate field data to spline points
    rbf = RBFInterpolator(xy, fvar, kernel='thin_plate_spline')
    vs  = rbf(xys)

    '''
    # Use RBF interpolation to interpolate field data to spline points
    try:
        rbf = RBFInterpolator(xy, fvar, kernel='thin_plate_spline')
        vs  = rbf(xys)
    except (np.linalg.LinAlgError, ValueError, Exception) as e:
        # Find very close points (excluding self-comparisons)
        distances = squareform(pdist(xy))
        np.fill_diagonal(distances, np.inf)
        close_points = np.where(distances < 1e-10)
        
        if len(close_points[0]) > 0:
            print(f"Found {len(close_points[0])//2} pairs of nearly duplicate points")
            idx0, idx1 = close_points[0]
            print(xy[idx0,:])
            print(xy[idx1-1:idx1+2,:])
            print(np.mean(xy[[idx1-1,idx1+1],:],axis=0))
    
        rbf = RBFInterpolator(xy, fvar, kernel='linear')
        vs  = rbf(xys)
        #vs  = np.zeros((ns, fvar.shape[1]))
    '''

    # Check conservation of field quantities
    interpolation_validation_dict = validate_interpolation(xy, np.abs(fvar[:,0]), xys, np.abs(vs[:,0]))
    if 100*interpolation_validation_dict['relative_conservation_error']>5:
        print(100*interpolation_validation_dict['relative_conservation_error'])
        print(np.linalg.norm(vs))

    return xys, vs, s

def parameterize_airfoil_curve(df, ns=100):
    '''
    Stitch together multiple fundamental functions to parameterize airfoils using B-splines.
    Paramterize entire airfoil, as well as top and bottom separately
    '''
    # Fit raw data to B-Splines
    x, y = df['CoordinateX'].values, df['CoordinateY'].values
    xspline, yspline, s, xtip, ytip, idx_tip = fit_raw_Bspline(df)
    x, y, xtip, ytip = rotate_scale(x, y, xtip, ytip)

    # Fit B-Spline to processed coordinate data
    xspline, yspline, s, tck = fit_processed_Bspline(x, y)
    xyspline = np.concatenate((xspline.reshape(-1,1), yspline.reshape(-1,1)), axis=1)

    # Get spline interpolation point falling closest to trailing tip of original data
    i = closest_point(xspline, yspline, xtip, ytip)
    midpoint = splev(np.array(s[i]), tck)
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    
    # sample spline along top and bottom
    sbottom = np.linspace(0, s[i], ns)
    stop = np.linspace(s[i], 1., ns)
    xspline_b, yspline_b = splev(sbottom, tck)
    xspline_t, yspline_t = splev(stop, tck)
    
    xyspline_b = np.concatenate((xspline_b.reshape(-1,1), yspline_b.reshape(-1,1)), axis=1)
    xyspline_t = np.concatenate((xspline_t.reshape(-1,1), yspline_t.reshape(-1,1)), axis=1)

    return xyspline, xyspline_b, xyspline_t, s, sbottom, stop, midpoint, xy, xtip, ytip

def calculate_geom_distance(df1, df2):
    '''
    Calcualte distance between two airfoil geometries by summing the pointwise Euclidesan distance between 
    the (x,y) locations at a common parameterization length s
    '''
    _, xys_b_1, xys_t_1, _, _, _, _, _, _, _ = parameterize_airfoil_curve(df1)
    _, xys_b_2, xys_t_2, _, _, _, _, _, _, _ = parameterize_airfoil_curve(df2)
    
    dist_b = np.sum(np.sqrt(np.sum((xys_b_1 - xys_b_2)**2, axis=1)))
    dist_t = np.sum(np.sqrt(np.sum((xys_t_1 - xys_t_2)**2, axis=1)))
    dist = dist_b + dist_t

    return dist

def validate_interpolation(original_points, field_values, spline_points, interpolated_values):
    """
    Validate interpolation quality for field quantities along a 1D curve in 2D space
    
    Parameters:
    -----------
    original_points : ndarray, shape (n_points, 2)
        Original points along the curve (x, y coordinates)
    field_values : ndarray, shape (n_points,)
        Field values at original points
    spline_points : ndarray, shape (n_spline_points, 2)
        B-spline points where values were interpolated
    interpolated_values : ndarray, shape (n_spline_points,)
        Interpolated field values at spline points
        
    Returns:
    --------
    dict : Dictionary containing validation metrics
    """
    def compute_arc_length_elements(points):
        """Compute differential arc length elements between consecutive points"""
        diff_vectors = np.diff(points, axis=0)  # [dx, dy] for each segment
        return np.sqrt(np.sum(diff_vectors**2, axis=1))  # dl = sqrt(dx^2 + dy^2)
    
    # Compute arc length elements for both point sets
    original_dl = compute_arc_length_elements(original_points)
    spline_dl = compute_arc_length_elements(spline_points)
    
    # Compute integral along the curve using arc length elements
    # We use average of adjacent values multiplied by segment length
    original_integral = np.sum(
        0.5 * (field_values[1:] + field_values[:-1]) * original_dl
    )
    interpolated_integral = np.sum(
        0.5 * (interpolated_values[1:] + interpolated_values[:-1]) * spline_dl
    )
    
    # Compute conservation error
    conservation_error = np.abs(original_integral - interpolated_integral)
    relative_error = conservation_error / np.abs(original_integral)
    
    # Check smoothness along the curve
    # Compute gradients with respect to arc length
    cumulative_length_orig = np.concatenate([[0], np.cumsum(original_dl)])
    cumulative_length_spline = np.concatenate([[0], np.cumsum(spline_dl)])
    
    # Gradient computation using arc length parameterization
    original_gradients = np.gradient(field_values, cumulative_length_orig)
    interpolated_gradients = np.gradient(interpolated_values, cumulative_length_spline)
    
    # Compute smoothness metrics
    smoothness_metric = {
        'original_gradient_variance': np.var(original_gradients),
        'interpolated_gradient_variance': np.var(interpolated_gradients)
    }
    
    # Check local conservation properties
    # Divide curve into segments and check conservation in each
    n_segments = 10
    local_errors = []
    
    orig_total_length = cumulative_length_orig[-1]
    spline_total_length = cumulative_length_spline[-1]
    
    for i in range(n_segments):
        # Define segment bounds in terms of normalized arc length
        lower = i / n_segments
        upper = (i + 1) / n_segments
        
        # Find points within this segment for both datasets
        orig_mask = ((cumulative_length_orig / orig_total_length) >= lower) & \
                   ((cumulative_length_orig / orig_total_length) < upper)
        spline_mask = ((cumulative_length_spline / spline_total_length) >= lower) & \
                     ((cumulative_length_spline / spline_total_length) < upper)
        
        # Compute local integrals
        if np.sum(orig_mask[:-1]) > 0 and np.sum(spline_mask[:-1]) > 0:
            local_orig_integral = np.sum(
                0.5 * (field_values[1:][orig_mask[:-1]] + 
                      field_values[:-1][orig_mask[:-1]]) * 
                original_dl[orig_mask[:-1]]
            )
            local_interp_integral = np.sum(
                0.5 * (interpolated_values[1:][spline_mask[:-1]] + 
                      interpolated_values[:-1][spline_mask[:-1]]) * 
                spline_dl[spline_mask[:-1]]
            )
            
            local_errors.append(np.abs(local_orig_integral - local_interp_integral) / 
                              np.abs(local_orig_integral))
    
    return {
        'global_conservation_error': conservation_error,
        'relative_conservation_error': relative_error,
        'smoothness_metrics': smoothness_metric,
        'local_conservation_errors': local_errors,
        'max_local_error': max(local_errors) if local_errors else None,
        'mean_local_error': np.mean(local_errors) if local_errors else None
    }