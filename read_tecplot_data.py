import numpy as np
import pandas as pd
from reorder import reorder_coords

def read_2d_slice_data(fname):
    # Read in 2D data airfoil data from tecplot formatted file named in fname
    file_path = fname
    with open(file_path, 'r') as f:
        s = f.readlines()
    
    # Extract 2D surface data
    for i, line in enumerate(s):
        if line.strip().startswith('DATAPACKING=POINT'):
            nnod = int(s[i-1].split('=')[1].split()[0].split(',')[0])
            nelm = int(s[i-1].split('=')[2].split()[0].split(',')[0])
    
            d2_dat = s[i+1:i+1+nnod]
            
            d2_dat = np.genfromtxt(d2_dat, dtype=float)
    
            break
    
    d2_df = pd.DataFrame(d2_dat[:,[0,1,2,6,7,8,9]], columns=['CoordinateX', 'CoordinateY', 'CoordinateZ', 'VX', 'VY', 'VZ', 'CP'])
    
    connectivity = np.genfromtxt(s[i+1+nnod:], dtype=float)
    df_conn = pd.DataFrame(connectivity, columns=['NodeC1', 'NodeC2'])
    
    result = pd.concat([d2_df, df_conn], axis=1)
    d2_df = result.reindex(d2_df.index)
    
    index_list = list(reorder_coords(d2_df, return_indices=True))
    d2_df = d2_df.loc[index_list]

    return d2_df

def read_3d_slice_data(fname):
    '''
    Take in file containing 3D slice data and return dataframes of the data for each slice as a list
    '''
    with open(fname, 'r') as f:
        s = f.readlines()
        
    # Find the lines where the zone data starts and ends
    start = []
    nzones=0

    nnods = []
    nelms = []
    data = []
    zcoords = []

    for i, line in enumerate(s):
        if line.strip().startswith('DATAPACKING=POINT'):
            start.append(i+1)
            nnod = int(s[i-1].split('=')[1].split()[0])
            nelm = int(s[i-1].split('=')[2].split()[0].split(',')[0])
            nnods.append(nnod)
            nelms.append(nelm)
            nzones += 1

            # Get zcoordinates
            z = float(s[i-2].split('=')[-1].split()[0][:-1])
            zcoords.append(z)

        else:
            continue

    # Sort cross sections according to z-coordinate
    sort_idx = np.argsort(zcoords).tolist()
    start   = [start[i] for i in sort_idx]
    nnods   = [nnods[i] for i in sort_idx]
    nelms   = [nelms[i] for i in sort_idx]
    zcoords = [zcoords[i] for i in sort_idx]

    # Read slice data in span-wise direction order extracting cross-section corrdinates and field variables
    for ind, nnod, nelm, zcoord in zip(start, nnods, nelms, zcoords):
        slice_dat = np.genfromtxt(s[ind:ind+nnod], dtype=float)
        if np.linalg.norm(slice_dat[:,2]-slice_dat[0,2])>1e-3:
            continue
        else:
            df = pd.DataFrame(slice_dat[:,[0,1,2,6,7,8,9]], columns=['CoordinateX', 'CoordinateY', 'CoordinateZ', 'VX', 'VY', 'VZ', 'CP'])
            connectivity = np.genfromtxt(s[ind+nnod:ind+nnod+nelm], dtype=float)
            df_conn = pd.DataFrame(connectivity, columns=['NodeC1', 'NodeC2'])
            result = pd.concat([df, df_conn], axis=1)
            df = result.reindex(df.index)
            index_list = list(reorder_coords(df, return_indices=True))
            df = df.loc[index_list]

            df['zcoord'] = zcoord
            
            data.append(df)

    return data