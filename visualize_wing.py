import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_3D_wing_sections(data, fname):
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D subplot on the left
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D subplot on the right
    ax2 = fig.add_subplot(122)


    # Calculate the color for each dataset
    cmap = plt.get_cmap('RdBu')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]

    for color, dat in zip(colors, data):
        x = dat['CoordinateX'].values
        y = dat['CoordinateY'].values
        vx = dat['VX'].values
        vy = dat['VY'].values
        p  = dat['CP'].values
        z = dat['zcoord'].values

        # 3D plot
        ax1.scatter(z, x, y, color=color, s=10, label=f'{z[0]:.2f}')
        ax1.plot(z, x, y, color=color, )
        
        # 2D projection plot (x-y plane)
        ax2.scatter(x, y, color='k', s=10)
        ax2.plot(x, y, color='k')


    # 3D plot settings
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.view_init(azim=45, elev=30)
    #ax1.set_title('3D View', fontsize=14)
    
    # 2D plot settings
    ax2.grid()
    #ax2.set_aspect('equal')
    ax2.set_title('X-Y Projection', fontsize=14)
    
    # Add labels
    ax1.set_xlabel('Z')
    ax1.set_ylabel('X')
    ax1.set_zlabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    plt.savefig(fname)
    plt.close()

    return None

def plot_2D_airfoil_section(data, fname):
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 10))
    
    # 2D subplot on the right
    ax1 = fig.add_subplot(111)

    data = data[0]
    x = data['CoordinateX'].values
    y = data['CoordinateY'].values
    vx = data['VX'].values
    vy = data['VY'].values
    p  = data['CP'].values

    # 3D plot
    ax1.scatter(x, y, color='k', s=10)
    ax1.plot(x, y, color='k')


    # 3D plot settings
    ax1.grid()
    ax1.set_aspect('equal')
    
    # Add labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    plt.savefig(fname)
    plt.close()

    return None

def plot_3D_Wing_Splines(geom_array, index_lookup, data_type):
    for geo_data, index in zip(geom_array, index_lookup):

        if index is None:
            break
        
        # Visualize spline geometry fit
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Calculate the color for each dataset
        cmap = plt.get_cmap('RdBu')
        colors = [cmap(i) for i in np.linspace(0, 1, geo_data.shape[0])]

        for i in range(geo_data.shape[0]):
            xs = geo_data[i, ...]
            # 3D plot
            ax.plot(xs[:,2], xs[:,0], xs[:,1], color=colors[i])
            ax.scatter(xs[100,2], xs[100,0], xs[100,1], color=colors[i], s=50)
            ax.scatter(xs[0,2], xs[0,0], xs[0,1], color=colors[i], s=50)


        # 3D plot settings
        ax.grid()
        ax.set_aspect('equal')
        ax.view_init(azim=45, elev=30)

        # Add labels
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
                    
        # Save and close
        plt.savefig('images/'+data_type+'/wing_'+index+'.png')
        plt.close()

    return None

def plot_3D_Field_Splines(geom_array, field_array, index_lookup, data_type):
    for geo_data,field_data, index in zip(geom_array, field_array, index_lookup):

        if index is None:
            break
        
        # Visualize spline geometry fit
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Calculate the color for each dataset
        cmap = plt.get_cmap('RdBu')
        colors = [cmap(i) for i in np.linspace(0, 1, geo_data.shape[0])]

        for i in range(geo_data.shape[0]):
            xs = geo_data[i, ...]
            vs = field_data[i, ...]
            # 3D plot
            ax.plot(xs[:,2], xs[:,0], vs[:,0], color=colors[i])

        # 3D plot settings
        ax.grid()
        ax.set_aspect('equal')
        ax.view_init(azim=45, elev=30)

        # Add labels
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
                    
        # Save and close
        plt.savefig('images/'+data_type+'/field_'+index+'.png')
        plt.close()

    return None

'''
def plot_2D_Wing_Splines(geom_array, index_lookup):
    for geo_data, index in zip(geom_array, index_lookup):

        if index is None:
            break
        
        # Visualize spline geometry fit
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Calculate the color for each dataset
        cmap = plt.get_cmap('RdBu')
        colors = [cmap(i) for i in np.linspace(0, 1, geo_data.shape[0])]

        for i in range(geo_data.shape[0]):
            xs = geo_data[i, ...]
            # 3D plot
            ax.plot(xs[:,2], xs[:,0], xs[:,1], color=colors[i])
            ax.scatter(xs[100,2], xs[100,0], xs[100,1], color=colors[i], s=50)
            ax.scatter(xs[0,2], xs[0,0], xs[0,1], color=colors[i], s=50)


        # 3D plot settings
        ax.grid()
        ax.set_aspect('equal')
        ax.view_init(azim=45, elev=30)

        # Add labels
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
                    
        # Save and close
        plt.savefig('images/wing_'+index+'.png')
        plt.close()

    return None

def plot_2D_Field_Splines(geom_array, field_array, index_lookup):
    for geo_data,field_data, index in zip(geom_array, field_array, index_lookup):

        if index is None:
            break
        
        # Visualize spline geometry fit
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Calculate the color for each dataset
        cmap = plt.get_cmap('RdBu')
        colors = [cmap(i) for i in np.linspace(0, 1, geo_data.shape[0])]

        for i in range(geo_data.shape[0]):
            xs = geo_data[i, ...]
            vs = field_data[i, ...]
            # 3D plot
            ax.plot(xs[:,2], xs[:,0], vs[:,0], color=colors[i])

        # 3D plot settings
        ax.grid()
        ax.set_aspect('equal')
        ax.view_init(azim=45, elev=30)

        # Add labels
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
                    
        # Save and close
        plt.savefig('images/field_'+index+'.png')
        plt.close()

    return None
'''