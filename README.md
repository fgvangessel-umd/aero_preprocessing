# Usage

You can preprocess raw airfoil/wing tecplot window in two steps.

1. Gather all available data

``python gather_slice_data.py --output_file 2D_Data/2D_surface_data.pickle --data_dirs 2D_Data/cases/ --data_type 2D``

    *or*
  
``python gather_slice_data.py --output_file 3D_Data/3D_surface_data.pickle --data_dirs 3D_Data/cases/ --data_type 3D``

2. Apply processing (e.g. B-Spline fitting and RBF interpolation of field data) to all aggregated tecplot data

``python wing_field_geometry_analysis.py --input_file 2D_Data/2D_surface_data.pickle --output_dir 2D_Data/ --data_type 2D``
   
    *or*
  
``python wing_field_geometry_analysis.py --input_file 3D_Data/3D_surface_data.pickle --output_dir 3D_Data/ --data_type 3D``

