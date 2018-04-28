import numpy as np

# REMEMBER THIS IS IN METERS <3

RADIUS_OF_EARTH = 6371e3
DEGREES = 360

def to_global_system(plane_latlong, obj_xy, plane_orientation):
    '''Convert plane latlong, plane_orientation, and object_xy to latlong.
    
    plane_orientation is relative to north

    NOTE: This function assumes the curvature is negligible!
    '''
    obj_latlong = to_latlong(obj_xy)
    world_rm = rot_mat(-to_rad(plane_orientation))

    return plane_latlong + world_rm.dot(obj_latlong)
    


def to_xy(latlong):
    ''' Converts latlong to xy.
    
    NOTE: This function assumes the curvature is negligible!
    '''
    return 2*np.pi*RADIUS_OF_EARTH*origin_latlong/DEGREES

def to_latlong(xy):
    ''' Converts latlong to xy.
    
    NOTE: This function assumes the curvature is negligible!
    '''
    return DEGREES*xy/(2*np.pi*RADIUS_OF_EARTH)

def to_rad(theta):
    return 2*np.pi*theta/DEGREES

def rot_mat(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])