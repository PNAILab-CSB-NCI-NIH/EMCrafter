import numpy as np
from scipy.ndimage import gaussian_filter

def circular_mask(shape, radius=80, soft=10, center=None):
    """
    Creates a circular mask centered at 'center' (default: center of image)
    with a hard edge at 'radius'. If 'soft' is specified, the mask is softened
    over a distance of 'soft' pixels from the edge using a gaussian filter.

    Returns two masks: an inner mask, and an outer mask.

    Parameters
    ----------
    shape : tuple of int
        Shape of the mask to be created.
    radius : int, optional
        Radius of the mask. Default is 80.
    soft : int, optional
        Distance over which the soft edge is applied. Default is 10.
    center : tuple of int, optional
        Center of the mask. Default is the center of the image.

    Returns
    -------
    inner_mask, outer_mask : 2D arrays
        The inner and outer masks, respectively.
    """
    h, w = shape
    if center is None:
        center = (h / 2, w / 2)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = np.ones((h, w))
    mask[dist >= radius] = 0.0

    soft_mask = gaussian_filter(mask.astype(float), sigma=soft)
    mask[dist >= radius + soft] = 0.0
    mask[dist < radius - soft] = 1.0

    inner_mask = soft_mask.copy()
    outer_mask = np.abs(soft_mask -1.)

    return inner_mask, outer_mask

def time_format(start_time, end_time):
    """
    Format a time difference from start_time to end_time.

    Parameters
    ----------
    start_time : float
        The start time of the time difference.
    end_time : float
        The end time of the time difference.

    Returns
    -------
    str
        A string representation of the time difference
    """
    delta = end_time - start_time
    if delta < 60:
        return f"{format(delta, '.2f')}s"
    elif delta < 3600:
        return f"{int(delta/60)}m{int(delta%60)}s"
    else:
        return f"{int(delta/3600)}h{int((delta%3600)/60)}m{int((delta%3600)%60)}s"
