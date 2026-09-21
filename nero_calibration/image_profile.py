"""RGB acquisition and crop coordinates shared by calibration and localization."""
import copy


def profile_options(profile=None):
    p = profile or {}
    resolution = p.get('color_resolution', [640, 480])
    fps = p.get('fps', 15)
    crop = p.get('crop_xywh')
    if resolution not in ([640, 480], [1280, 720]) or any(type(x) is not int for x in resolution):
        raise ValueError('Unsupported color_resolution')
    if type(fps) is not int or fps not in (6, 15, 30):
        raise ValueError('Invalid camera fps')
    if crop is not None:
        if (not isinstance(crop, (list, tuple)) or len(crop) != 4
                or any(type(x) is not int for x in crop)):
            raise ValueError('crop_xywh requires four integers')
        x,y,w,h=crop
        if min(x,y)<0 or min(w,h)<=0 or x+w>resolution[0] or y+h>resolution[1]:
            raise ValueError('Crop outside RGB image')
    return resolution, fps, crop


def crop_intrinsics(intrinsics, crop):
    result=copy.deepcopy(intrinsics)
    if crop is not None:
        x,y,w,h=crop
        result.update(width=w,height=h,cx=result['cx']-x,cy=result['cy']-y)
    return result


def crop_image(image, crop):
    if crop is None:
        return image.copy()
    x,y,w,h=crop
    if x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('Crop outside image')
    return image[y:y+h,x:x+w].copy()
