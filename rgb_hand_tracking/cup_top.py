"""Read-only RGB cup-top localization with explicit geometry or YOLO backend.

The fitted edge is an internal top-face boundary; the whole-body mask centroid
is only a diagnostic. Pixel geometry cannot yet authorize robot motion.
"""
import math

import cv2
import numpy as np


class CupTopDetector:
    """Use ``segmentor.infer(bgr)`` and its explicit model ``layout``.

    ``best.q.onnx`` has cap=0 in ``standard2_cap_ground``. The established
    COCO export has cup=41. ``green_top`` is an independent explicit backend:
    identity is the known single green dice cup in a cleared workspace, and
    valid only means internal-top-edge pixel geometry passed its checks.
    It never reports model confirmation or valid physical motion geometry.
    """

    def __init__(self, segmentor=None, candidate_classes=None, backend='yolo_confirmed'):
        if backend not in ('yolo_confirmed', 'green_top'):
            raise ValueError('Unknown cup-top backend')
        self.backend = backend
        self.segmentor = segmentor
        self.candidate_classes = ()
        if backend == 'green_top':
            return
        if segmentor is None:
            raise ValueError('A YOLO segmentor is required for cup identity confirmation')
        self.segmentor = segmentor
        layout = getattr(segmentor, 'layout', None)
        if candidate_classes is None:
            if layout == 'standard2_cap_ground':
                candidate_classes = (0,)
            elif layout == 'spacemit13_coco':
                candidate_classes = (41,)
            else:
                raise ValueError('Unknown model layout; pass explicit candidate_classes')
        self.candidate_classes = tuple(candidate_classes)

    def process(self, bgr):
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
                or bgr.ndim != 3 or bgr.shape[2] != 3):
            raise ValueError('Expected uint8 BGR image')
        result = {'valid': False, 'reason': None, 'center_px': None,
                  'ellipse_px': None, 'quality': {}, 'motion_target_valid': False,
                  'source': ('green_top_geometry' if self.backend == 'green_top'
                             else 'yolo_confirmed_rgb_internal_top_edge'),
                  'model_confirmed': False,
                  'coordinate_frame': 'image_pixels',
                  'physical_accuracy_valid': False}
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 55, 35), (90, 255, 255)) != 0
        if self.backend == 'green_top':
            candidates = self._green_geometry_candidates(green)
            result['quality']['green_geometry_candidate_count'] = len(candidates)
            if len(candidates) != 1:
                result['reason'] = ('no_round_green_geometry_candidate' if not candidates
                                    else 'multiple_green_geometry_candidates')
                return result
            body, descriptor = candidates[0]
            result['green_body'] = descriptor
            return self._finish_top_fit(bgr, body, result, combine_edge_fragments=True)
        instances = self.segmentor.infer(bgr)
        candidates = []
        for item in instances:
            mask = np.asarray(item['mask'], dtype=bool)
            if mask.shape != bgr.shape[:2]:
                raise ValueError('YOLO mask shape differs from input frame')
            count = int(mask.sum())
            fraction = float(green[mask].mean()) if count else 0.
            if (item['class_id'] in self.candidate_classes and count >= 100
                    and fraction >= .35):
                candidates.append((item, fraction))
        result['quality']['confirmed_cup_count'] = len(candidates)
        if len(candidates) != 1:
            result['reason'] = ('no_green_yolo_candidate' if not candidates
                                else 'multiple_green_yolo_candidates')
            return result
        item, fraction = candidates[0]
        result['model_confirmed'] = True
        result['instance'] = {'class_id': int(item['class_id']),
                              'score': float(item['score']),
                              'bbox_xyxy': [float(x) for x in item['bbox_xyxy']],
                              'green_fraction': fraction}
        body = self._connected_green_body(green, item)
        if body is None:
            result['reason'] = 'no_connected_green_body'
            return result
        return self._finish_top_fit(bgr, body, result)

    @classmethod
    def _finish_top_fit(cls, bgr, body, result, combine_edge_fragments=False):
        ys, xs = np.nonzero(body)
        result['body_center_px_diagnostic'] = [float(xs.mean()), float(ys.mean())]
        fits = cls._fit_top_edges(bgr, body, combine_edge_fragments)
        result['quality']['distinct_top_edge_candidates'] = len(fits)
        if not fits:
            result['reason'] = 'no_supported_internal_top_edge'
            return result
        fits.sort(key=lambda fit: fit['score'], reverse=True)
        best = fits[0]
        # A second well-supported, geometrically different nested ellipse cannot
        # be assigned to the actual top face from this image alone.
        if len(fits) > 1 and fits[1]['score'] >= best['score'] * .9:
            result['reason'] = 'ambiguous_internal_top_edges'
            return result
        result['valid'] = True
        result['reason'] = None
        result['center_px'] = best['ellipse']['center_px']
        result['ellipse_px'] = best['ellipse']
        result['edge_points_px'] = best['edge_points_px']
        result['quality'].update(best['quality'])
        return result

    @staticmethod
    def _green_geometry_candidates(green):
        """Locate known green round objects; reject rectangular board borders.

        Identity here comes from the explicit single-green-cup scene contract,
        not a learned class. Multiple plausible objects remain ambiguous.
        """
        closed = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        candidates = []
        for label in range(1, count):
            x, y, width, height, pixels = (int(v) for v in stats[label])
            if pixels < 200 or min(width, height) < 20:
                continue
            if x == 0 or y == 0 or x+width == green.shape[1] or y+height == green.shape[0]:
                continue
            body = labels == label
            contours, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            if area <= 0 or perimeter <= 0 or hull_area <= 0:
                continue
            circularity = 4*math.pi*area/(perimeter*perimeter)
            solidity = area/hull_area
            fill_fraction = pixels/area
            corners = len(cv2.approxPolyDP(contour, .02*perimeter, True))
            rectangle = cv2.minAreaRect(contour)
            axes = rectangle[1]
            axis_ratio = min(axes)/max(axes) if max(axes) > 0 else 0.
            if (circularity < .40 or solidity < .82 or fill_fraction < .60
                    or axis_ratio < .45 or corners < 6):
                continue
            candidates.append((body, {'bbox_xyxy': [x, y, x+width, y+height],
                                      'area_px': pixels, 'circularity': circularity,
                                      'solidity': solidity, 'fill_fraction': fill_fraction,
                                      'polygon_vertex_count': corners,
                                      'identity_contract': 'single known green dice cup in cleared workspace'}))
        return candidates

    @staticmethod
    def _connected_green_body(green, item):
        h, w = green.shape
        x1, y1, x2, y2 = item['bbox_xyxy']
        pad = max(4, int(.15 * max(x2-x1, y2-y1)))
        roi = np.zeros_like(green)
        roi[max(0, int(y1)-pad):min(h, int(math.ceil(y2))+pad),
            max(0, int(x1)-pad):min(w, int(math.ceil(x2))+pad)] = True
        # Small JPEG holes are closed without using model mask as a top edge.
        candidate = cv2.morphologyEx((green & roi).astype(np.uint8),
                                    cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        model_mask = np.asarray(item['mask'], dtype=bool)
        ranked = [(int(((labels == i) & model_mask).sum()), i)
                  for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 100]
        if not ranked:
            return None
        overlap, selected = max(ranked)
        return labels == selected if overlap >= 50 else None

    @staticmethod
    def _ellipse_samples(ellipse, count=180):
        (cx, cy), (a, b), angle = ellipse
        theta = np.linspace(0, 2*np.pi, count, endpoint=False)
        points = np.column_stack((a*.5*np.cos(theta), b*.5*np.sin(theta)))
        alpha = math.radians(angle)
        rotation = np.array([[math.cos(alpha), -math.sin(alpha)],
                             [math.sin(alpha), math.cos(alpha)]])
        return points @ rotation.T + [cx, cy]

    @classmethod
    def _fit_top_edges(cls, bgr, body, combine_edge_fragments=False):
        ys, xs = np.nonzero(body)
        x0, y0 = max(0, int(xs.min())-3), max(0, int(ys.min())-3)
        xend = min(bgr.shape[1], int(xs.max())+4)
        yend = min(bgr.shape[0], int(ys.max())+4)
        crop = bgr[y0:yend, x0:xend]
        # Both luminance and green retain the top-wall contrast differently
        # under camera exposure/JPEG variation. Every fit uses actual edges.
        planes = [('luminance', cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)),
                  ('green', crop[:, :, 1])]
        crop_body = body[y0:yend, x0:xend]
        if combine_edge_fragments:
            # Dark facets can fall outside the HSV threshold while still being
            # inside the connected object's silhouette. Filling only enclosed
            # holes makes containment independent of those color dropouts.
            silhouette = np.zeros_like(crop_body, dtype=np.uint8)
            contours, _ = cv2.findContours(crop_body.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
            containment_body = silhouette != 0
        else:
            containment_body = crop_body
        body_area = int(body.sum())
        results = []
        for channel_name, plane in planes:
            gray = cv2.GaussianBlur(plane, (3, 3), 0)
            for low, high in ((4, 12), (8, 24), (12, 36), (20, 60)):
                edges = cv2.Canny(gray, low, high)
                contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
                if combine_edge_fragments:
                    edge_y, edge_x = np.nonzero((edges != 0) & containment_body)
                    edge_points = np.column_stack((edge_x, edge_y)).astype(float)
                for contour in contours:
                    if len(contour) < 30:
                        continue
                    ellipse = cv2.fitEllipse(contour)
                    points = contour[:, 0, :].astype(float)
                    if combine_edge_fragments:
                        refined = cls._combine_edge_fragments(ellipse, points, edge_points)
                        if refined is None:
                            continue
                        ellipse, points = refined
                    (cx, cy), (a, b), angle = ellipse
                    if min(a, b) < 15 or min(a, b)/max(a, b) < .45:
                        continue
                    area_ratio = math.pi*a*b/(4*body_area)
                    # The outer silhouette includes the lower wall/base, and must
                    # not be substituted for the requested elevated top edge.
                    if not .18 <= area_ratio <= .80:
                        continue
                    samples = cls._ellipse_samples(ellipse)
                    pixels = np.rint(samples).astype(int)
                    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < gray.shape[1])
                              & (pixels[:, 1] >= 0) & (pixels[:, 1] < gray.shape[0]))
                    if not inside.all() or containment_body[pixels[:, 1], pixels[:, 0]].mean() < .97:
                        continue
                    alpha = math.radians(angle)
                    rotation = np.array([[math.cos(alpha), -math.sin(alpha)],
                                         [math.sin(alpha), math.cos(alpha)]])
                    local = (points-[cx, cy]) @ rotation
                    normalized = local / [a*.5, b*.5]
                    residual = np.abs(np.linalg.norm(normalized, axis=1)-1)*min(a, b)*.5
                    inliers = residual <= max(2., min(a, b)*.035)
                    if inliers.sum() < 25:
                        continue
                    bins = np.floor((np.arctan2(normalized[inliers, 1], normalized[inliers, 0])
                                     + np.pi)*36/(2*np.pi)).astype(int) % 36
                    coverage = len(np.unique(bins))/36.
                    p80 = float(np.quantile(residual, .8))
                    tolerance = max(2., min(a, b)*.035)
                    support = float((distance[pixels[:, 1], pixels[:, 0]] <= tolerance).mean())
                    if coverage < .60 or p80 > tolerance or support < .60:
                        continue
                    quality = {'angular_coverage': coverage, 'edge_support_fraction': support,
                               'fit_residual_median_px': float(np.median(residual)),
                               'fit_residual_p80_px': p80, 'top_to_body_area_ratio': area_ratio,
                               'canny_thresholds': [low, high], 'edge_channel': channel_name,
                               'note': 'ellipse fit to visible internal top boundary; pixel geometry only'}
                    if combine_edge_fragments:
                        quality['combined_same_frame_edge_fragments'] = True
                    candidate = {'ellipse': {'center_px': [float(cx+x0), float(cy+y0)],
                                             'diameters_px': [float(a), float(b)],
                                             'angle_deg': float(angle)},
                                 'quality': quality,
                                 'edge_points_px': (points[inliers]+[x0, y0]).tolist(),
                                 'score': coverage*.45+support*.45+.1/(1+p80)}
                    # Canny traces both sides of an edge and repeats it at multiple
                    # thresholds. Keep the best fit once for each geometric edge.
                    duplicate = None
                    for index, previous in enumerate(results):
                        p = previous['ellipse']
                        if (np.linalg.norm(np.asarray(p['center_px'])-candidate['ellipse']['center_px'])
                                <= max(3., min(a, b)*.06)
                                and np.max(np.abs(np.asarray(p['diameters_px'])/[a, b]-1)) < .15):
                            duplicate = index
                            break
                    if duplicate is None:
                        results.append(candidate)
                    elif candidate['score'] > results[duplicate]['score']:
                        results[duplicate] = candidate
        return results

    @staticmethod
    def _combine_edge_fragments(ellipse, contour_points, edge_points):
        """Refit measured pixels near a contour hypothesis, without inventing edges.

        A Canny trace can break into separate arcs because of quantization or
        weak local contrast. All final coverage/residual/support checks still
        apply to the combined actual image-edge pixels.
        """
        (cx, cy), (a, b), angle = ellipse
        if min(a, b) < 15 or min(a, b)/max(a, b) < .45:
            return None
        alpha = math.radians(angle)
        rotation = np.array([[math.cos(alpha), -math.sin(alpha)],
                             [math.sin(alpha), math.cos(alpha)]])
        tolerance = max(2., min(a, b)*.035)
        normalized = (contour_points-[cx, cy]) @ rotation/[a*.5, b*.5]
        residual = np.abs(np.linalg.norm(normalized, axis=1)-1)*min(a, b)*.5
        if np.quantile(residual, .8) > tolerance:
            return None
        normalized = (edge_points-[cx, cy]) @ rotation/[a*.5, b*.5]
        residual = np.abs(np.linalg.norm(normalized, axis=1)-1)*min(a, b)*.5
        selected = edge_points[residual <= tolerance]
        if len(selected) < 25:
            return None
        return cv2.fitEllipse(selected.astype(np.float32)), selected
