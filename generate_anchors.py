import numpy as np

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.
    Args:
        base_size:
        ratios:
        scales:
    Returns:
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors
pyramid_levels = [3, 4, 5, 6, 7]
sizes=[32, 64, 128, 256, 512]
strides=[8, 16, 32, 64, 128],
# ratio=h/w
ratios=np.array([0.634, 1, 1.577])
scales= np.array([0.4, 0.506, 0.641])
for idx, p in enumerate(pyramid_levels):
    anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
    # shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
    # all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

#all_anchors = np.expand_dims(all_anchors, axis=0)
#anchors = generate_anchors(base_size=16, ratios=ratios, scales=scales)
    n_scales = len(scales)
    for ind, an in enumerate(anchors):
        if ind%n_scales == 0:
            print("anchors for scale", scales[int(ind/n_scales)], ":")
        print("\t", an)