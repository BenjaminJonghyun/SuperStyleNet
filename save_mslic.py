
import maskslic as seg
import scipy.interpolate as interp
import numpy as np


def _adaptive_interp(input, num_of_style_feature):
    if len(input) < 3:
        return input
    if len(input) < num_of_style_feature:

        x = np.linspace(0, num_of_style_feature - 1, num=len(input))
        x_new = np.linspace(0, num_of_style_feature - 1, num=num_of_style_feature)

        interp_out = interp.InterpolatedUnivariateSpline(x, input)
        output = interp_out(x_new)
    else:
        output = input

    return output


def _encoding(label_field, image, num_of_vectors, bg_label=-1):
    '''
    Generating Style codes
    :param label_field: super-pixel output
    :param image: style image
    :param bg_label: background label in super-pixel
    :return: style codes
    '''
    #lab = color.rgb2yuv(image)
    lab = image
    #lab = color.rgb2lab(image)
    l = []
    a = []
    b = []
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
    for label in labels:
        mask = (label_field == label).nonzero()
        feature = lab[mask].mean(axis=0)
        l.append(feature[0])
        a.append(feature[1])
        b.append(feature[2])
    l = np.reshape(l, (-1))
    l = _adaptive_interp(l, num_of_vectors)
    a = np.reshape(a, (-1))
    a = _adaptive_interp(a, num_of_vectors)
    b = np.reshape(b, (-1))
    b = _adaptive_interp(b, num_of_vectors)
    out = np.reshape([l, a, b], (-1))

    out = _adaptive_interp(out, 512)
    out = np.reshape(out, (-1,))


    return out


def _style_encoder(images, masks, num_of_style=512, num_of_vectors=128):
    style_vectors = []
    null = np.zeros(num_of_style)
    for i in range(len(images)):
        styles = []
        for j in range(np.shape(masks)[-1]):
            num_of_component_pixel = np.count_nonzero(masks[i, :, :, j])
            if num_of_component_pixel > 0:
                try:
                    m_slic = seg.slic(images[i], compactness=10, seed_type='nplace', mask=masks[i, :, :, j],
                                      n_segments=num_of_vectors, recompute_seeds=True, enforce_connectivity=False)
                    style_vector = _encoding(m_slic, images[i], num_of_vectors)
                    styles.append(style_vector)
                except:
                    styles.append(null)
            else:
                styles.append(null)
        style_vectors.append(styles)
    style_vectors = np.reshape(style_vectors, (np.shape(masks)[-1], len(images), 1, 1, num_of_style))

    return style_vectors

