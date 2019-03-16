import numpy as np
from mrcnn.visualize import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes
import seaborn as sns


def display_image(image, title="", figsize=(16, 16), ax=None):
    """
    title: (optional) Figure title
    figsize: (optional) the size of the image
    """

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    ax.imshow(image.astype(np.uint8))

    if auto_show:
        plt.show()


def get_axis(rows=1, cols=1, size=32):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def display_instance_outlines(image, masks,
                              colors=None,
                              linewidth=0.5,
                              alpha=1,
                              linestyle="-",
                              dpi=300):
    """
    masks: List of masks
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    N = len(masks)
    if not N:
        print("\n*** No instances to display *** \n")

    # Generate random colors
    colors = colors or random_colors(N)
    # Show image.
    height, width = image.shape[:2]
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches((width / dpi, height / dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image.astype(np.uint8))

    # Show area outside image boundaries.
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)

    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[i]

        mask = binary_fill_holes(mask)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        linestyle=linestyle)
            ax.add_patch(p)

    return ax.figure


def plot_size_distributions(sizedistributions, captions,
                            density=True,
                            number_in_legend=True,
                            bins="auto",
                            alpha = 0.5,
                            fill=True,
                            histtype="step",
                            **kwargs):
    
    # Set a default for the color
    if "color" in kwargs:
        color = kwargs["color"]
        del(kwargs["color"])
    else:
        number_of_sizedistributions = len(sizedistributions)
        color = sns.color_palette("viridis",number_of_sizedistributions)
    
    sizes_list = [sizedistribution.sizes for sizedistribution in sizedistributions]
    
    labels = list()
    
    for sizedistribution, caption in zip(sizedistributions, captions):
        d_g = sizedistribution.geometric_mean
        s_g = sizedistribution.geometric_standard_deviation

        label = caption + "$d_\mathrm{{g}} = {:.0f}\mathrm{{px}}$; $\sigma_\mathrm{{g}} = {:.2f}$".format(d_g, s_g)
        
        if number_in_legend:
            N = sizedistribution.number_of_particles
            label = label + "; $N = {:d}$".format(N)
            
        labels += [label]
            
    histogram_n, histogram_bins, _ = plt.hist(sizes_list,
                                           bins=bins,
                                           density=density,
                                           label=labels,
                                           alpha=alpha,
                                           fill=fill,
                                           histtype=histtype,
                                           color=color,
                                           **kwargs)
        
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc="lower left",
               ncol=1,
               mode="expand",
               borderaxespad=0.)
    ax = plt.gca()
    plt.xlabel("Diameter [px]")

    plt.xlim(left=0)

    if density:
        plt.ylabel("Probability Density [a.u.]")
    else:
        plt.ylabel("Count [a.u.]")

    return ax, histogram_n, histogram_bins
