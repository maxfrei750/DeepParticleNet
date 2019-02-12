import numpy as np
from mrcnn.visualize import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes


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
                              figsize=(16, 16),
                              ax=None,
                              colors=None,
                              linewidth=4,
                              alpha=1,
                              linestyle="-"):
    """
    masks: [height, width, num_instances]
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    N = masks.shape[2]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]

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
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def plot_size_distributions(sizedistributions, captions,
                            plot_density=True,
                            bins=np.logspace(np.log10(10), np.log10(150), 15)):
    for sizedistribution, caption in zip(sizedistributions, captions):
        d_g = sizedistribution.geometric_mean
        s_g = sizedistribution.geometric_standard_deviation
        N = sizedistribution.number_of_particles

        legend_string = caption + "\n$d_g = {:.0f}\mathrm{{px}}$; $\sigma_g = {:.2f}$; $N = {:d}$".format(d_g, s_g, N)

        plt.hist(sizedistribution.sizes,
                 bins=bins,
                 density=plot_density,
                 label=legend_string,
                 alpha=0.5,
                 fill=True,
                 histtype='step')

    plt.grid(True)
    plt.legend(loc="upper left")
    ax = plt.gca()
    ax.set_xscale("log", nonposx="clip")
    plt.xlabel("Diameter [px]")

    if plot_density:
        plt.ylabel("Probability Density [a.u.]")
    else:
        plt.ylabel("Count [#]")

    plt.tight_layout()
    plt.show()