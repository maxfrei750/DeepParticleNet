from mrcnn.visualize import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes
import seaborn as sns


def display_image(image, title="", figsize=(16, 16), ax=None):
    """Display an image.

    :param image: Image to be displayed.
    :param title: Title to be displayed (default: "").
    :param figsize: Size of the figure in inches (default: (16, 16)).
    :param ax: Axis to use for the plot (default: None, creates a new axis).
    :return: nothing
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
    """Return a Matplotlib Axes array to be used in all visualizations in the notebook. Provide a central point to
    control graph sizes. Adjust the size attribute to control how big to render images.

    :param rows: Number of subplot rows (default: 1).
    :param cols: Number of subplot columns (default: 1).
    :param size: Size of the subplots (default: 32).
    :return: Axis object.
    """

    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def display_instance_outlines(image, masks,
                              colors=None,
                              linewidth=0.5,
                              alpha=1,
                              linestyle="-",
                              dpi=300):
    """Display an image with overlayed outlines of the detected instances.

    :param image: Original image.
    :param masks: List of instance masks.
    :param colors: List of colors for each object (default: None, use random colors).
    :param linewidth: Width of the outlines (default: 0.5).
    :param alpha: Opacity of the outlines (default: 1).
    :param linestyle: Style of the outlines (default: "-").
    :param dpi: Resolution of the output image (default: 300)
    :return: Figure handle.
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
                            d_g_in_legend=True,
                            sigma_g_in_legend=True,
                            bins="auto",
                            alpha=0.5,
                            fill=True,
                            histtype="step",
                            ncol=2,
                            **kwargs):
    """Plot a list of SizeDistribution objects.

    :param sizedistributions: List of SizeDistribution objects.
    :param captions: List of captions. One for each SizeDistribution object.
    :param density: Plot the probability density distribution (Default: True).
    :param number_in_legend: Display the number of instances of each distribution in the legend (Default: True).
    :param d_g_in_legend: Display the geometric mean of each distribution in the legend (Default: True).
    :param sigma_g_in_legend: Display the geometric standard deviation of each distribution in the legend
                              (Default: True).
    :param bins: List of bin edges, number of bins or binning mode (Default: "auto").
    :param alpha: Opacity of the histograms (Default: 1).
    :param fill: Whether or not to fill the histograms (Default: True).
    :param histtype: Histogram type. For further information, see matplotlib.pyplot.hist (Default: "step").
    :param ncol: Number of columns of the legend (Default: 2).
    :param kwargs: Additional arguments to be passed to the matplotlib.pyplot.hist function.
    :return: Axis object, histogram counts, histogram bins.
    """
    
    # Set a default for the color
    if "color" in kwargs:
        colors = kwargs["color"]
        del(kwargs["color"])
    else:
        number_of_sizedistributions = len(sizedistributions)
        colors = sns.color_palette("viridis",number_of_sizedistributions)
    
    sizes_list = [sizedistribution.sizes for sizedistribution in sizedistributions]
    
    labels = list()
    
    for sizedistribution, caption in zip(sizedistributions, captions):
        label = caption

        if d_g_in_legend or sigma_g_in_legend or number_in_legend:
            label += " ("

        if d_g_in_legend:
            d_g = sizedistribution.geometric_mean
            label += "$d_\mathrm{{g}} = {:.0f}\mathrm{{px}}$; ".format(d_g)
            
        if sigma_g_in_legend:
            s_g = sizedistribution.geometric_standard_deviation
            label += "$\sigma_\mathrm{{g}} = {:.2f}$; ".format(s_g)
        
        if number_in_legend:
            N = sizedistribution.number_of_particles
            label += "$N = {:d}$; ".format(N)
            
        if d_g_in_legend or sigma_g_in_legend or number_in_legend:
            label = label[:-2] + ")"
            
        labels += [label]
        
    # Reverse colors, labels and sizes_list because they are again reversed by the hist function.
    colors.reverse()
    sizes_list.reverse()
    labels.reverse()
            
    histogram_n, histogram_bins, _ = plt.hist(sizes_list,
                                           bins=bins,
                                           density=density,
                                           label=labels,
                                           alpha=alpha,
                                           fill=fill,
                                           histtype=histtype,
                                           color=colors,
                                           **kwargs)
    
    # Reverse outputs
    histogram_n.reverse()
        
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc="lower left",
               ncol=ncol,
               mode="expand",
               borderaxespad=0.)
    ax = plt.gca()
    plt.xlabel("$d/\mathrm{px}$")

    plt.xlim(left=0)

    if density:
        plt.ylabel("$P(d)$")
    else:
        plt.ylabel("$N$")

    return ax, histogram_n, histogram_bins
