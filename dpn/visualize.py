from mrcnn.visualize import *

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