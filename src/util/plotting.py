import numpy as np

def relabel_axis_with_images(fig, ax, x_images, y_images, n_x=None, n_y=None):
    xl, yl, xh, yh=np.array(ax.get_position()).ravel()
    w=xh-xl
    h=yh-yl

    if n_x is None:
        n_x = len(x_images)
    if n_y is None:
        n_y = len(y_images)
    h_step = h/n_y
    w_step = w/n_x
    size = h_step * 0.95

    yp = yh + 0.5 * h_step
    for img in y_images:
        yp=yp-h_step

        ax1=fig.add_axes([xl-0.5*w_step-size*0.6, yp-size*0.5, size, size])
        ax1.axison = False
        imgplot = ax1.imshow(img)

    xp = xl-0.5*w_step
    for img in x_images:
        xp=xp+w_step

        ax1=fig.add_axes([xp-size*0.5, yh+0.1*size, size, size])
        ax1.axison = False
        imgplot = ax1.imshow(img)