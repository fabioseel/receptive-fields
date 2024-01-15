import numpy as np

def relabel_axis_with_images(fig, ax, x_images, y_images):
    xl, yl, xh, yh=np.array(ax.get_position()).ravel()
    w=xh-xl
    h=yh-yl

    h_step = h/len(y_images)
    w_step = w/len(x_images)
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