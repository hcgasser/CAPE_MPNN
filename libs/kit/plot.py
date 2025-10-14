""" offers some helper functions for plotting """

import numpy as np
from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

A4_width = 8.27
A4_height = 11.69

def rm_axes_elements(axes, elements, y_axis=True, x_axis=True):
    """removes elements from the axes

    :param axes: list of axes to remove the elements from
    :param elements: elements to remove (e.g. "ticks,ticklabels,plain,labels,grid,legend")
    :param y_axis: whether to remove the elements from the y-axis
    :param x_axis: whether to remove the elements from the x-axis
    """

    elements = elements.split(",")

    # pylint: disable=unidiomatic-typecheck
    if type(axes) != list:
        axes = [axes]
    for ax in axes:
        if ax is not None:
            if "ticks" in elements:
                if x_axis:
                    ax.set_xticks([])
                if y_axis:
                    ax.set_yticks([])
            if "ticklabels" in elements:
                if x_axis:
                    ax.set_xticklabels("")
                if y_axis:
                    ax.set_yticklabels("")
            if "plain" in elements:
                ax.axis("off")
            if "labels" in elements:
                if x_axis:
                    ax.set_xlabel("")
                if y_axis:
                    ax.set_ylabel("")
            if "grid" in elements:
                if x_axis:
                    ax.xaxis.grid(False)
                if y_axis:
                    ax.yaxis.grid(False)
            if "legend" in elements:
                ax.legend_.remove()


def plot_text(text, ax, font_scale=1.0, rotation=0, x_pos=0.5, y_pos=1.0):
    """plots text in the specified axes"""

    rm_axes_elements(ax, "plain")
    ax.text(
        x_pos,
        y_pos,
        text,
        fontsize=plt.rcParams["font.size"] * 1.5 * font_scale,
        ha="center",
        va="top",
        rotation=rotation,
        rotation_mode='anchor'
    )


def plot_legend_patches(legend, ax, location="center"):
    """plots a legend in the specified axes

    the legend is showing patches with the specified colors and labels
    """

    patches = []
    for key, value in legend.items():
        patches.append(mpatches.Patch(color=value, label=key))

    ax.legend(
        handles=patches,  # loc='upper right')
        loc=location,
        fancybox=False,
        shadow=False,
        ncol=1,
    )

    rm_axes_elements(ax, "plain")


def plot_legend_scatter(ax, labels, markers, colors, **kwargs):
    """plots a legend in the specified axes

    the legend is showing scatter plot elements with the specified colors and labels
    """

    legend_elements = [
        plt.scatter([], [], label=l, linewidth=2, marker=m, color=c)
        for l, m, c in zip(labels, markers, colors)
    ]
    ax.legend(handles=legend_elements, title=None, fancybox=False, **kwargs)


def plot_legend(ax, markers, colors, **kwargs):
    assert len(set(markers.keys()) ^ set(colors.keys())) == 0

    _labels, _markers, _colors = [], [], []
    for l in markers.keys():
        _labels.append(l)
        _markers.append(markers[l])
        _colors.append(colors[l])

    plot_legend_scatter(ax, _labels, _markers, _colors, **kwargs)


def plot_green_red_areas(top, bottom, good, ax, alpha=0.01):
    # Add a red area for values above 0
    color_positive, color_negative = ("green", "red") if good > 0 else ("red", "green")
    
    for lower in np.linspace(top, 0, 50):
        ax.axhspan(lower, top, facecolor=color_positive, alpha=alpha)
    
    for upper in np.linspace(bottom, 0, 50):
        ax.axhspan(bottom, upper, facecolor=color_negative, alpha=alpha)


def plot_borders_in_stripplot(cnt, ax):
    for category in range(cnt - 1):
        ax.axvline(category + 0.5, color='gray', linestyle='--')


def spider_plot(df_values,
                fill=None,
                area=None, ax=None,
                ylim=None, ysections=None,
                face_center=True, fy=1.15):
    """ plots a spider web figure

    :param df_values: indices are the categories and columns are the series
    :param fill:
    :param area:
    :param ax:
    :param ylim:
    :param ysections:
    :param face_center:
    :param fy:
    :return:
    """

    if area is None and ax is None:
        fig = plt.figure(figsize=(A4_width, A4_width))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
    elif ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(area, projection='polar')

    categories = list(df_values.index)

    N = len(categories)  # number of variables
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    for c in df_values.columns:  # plot all series
        # plot data
        values = list(df_values[c])
        values += [values[0]]
        ax.plot(angles, values, linewidth=2, linestyle='solid')

        # fill area
        if fill is not None and fill[i]:
            ax.fill(angles, values, 'b', alpha=0.1)

    # add value labels inside
    ax.set_rlabel_position(0)
    # plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
    if ylim is not None:
        ax.set_ylim(ylim)
        if ysections is not None:
            sec = np.array(range(ysections+1))*(ylim[1] - ylim[0])/ysections
            sec = sec[:-1]
            ax.set_yticks(sec)
            ax.set_yticklabels([f"{s:.2f}" for s in sec], color="grey", size=7)
    # plt.ylim(0, 5)

    # add category labels outside the circle
    if not face_center:
        ax.set_xticks(angles[:-1], categories)
    else:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        for i, category in enumerate(categories):
            angle = angles[i]
            angle_label = (-pi / 2 + angle) if angle <= pi else (-pi / 2 + angle - pi)
            rotation = np.degrees(angle_label)
            ha = 'center'
            va = 'center'  #'bottom' if angle <= pi / 2 or angle > 3 * pi / 2 else 'top'
            ax.text(
                angle,
                ax.get_ylim()[1]*fy,
                f"{category}",  # _{angle:.2f}_{rotation:.2f}",
                size=12,
                horizontalalignment=ha,
                verticalalignment=va,
                rotation=rotation,
                rotation_mode='anchor'
            )

    return ax


def broken_axis_plot(
        plot_function, kwargs, y_lims, height_ratios, area, axes=None, hspace=0.025,
        zigzag_top=False, zigzag_bottom=True
    ):
    """ plots a broken axis plot

    example: 
        np.random.seed(0)
        data = np.random.normal(1, 1, 100)  # Most data points between -3 and 5
        outliers = np.array([1000, 2000, 3000])  # Outliers
        data = np.concatenate((data, outliers))

        data2 = np.random.normal(1, 1, 100)  # Most data points between -3 and 5
        outliers2 = np.array([1500, 2500, 3000])  # Outliers
        data2 = np.concatenate((data, outliers))

        fig = plt.figure(figsize=(A4_width, A4_height))
        gs = mpl.gridspec.GridSpec(
            1,
            1,
            height_ratios=[1],
            width_ratios=[1],
            wspace=0.1, hspace=0.1)

        axes= broken_axis_plot(sns.stripplot, {'data': data, 'color': "blue", 'size': 5}, 
                    [(900, 3100), (-3, 5)], [1,9], gs[0,0]
                )

        broken_axis_plot(sns.stripplot, {'data': data, 'color': "red", 'size': 5}, 
                    [(900, 3100), (-3, 5)], [1,9], gs[0,0], axes=axes
                )
    
    """

    fig = plt.gcf()
    grid_spec = mpl.gridspec.GridSpecFromSubplotSpec(
        len(y_lims), 1, subplot_spec=area, height_ratios=height_ratios, hspace=hspace
    )
    _axes = [] if axes is None else axes

    zigzag_x = np.linspace(-0.02, 1.02, 52)  # 50 points across the x-axis
    for i, y_lim in enumerate(y_lims):
        if axes is None:
            _axes.append(fig.add_subplot(grid_spec[i], sharex=(_axes[0] if len(_axes) > 0 else None)))
        ax = _axes[i]
        plot_function(ax=ax, **kwargs)
        ax.set_ylim(y_lim)        

        d_x = 0.015   # Size of diagonal lines
        d_y = 0.015 / height_ratios[i]
        
        _kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

        if i == 0:  # upper axis
            ax.xaxis.tick_top()
        else:
            ax.spines['top'].set_visible(False)

            # top breaks
            if zigzag_top:
                zigzag_y_top = np.tile([1-d_y, 1+d_y], len(zigzag_x) // 2)
                ax.plot(zigzag_x, zigzag_y_top, **_kwargs)
            else:
                ax.plot((-d_x, + d_x), (1 - d_y, 1 + d_y), **_kwargs)
                ax.plot((1 - d_x, 1 + d_x), (1 - d_y, 1 + d_y), **_kwargs)
        
        if i == (len(y_lims) - 1):  # lower axis
            ax.xaxis.tick_bottom()
        else:
            ax.spines['bottom'].set_visible(False)

            # bottom breaks
            if zigzag_bottom:
                zigzag_y_bottom = np.tile([-d_y, +d_y], len(zigzag_x) // 2)
                ax.plot(zigzag_x, zigzag_y_bottom, **_kwargs)
            else:
                ax.plot((-d_x, +d_x), (-d_y, +d_y), **_kwargs)         # Top-left diagonal line
                ax.plot((1 - d_x, 1 + d_x), (-d_y, +d_y), **_kwargs)   # Top-right diagonal line

    return _axes
    
