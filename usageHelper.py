import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_lines(lines, figsize=(18, 18), margin=10, plot_points = True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#111155')
    end_points = []
    for line in lines:
        end_points.append((line[0].x, line[0].y))
        end_points.append((line[-1].x, line[-1].y))

        xdata = []
        ydata = []
        for point in line:
            xdata.append(point.x)
            ydata.append(point.y)

        plt_line = matplotlib.lines.Line2D(xdata, ydata, linewidth=1, axes=ax, color='#ff2222', alpha=.15)

        ax.add_line(plt_line)

    # Limits
    end_points_arr = np.array(end_points)
    x_min = end_points_arr[:, 0].min() - margin
    x_max = end_points_arr[:, 0].max() + margin
    y_min = end_points_arr[:, 1].min() - margin
    y_max = end_points_arr[:, 1].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # / Limits

    if plot_points:
        ax.scatter(end_points_arr[:, 0], end_points_arr[:, 1], s=10, c='#ffee00')

    ax.set_aspect(1.0)
    plt.show()
    plt.close()


import geopandas as gpd
from shapely.geometry import LineString, Point
import contextily as ctx

def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))


def plot_lines_on_map(input_lines, figsize=(18, 18), footer=None, save_filename=None):
    lines = []
    for edge_idx, edge in enumerate(input_lines):
        points = []

        for point in edge:
            points.append(Point(point.x, point.y))

        lines.append(LineString(points))

    gdf = gpd.GeoSeries(lines)
    gdf.crs = {'init': 'epsg:5361'}
    gdf = gdf.to_crs({'init': 'epsg:3857'})
    ax = gdf.plot(linewidth=.2, color='#ff2222', figsize=figsize, alpha=0.3)
    add_basemap(ax, zoom=12)
    #ax.set_xlim(-7885000, -7840000)
    # ax.set_ylim(-33.65, -33.3)
    plt.axis('off')
    if footer:
        ax.set_title(footer)
    #plt.annotate('texto', (0, 0), (.5, .01), fontsize=14, textcoords='axes fraction')
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
    plt.show()
    plt.close()