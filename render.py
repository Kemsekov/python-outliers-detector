import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def __color_scale(x):
    min_x = np.min(x)
    max_x = np.max(x)
    scale = max_x-min_x
    scale = 1 if scale == 0 else scale
    return ((x - np.min(x)) / scale * 255).astype(int)

import numpy as np
import plotly.graph_objects as go

def plot_3d_rgb(data_array: np.ndarray, title: str, axis_titles: list[str], 
                dot_size=5, template='plotly_dark', axis_sizes=None):
    # Ensure the input is an ndarray
    if not isinstance(data_array, np.ndarray) or data_array.shape[1] < 3:
        raise ValueError("The input must be an ndarray with at least 3 columns.")
    
    # Normalize the color dimensions to be between 0 and 255
    if data_array.shape[1] > 3:
        red = __color_scale(data_array[:, 3])
    else:
        red = np.ones(data_array.shape[0]) * 255
    
    if data_array.shape[1] > 4:
        green = __color_scale(data_array[:, 4])
    else:
        green = np.ones(data_array.shape[0]) * 255

    if data_array.shape[1] > 5:
        blue = __color_scale(data_array[:, 5])
    else:
        blue = np.ones(data_array.shape[0]) * 255

    # Combine the color dimensions into a single color value for each point
    colors = ['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(red, green, blue)]
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data_array[:, 0],
        y=data_array[:, 1],
        z=data_array[:, 2],
        mode='markers',
        marker=dict(
            size=dot_size,
            color=colors,  # Set color to the RGB values
            opacity=0.7
        )
    )])
    
    # Define the axis ranges if axis_sizes are provided
    xaxis_range = axis_sizes[0] if axis_sizes else None
    yaxis_range = axis_sizes[1] if axis_sizes else None
    zaxis_range = axis_sizes[2] if axis_sizes else None
    
    # Update the layout of the plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=axis_titles[0], range=xaxis_range),
            yaxis=dict(title=axis_titles[1], range=yaxis_range),
            zaxis=dict(title=axis_titles[2], range=zaxis_range)
        ),
        width=700,
        height=600,
        template=template
    )
    
    fig.show()

# Example of axis_sizes usage:
# axis_sizes = [(0, 10), (0, 20), (0, 30)]


def plot_2d_rgb(data_array: np.ndarray, title: str, axis_titles: list[str],dot_size=5,template='plotly_dark',axis_sizes=None):
    # Ensure the input is an ndarray
    if not isinstance(data_array, np.ndarray) or data_array.shape[1] < 2:
        raise ValueError("The input must be an ndarray with at least 2 columns.")

    # Normalize the color dimensions to be between 0 and 255
    if data_array.shape[1] > 2:
        red = __color_scale(data_array[:, 2])
    else:
        red = np.ones(data_array.shape[0]) * 255

    if data_array.shape[1] > 3:
        green = __color_scale(data_array[:, 3])
    else:
        green = np.ones(data_array.shape[0]) * 255

    if data_array.shape[1] > 4:
        blue = __color_scale(data_array[:, 4])
    else:
        blue = np.ones(data_array.shape[0]) * 255

    # Combine the color dimensions into a single color value for each point
    colors = ['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(red, green, blue)]

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=data_array[:, 0],
        y=data_array[:, 1],
        mode='markers',
        marker=dict(
            size=dot_size,
            color=colors,  # Set color to the RGB values
            opacity=0.7
        )
    )])

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        width=700,
        height=600,
        template=template
    )

    fig.show()