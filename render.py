import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_3d_rgb(data_array : np.ndarray,title : str, axis_titles : list[str]):
    # Ensure the input is an ndarray
    if not isinstance(data_array, np.ndarray) or data_array.shape[1] < 3:
        raise ValueError("The input must be an ndarray with at least 3 columns.")
    
    # Normalize the color dimensions to be between 0 and 255
    color_scale = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255).astype(int)
    
    if data_array.shape[1]>3:
        red = color_scale(data_array[:, 3])
    else:
        red = np.ones(data_array.shape[0])*255
    
    if data_array.shape[1]>4:
        green = color_scale(data_array[:, 4])
    else:
        green = np.ones(data_array.shape[0])*255

    if data_array.shape[1]>5:
        blue = color_scale(data_array[:, 5])
    else:
        blue = np.ones(data_array.shape[0])*255

    # Combine the color dimensions into a single color value for each point
    colors = ['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(red, green, blue)]
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data_array[:, 0],
        y=data_array[:, 1],
        z=data_array[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,  # Set color to the RGB values
            opacity=0.7
        )
    )])
    
    # Update the layout of the plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2]
        ),
        width=700,
        height=600,
        template='plotly_dark'
    )
    
    fig.show()

def plot_2d_rgb(data_array: np.ndarray, title: str, axis_titles: list[str]):
    # Ensure the input is an ndarray
    if not isinstance(data_array, np.ndarray) or data_array.shape[1] < 2:
        raise ValueError("The input must be an ndarray with at least 2 columns.")

    # Normalize the color dimensions to be between 0 and 255
    color_scale = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255).astype(int)

    if data_array.shape[1] > 2:
        red = color_scale(data_array[:, 2])
    else:
        red = np.ones(data_array.shape[0]) * 255

    if data_array.shape[1] > 3:
        green = color_scale(data_array[:, 3])
    else:
        green = np.ones(data_array.shape[0]) * 255

    if data_array.shape[1] > 4:
        blue = color_scale(data_array[:, 4])
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
            size=4,
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
        template='plotly_dark'
    )

    fig.show()