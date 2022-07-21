""" Routines to generate movie for movie_coil2D re-play"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from environments.dm_control.utils import episode_data
import tempfile
from IPython.display import Image, display

def renderer(env, outpath = './', filename = 'movie', save=False, display=True):
    # Pack data from simulation
    data = episode_data.pack_datadict(env.task.datadict)
    
    # Initialize the movie
    fig, ax = plt.subplots()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')

    # Plot boundary constraints for the coil
    t = np.linspace(0,2*3.14,100)
    ax.plot(env.physics.par_dict['r_b']*np.sin(t),env.physics.par_dict['r_b']*np.cos(t), 'b')

    # Plot moving coil and fixed coils
    circle =  ax.plot([], [], 'ro', markersize = 10)[0]
    red_cross = ax.plot([], [], 'rx', markersize = 10)[0]
    rectangle = [ax.plot(x[0],x[1], 'bs', markersize = 20)[0] for x in env.physics.par_dict['x_c']]

    # Get max/min action value
    Imin = env.task.par_dict['minIp']
    Imax = env.task.par_dict['maxIp']

    # Converting coil current to r,g,b
    # I = Imax -> red, I = Imin -> blue
    I2canonical = lambda x: (x - Imin)/(Imax-Imin)
    color = lambda x: np.array([1,0,0]) + I2canonical(x)*(np.array([0,0,1]) - np.array([1,0,0]))

    # Set color for moving coil
    circle.set_color(tuple(color(env.physics.par_dict['Ip'])) )

    animate = lambda it: update_plot(ax, data, it, circle, red_cross, rectangle, color)
    
    if display:
        manimation.FuncAnimation(fig, animate, frames=5)            

    if save:
        # Define the meta data for the movie
        FFMpegWriter = manimation.writers['pillow']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Moving coil 2D plotting')
        writer = FFMpegWriter(fps=5, metadata=metadata)

        filepath = outpath + '/' + filename + '.gif'

        # Update the frames for the movie
        with writer.saving(fig,  filepath, 100):
            for it in range(len(data['time'])):
                animate(it)
                writer.grab_frame()

def update_plot(ax, data, it, circle, red_cross, rectangle, color):
    circle.set_data(data['state'][it,0], data['state'][it,1])
    red_cross.set_data(data['reference'][it,0], data['reference'][it,1])
    for ic in range(data['action'].shape[1]):
        rectangle[ic].set_color(tuple(color(data['action'][it,ic])))
        

def display_movie_ipynb(env):
    """Convenience function to display movie in Notebook"""
    with tempfile.TemporaryDirectory() as dir_path:
        tmp_fname = "movie"
        renderer( env, outpath = dir_path, filename=tmp_fname, save=True, display=False )
        img = Image(dir_path + '/' + tmp_fname + '.gif')
        display(img)
