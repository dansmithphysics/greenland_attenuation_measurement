import matplotlib.pyplot as plt
import numpy as np
import copy 
from multiprocessing import Pool, freeze_support

def step_length(z, ice_layer):
    step_length_ = 1.0
    return step_length_

def index_function(z, ice_layer, model_top, model_bottom, homo):
    if(homo):
        return model_top
    else:
        start = model_top
        climb = model_bottom - model_top
        return  start + climb * (1.0 - np.exp(-0.0134459 * np.abs(z)))

#def throw_for_pool(args):
#    return throw(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    
def throw(dir0s, x0s, nsteps, ice_layer, homo, model_top, model_bottom):
    
    pos = np.array([x0s[0], x0s[1], x0s[2]])
    total_length = 0.0
    norm_vec = np.array([0.0, 0.0, 1.0]) # Vector norm
        
    i_vec = np.array([np.sin(dir0s[1]) * np.cos(dir0s[0]), np.cos(dir0s[1]) * np.cos(dir0s[0]), np.sin(dir0s[0])])
    t_vec = np.array([0.0, 0.0, 0.0])

    n_old = index_function(pos[2], ice_layer, model_top, model_bottom, homo)
    for i in range(nsteps):
            
        n1 = n_old
        n2 = index_function(pos[2], ice_layer, model_top, model_bottom, homo)
        
        if(pos[2] < ice_layer):
            i_vec[2] *= -1
            pos[2] = ice_layer
            n2 = index_function(pos[2], ice_layer, model_top, model_bottom, homo)
                
        # Stupid solution to not having the correct normal vector sign
        going_down = False
        if(i_vec[2] < 0.0):
            going_down = True
                
        if not(going_down):
            i_vec[2] *= -1
            break # breaking here, to save computational time
        
        t_vec = np.sqrt(1.0 - np.power(n1 / n2, 2.0)*(1.0 - np.power(np.dot(norm_vec, i_vec), 2.0))) * norm_vec
        t_vec += (n1/n2)*(i_vec - np.dot(norm_vec, i_vec)*norm_vec)

        t_vec /= np.sqrt(np.sum(np.power(t_vec, 2.0)))
        
        if(going_down):
            t_vec[2] *= -1.0 
            
        step_length_ = step_length(pos[2], ice_layer)

        total_length += step_length_
            
        pos += step_length_ * t_vec
        i_vec = copy.deepcopy(t_vec)
        n_old = n2
       

        if(pos[2] > 0.0 and not(going_down)):
            break

    return pos, total_length
        
def main(homo, model_top, model_bottom):
    
    nrays = 200
    
    # starting points and direction
    x0s = [0.0, 0.0, 0.0]

    nsteps = 50000
    ice_layer = -300.0 # made the layer just after the firn
    
    last_points_x, last_points_y, total_lengths = [], [], []
    args = []
    dir_steps_0 = np.linspace(-np.deg2rad(90.0), -np.deg2rad(45.0), nrays) # zenith
    for dir_0_0 in dir_steps_0:
        dir_steps_1 = np.linspace(np.deg2rad(0.0), np.deg2rad(90.0), int(nrays * np.deg2rad(90.0) * np.cos(dir_0_0))) # azimuth    
        for dir_0_1 in dir_steps_1:
            dir0s = [dir_0_0, dir_0_1]        
            args += [(dir0s, x0s, nsteps, ice_layer, homo, model_top, model_bottom)]

            
    with Pool(processes=4) as pool:
        L = pool.starmap(throw, args)
        
        for i in range(len(L)):
            pos, total_length = L[i]
            last_points_x += [pos[0] * 2.0]
            last_points_y += [pos[1] * 2.0]
            total_lengths += [total_length * 2.0]

    return last_points_x, last_points_y
                                        
if __name__ == "__main__":
    freeze_support()

    nthrows = 1
    model_top = 1.4 #np.random.uniform(1.3, 1.4, nthrows)
    model_bottom = 1.78 # np.random.normal(1.78, 0.03, nthrows)
    
    last_points_x_homo, last_points_y_homo = main(homo = True, model_top = model_top, model_bottom = model_bottom)
    last_points_x, last_points_y = main(homo = False, model_top = model_top, model_bottom = model_bottom)

    np.savez("./ray_tracing_results/data_focus_3d_homo_raytracer_posese_model.npy", x = last_points_x_homo, y = last_points_y_homo)
    np.savez("./ray_tracing_results/data_focus_3d_inhomo_raytracer_posese_model.npy", x = last_points_x, y = last_points_y)
        
    fig = plt.figure()
    ax = fig.add_subplot()
    circle = plt.Circle((0, 0), 244.0, color='red', fill=False)
    ax.add_patch(circle)
    ax.scatter(last_points_x_homo, last_points_y_homo)
    ax.set_aspect('equal')

    fig = plt.figure()
    ax = fig.add_subplot()
    circle = plt.Circle((0, 0), 244.0, color='red', fill=False)
    ax.add_patch(circle)
    ax.scatter(last_points_x, last_points_y)
    ax.set_aspect('equal')
    
    plt.show()
