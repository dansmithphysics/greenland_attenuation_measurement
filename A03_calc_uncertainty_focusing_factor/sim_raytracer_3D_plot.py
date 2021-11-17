import numpy as np
import matplotlib.pyplot as plt

homo_data = np.load("./ray_tracing_results/data_focus_3d_homo_raytracer_posese_model.npy.npz")
inhomo_data = np.load("./ray_tracing_results/data_focus_3d_inhomo_raytracer_posese_model.npy.npz")

homo_x = homo_data['x']
homo_y = homo_data['y']
inhomo_x = inhomo_data['x']
inhomo_y = inhomo_data['y']

n_homo = np.sum(np.sqrt(np.square(homo_x) + np.square(homo_y)) < 400.0)
n_inhomo = np.sum(np.sqrt(np.square(inhomo_x) + np.square(inhomo_y)) < 400.0)

print(n_homo, n_inhomo, np.sqrt(n_inhomo / n_homo))

h_homo, xedges, yedges, thing = plt.hist2d(homo_x,
                                           homo_y,
                                           range = ((0.0, 600.0), (0.0, 600.0)),
                                           bins = (10, 10))
h_inhomo, xedges, yedges, thing = plt.hist2d(inhomo_x,
                                             inhomo_y,
                                             range = ((0.0, 600.0), (0.0, 600.0)),
                                             bins = (10, 10))

plt.close()

plt.figure()
plt.imshow(np.sqrt(h_inhomo / h_homo), vmin = 0.0, vmax = 2.0)
plt.colorbar()
plt.show()
