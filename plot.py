import matplotlib.pyplot as plt
import numpy as np

def  plot_images(X_true,X_output):
    plt.figure(figsize = (8,12))
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(X_true[i].reshape(28,28), vmin =0, vmax =1, cmap ="gray")
        plt.title("True_number")
        plt.colorbar()

        plt.subplot(5,2,2*i+2)
        plt.imshow(X_output[i].reshape(28,28),vmin = 0, vmax =1, cmap = "gray")
        plt.title("Reconstruction_number")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('picture1')
    plt.close()

def plot_2d_scatter(x,y,test_labels):
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, c = np.argmax(test_labels,1), marker ='.', edgecolor = 'none', cmap = discrete_cmap('jet'))
    plt.colorbar()
    plt.grid()
    plt.savefig('picture2')
    plt.close()

def discrete_cmap(base_cmap =None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,10))
    cmap_name = base.name + str(10)
    return base.from_list(cmap_name,color_list,10)
