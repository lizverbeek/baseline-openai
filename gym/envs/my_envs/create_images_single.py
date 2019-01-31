import numpy as np

m = 100
n_images = 1000

def sample_target(m):
    """
    Sample a random (x, y) coordinate from the m x m grid
    """
    m_range = np.arange(0, m)
    x = np.random.choice(m_range)
    y = np.random.choice(m_range)
    target = (x, y)
    return target

def color_radius(radius, color, target, image):
    """
    Set all the pixels in the given radius around the target to the color
    """
    x_start = np.min([np.max([0, target[0]-radius]), m])
    x_end = np.min([np.max([0, target[0]+radius]), m])
    y_start = np.min([np.max([0, target[1]-radius]), m])
    y_end = np.min([np.max([0, target[1]+radius]), m])
    image[x_start:x_end, y_start:y_end] = color
    return image

def color_image(image, target):
    """
    Change the pixel values in the image by smoothly interpolating between maximum and minimum colors
    """
    cold_radius = 100
    cold_color = 0.
    warm_radius = 1
    warm_color = 5.
    n_separations = 100
    
    radius_steps = (cold_radius - warm_radius)/n_separations
    color_steps = (cold_color + warm_color)/n_separations
    
    for i in range(n_separations):
        i_radius = int(cold_radius - (i+1) * radius_steps)
        i_color = cold_color + i * color_steps
        image = color_radius(i_radius, i_color, target, image)
    return image

def make_images():
    training_images = np.zeros((int(0.8*n_images), m, m))
    test_images = np.zeros((int(0.2*n_images), m, m))
    for j in range(n_images):
        image = np.zeros((m, m))
        target = sample_target(m)
        image = color_image(image, target)
        if j < int(0.8*n_images):
            training_images[j] = image
        else:
            i = j - int(0.8*n_images)
            test_images[i] = image

    training_images.dump("training_images.npy")
    test_images.dump("test_images.npy")
