import matplotlib.pyplot as plt


def draw_map(data):
    plt.rcParams["figure.figsize"] = (0.75 * 6, 0.75 * 4)
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.show()


def draw_picture(data):
    plt.rcParams["figure.figsize"] = (0.75 * 6, 0.75 * 4)
    plt.imshow(data, cmap='summer')
    plt.show()
