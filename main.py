import numpy as np
import math
import matplotlib.pyplot as plt


def denoise_image(filename, burn_in_steps, total_samples, logfile):
    print("Entre a denoise")
    posterior = get_posterior(filename, burn_in_steps,
                              total_samples, logfile=logfile)
    denoised = np.zeros(posterior.shape, dtype=np.float64)
    denoised[posterior > 0.5] = 1
    print("FIN metodo denoise_image")
    return denoised[1:-1, 1:-1]


def load_image(filename):
    my_img = plt.imread(filename)
    img_gray = np.dot(my_img[..., :3], [0.2989, 0.5870, 0.1140])
    img_gray = np.where(img_gray > 0.5, 1, -1)
    img_padded = np.zeros([img_gray.shape[0] + 2, img_gray.shape[1] + 2])
    img_padded[1:-1, 1:-1] = img_gray
    return img_padded


def get_posterior(filename, burn_in_steps, total_samples, logfile):
    X = load_image(filename)  # Imagen es un array de ceros y unos
    # Un array como la imagen, pero lleno de ceros.
    posterior = np.zeros(X.shape)
    print(X.shape)
    # Y es array como la imagen, pero lleno aleatoriamente con 1 o -1
    Y = np.random.choice([1, -1], size=X.shape)
    energy_list = list()
    for step in range(burn_in_steps + total_samples):
        print("Iteracion"+str(step))
        # itera de 1 a filas*columnas (de la imagen) - 1
        for i in range(1, Y.shape[0]-1):  # itera de 1 a #filas
            for j in range(1, Y.shape[1]-1):  # itera de 1 a #columnas
                y = sample_y(i, j, Y, X)  # ij es un pixel
                Y[i, j] = y
                if y == 1 and step >= burn_in_steps:
                    posterior[i, j] += 1
        energy = -np.sum(np.multiply(Y, X))*ITA-(np.sum(np.multiply(
            Y[:-1], Y[1:]))+np.sum(np.multiply(Y[:, :-1], Y[:, 1:])))*BETA
        if step < burn_in_steps:
            energy_list.append(str(step) + "\t" + str(energy) + "\tB")
        else:
            energy_list.append(str(step) + "\t" + str(energy) + "\tS")
    posterior = posterior / total_samples

    file = open(logfile, 'w')
    print("Abri logfile")
    for element in energy_list:
        file.writelines(element)
        file.write('\n')
    file.close()
    return posterior


def sample_y(i, j, Y, X):
    # markov_blanket es (115,230,100,90,pixel)
    markov_blanket = [Y[i - 1, j], Y[i, j - 1],
                      Y[i, j + 1], Y[i + 1, j], X[i, j]]
    w = ITA * markov_blanket[-1] + BETA * sum(markov_blanket[:4])
    prob = 1 / (1 + math.exp(-2*w))  # 0.63
    # print(prob)
    return (np.random.rand() < prob) * 2 - 1


def plot_energy(filename):
    x = np.genfromtxt(filename, dtype=None, encoding='utf8')
    its, energies, phases = zip(*x)
    its = np.asarray(its)
    energies = np.asarray(energies)
    phases = np.asarray(phases)
    burn_mask = (phases == 'B')
    samp_mask = (phases == 'S')
    assert np.sum(burn_mask) + np.sum(samp_mask) == len(x), 'Found bad phase'
    its_burn, energies_burn = its[burn_mask], energies[burn_mask]
    its_samp, energies_samp = its[samp_mask], energies[samp_mask]
    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_samp, energies_samp, 'b')
    plt.title("energy")
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.legend([p1, p2], ['burn in', 'sampling'])
    plt.savefig('%s.png' % filename)
    plt.close()


def save_image(denoised_image):
    plt.imshow(denoised_image, cmap='gray')
    plt.title("denoised image")
    plt.savefig('denoise_image.png')
    plt.close()


if __name__ == '__main__':
    ITA = 1
    BETA = 1
    total_samples = 10
    burn_in_steps = 1
    logfile = "logfile"
    denoised_img = denoise_image(
        "pericos_sucios.png", burn_in_steps=burn_in_steps, total_samples=total_samples, logfile=logfile)
    plot_energy(logfile)
    save_image(denoised_img)
