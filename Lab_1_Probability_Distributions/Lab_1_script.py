import numpy as np
import argparse
import matplotlib.pyplot as plt

"""
Task 1 pdf function
"""
def pdf(x, mean=0.0, std=1.0):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)


"""
1.c run from cmd
uncomment below to enable 
"""
def run_from_console():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mean', type=float)
        parser.add_argument('-s', '--std', type=float)
        args = parser.parse_args()

        print(f"Mean: {args.mean}, Std: {args.std} used to calculate pdf:")
        x = np.linspace(-6, 6, 100)  # no x was specified for the cmd call so I use the one from T2
        print(pdf(x, args.mean, args.std))
    except Exception as e:
        print(e)


run_from_console()

"""
2.a
"""
def plot_pdf(x, mean, std, filename):
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f'PDF with Mean: {mean} and Std: {std}')
    plt.plot(x, pdf(x, mean, std))
    plt.savefig(f"./output/{filename}.png")
    plt.show()

"""
3.a
"""
def draw_normal():
    return np.random.normal(10.0, 4.0, 1000000)

"""
3.b
"""
def draw_uniform():
    return np.random.uniform(0, 20, 1000000)

"""
3.d
"""
def plot_histograms(normal_sample, uniform_sample, filename):
    plt.hist(normal_sample, bins=100, density=True, alpha=0.5, label='Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.axvline(x=np.mean(normal_sample), color='red', linestyle='--', label="Mean")
    plt.axvline(x=np.mean(normal_sample) - np.std(normal_sample), color='green', linestyle='--', label="Std")
    plt.axvline(x=np.mean(normal_sample) + np.std(normal_sample), color='green', linestyle='--', label="Std")
    plt.title('Histogram of the Normal Distribution (n=1000000, m=10, std=4')
    plt.legend()
    plt.savefig(f"./output/{filename}_normal.png")
    plt.legend()
    plt.show()

    plt.hist(uniform_sample, bins=100, density=True, color='red', alpha=0.5, label='Uniform Distribution')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.axvline(x=np.mean(uniform_sample), color='red', linestyle='--', label="Mean")
    plt.axvline(x=np.mean(uniform_sample) - np.std(uniform_sample), color='green', linestyle='--', label="Std")
    plt.axvline(x=np.mean(uniform_sample) + np.std(uniform_sample), color='green', linestyle='--', label="Std")
    plt.title('Histogram of the Uniform Distribution (n=1000000, intervall:[0, 20]')
    plt.legend()
    plt.savefig(f"./output/{filename}_uniform.png")
    plt.show()
