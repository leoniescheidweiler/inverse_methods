import numpy as np
import matplotlib.pyplot as plt


def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x - mu)**2/(sigma**2))


# plot different distributions

N = 100000

uniform = np.random.uniform(0, 1, N)
poisson1 = np.random.poisson(5, N)
poisson2 = np.random.poisson(20, N)
normal = np.random.normal(3, 2, N)

xfine = np.linspace(-10, 10, 1000)
yfine = gauss(xfine, 3, 2)

plt.title("some distributions")
plt.hist(uniform, alpha=.5, density=True, label="uniform")
plt.hist(poisson1, alpha=.5, density=True, label="poisson mu=5")
plt.hist(poisson2, alpha=.5, density=True, label="poisson mu=20")
plt.hist(normal, alpha=.5, density=True, color="red", label="normal mu=3 sigma=2")
plt.plot(xfine, yfine, color="red", label="normal mu=3 sigma=2 exact")
plt.legend()
plt.show()



# show poisson -> gauss

mu_2 = 2
poisson_2 = np.random.poisson(mu_2, N)
xfine_2 = np.linspace(-3, 8, 1000)
yfine_2 = gauss(xfine_2, mu_2, np.sqrt(mu_2))

mu_10 = 5
poisson_10 = np.random.poisson(mu_10, N)
xfine_10 = np.linspace(-2.5, 22.5, 1000)
yfine_10 = gauss(xfine_10, mu_10, np.sqrt(mu_10))

mu_100 = 100
poisson_100 = np.random.poisson(mu_100, N)
xfine_100 = np.linspace(60, 140, 1000)
yfine_100 = gauss(xfine_100, mu_100, np.sqrt(mu_100))

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("poisson with large mu")

ax[0].hist(poisson_2, alpha=.5, density=True, color="red", label="poisson mu=2")
ax[0].plot(xfine_2, yfine_2, color="red", label="normal mu=2, sigma=sqrt(2) exact")
ax[0].set_ylim(0, np.max(yfine_2)*1.75)
ax[0].legend()

ax[1].hist(poisson_10, alpha=.5, density=True, color="red", label="poisson mu=10")
ax[1].plot(xfine_10, yfine_10, color="red", label="normal mu=10, sigma=sqrt(10) exact")
ax[1].set_ylim(0, np.max(yfine_10)*1.5)
ax[1].legend()

ax[2].hist(poisson_100, alpha=.5, density=True, color="red", label="poisson mu=100")
ax[2].plot(xfine_100, yfine_100, color="red", label="normal mu=100, sigma=sqrt(100) exact")
ax[2].set_ylim(0, np.max(yfine_100)*1.5)
ax[2].legend()

plt.show()



# sum of two normal distributions

xfine = np.linspace(0, 10, N)

pdf_combined = gauss(xfine, 3, 1) + gauss(xfine, 6, 2)
pdf_combined /= np.sum(pdf_combined)

yfine = 0.5 * (gauss(xfine, 3, 1) + gauss(xfine, 6, 2))

normal = np.random.choice(xfine, size=N, p=pdf_combined)

plt.title("combined pdf")
plt.hist(normal, density=True, alpha=.5, color="red", label="drawn distribution")
plt.plot(xfine, yfine, color="red", label="exact")
plt.legend()
plt.show()
