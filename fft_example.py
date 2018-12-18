from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np

x = np.zeros(500)
x[100:150] = 1

X = fftpack.fft(x)

freq = fftpack.fftfreq(len(x))
freq = fftpack.fftshift(freq)

f, (ax0, ax1) = plt.subplots(2, 1, sharex=False)

ax0.plot(x)
ax0.set_ylim(-0.1, 1.1)

ax1.plot(freq, fftpack.fftshift(np.abs(X)))
# ax1.plot(np.abs(X))
ax1.set_ylim(-5, 55)

# plt.figure()
# plt.plot(freq, fftpack.fftshift(np.abs(X)))
