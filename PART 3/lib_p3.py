import math
import numpy as np
import scipy
from scipy import sparse
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

def matrix(N, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4):
		A = np.zeros((2*N, 2*N))
		for i in range(N):
			A[i, i] = a1
			A[i, i+N] = a2
			A[i+N, i] = a3
			A[i+N, i+N] = a4
		for i in range(N-1):
			A[i, i+1] = b1
			A[i, i+N+1] = b2
			A[i+N, i+1] = b3
			A[i+N, i+N+1] = b4
			A[i+1, i] = c1
			A[i+1, i+N] = c2
			A[i+N+1, i] = c3
			A[i+N+1, i+N] = c4
		return A

def crank_nicolson_mod_alpha_omega(init_cond_Br, init_cond_Bphi, M, N, T, L, eta_T, q, Omega, alpha):
	x0, xL = -L, L
	dx = (xL - x0)/(M-1)
	t0, tF = 0, T 
	dt = (tF - t0)/(N-1)

	xspan = np.linspace(x0, xL, M)
	tspan = np.linspace(t0, tF, N)

	# Coefficients for the matrix A and B
	nu = eta_T*dt/(2*dx**2)
	mu = alpha*dt/(2*dx)

	A = matrix(len(xspan), 1+2*nu, -mu, q*Omega*dt/2, 1+2*nu, -nu, mu, 0, -nu, -nu, 0, 0, -nu)
	B = matrix(len(xspan), 1-2*nu, mu, -q*Omega*dt/2, 1-2*nu, nu, -mu, 0, nu, nu, 0, 0, nu)

	# Initialize temperature array
	U = np.zeros((2*M, N))

	# Initial condition
	for i in range(M):
		U[i, 0] = init_cond_Br[i]
		U[M+i, 0] = init_cond_Bphi[i]

	A_inv = np.linalg.inv(A)

	for j in range(1, N):
		U[:, j] = np.dot(A_inv, np.dot(B, U[:, j - 1]))
		U[0, j] = init_cond_Br[0]
		U[M-1, j] = init_cond_Br[-1]
		U[M, j] = init_cond_Bphi[0]
		U[-1, j] = init_cond_Bphi[-1]

	return (U, tspan, xspan)


def crank_nicolson_mod_alpha2_omega(init_cond_Br, init_cond_Bphi, M, N, T, L, eta_T, q, Omega, alpha):
	x0, xL = -L, L
	dx = (xL - x0)/(M-1)
	t0, tF = 0, T 
	dt = (tF - t0)/(N-1)

	xspan = np.linspace(x0, xL, M)
	tspan = np.linspace(t0, tF, N)

	# Coefficients for the matrix A and B
	nu = eta_T*dt/(2*dx**2)
	mu = alpha*dt/(2*dx)

	A = matrix(len(xspan), 1+2*nu, -mu, mu, 1+2*nu, -nu, mu, -mu, -nu, -nu, 0, 0, -nu)
	B = matrix(len(xspan), 1-2*nu, mu, -mu, 1-2*nu, nu, -mu, mu, nu, nu, 0, 0, nu)

	# Initialize temperature array
	U = np.zeros((2*M, N))

	# Initial condition
	for i in range(M):
		U[i, 0] = init_cond_Br[i]
		U[M+i, 0] = init_cond_Bphi[i]

	A_inv = np.linalg.inv(A)

	for j in range(1, N):
		U[:, j] = np.dot(A_inv, np.dot(B, U[:, j - 1]))
		U[0, j] = init_cond_Br[0]
		U[M-1, j] = init_cond_Br[-1]
		U[M, j] = init_cond_Bphi[0]
		U[-1, j] = init_cond_Bphi[-1]

	return (U, tspan, xspan)


def plot_B(U, tspan, xspan, T, L, name):

	# 2D Heat Map
	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	im = axs[0].imshow(U, extent=[0, T, -L, L], origin='lower', aspect='auto', cmap='magma')
	# Add contours
	#contours = axs[0].contour(U, extent=[0, T, 0, L], colors='white', linestyles='dashed')
	#axs[0].clabel(contours, inline=True, fontsize=8)
	fig.colorbar(im, ax=axs[0], label=f'$B_{str(name)}$')
	axs[0].set_xlabel('T')
	axs[0].set_ylabel('Z')
	axs[0].set_title(f'Diffusion equation solution for $B_{str(name)}$(Z,T)')


	# 3D Surface Plot
	T, X = np.meshgrid(tspan, xspan)

	for spine in axs[1].spines.values():
		spine.set_visible(False)
	axs[1].set(xticklabels=[], yticklabels=[])
	axs[1].xaxis.set_ticks_position('none')
	axs[1].yaxis.set_ticks_position('none')


	ax2 = fig.add_subplot(122, projection='3d')
	surf = ax2.plot_surface(X, T, U, cmap='magma')

	ax2.set_xlabel('Z')
	ax2.set_ylabel('T')
	ax2.set_zlabel(f'$B_{str(name)}$')
	ax2.set_title(f'Surface plot of $B_{str(name)}$(Z,T)')
	fig.colorbar(surf, ax=ax2)

	plt.show()

def plot_B_log(U, tspan, xspan, T, L, name):
	# Apply symlog transformation to U
	U_log = np.sign(U) * np.log1p(np.abs(U))

	# 2D Heat Map
	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	im = axs[0].imshow(U_log, extent=[0, T, -L, L], origin='lower', aspect='auto', cmap='magma')
	fig.colorbar(im, ax=axs[0], label=f'symlog($B_{str(name)}$)')
	axs[0].set_xlabel('T')
	axs[0].set_ylabel('Z')
	axs[0].set_title(f'Diffusion equation solution for symlog($B_{str(name)}$(Z,T))')

	# 3D Surface Plot
	T, X = np.meshgrid(tspan, xspan)

	for spine in axs[1].spines.values():
		spine.set_visible(False)
	axs[1].set(xticklabels=[], yticklabels=[])
	axs[1].xaxis.set_ticks_position('none')
	axs[1].yaxis.set_ticks_position('none')

	ax2 = fig.add_subplot(122, projection='3d')
	surf = ax2.plot_surface(X, T, U_log, cmap='magma')

	ax2.set_xlabel('Z')
	ax2.set_ylabel('T')
	ax2.set_zlabel(f'symlog($B_{str(name)}$)')
	ax2.set_title(f'Surface plot of symlog($B_{str(name)}$(Z,T))')
	fig.colorbar(surf, ax=ax2)

	plt.show()


def plot_B_animation(U, tspan, xspan, T, L, name, num):
	fac = 4
	fig, ax = plt.subplots()
	ax.set_xlim(-L, L)
	ax.set_xlabel('Z')
	ax.set_ylabel(f'$B_{str(name)}$(Z,T)')
	ax.set_title(f'$B_{str(name)}$(Z,T)')
	ax.grid(True)

	line, = ax.plot([], [], lw=2)
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

	def init():
		line.set_data([], [])
		time_text.set_text('')
		return line, time_text,

	def animate(i):
		line.set_data(xspan, U[:, i*fac])
		val = np.ceil(np.max(np.abs(U[:, i*fac])) / 5) * 5
		ax.set_ylim(-1*val, val)  # Update y-axis limits here
		time_text.set_text('Time = %.1f' % tspan[i*fac])
		return line, time_text,

	ani = animation.FuncAnimation(fig, animate, frames=int(len(tspan)/fac), init_func=init, blit=True)
	ani.save(name+num+'.mp4', writer=FFMpegWriter(fps=20))  # Use FFMpegWriter and save as MP4

def plot_pB(U, tspan, xspan, T, L, name):

	# 2D Heat Map
	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	im = axs[0].imshow(U, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='magma')
	# Add contours
	#contours = axs[0].contour(U, extent=[0, T, 0, L], colors='white', linestyles='dashed')
	#axs[0].clabel(contours, inline=True, fontsize=8)
	fig.colorbar(im, ax=axs[0], label=f'$B_{str(name)}$')
	axs[0].set_xlabel('T')
	axs[0].set_ylabel('Z')
	axs[0].set_title('Diffusion equation solution for $p_{B}$(Z,T)')


	# 3D Surface Plot
	T, X = np.meshgrid(tspan, xspan)

	for spine in axs[1].spines.values():
		spine.set_visible(False)
	axs[1].set(xticklabels=[], yticklabels=[])
	axs[1].xaxis.set_ticks_position('none')
	axs[1].yaxis.set_ticks_position('none')


	ax2 = fig.add_subplot(122, projection='3d')
	surf = ax2.plot_surface(X, T, U, cmap='magma')

	ax2.set_xlabel('Z')
	ax2.set_ylabel('T')
	ax2.set_zlabel('$p_{B}$')
	ax2.set_title('Surface plot of $p_{B}$(Z,T)')
	fig.colorbar(surf, ax=ax2)

	plt.show()

def plot_pB_animation(U, tspan, xspan, T, L, name, num):
	fac = 4
	fig, ax = plt.subplots()
	ax.set_xlim(-L, L)
	ax.set_ylim(-100, 100)
	ax.set_xlabel('Z')
	ax.set_ylabel('Pitch Angle $p_{B}$')
	ax.set_title('$p_{B}$(Z,T)')
	ax.grid(True)

	line, = ax.plot([], [], lw=2)
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

	def init():
		line.set_data([], [])
		time_text.set_text('')
		return line, time_text,

	def animate(i):
		line.set_data(xspan, U[:, i*fac])
		time_text.set_text('Time = %.1f' % tspan[i*fac])
		return line, time_text,

	ani = animation.FuncAnimation(fig, animate, frames=int(len(tspan)/fac), init_func=init, blit=True)
	ani.save(name+num+'.mp4', writer=FFMpegWriter(fps=20))  # Use FFMpegWriter and save as MP4


def find_local_maxima(x, y):
	n = int(len(x)/3)
	#n=1000
	x = x[n:]
	y = y[n:]
	x_maxima = []
	y_maxima = []
	for i in range(1, len(y) - 1):
		if y[i] > y[i - 1] and y[i] > y[i + 1]:
			x_maxima.append(x[i])
			y_maxima.append(y[i])
	return np.array(x_maxima), np.array(y_maxima)

def get_decay_rate(x, y, do_print=True):

	x, y = find_local_maxima(x, y)

	y = np.log10(y)  # Logarithm applied
	slope, intercept = np.polyfit(x, y, 1)

	if slope < 0 and do_print:
		print(r"Decay rate:", format(-1*slope, ".3e"))
	elif slope >= 0 and do_print:
		print(r"Growth rate:", format(slope, ".3e"))

	return slope

def bisection(f, a, b, eps=1e-6):
	counter = 1
	COUNT = []
	VAL = []
	if f(a)*f(b) == 0.0:
		if f(a)==0.0:
			return a
		else:
			return b

	c = (a+b)/2
	while np.abs(f(c)) > eps: # checking if the accuracy is achieved

		c = (a+b)/2
		if (f(a)*f(c)) <= 0.0: # Check if the root is properly bracketted
			b = c
		else:
			a = c
		if counter > 100:
			print('Maximum iterations reached.')
			break
		counter += 1
		COUNT.append(counter)
		VAL.append(c)

	return c, COUNT, VAL
