import math
import numpy as np
import scipy
from scipy import sparse
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt


#Crank Nicolson method
def crank_nicolson_1d(M, N, eta_T, u_initial, T, L):
	x0, xL = -L, L
	dx = (xL - x0)/(M-1)
	t0, tF = 0, T 
	dt = (tF - t0)/(N-1)

	alpha = eta_T

	a0 = 1 + 2*alpha
	c0 = 1 - 2*alpha

	xspan = np.linspace(x0, xL, M)
	tspan = np.linspace(t0, tF, N)

	# Create the main diagonal for the left-hand side matrix with all elements as a0
	maindiag_a0 = a0*np.ones((1,M))

	# Create the off-diagonal for the left-hand side matrix with all elements as -alpha
	offdiag_a0 = (-alpha)*np.ones((1, M-1))

	# Create the main diagonal for the right-hand side matrix with all elements as c0
	maindiag_c0 = c0*np.ones((1,M))

	# Create the off-diagonal for the right-hand side matrix with all elements as alpha
	offdiag_c0 = alpha*np.ones((1, M-1))

	# Create the left-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	a = maindiag_a0.shape[1]

	# Create a list of the diagonals
	diagonalsA = [maindiag_a0, offdiag_a0, offdiag_a0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	A = sparse.diags(diagonalsA, [0,-1,1], shape=(a,a)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	A[0,1] = (-2)*alpha
	A[M-1,M-2] = (-2)*alpha

	# Create the right-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	c = maindiag_c0.shape[1]

	# Create a list of the diagonals
	diagonalsC = [maindiag_c0, offdiag_c0, offdiag_c0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	Arhs = sparse.diags(diagonalsC, [0,-1,1], shape=(c,c)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	Arhs[0,1] = 2*alpha
	Arhs[M-1,M-2] = 2*alpha

	#nitializes matrix U
	U = np.zeros((M, N))

	#Initial conditions
	U[:,0] = u_initial(xspan)

	#Boundary conditions
	f = np.arange(1, N+1)
	U[0,:] = 0
	f = U[0,:]
	
	g = np.arange(1, N+1)
	U[-1,:] = 0
	g = U[-1,:]
	
	#k = 1
	for k in range(1, N):
		ins = np.zeros((M-2,1)).ravel()
		b1 = np.asarray([f[k], g[k]])
		b1 = np.insert(b1, 1, ins)
		b2 = np.matmul(Arhs, np.array(U[0:M, k-1]))
		b = b1 + b2  # Right hand side
		U[0:M, k] = np.linalg.solve(A,b)  # Solving x=A\b
		#Boundary Condition Reset
		U[0,:] = 0
		U[-1,:] = 0
	
	return (U, tspan, xspan)



def plot_B(U, tspan, xspan, T, L, name):

	# 2D Heat Map
	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	im = axs[0].imshow(U, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='magma')
	# Add contours
	contours = axs[0].contour(U, extent=[0, T, 0, L], colors='white', linestyles='dashed')
	axs[0].clabel(contours, inline=True, fontsize=8)
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


def plot_B_animation(U, tspan, xspan, T, L, name):
	fac = 5
	fig, ax = plt.subplots()
	ax.set_xlim(-L, L)
	ax.set_ylim(0, 2)
	ax.set_xlabel('Z')
	ax.set_ylabel(f'$B_{str(name)}$(Z,T)')
	ax.set_title(f'$B_{str(name)}$(Z,T)')

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
	ani.save(name+'.mp4', writer=FFMpegWriter(fps=60))  # Use FFMpegWriter and save as MP4


def plot_pB(U, tspan, xspan, T, L, name):

	# 2D Heat Map
	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	im = axs[0].imshow(U, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='magma')
	# Add contours
	contours = axs[0].contour(U, extent=[0, T, 0, L], colors='white', linestyles='dashed')
	axs[0].clabel(contours, inline=True, fontsize=8)
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

def plot_pB_animation(U, tspan, xspan, T, L, name):
	fac = 5
	fig, ax = plt.subplots()
	ax.set_xlim(-L, L)
	ax.set_ylim(0, 100)
	ax.set_xlabel('Z')
	ax.set_ylabel('Pitch Angle $p_{B}$')
	ax.set_title('$p_{B}$(Z,T)')

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
	ani.save(name+'.mp4', writer=FFMpegWriter(fps=60))  # Use FFMpegWriter and save as MP4

