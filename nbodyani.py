import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.animation as animation

G = 1
q_n = 100 # number of bodies 
N = 10000
tau = 1e4
dt = tau/float(N-1)
epsilon = 1e-6 #1e-3 är hyffsad 1e-4 ännu bättre

class Body:
    '''Class representing a 2D particle'''
    def __init__(self, x, y, vx, vy, ax, ay, m):

        self.r = np.array([x, y], dtype=float)
        self.v = np.array([vx, vy], dtype=float)
        self.a = np.array([ax, ay], dtype=float)
        self.m = m
   

    def acc(self, other):
        '''
        newtons law of universal gravitation
        and newtons 2nd law gives a = F/m
        '''
        d = (self.r - other.r)

        if not np.all(d):

            tmp = 0

        else:

            tmp = -((G * other.m)/(np.linalg.norm(d + epsilon))**2)*(d/np.linalg.norm(d + epsilon))

        self.a += tmp


    def leapfrog(self):
        '''leapfrog integrator, for solving the movement
        and velocity of a given particle with an acceleration.
        '''
        v_next = self.v + self.a * dt
        r_next = self.r + v_next * dt

        self.r = r_next
        self.v = v_next
        self.a -= self.a

        return r_next

particles = [Body(np.random.uniform(-1e4, 1e4), np.random.uniform(-1e4, 1e4), np.random.uniform(0, 0), np.random.uniform(0, 0), 0, 0, 1e3) for i in range(q_n)] #1e3 pos

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax = plt.axes(xlim=(-100000,100000), ylim=(-100000, 100000))
line, = ax.plot([], [], '.', alpha=1, markersize=10)

def init():
    line.set_data([], [])
    return line,


def animate(i):

    pairs = combinations(range(q_n), 2)
    
    for j, k in pairs:
        
        Body.acc(particles[j], particles[k])
    
    x_position = []
    y_position = []

    for n in range(q_n):

        tmppos = Body.leapfrog(particles[n])
        '''xdata[n][i] = tmp1[0]
        ydata[n][i] = tmp1[1]
        vxdata[n][i] = tmp2[0]
        vydata[n][i] = tmp2[1]'''
        x_position.append(tmppos[0])
        y_position.append(tmppos[1])

        

    line.set_data(x_position, y_position)
    
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=5, blit=True)
anim.save('fuckoff.mp4', writer=writer, dpi=200)
       