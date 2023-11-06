import sys
import time
import numpy as np
from numba import jit, prange

np.random.seed(42)

@jit(nopython=True, parallel=True)
def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr

def timeInfo(nsteps,Ts,runtime,nmax):
    version = "Numba"
    filename = "outputs/timeInfo.csv"
    FileOut = open(filename,"a")
    print("{:d}x{:d}, {:d}, {:5.3f}, {:8.6f}, {}".format(nmax,nmax, nsteps, Ts, runtime, version),file=FileOut)
    FileOut.close()
    
@jit(nopython=True)
def one_energy(arr, ix, iy, nmax):
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    return en

@jit(nopython=True, parallel=True)
def all_energy(arr, nmax):
    enall = 0.0
    for i in prange(nmax):
        for j in prange(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall

@jit(nopython=True)
def get_order(arr,nmax):
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)

    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()

@jit(nopython=True)
def MC_step(arr,Ts,nmax,xran,yran,aran):
    accept = 0
    
    for i in prange(nmax):
        for j in prange(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)

def main(program, nsteps, nmax, temp):
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        scale = 0.1+temp
        xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
        yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
        aran = np.random.normal(scale=scale, size=(nmax,nmax))
        
        ratio[it] = MC_step(lattice,temp,nmax, xran, yran, aran)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    timeInfo(nsteps, temp, runtime, nmax)

if __name__ == '__main__':
    if int(len(sys.argv)) == 4:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE>".format(sys.argv[0]))
