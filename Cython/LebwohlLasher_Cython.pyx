import sys
import time
import datetime
import numpy as np

# cython imports
cimport cython
cimport numpy as np
from libc.math cimport sin, cos, pi, exp

cdef np.ndarray initdat(int nmax):
    cdef np.ndarray[double, ndim=2] arr = np.random.random_sample((nmax,nmax))*2.0*pi
    return arr

cdef float one_energy(np.ndarray[double, ndim=2] arr, int ix, int iy, int nmax):
    cdef float en = 0.0
    cdef int ixp = (ix+1)%nmax 
    cdef int ixm = (ix-1)%nmax 
    cdef int iyp = (iy+1)%nmax 
    cdef int iym = (iy-1)%nmax 

    cdef float ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    return en

cdef float all_energy(np.ndarray[double, ndim=2] arr, int nmax):
    cdef float enall = 0.0
    cdef int i,j

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

cdef float get_order(np.ndarray[double, ndim=2] arr,int nmax):
    cdef int a, b, i, j

    cdef double[:,:] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:,:] delta = np.eye(3, dtype=np.float64)
    cdef double[:,:,:] lab = np.empty((3, nmax, nmax), dtype=np.float64)
    cdef double[:] eigenvalues
    cdef float scalar = (2.0*nmax*nmax)

    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3.0*lab[a,i,j]*lab[b,i,j] - delta[a,b]
            Qab[a,b] /= scalar

    eigenvalues = np.linalg.eigvals(Qab)

    return np.max(eigenvalues)

cdef float MC_step(np.ndarray[double, ndim=2] arr, double Ts,int nmax):
    cdef float scale = 0.1+Ts
    cdef float accept = 0.0

    cdef int[:, :] xran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    cdef int[:, :] yran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    cdef double[:, :]aran = np.random.normal(scale=scale, size=(nmax,nmax))

    cdef int i, j, ix, iy
    cdef float ang, en0, en1, boltz, rand_num

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = exp( -(en1 - en0) / Ts )#
                rand_num = np.random.uniform(0.0, 1.0)

                if boltz >= rand_num:
                    accept += 1
                else:
                    arr[ix,iy] -= ang

    return accept/(nmax*nmax)

def timeInfo(nsteps, Ts, runtime, nmax):
    filename = "timeInfo_Cy.csv"
    with open(filename, "a") as FileOut:
        FileOut.write("{:d}x{:d}, {:d}, {:5.3f}, {:8.6f}\n".format(nmax, nmax, nsteps, Ts, runtime))

def main(program, int nsteps, int nmax, double temp):
    lattice = initdat(nmax)

    cdef:
        double[:] energy = np.zeros(nsteps + 1)
        double[:] ratio = np.zeros(nsteps + 1)
        double[:] order = np.zeros(nsteps + 1)

    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5  # ideal value
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial

    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax, nsteps, temp, order[nsteps - 1], runtime))
    
    # Add the timeInfo function call to log the information
    timeInfo(nsteps, temp, runtime, nmax)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE>".format(sys.argv[0]))