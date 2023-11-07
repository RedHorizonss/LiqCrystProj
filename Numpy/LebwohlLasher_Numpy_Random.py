import sys
import time
import numpy as np

np.random.seed(42)

def initdat(nmax):
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr

def timeInfo(nsteps,Ts,runtime,nmax, plateu):
    filename = "Numpy/timeInfo_Random.csv"
    FileOut = open(filename,"a")
    print("{:d}x{:d}, {:d}, {:5.3f}, {:8.6f}, {:d}".format(nmax,nmax, nsteps, Ts, runtime, plateu),file=FileOut)
    FileOut.close()

def all_energy(arr,nmax):
    i, j = np.meshgrid(np.arange(nmax), np.arange(nmax))
    results = one_energy(arr,i,j,nmax)
    enall = np.sum(results)

    return enall

###I do not know how to vectorise this code
def get_order(arr,nmax):
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)

    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
                    
    Qab /= (2 * nmax * nmax)
    eigenvalues, _ = np.linalg.eig(Qab)
    return eigenvalues.max()

def one_energy(arr,ix,iy,nmax):
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang_all = arr[ix, iy] - np.array([arr[ixp, iy], arr[ixm, iy], arr[ix, iyp], arr[ix, iym]])

    en_cont = 0.5*(1.0-3.0*np.cos(ang_all)**2)

    en = np.sum(en_cont)
    
    return en

def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts  # 'scale' determines the width of angle changes
    accept = 0  # Initialize the acceptance counter

    # Generate randomized ix and iy coordinates for each lattice site
    random_ix = np.random.randint(0, high=nmax, size=(nmax, nmax))
    random_iy = np.random.randint(0, high=nmax, size=(nmax, nmax))
    
    # Generate random angles for all lattice sites at once
    random_angles = np.random.normal(scale=scale, size=(nmax, nmax))

    for ix, iy, ang in zip(random_ix.ravel(),random_iy.ravel(), random_angles.ravel()):
        arr_to_change = arr.copy()
        arr_to_change[ix, iy] += ang
        
        # Calculate the energy of the current lattice configuration at (ix, iy)
        en0 = one_energy(arr, ix, iy, nmax)
        en1 = one_energy(arr_to_change, ix, iy, nmax)

        if en1 <= en0 or np.exp(-(en1 - en0) / Ts) >= np.random.uniform(0.0, 1.0):
            # If the new configuration has lower energy, accept the change
            accept += 1
            arr[ix, iy] = arr_to_change[ix, iy]

    # Calculate and return the acceptance ratio for the current MC step
    return accept / (nmax * nmax)

def main(program, nsteps, nmax, temp):
    lattice = initdat(nmax)

    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)

    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice,nmax)
    
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    indices = np.argwhere(order > 0.9)

    if indices.size > 0:
        # Get the index of the first instance
        plateu = indices[0][0]
        print("Index of the first number > 0.9:", plateu)
        print("Value at that index:", order[plateu])
    else:
        plateu = 0
    
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    timeInfo(nsteps, temp, runtime, nmax, plateu)

if __name__ == '__main__':
    if int(len(sys.argv)) == 4:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE>".format(sys.argv[0]))

