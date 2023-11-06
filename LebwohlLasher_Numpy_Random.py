import sys
import time
import numpy as np
import datetime

np.random.seed(42)

def initdat(nmax):
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr

def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "outputs/LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.3f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()

def timeInfo(nsteps,Ts,runtime,nmax):
    version = "Numpy_Random"
    filename = "outputs/timeInfo.csv"
    FileOut = open(filename,"a")
    print("{:d}x{:d}, {:d}, {:5.3f}, {:8.6f}, {}".format(nmax,nmax, nsteps, Ts, runtime, version),file=FileOut)
    FileOut.close()

def one_energy(arr,ix,iy,nmax):
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang_all = arr[ix, iy] - np.array([arr[ixp, iy], arr[ixm, iy], arr[ix, iyp], arr[ix, iym]])

    en_cont = 0.5*(1.0-3.0*np.cos(ang_all)**2)

    en = np.sum(en_cont)
    
    return en

def all_energy(arr,nmax):
    i, j = np.meshgrid(np.arange(nmax), np.arange(nmax))
    results = one_energy(arr,i,j,nmax)
    enall = np.sum(results)

    return enall

###I do not know how to vectorise this code specifically?? Everytime I do I get a different number from the original code
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

def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts  # 'scale' determines the width of angle changes
    accept = 0  # Initialize the acceptance counter

    # Create a 1D array with all lattice site indices in a serialized order
    site_indices = np.arange(nmax * nmax)

    # Shuffle the site_indices to visit them in a random order
    np.random.shuffle(site_indices)

    for site_index in site_indices:
        # Convert the 1D index back to 2D lattice coordinates
        ix = site_index // nmax
        iy = site_index % nmax

        # Generate a random angle for this site
        ang = np.random.normal(scale=scale)

        # Calculate the energy of the current lattice configuration at (ix, iy)
        en0 = one_energy(arr, ix, iy, nmax)

        # Attempt to change the lattice configuration by adding the random angle
        arr[ix, iy] += ang

        # Calculate the energy of the new lattice configuration at (ix, iy)
        en1 = one_energy(arr, ix, iy, nmax)

        if en1 <= en0:
            # If the new configuration has lower energy, accept the change
            accept += 1
        else:
            # Now apply the Monte Carlo test - compare
            # exp(-(E_new - E_old) / Ts) >= rand(0, 1)
            boltz = np.exp(-(en1 - en0) / Ts)

            if boltz >= np.random.uniform(0.0, 1.0):
                # If the Boltzmann factor is greater than or equal to a random number, accept the change
                accept += 1
            else:
                # If the change is not accepted, revert the lattice configuration
                arr[ix, iy] -= ang

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

    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
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

