import pytest
import numpy as np
from LebwohlLasher_Numba import initdat, one_energy, all_energy, get_order, MC_step

##TODO: add a way to check the last like of the most recent file or maybe just the output.

#Setting the lattice sze as a global number to be used.
# This is because the code doesn't have any error catch statements and can only take a positive number
nmax = 5

# Fixture that sets a fixed random seed for reproducible results
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)

def test_initdat():
    # Test the initialization of the lattice
    arr = initdat(nmax)
    # Check the shape of the resulting lattice
    assert arr.shape == (nmax, nmax)
    # Ensure values are within a valid range
    assert (0 <= arr).all() and (arr <= 2 * np.pi).all() 
    
def test_one_energy():
    # Test the energy calculation for a specific lattice site
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    # Coordinates within the lattice
    ix, iy = 2, 2
    # Calculate the energy at the specified coordinates
    energy = one_energy(arr, ix, iy, nmax)  
    assert energy == -1.0512936529391477  
    
def test_all_energy():
    # Test the total energy calculation for the entire lattice
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    # Calculate the total energy
    total_energy = all_energy(arr, nmax)  
    assert total_energy == -22.709752842971472 
    
def test_get_order():
    # Test the order parameter calculation for the lattice
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    print(arr)
    print(nmax)
    # Calculate the order parameter
    order = get_order(arr, nmax)
    assert order == 0.3238593394231049

def test_MC_step():
    # Test a Monte Carlo step
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    # Set the temperature
    temp = 0.5 
    # Perform a Monte Carlo step and get the acceptance ratio
    acceptance_ratio = MC_step(arr, temp, nmax)  
    #Check if the acceptance ratio is within the valid range [0, 1]
    assert acceptance_ratio == 0.52
