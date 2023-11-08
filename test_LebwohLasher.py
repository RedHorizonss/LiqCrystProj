import pytest
import Original.LebwohlLasher as LebwohlLasher #Can be chahnged to test Numpy codes and MPI codes

#TODO: Add a test for the actual values of the system

@pytest.fixture(params=[(5, LebwohlLasher.initdat(5)),
                        (10, LebwohlLasher.initdat(10)),
                        (20, LebwohlLasher.initdat(20))],
                ids=["nmax=5", "nmax=10", "nmax=20"]) #We can identify if which tests we are running

def nmax_and_lattice(request):
    return request.param

def test_initdat(nmax_and_lattice):
    nmax, lattice = nmax_and_lattice
    #checks if its producing the right shape
    assert lattice.shape == (nmax, nmax)

def test_one_energy(nmax_and_lattice):
    nmax, lattice = nmax_and_lattice
    #picks the middle point of nmax
    ix, iy = nmax // 2, nmax // 2
    energy = LebwohlLasher.one_energy(lattice, ix, iy, nmax)
    #checks if its the right type
    assert isinstance(energy, float)

def test_all_energy(nmax_and_lattice):
    nmax, lattice = nmax_and_lattice
    energy = LebwohlLasher.all_energy(lattice, nmax)
    #checks if its the right type
    assert isinstance(energy, float)

def test_get_order(nmax_and_lattice):
    nmax, lattice = nmax_and_lattice
    order = LebwohlLasher.get_order(lattice, nmax)
    #checks if its the right type
    assert isinstance(order, float)

def test_MC_step(nmax_and_lattice):
    nmax, lattice = nmax_and_lattice
    Ts = 0.5  # Reduced temperature
    acceptance_ratio = LebwohlLasher.MC_step(lattice, Ts, nmax)
    #checks if its in the right region i did put equals to because 
    #complete order and complete disorder doesnt make physical sense
    assert 0.0 < acceptance_ratio < 1.0
    
