from ase.data import atomic_numbers
from ase.units import Bohr
from ase import Atoms
import pandas as pd
from ase.io import read, write
def attach_bader_charges(atoms:Atoms, ACF_path:str, displacement:float=1e-3)->Atoms:
    """
    Function used to attached the Bader charges to the atoms object.
    Note: The zval is given by the POTCAR from the VASP 6.4

    Args:
        atoms (Atoms): ASE atoms object
        ACF_path (str): Path to the ACF.dat file
        displacement (float): Displacement to test if the positions match

    Returns:
        atoms (Atoms): ASE atoms object with the Bader charges attached
    
    """

    # Define the zval, given by the POTCAR from VASP 6.4 
    zval = {'Na':7,'Fe':14,'O':6,'P':5,'Mn':13,'Co':9,'Ni':16,'Si':4,'S':6}

    # Load ACF file 
    df = pd.read_csv(ACF_path,skiprows=2,skipfooter=4,sep='\s+',header=None,index_col=0,engine='python') # Read the ACF.dat file and skip the first two lines and the last 4 lines
    df.columns = ['X','Y','Z','CHARGE','MIN DIST','ATOMIC VOL'] # Add the header according to ACF.dat
    acf_charge = df['CHARGE'].values # Get the charges from the ACF.dat file
    acf_pos = df[['X','Y','Z']].values # Get the positions from the ACF.dat file

    # Add charges and test if the positions match
    for i, a in enumerate(atoms):
        a.charge = acf_charge[i] -zval[a.symbol] # Add the charge to the atom object
        # Test if the atom positions match
        if displacement is not None:
            norm = np.linalg.norm(a.position - acf_pos[i])
            norm2 = np.linalg.norm(a.position - acf_pos[i] * Bohr)
            assert norm < displacement or norm2 < displacement
    #atoms.set_initial_charge = df['CHARGE'].values # Add the charge to the atoms object
    return atoms

structure = read('POSCAR') # Read the structure (Put in your path)
ACF_path = 'ACF.dat' # Path to the ACF.dat file

structure_bader = structure.copy() # Copy the structure

structure_bader = attach_bader_charges(structure_bader, ACF_path) # Attach the Bader charges

structure.arrays['bader_charge'] = structure_bader.get_initial_charges() # Get the initial charges and add them to the atom object using the array dictionary

write('structure_with_bader.xyz',structure) # Save the structure with the Bader charges NEEDS TO BE xyz format

