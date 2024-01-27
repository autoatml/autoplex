from autoplex.regularization import label_stoichiometry_volume, calculate_hull_3D, get_e_distance_to_hull_3D, get_e_distance_to_hull, get_convex_hull
import numpy as np
import ase
import ase.io
from autoplex.utilities import flatten
from ase.units import GPa
from autoplex.cur import cur_select


def boltz(e, emin, kT):
    return np.exp(-(e-emin)/(kT))


def boltzhist_CUR(atoms,
                  isol_es=None,
                  bolt_frac=0.1, 
                  bolt_max_num=3000,
                  cur_num=100, 
                  kernel_exp=4, 
                  kT=0.3, 
                  energy_label='energy',
                  P=None,
                  descriptor=None
                  ):

    '''
    Select most diverse atoms from list based on chosen algorithm
    Parameters:
        atoms        :: list of ase.Atoms. Flattened if list of lists
        bolt_frac    :: fraction to control flat boltzmann selection number
        bolt_max_num :: max slected number by Boltzhistgram 
        descriptor   :: quippy descriptor string for CUR
        kernel_exp   :: exponent for dot-product SOAP kernel
        kT           :: eV for boltzmann weighting, (Kb x T)
        P            :: list of pressures at which Atoms have been optimisied, unit: GPa
    
    Returns:
        list of (copies of) selected atoms

    '''

    if isinstance(atoms[0], list):
        print('flattening')
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms

    if P is None:
        
        print('[log] pressures not supplied, attempting to use pressure in atoms dict')

        try:
            ps = np.array([at.info['pressure'] for at in fatoms])
        except:
            raise RuntimeError('No pressures, so can\'t Boltzmann weight')
    
    else:
        ps = P

    enthalpies = []

    at_ids = [atom.get_atomic_numbers() for atom in fatoms]
    ener_relative = np.array([(atom.info['energy'] - sum([isol_es[j] for j in at_ids[ct]])) / len(atom) for ct, atom in enumerate(fatoms)])
    for i, at in enumerate(fatoms):
        enthalpy = (ener_relative[i] + at.get_volume() * ps[i] * GPa) / len(at)
        enthalpies.append(enthalpy)
   
    enthalpies = np.array(enthalpies)
    min_H = np.min(enthalpies)
    config_prob = []
    histo = np.histogram(enthalpies)
    for H in enthalpies:
        bin_i = np.searchsorted(histo[1][1:], H, side='right')
        if bin_i == len(histo[1][1:]):
            bin_i = bin_i - 1
        if histo[0][bin_i] > 0.0:
            p = 1.0 / histo[0][bin_i]
        else:
            p = 0.0
        if kT > 0.0:
            p *= np.exp(-(H - min_H) / kT)
        config_prob.append(p)

    select_num = round(bolt_frac * len(fatoms))

    if select_num < bolt_max_num:
        select_num = select_num
    else:
        select_num = bolt_max_num

    config_prob = np.array(config_prob)
    selected_bolt_ats = []
    for _ in range(select_num):
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)  # cumulate prob
        rv = np.random.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        selected_bolt_ats.append(fatoms[config_i])
        # remove from config_prob by converting to list
        config_prob = np.delete(config_prob, config_i)
        # remove from other lists
        del fatoms[config_i]
        enthalpies = np.delete(enthalpies, config_i)

    ## implement CUR
    if cur_num < select_num:
        selected_atoms = cur_select(atoms=selected_bolt_ats, 
                                    selected_descriptor=descriptor,
                                    kernel_exp=kernel_exp, 
                                    select_nums=cur_num, 
                                    stochastic=True)
    else: 
        selected_atoms = selected_bolt_ats

    ase.io.write('boltzhist_CUR.extxyz', selected_atoms, parallel=False)

    return selected_atoms


def convexhull_CUR(atoms,
                   bolt_frac=0.1,
                   bolt_max_num=3000,
                   cur_num=100,
                   kernel_exp=4,
                   kT=0.5,
                   energy_label='REF_energy',
                   descriptor=None,
                   isol_es=None,
                   element_order=None,
                   scheme='linear-hull',
                   ):
    
    '''
    Select most diverse atoms from list based on chosen algorithm
    Parameters:
        atoms        :: list of ase.Atoms. Flattened if list of lists
        bolt_frac    :: fraction to control flat boltzmann selection number
        bolt_max_num :: max slected number by Boltzhistgram 
        descriptor   :: quippy descriptor string for CUR
        kernel_exp   :: exponent for dot-product SOAP kernel
        kT           :: eV for boltzmann weighting, (Kb x T)
    
    Returns:
        list of (copies of) selected atoms

    '''

    if isinstance(atoms[0], list):
        print('flattening')
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms

    if isol_es == None:
        raise KeyError('isol_es must be supplied for convexhull_CUR')

    ## 
    if scheme == 'linear-hull':
        hull, p = get_convex_hull(atoms, energy_name=energy_label)
        des = np.array([get_e_distance_to_hull(hull, 
                                               at, 
                                               energy_name=energy_label)
                                               for at in atoms])

    elif scheme == 'volume-stoichiometry':
        points = label_stoichiometry_volume(atoms, 
                                            isol_es=isol_es, 
                                            e_name=energy_label, 
                                            element_order=element_order)
        hull = calculate_hull_3D(points)

        des = np.array([get_e_distance_to_hull_3D(hull,
                                    at,
                                    isol_es=isol_es,
                                    energy_name=energy_label,
                                    element_order=element_order) 
                                    for at in atoms])
        print('it will be coming soon!')

    histo = np.histogram(des)
    config_prob = []
    min_ec = np.min(des)

    for ec in des:
        bin_i = np.searchsorted(histo[1][1:], ec, side='right')
        if bin_i == len(histo[1][1:]):
            bin_i = bin_i - 1
        if histo[0][bin_i] > 0.0:
            p = 1.0 / histo[0][bin_i]
        else:
            p = 0.0
        if kT > 0.0:
            p *= np.exp(-(ec - min_ec) / kT)
        config_prob.append(p)

    select_num = round(bolt_frac * len(fatoms))

    if select_num < bolt_max_num:
        select_num = select_num
    else:
        select_num = bolt_max_num

    config_prob = np.array(config_prob)
    selected_bolt_ats = []
    for _ in range(select_num):
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)  # cumulate prob
        rv = np.random.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        selected_bolt_ats.append(fatoms[config_i])
        # remove from config_prob by converting to list
        config_prob = np.delete(config_prob, config_i)
        # remove from other lists
        del fatoms[config_i]
        des = np.delete(des, config_i)

    ## implement CUR
    if cur_num < select_num:
        selected_atoms = cur_select(atoms=selected_bolt_ats, 
                                    selected_descriptor=descriptor,
                                    kernel_exp=kernel_exp, 
                                    select_nums=cur_num, 
                                    stochastic=True)
    else: 
        selected_atoms = selected_bolt_ats

    ase.io.write('boltzhist_CUR.extxyz', selected_atoms, parallel=False)

    return selected_atoms

    
    

# atoms=ase.io.read('quip_train.extxyz', index=':')
# convexhull_CUR(atoms,
#                 bolt_frac=0.1,
#                 bolt_max_num=10000,
#                 cur_num=100,
#                 kernel_exp=4,
#                 kT=0.5,
#                 energy_label='energy',
#                 descriptor=None,
#                 isol_es={14:-0.67334161},
#                 element_order=None,
#                 )

# atoms=ase.io.read('test.extxyz')
# print(type(atoms.get_atomic_numbers()[0]))

# descriptor = 'soap l_max=4' + \
#                 ' n_max=4' + \
#                 ' atom_sigma=0.5' + \
#                 ' cutoff=4' + \
#                 ' n_species=1' + \
#                 ' species_Z=14'  + \
#                 ' cutoff_transition_width=0.5' + \
#                 ' average =T' 
    
# atoms=ase.io.read('quip_train.extxyz', index=':')
# boltzhist_CUR(atoms, 
#               bolt_frac=0.1, 
#               bolt_max_num=10000,
#               cur_num=8, 
#               kernel_exp=4, 
#               kT=0.3, 
#               energy_label='REF_energy',
#               P=None,
#               descriptor=descriptor,
#               )