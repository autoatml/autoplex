# functions for automatic regularization and weighting of training data

from ase.io import read, write
import numpy as np
import traceback
from argparse import ArgumentError
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def set_sigma(atoms, etup, scheme='linear-hull',
              energy_name='REF_energy', force_name='REF_forces', virial_name='REF_virial', isol_es={},
              element_order=None):

    '''Handles automatic regularisation based on distance to convex hull, amongst other things
    Need to make sure this works for multi-stoichiometry systems

    Parameters:
        atoms :: (list of ase.Atoms) list of atoms objects to set reg. for. Usually fitting database
        etup :: (list of tuples) list of tuples of (min, max) values for energy, force, virial sigmas
        scheme :: (str) scheme to use for regularisation. Options are: linear_hull, volume-stoichiometry
        energy_name :: (str) name of energy key in atoms.info
        force_name :: (str) name of force key in atoms.arrays
        virial_name :: (str) name of virial key in atoms.info
        isol_es :: (dict) dictionary of isolated energies for each atomic number. Only needed for volume-x scheme
                    e.g. {14: '-163.0', 8:'-75.0'} for SiO2
    
    e.g. etup = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]
            [(emin, emax), (semin, semax), (sfmin, sfmax), (ssvmin, svmax)]'''


    sigs = [[], [], []]
    atoms_modi = []

    for at in atoms:
        if at.info['config_type'] == 'isolated_atom':
            # isol_es 
            at.info['energy_sigma'] = 0.0001
            at.calc = None  #  TODO side-effect alert
            try:
                del at.arrays[force_name]
                del at.info[virial_name]
            except:
                pass
            atoms_modi.append(at)
            continue

        if at.info['config_type'] == 'dimer':
            at.info['energy_sigma'] = 0.1
            at.info['force_sigma'] = 0.5
            try:
                del at.info[virial_name]
            except:
                pass
            atoms_modi.append(at)
            continue
    
    if scheme == 'linear-hull':
        # Use this one for a simple single-composition system. 
        # Makes a 2D convex hull of volume vs. energy and calculates distance to it
        print('Regularising with linear hull')
        hull, p = get_convex_hull(atoms, energy_name=energy_name)
        # print("got hull pts", [p1 for p1 in zip(hull[:, 0], hull[:, 1])])
        get_e_distance_func = get_e_distance_to_hull

    elif scheme == 'volume-stoichiometry':
        # Use this one for a binary or pseudo-binary-composition system.
        # Makes a 3D convex hull of volume vs. mole fraction vs. energy and calculates distance to it
        print('Regularising with 3D volume-mole fraction hull')
        if isol_es == {}:
            raise ArgumentError('Need to supply dictionary of isolated energies')

        p = label_stoichiometry_volume(atoms, isol_es, energy_name, element_order=element_order) # label atoms with volume and mole fraction
        hull = calculate_hull_3D(p) # calculate 3D convex hull
        get_e_distance_func = get_e_distance_to_hull_3D # function to calculate distance to hull (in energy)

    p = {}
    for group in sorted(set( [ at.info.get("gap_rss_group") for at in atoms if not at.info.get("gap_rss_nonperiodic") ] )):
        p[group] = []

    for at in atoms:
        try:
            # skip non-periodic configs, volume is meaningless
            if "gap_rss_nonperiodic" in at.info and at.info["gap_rss_nonperiodic"]:
                continue
            p[at.info.get("gap_rss_group")].append(at)
        except:
            pass

    for (group, atoms_group) in p.items():

        print('group:', group)

        for i, val in enumerate(atoms_group):
            de = get_e_distance_func(hull, val, energy_name=energy_name, isol_es=isol_es)
           
            if de > 20.0:
                # don't even fit if too high
                continue

            if group == 'initial':
                sigs[0].append(etup[1][1])
                sigs[1].append(etup[2][1])
                sigs[2].append(etup[3][1])
                val.info['energy_sigma'] = etup[1][1]
                val.info['force_sigma'] = etup[2][1]
                val.info['virial_sigma'] = etup[3][1]
                atoms_modi.append(val)
                continue

            if de <= etup[0][0]:
                sigs[0].append(etup[1][0])
                sigs[1].append(etup[2][0])
                sigs[2].append(etup[3][0])
                val.info['energy_sigma'] = etup[1][0]
                val.info['force_sigma'] = etup[2][0]
                val.info['virial_sigma'] = etup[3][0]
                atoms_modi.append(val)

            elif de >= etup[0][1]:
                sigs[0].append(etup[1][1])
                sigs[1].append(etup[2][1])
                sigs[2].append(etup[3][1])
                val.info['energy_sigma'] = etup[1][1]
                val.info['force_sigma'] = etup[2][1]
                val.info['virial_sigma'] = etup[3][1]
                atoms_modi.append(val)

            else:
                # rat = (de-etup[0][0]) / (etup[0][1]-etup[0][0])
                # e = rat*(etup[1][1]-etup[1][0]) + etup[1][0]
                # f = rat*(etup[2][1]-etup[2][0]) + etup[2][0]
                # v = rat*(etup[3][1]-etup[3][0]) + etup[3][0]
                [e, f, v] = piecewise_linear(de,[(0.1, [etup[1][0], etup[2][0], etup[3][0]]), (1.0, [etup[1][1], etup[2][1], etup[3][1]])])
                sigs[0].append(e)
                sigs[1].append(f)
                sigs[2].append(v)
                val.info['energy_sigma'] = e
                val.info['force_sigma'] = f 
                val.info['virial_sigma'] = v
                atoms_modi.append(val)

    e = np.array(sigs[0])
    f = np.array(sigs[1])
    v = np.array(sigs[2])

    print('Automatic regularisation statistics for {} structures:\n'.format(len(e)))
    print('{:>20s}{:>20s}{:>20s}{:>20s}{:>20s}'.format(
        '', 'Mean', 'Std', 'Nmin', 'Nmax'))
    print('{:>20s}{:>20.4f}{:>20.4f}{:>20d}{:>20d}'.format(
        'E', e.mean(), e.std(), len(e[e==e.min()]), len(e[e==e.max()])))
    print('{:>20s}{:>20.4g}{:>20.4f}{:>20d}{:>20d}'.format(
        'F', f.mean(), f.std(), len(f[f==f.min()]), len(f[f==f.max()])))
    print('{:>20s}{:>20.4g}{:>20.4f}{:>20d}{:>20d}'.format(
        'V', v.mean(), v.std(), len(v[v==v.min()]), len(v[v==v.max()])))

    return atoms_modi


def plot_convex_hull(all_points, hull_points):
    hull = ConvexHull(hull_points)

    plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=3, label='All Points')

    for i, simplex in enumerate(hull.simplices):
        if i == 0:
            plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], 'k-', label='Convex Hull')
        else:
            plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], 'k-')

    plt.xlabel('Volume')
    plt.ylabel('Energy')
    plt.title('Convex Hull with All Points')
    plt.legend()
    plt.show()


def get_convex_hull(atoms, energy_name='energy', **kwargs):
    '''Calculate simple linear convex hull of volume vs. energy
    Parameters:
                atoms :: (list) list of atoms objects
                energy_name :: (str) name of energy key in atoms.info (typically a DFT energy)

    Returns:
                the list of points in the convex hull (lower half only), and additionally all the points for testing purposes
    '''
    p = []
    ct = 0
    for at in atoms:
        if at.info['config_type'] == 'isolated_atom':
            continue
        elif at.info['config_type'] == 'dimer':
            continue
        try:
            v = at.get_volume()/len(at)
            e = at.info[energy_name]/len(at)
            p.append((v,e))
        except:
            ct += 1
    print('Convex hull failed to include {}/{} structures'.format(ct, len(atoms)))
    p = np.array(p)
    p = p.T[:, np.argsort(p.T[0])].T  # sort in volume axis
    
    hull = ConvexHull(p)  # generates full convex hull, we only want bottom half
    hull_points = p[hull.vertices]
    
    min_x_index = np.argmin(hull_points[:, 0])
    max_x_index = np.argmax(hull_points[:, 0])
    
    lower_half_hull = []
    i = min_x_index

    while True:
        lower_half_hull.append(hull.vertices[i])
        i = (i + 1) % len(hull.vertices)
        if i == max_x_index:
            lower_half_hull.append(hull.vertices[i])
            break

    lower_half_hull_points = p[lower_half_hull]

    lower_half_hull_points = lower_half_hull_points[lower_half_hull_points[:, 1] <= np.max(lower_half_hull_points[:, 1])]

    # plot_convex_hull(p, lower_half_hull_points)
    
    return lower_half_hull_points, p


def get_e_distance_to_hull(hull, at, energy_name='energy', **kwargs):
    '''Calculate the distance of a structure to the linear convex hull in energy
    Parameters: 
                hull_ps :: (np.array) points in the convex hull
                at :: (ase.Atoms) structure to calculate distance to hull
                energy_name :: (str) name of energy key in atoms.info (typically a DFT energy)'''
    v = at.get_volume()/len(at)
    e = at.info[energy_name]/len(at)
    tp = np.array([v, e])

    if isinstance(hull, ConvexHull):
        hull_ps = hull.points
    else:
        hull_ps = hull
    
    if any(np.isclose(hull_ps[:], tp).all(1)): # if the point is on the hull, return 0.0
        return 0.0
    
    nearest = np.searchsorted(hull_ps.T[0], tp, side='right')[0] # find the nearest convex hull point
    
    de = e - get_intersect( tp,    # get intersection of the vertical line (energy axis) and the line between the nearest hull points
                            tp + np.array([0,1]), 
                            hull_ps[(nearest-1) % len(hull_ps.T[0])], 
                            hull_ps[nearest % len(hull_ps.T[0])])[1]
    
    return de


def get_intersect(a1, a2, b1, b2):
    """ Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line"""
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return x/z, y/z


def get_x(at, element_order=None):
    '''Calculates the mole-fraction of a structure.
    Parameters:
                at :: (ase.Atoms) structure to calculate mole-fraction of
                element_order :: (list) list of atomic numbers in order of choice (e.g. [42, 16] for MoS2)
    Returns:
                (x2, x3...) :: (array of float) reduced mole-fraction of structure - first element n = 1-sum(others)
    '''

    el, cts = np.unique(at.get_atomic_numbers(), return_counts=True)

    if element_order is None and len(el) < 3: # compatibility with old version
        # print('using old version of get_x')
        if len(el) == 2:
            x = cts[1]/sum(cts)
        else:
            x = 1

    else: # new version, requires element_order, recommended for all new calculations
        if element_order is None:
            element_order = el # use default order
        not_in = [i for i in element_order if i not in el]
        for i in not_in:
            el = np.insert(el, -1, i)
            cts = np.insert(cts, -1, 0)

        cts = np.array([cts[np.argwhere(el == i).squeeze()] for i in element_order])  
        el = np.array([el[np.argwhere(el == i).squeeze()] for i in element_order])

        x = cts[1:]/sum(cts)
        # print(x, at)
        
    # print(at)
    return x


def label_stoichiometry_volume(ats, isol_es, e_name, element_order=None):
    '''Calcuate the stoichiometry, energy, and volume coordinates for forming the convex hull
    Parameters:
                ats :: (list) list of atoms objects
                isol_es :: (dict) dictionary of isolated atom energies
                e_name :: (str) name of energy key in atoms.info (typically a DFT energy)
    '''
    p = []
    for ct, at in enumerate(ats):
        try:
            v = at.get_volume()/len(at)
            # make energy relative to isolated atoms
            e = (at.info[e_name]-sum([isol_es[j] for j in at.get_atomic_numbers()]))/len(at)
            x = get_x(at, element_order=element_order)
            # print(x)
            p.append(np.hstack((x,v,e)))
        except:
            traceback.print_exc()
            pass
    p = np.array(p)
    p = p.T[:, np.argsort(p.T[0])].T

    return p


def point_in_triangle_2D(p1, p2, p3, pn):
    '''Check if a point is inside a triangle in 2D
    Parameters:
                p1 :: (tuple) coordinates of first point
                p2 :: (tuple) coordinates of second point
                p3 :: (tuple) coordinates of third point
                pn :: (tuple) coordinates of point to check
    '''
    ep = 1e-4
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = pn

    denominator = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
    c = 1 - a - b

    return 0-ep <= a and a <= 1+ep and 0-ep <= b and b <= 1+ep and 0-ep <= c and c <= 1+ep

def point_in_triangle_ND(pn, *preg):
    '''Check if a point is inside a region of hyperplanes in N dimensions - make a little convex hull in N-1 D and check this
    Parameters:
                pn: point to check (in ND)
                *preg: list of points defining (in ND) to check against
    '''
    hull = Delaunay(preg)
    return hull.find_simplex(pn) >= 0


def calculate_hull_3D(p):
    '''Calculate the convex hull in 3D'''
    p0 = np.array([(p[:, i].max() - p[:, i].min())/2 + p[:, i].min() for i in range(2)] + [-1e6])  # test point to get the visible facets from below
    pn = np.vstack((p0, p))

    hull = ConvexHull(pn, qhull_options='QG0')
    hull.remove_dim = []

    return hull


def calculate_hull_ND(p):
    '''Calculate the convex hull in ND (N>=3)'''
    
    p0 = np.array([(p[:, i].max() - p[:, i].min())/2 + p[:, i].min() for i in range(p.shape[1]-1)] + [-1e6])  # test point to get the visible facets from below
    pn = np.vstack((p0, p))
    remove_dim = []

    for i in range(p.shape[1]):
        if np.all(p.T[i, 0] == p.T[i, :]):
            pn = np.delete(pn, i, axis=1)
            print(f'Convex hull lower dimensional - removing dimension {i}')
            remove_dim.append(i)
            # print(pn)

    hull = ConvexHull(pn, qhull_options='QG0')
    print('done calculating hull')
    hull.remove_dim = remove_dim

    return hull


def get_e_distance_to_hull_3D(hull, at, isol_es=None, energy_name='energy', element_order=None):
    '''Calculate the energy distance to the convex hull in 3D
    '''
    x = get_x(at, element_order=element_order)
    e = (at.info[energy_name]-sum([isol_es[j] for j in at.get_atomic_numbers()]))/len(at)
    v = at.get_volume()/len(at)

    sp = np.hstack([x, v, e])
    for i in hull.remove_dim:
        sp = np.delete(sp, i)

    if len(sp[:-1]) == 1:
            # print('doing convexhull analyis in 1D')
            de = get_e_distance_to_hull(hull, at, energy_name=energy_name)

            return de

    for ct, visible_facet in enumerate(hull.simplices[hull.good]):

        # print(sp[:-1])
        # print(*hull.points[visible_facet][:, :-1])

        if point_in_triangle_ND(sp[:-1], *hull.points[visible_facet][:, :-1]):
            
            n_3 = hull.points[visible_facet]
            e = sp[-1]

            norm = np.cross(n_3[2]-n_3[0], n_3[1]-n_3[0])
            norm = norm/np.linalg.norm(norm) # plane normal
            D = np.dot(norm, n_3[0]) # plane constant

            de = e - (D - norm[0]*sp[0] - norm[1]*sp[1])/norm[2]

            return de
    
    print('Failed to find distance to hull')
    return 1e6


def piecewise_linear(x, vals):
    i = np.searchsorted([v[0] for v in vals], x)
    f0 = (vals[i][0]-x)/(vals[i][0]-vals[i-1][0])
    return f0*np.array(vals[i-1][1]) + (1.0-f0)*np.array(vals[i][1])

