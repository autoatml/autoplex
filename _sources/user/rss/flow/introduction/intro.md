(introduction)=

# Brief overview

The random structure searching (RSS) approach was initially proposed for predicting crystal structures by generating randomized, sensible structures and optimising them via first-principles calculations ([Phys. Rev. Lett. 97, 045504 (2006)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.045504) and [J. Phys.: Condens. Matter 23, 053201 (2011)](https://iopscience.iop.org/article/10.1088/0953-8984/23/5/053201)). Recently, RSS was expanded into a methodology for exploring and learning potential-energy surfaces from scratch ([Phys. Rev. Lett. 120, 156001 (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.156001) and [npj Comput. Mater. 5, 99 (2019)](https://www.nature.com/articles/s41524-019-0236-6)). Enhanced with physics-inspired sampling methods, such as Boltzmann-probability biased histograms and CUR, this approach ensures both the significance (low-energy) and diversity of the structures being searched.

## Features

Generating datasets for potential fitting through classical or AIMD simulations is common practice. However, such methods heavily rely on the initial input structures, often limiting the exploration to a narrow phase space. This is because they struggle to overcome the energy barriers associated with phase transitions. As a result, a single (AI)MD trajectory typically works only for specific systems, such as single phases with fixed stoichiometry. In contrast, RSS enables access to a much larger configurational space due to the high diversity of initial random structures. Overall, RSS has the following key features:

- **Broad configurational space exploration**  
  RSS ensures structure diversity by exploring a broad configurational space. This makes it suitable for generating general-purpose potentials or pre-trained models.

- **Low cost**  
  RSS eliminates dependence on AIMD, relying instead on ML-driven iterative optimization. As a result, it only requires single-point DFT calculations, largely reducing the computational cost.

- **Energy-based sampling**  
  By employing energy-weighted sampling, RSS can generate physically reasonable training sets.

- **Versatile applicability**  
  RSS can effortlessly handle varying elements, stoichiometries, and densities, making it highly adaptable to different material systems. To date, RSS has been demonstrated to produce accurate potentials for elemental, binary, and ternary material systems. 
