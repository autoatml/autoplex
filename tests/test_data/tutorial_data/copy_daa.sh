mkdir inputs
mkdir outputs
cp POSCAR.gz inputs
echo "Sn"  >inputs/POTCAR.spec
gzip inputs/POTCAR.spec
cp INCAR.gz inputs
cp inputs/* outputs
cp CONTCAR* outputs
cp OUTCAR* outputs
cp vasp* outputs

