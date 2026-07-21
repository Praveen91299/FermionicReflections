# Fermionic Reflections  

Code to create and optimize ansatz consisting of Fermionic Reflection generators proposed in the Manuscript ``On the Feasibility of Exact Transformation of Many-Body Hamiltonians".  

## Thesis related (DO NOT CHANGE THIS ENTRY/FILES)  
`h4_dress_trunc.py` provides the numerical example in Chapter 3, section 3. Saved data and plots can be found at `thesis_data`.   
`prove_rank2_reflections.py` is a numerical test of exhaustive search for fermionic reflections of fermionic rank 2. `RANK2_REFLECTION_PROOF.md` is a explainer note for the script.  

## How to
Main ansatz class can be found at `Refl.reflection_ansatz.py`. See class definition for instructions on usage. See script `h4_dress_trunc.py` for demonstration.  

### Required python packages
openfermion, openfermionpyscf, numpy, scipy, opt_einsum, copy  

Optional: optimparallel (for orbital rotation optimization), pickle (for saving files)  

## Developed by  
Praveen Jayakumar
