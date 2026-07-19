# thesis energy plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fci_e = -2.1663874486347625#molecule.fci_energy
hf_energy = -2.0985459369977635
#refl_errs = [abs(e - fci_e) for e in energies]

#refl 1e-3 trunc err
filename = './saved/data/H4_1.0_energies_refl_feb22'
df = pd.read_csv(filename+'.csv')
refl_e3_energies = df['energy'][:]
refl_e3_errs = [abs(np.float(e[1:10]) - fci_e) for e in refl_e3_energies]

filename = './saved/data/H4_1.0_energies_refl_1e-4'
df = pd.read_csv(filename+'.csv')
refl_e4_energies = df['energy'][:]
refl_e4_errs = [abs(np.float(e[1:10]) - fci_e) for e in refl_e4_energies]

filename = './saved/data/H4_1.0_energies_refl_1e-5'
df = pd.read_csv(filename+'.csv')
refl_e5_energies = df['energy'][:]
refl_e5_errs = [abs(e - fci_e) for e in refl_e5_energies]

filename = './saved/data/H4_1.0_energies_all_refl_1e-5'
df = pd.read_csv(filename+'.csv')
refl_all_e5_energies = df['energy'][:]
refl_all_e5_errs = [abs(e - fci_e) for e in refl_all_e5_energies]

filename = './saved/data/h4_energies_qcc_Feb23'
df = pd.read_csv(filename+'.csv')
qcc_energies = df['energy'][1:]
qcc_errs = [abs(e - fci_e) for e in qcc_energies]

hf_err = abs(hf_energy - fci_e)

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
plt.tick_params(which='major', axis='y', length=10)
plt.tick_params(which='minor', axis='y', length=5)


#plt.plot([hf_err] + refl_e4_errs, '--+', label=r"FR, $\epsilon_{\mathrm{c}} = 10^{-4}$")
#plt.plot([hf_err] + refl_e5_errs, '--x', label=r"FR, $\epsilon_{\mathrm{c}} = 10^{-5}$")
plt.plot([hf_err] + refl_all_e5_errs, '--x', label=r"FR")
#plt.plot([hf_err] + refl_e3_errs, label=r"Refl $\epsilon_{tr} = 10^{-3}$")
plt.plot([hf_err] + qcc_errs, '-*', label=r'iQCC', color='black')
plt.axhline(y=1.59e-3, color='grey', linestyle='--')

plt.legend(frameon=False)
plt.xticks(list(range(0, 11)), list(range(0, 11)))
plt.yscale("log")
plt.ylabel("Absolute Energy Errors (Hartree)")
plt.xlabel("Number of generators")
plt.savefig('./saved/plots/H4_refl_all_qcc.pdf', dpi=300)
plt.show()