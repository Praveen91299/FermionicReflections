# thesis gradient plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = './saved/data/H4_1.0_energies_all_refl_1e-5'
df = pd.read_csv(filename+'.csv')
refl_all_e5_grad = df['gradients'][:]
#refl_all_e5_errs = [abs(e - fci_e) for e in refl_all_e5_energies]

filename = './saved/data/h4_energies_qcc_Feb23_grads'
df = pd.read_csv(filename+'.csv')
qcc_grads = df['gradients'][:]
#qcc_errs = [abs(e - fci_e) for e in qcc_energies]

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
plt.tick_params(which='major', axis='y', length=10)
plt.tick_params(which='minor', axis='y', length=5)


#plt.plot([hf_err] + refl_e4_errs, '--+', label=r"FR, $\epsilon_{\mathrm{c}} = 10^{-4}$")
#plt.plot([hf_err] + refl_e5_errs, '--x', label=r"FR, $\epsilon_{\mathrm{c}} = 10^{-5}$")
plt.plot(list(range(1, 11)), refl_all_e5_grad, 'x', label=r"FR")
#plt.plot([hf_err] + refl_e3_errs, label=r"Refl $\epsilon_{tr} = 10^{-3}$")
plt.plot(list(range(1, 11)), qcc_grads[:-1], '*', label=r'iQCC', color='black')

plt.legend(frameon=False)
plt.xticks(list(range(1, 11)), list(range(1, 11)))
#plt.yscale("log")
plt.ylabel("Maximum candidate Gradient (Hartree)")
plt.xlabel("Generator Index")
plt.savefig('./saved/plots/H4_refl_all_qcc_grad.pdf', dpi=1000)
plt.show()