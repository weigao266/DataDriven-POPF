from pypower.api import case9, ppoption, runpf, printpf

ppc = case9()
ppopt = ppoption(PF_ALG=2)
r = runpf(ppc, ppopt)
printpf(r)

help runpf