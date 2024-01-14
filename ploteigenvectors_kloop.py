import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def plot_for_k(k_select):
    f = h5.File('eigenvectors.h5','r')
    evals = f['exciton_data/eigenvalues'][()]
    evc = f['exciton_data/eigenvectors'][()]
    evc = evc[0,:,:,:,:,0,:]
    (nS, nk, nc, nv, _) = evc.shape

    plt.figure(figsize=(16,8))
    plt.subplots_adjust(left=0.22, right=0.94, top=0.96, bottom=0.22 , wspace=0, hspace=0)

    if k_select == -1:
        pass
    else:
        evc = evc[:,[k_select],:,:,:]

    for iN_S in range(nS):
        if e_min<evals[iN_S]<e_high:
            temp_contrib_cv = np.sum(abs(evc[iN_S,:,:,:,0]+evc[iN_S,:,:,:,1]*1j)**2, axis=0) # Summarize_eigenvectors and Ploteigenvectors - For a certain k and c, calculating the sum of Acvk across all vs, and vice versa, this is the correct way to do it, because for a certain k, valence band pdos idoes not change when we fix a v, and vice versa, But you ignore the correlation between c and v, thus you miss the ECF, but you can only plot ECF_noshift
            temp_contrib_v = np.sum(temp_contrib_cv, axis=0)
            temp_contrib_c = np.sum(temp_contrib_cv, axis=1)
            #plt.scatter(np.ones(nc) * evals[iN_S], np.arange(1,  nc+1, 1), color='b', s=temp_contrib_c * scale)
            #plt.scatter(np.ones(nv) * evals[iN_S], np.arange(-1,-nv-1,-1), color='r', s=temp_contrib_v * scale)
            # Store and temp_contrib_c and temp_contrib_v into a file for different k for each N_S
            np.savetxt(f'contrib_c_k={k_select}_N_S={iN_S}.txt', temp_contrib_c)
            np.savetxt(f'contrib_v_k={k_select}_N_S={iN_S}.txt', temp_contrib_v)

    #plt.yticks([-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8],('8','7','6','5','4','3','2','1','1','2','3','4','5','6','7','8'))
    #plt.ylabel('Contributing Band',fontsize=25,font='Helvetica')
    #plt.xlabel('Exciton Energy (eV)',fontsize=25, font='Helvetica')
    #plt.tick_params(axis='x', which='major', labelsize=18)
    #plt.tick_params(axis='y', which='major', labelsize=18)
    #ylim = plt.ylim(-21,21)
    #xlim = plt.xlim(2.98,3.1)
    #plt.savefig(f'evec_component_k={k_select}.png',dpi=400)
    #plt.close()

scale = 10E2
e_min = 0
e_high = 5

for k in range(-1,0):
    plot_for_k(k)
