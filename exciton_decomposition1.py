# %% [markdown]
# # Exciton Decomposition Analysis
# 
# There are qualititative and quantative analysis, for qualititive you can use summarize eigenvectors, just know the largest band to band transition and largest k point, then you will learn the band composition into parts.
# 
# For ploteigenvectors.py is also a qualititive analysis, summarize eigenvectors is also semi-qualiative, because of limited information: 
# 
# It only tells you about wtot, wmax and ikmax for different band to band transitions for each Exciton S - You cannot resolve K-point from this information, thus cannot safely project to atomic orbitals.
# 
# `wtot = sum_k |A_vck|^2. wmax = max_k |A_vck|^2. |A_vc (ikmax)|^2 = wmax.`
# 
# 
# **To do a proper quantitative approach:**
# 
# Read eigenvectors.h5 to extract K resolved decomposition into different band to band transition, in combination with pdos information, I can calculate the percentage for perovskite/non-perovskite of electrons and holes for the every excitons, similar information are also calculated when I did the ECF-noshift plot, where I can see the localization of electrons and holes for each exciton in real space.

# %% [markdown]
# file:///Users/yinanchen/PhD/2022_Intergrowth_Hema/APD1_Na/absorption/k104/evec_component_k=0.png
# file:///Users/yinanchen/PhD/2022_Intergrowth_Hema/APD1_Na/absorption/k104/evec_component_k=-1.png
# file:///Users/yinanchen/PhD/2022_Intergrowth_Hema/APD2_Pb/corrected_H/absorption/evec_component_k=-1.png

# %%
import pandas as pd
import numpy as np

# %%
# Things to modify by hand

# folded symmetry reduced BZ
kgrid_file = 'kgrid.log_cocn.bak'
kgrid_file = 'kgrid.log_CYS.bak'
kgrid_file = 'kgrid.log_APD2.bak'
kgrid_file = 'kgrid.log_APD1.bak'

# expanded full BZ
bandstructure_file = 'bandstructure.dat_cocn.bak'
bandstructure_file = 'bandstructure.dat_CYS.bak'
bandstructure_file = 'bandstructure.dat_APD2.bak'
bandstructure_file = 'bandstructure.dat_APD1.bak'

fermi = 280 # ! for HOCN
fermi = 304   # ! for COCN
fermi = 544   # ! for CYS
fermi = 408 # ! VBM - start + 1 , hard-coded for APD2_Li now, read from projbands file
fermi = 432   # ! for APD1_Na

# Eigenvectors are using  k point list in bandstructure.dat, expanded full BZ
eigenvector_file = 'eigenvectors.h5_hocn.bak'
eigenvector_file = 'eigenvectors.h5_cocn.bak'
eigenvector_file = 'eigenvectors.h5.CYS.bak'
eigenvector_file = 'eigenvectors.h5.APD2.bak'
eigenvector_file = 'eigenvectors.h5.APD1.bak'
#eigenvector_file = 'eigenvectors_APD1_relax.h5'

scf_file = 'relax.out_cocn.bak'
scf_file = 'scf.out_CYS.bak'
scf_file = 'scf.out_APD2_Li.bak'
scf_file = 'scf.out_APD1.bak'

# On a full uniform grid of BZ, symmetry reduced
projbands_file = 'cl.projbands_hocn.bak'
projbands_file = 'cl.projbands_cocn.bak'
projbands_file = 'cl.projbands.CYS.bak'
projbands_file = 'cl.projbands_APD2_Li.bak'
projbands_file = 'cl.projbands.apd1.bak'
#projbands_file = 'cl.projbands_APD1_relax.bak'

e_min = 0
e_high = 5

# %% [markdown]
# ## Step 0: Find the K list correspondance from bandstructure.dat (unfolded) with the K list from kgrid.log (folded)

# %% [markdown]
# ### 0.1 Process K list from Kgrid.log (folded)

# %%
## Process the kgrid.log, find the correpondance with uniform grid and irrduciible grid used to calculate WFN_fi

def process_kgrid_WFN(file_content):
    lines = file_content.split('\n')

    for i, line in enumerate(lines):
        if "k-points in the original uniform grid" in line:
            # Next line contains the number of rows to read
            num_rows = int(lines[i + 1].strip())
            # Data starts from the next line
            start_line = i + 2
            break

    data = []
    for line in lines[start_line:start_line + num_rows]:
        # Split the line into columns and convert to appropriate types
        columns = line.split()
        row = [int(columns[0])] + [float(c) for c in columns[1:5]] + [int(columns[5]), columns[6]]
        data.append(row)
    unfolded_idx = {}
    
    cnt = 0
    processed_data = []

    for row in data:
        row_number = row[0]
        reference_row = row[5]

        # If the sixth column is 0, increment the count
        if reference_row == 0:
            unfolded_idx[row_number] = cnt
            cnt = cnt + 1
        else:
            # If the sixth column is not 0, set the count to the count of the referenced row
            unfolded_idx[row_number] = unfolded_idx[reference_row]
        processed_data.append(row + [unfolded_idx[row_number]])

    return processed_data

processed_data = process_kgrid_WFN(open(kgrid_file).read())
# the last column and kx, ky, kz
processed_data = np.array(processed_data)[:, [1, 2, 3, 7]].astype(float)
nk = processed_data.shape[0]
print('nk = ', nk)
print(processed_data.shape)
processed_data

# %% [markdown]
# ### 0.2 Process bandstructure.dat (unfolded)

# %%
def read_unfolded(file_path):
    matrix = []
    last_spin, last_band = None, None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue  
            spin, band, kx, ky, kz = parts[:5]

            # check if we have a new spin or band
            if last_spin is not None and (spin != last_spin or band != last_band):
                break
            # update last spin and band
            last_spin, last_band = spin, band

            matrix.append([float(kx), float(ky), float(kz)])

    return np.array(matrix)

data_matrix = read_unfolded(bandstructure_file)
print(data_matrix.shape)


# %%

#reciprocal_lattice_vectors and alat are read from relax.out or scf.out

def extract_specific_data(file_path):
    with open(file_path, 'r') as file:
        alat = None
        b_vectors = []

        for line in file:
            if 'lattice parameter (alat)' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    alat = float(parts[1].split()[0])

            if 'reciprocal axes' in line:
                for i in range(3):
                    b_vectors.append(file.readline().split()[3:6])
            # ! Heavily depends on the format of the file
            if alat is not None and len(b_vectors) == 3:
                break

    return alat, np.array(b_vectors, dtype=float)

alat, b_vectors = extract_specific_data(scf_file)
print(alat)
print(b_vectors)
two_pi_over_alat = 2 * np.pi / alat
reciprocal_lattice_vectors = np.array(b_vectors) * two_pi_over_alat
reciprocal_lattice_vectors

# %%
# Convert cartesian coordinates to fractional coordinates
def cartesian_to_fractional(cartesian_coordinates, reciprocal_lattice_vectors):
    return np.dot(cartesian_coordinates, np.linalg.inv(reciprocal_lattice_vectors))

def fractional_to_cartesian(fractional_coordinates, reciprocal_lattice_vectors):
    return np.dot(fractional_coordinates, reciprocal_lattice_vectors)

fractional_coordinates = cartesian_to_fractional(data_matrix, reciprocal_lattice_vectors)
# Around to 3 decimals
fractional_coordinates = np.around(fractional_coordinates, decimals=3)
print(fractional_coordinates.shape)
fractional_coordinates

# %%
# Convert fractional coordinates to the BZ within (0,0,0) to (1,1,1) by translating the coordinates
def translate_to_BZ(fractional_coordinates):
    return fractional_coordinates - np.floor(fractional_coordinates)

translated_coordinates = translate_to_BZ(fractional_coordinates)
translated_coordinates

# %%
# Sort the translated coordinates first by kx, then by ky, then by kz, and keeping the original indices
def sort_by_k(translated_coordinates):
    indices = np.lexsort((translated_coordinates[:,2], translated_coordinates[:,1], translated_coordinates[:,0]))
    return translated_coordinates[indices], indices

sorted_coordinates, indices = sort_by_k(translated_coordinates)
print(sorted_coordinates[:20])
print(indices[:20])

# %%
# append the indices to the processed_data
processed_kgrid_wfn_eigenvector = np.hstack((processed_data, indices.reshape(-1,1)))
processed_kgrid_wfn_eigenvector[:20]

# %%
# Check both indices are correct, the first one should be folded index, the second one is the unfolded index
print(max(processed_kgrid_wfn_eigenvector[:,4]))
print(max(processed_kgrid_wfn_eigenvector[:,3]))
print(len(processed_kgrid_wfn_eigenvector))

# %% [markdown]
# ## Step 1: Decomposition of excitons into the contribution of band-to-band transitions at different k points
# 
# Specifically, I read and store sum_v/c |Acvk|^2 for each (S,k) pair for each band in c, v respectively. Then I can also plot the distribution of the contribution of each band-to-band transition to the exciton wavefunction as I did using the script plot_eigenvectors_kloop.py.

# %% [markdown]
# file:///Users/yinanchen/PhD/2022_Intergrowth_Hema/APD1_Na/absorption/k104/evec_component_k=0.png
# file:///Users/yinanchen/PhD/2022_Intergrowth_Hema/APD1_Na/absorption/k104/evec_component_k=-1.png

# %%
with open(projbands_file, 'r') as f:
    lines = f.readlines()

comments = []
for line in lines:
    if line.startswith('#'):
        comments.append(line)
    else:
        break

data = []
for line in lines:
    if not line.startswith('#'):
        data.append(line)


# %%
# convert the list of strings to a list of lists, 2D array
data = [line.split() for line in data]

# %%
# conver the 2d array to a dataframe, and convert the strings to floats
df = pd.DataFrame(data).astype(float)

# %%
df
# k_index, K_length, E-Ef, TOTAL, projection on a,\nu

# %%
import h5py
with h5py.File(eigenvector_file, 'r') as f:
    evals_shape = f['exciton_data/eigenvalues'].shape
    print("Eigenvalues shape:", evals_shape)
    
    evc_shape = f['exciton_data/eigenvectors'].shape
    print("Eigenvectors shape:", evc_shape)

# Shape is _,  nS, nK, nc, nv, _, (real+imag)=2

# %% [markdown]
# ### Read only the partial information from eigenvectors.h5

# %%
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

contrib_c_dict = {}
contrib_v_dict = {}

def plot_for_k(eigenvector_file, k_select, iN_S_select, e_min=None, e_high=None):
    with h5.File(eigenvector_file, 'r') as f:
        # 只读取特定iN_S的eigenvalues
        evals = f['exciton_data/eigenvalues'][iN_S_select]
        # 只读取特定iN_S的eigenvectors
        evc = f['exciton_data/eigenvectors'][0,iN_S_select,:,:,:,0,:]
        nk, nc, nv, _ = evc.shape

        if k_select != -1:
            evc = evc[[k_select],:,:,:]

        if e_min<evals<e_high:
            temp_contrib_cv = np.sum(abs(evc[:,:,:,0]+evc[:,:,:,1]*1j)**2, axis=0) 
            # Summarize_eigenvectors and Ploteigenvectors - For a certain k and c, calculating the sum of Acvk across all vs, and vice versa, this is the correct way to do it, because for a certain k, valence band pdos idoes not change when we fix a v, and vice versa, But you ignore the correlation between c and v, thus you miss the ECF, but you can only plot ECF_noshift
            temp_contrib_v = np.sum(temp_contrib_cv, axis=0)
            temp_contrib_c = np.sum(temp_contrib_cv, axis=1)

            contrib_c_dict[(iN_S_select,k_select)] = temp_contrib_c
            contrib_v_dict[(iN_S_select,k_select)] = temp_contrib_v
        
        return 1, nk, nc, nv  # 由于只读取一个iN_S，nS始终为1

# %%
iN_S_select =1
for k in range(0,nk):
    nS, nk, nc, nv = plot_for_k(eigenvector_file,k,iN_S_select,0,5)
    #print(contrib_c_dict)
    #print(contrib_v_dict)
print(nk, nc, nv, nS)

# %% [markdown]
# ### Read the full information from eigenvectors.h5

# %%
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
contrib_c_dict = {}
contrib_v_dict = {}
def plot_for_k(k_select):
    f = h5.File(eigenvector_file,'r')
    evals = f['exciton_data/eigenvalues'][()]
    evc = f['exciton_data/eigenvectors'][()]
    evc = evc[0,:,:,:,:,0,:]
    (nS, nk, nc, nv, _) = evc.shape

    if k_select == -1:
        pass
    else:
        evc = evc[:,[k_select],:,:,:]

    for iN_S in range(nS):
        if e_min<evals[iN_S]<e_high:
            temp_contrib_cv = np.sum(abs(evc[iN_S,:,:,:,0]+evc[iN_S,:,:,:,1]*1j)**2, axis=0) # Summarize_eigenvectors and Ploteigenvectors - For a certain k and c, calculating the sum of Acvk across all vs, and vice versa, this is the correct way to do it, because for a certain k, valence band pdos idoes not change when we fix a v, and vice versa, But you ignore the correlation between c and v, thus you miss the ECF, but you can only plot ECF_noshift
            # a matrix of nc x nv
            temp_contrib_v = np.sum(temp_contrib_cv, axis=0) # sum over c
            temp_contrib_c = np.sum(temp_contrib_cv, axis=1) # sum over v

            contrib_c_dict[(iN_S,k_select)] = temp_contrib_c
            contrib_v_dict[(iN_S,k_select)] = temp_contrib_v
    return nS, nk, nc, nv #, temp_contrib_cv


# %%
temp_contrib_cv = None
for k in range(0,nk):
    nS, nk, nc, nv = plot_for_k(k)
    #print(contrib_c_dict)
    #print(contrib_v_dict)
print(nk)

# %%
print(len(contrib_c_dict))
contrib_c_dict
# (i_N_S, k): [nc] an array of length nc, sum of |Acvk|^2 for each c, total of k x N_s x nc data points

# %%
nS, nk, nc, nv

# %%
# find the corresponding pdos for each k from df, this step is to check
# Count the number of k points in df, should be consistent with nk
n_k = len(df[0].unique())
n_k

# %% [markdown]
# ## Step 2: Calculate the kpdos on a WFN_fi calculation
# 
# Now that we have the decomposition of excitons into the contribution of band-to-band transitions for each (S,k) pair.
# 
# For every k, find the corresponding pdos from K-resolved Projected DOS on WFN_fi.

# %%
# Before that, we need to isolating only the middle nv+nc rows for each k, 
# corresponding to the nv valence and nc conduction bands

# First group df by the first column, then operate on each group
# For each group, take out the middle nv+nc rows around fermi, i.e. fermi - nv : fermi + nc
# ! Locate the position of fermi, i.e. the position where df[2] changes from negative to positive. 

conduction_df = df.groupby(0).apply(lambda x: x.iloc[fermi:fermi+nc,:])
valence_df = df.groupby(0).apply(lambda x: x.iloc[fermi-nv:fermi,:])
reversed_valence_df = valence_df.iloc[::-1,:]
reversed_valence_df.shape

# %%
# This step is to check the vs are negative and cs are positive
# To ensure that the hard-coded fermi is correct

valence_df.loc[1]
print(valence_df.loc[1][2].values)
print(reversed_valence_df.loc[1][2].values)
print(conduction_df.loc[1][2].values)

# %%
# To correspond to the contribution of valence bands (counting from the VBM) and conduction bands (counting from the CBM)
# We need to reverse the order of the valence bands
# For the 41st Kpoint
reversed_valence_df.loc[400]
#conduction_df.loc[400]
# K_index, K_length, E-Ef, TOTAL, projection on a,\nu, Here I haven't normalized the TOTAL to 1.

# %%
# Normalize the reversed_valence_df and conduction_df by the sum of the pdos of each k (the 3rd column)

reversed_valence_df = reversed_valence_df.apply(lambda x: x/x[3], axis=1)
conduction_df = conduction_df.apply(lambda x: x/x[3], axis=1)
reversed_valence_df.loc[41]
conduction_df.loc[41]

# %% [markdown]
# ## Step 3: Combine the knowledge of K resolved pdos and exciton decomposition into bands
# 
# For each k point, find out the decomposition of bands into atomic orbitals, below is for a specific S,

# %%
# Now we use contrib_v_dict[(s,k)] as the weight to multiply with the pdos of the valence band at k = k_select
# Then we sum over all k points, and we get the pdos of the valence band

# ! Note, Question: should we normalize the weight? i.e. divided by the sum of the weights over all vs or cs for each k point?
# ! Answer: No, because each k point in BZ should have a different total weight as seen by the plot_eigenvectors.py plots,
# #! so we should not normalize the weight for each k point.

# ! Also, I am not normalizing pdos (not divided by TOTAL (ranging from 0.7 - 0.9) in column 4), 
# ! using their original pdos at (k, n), where n is the band index.
# ! Correct: We should normalize the pdos by the TOTAL
iN_S_select = 1
result_df_valence = pd.DataFrame()
result_df_conduction = pd.DataFrame()
for k_select in range(len(processed_kgrid_wfn_eigenvector)):
    k_select_kpdos =  int(processed_kgrid_wfn_eigenvector[k_select][3])
    k_select_eigenvector = int(processed_kgrid_wfn_eigenvector[k_select][4])
    weighted_sum_valence = reversed_valence_df.loc[k_select_kpdos+1].mul(contrib_v_dict[(iN_S_select
,k_select_eigenvector)], axis=0).sum()
    #if k_select == 1:
        #print(reversed_valence_df.loc[k_select_kpdos+1])
        #print(contrib_v_dict[(iN_S_select,k_select_eigenvector)])
        #print(weighted_sum_valence)
    # \ / contrib_v_dict[(iN_s_select,k_select)].sum()
    # nc x a,nu  .mul 1 x nc = 1 x a,nu
    # 20 x 684 .mul 1 x 20 = 1 x 684  
    # This should be a single row dataframe, representing the projection onto a, nu at this k
    result_df_weighted_sum_valence = pd.DataFrame([weighted_sum_valence])

    weighted_sum_conduction = conduction_df.loc[k_select_kpdos+1].mul(contrib_c_dict[(iN_S_select
,k_select_eigenvector)], axis=0).sum()
    # \ / contrib_c_dict[(iN_s_select,k_select)].sum()
    result_df_weighted_sum_conduction = pd.DataFrame([weighted_sum_conduction])

    # append all k points into a single dataframe
    #result_df = result_df.append(result_df_weighted_sum)
    result_df_valence = pd.concat([result_df_valence, result_df_weighted_sum_valence], axis=0)
    result_df_conduction = pd.concat([result_df_conduction, result_df_weighted_sum_conduction], axis=0)
# Result_df_conduction is a K-resolved decomposition of hole/electron onto a, nu
# sum over all k points, row-wise

print(result_df_conduction)
#print(result_df_valence.sum(axis=0))
print(result_df_conduction.sum(axis=0))

# %% [markdown]
# This is a for all the excitons

# %%
ns_accumulated_df_valence = pd.DataFrame()
ns_accumulated_df_conduction = pd.DataFrame()
for s_select in range(nS):
#for s_select in range(iN_S_select,iN_S_select+1):
    accumulated_df_valence = pd.DataFrame()
    accumulated_df_conduction = pd.DataFrame()
    for k_select in range(len(processed_kgrid_wfn_eigenvector)):
        k_select_kpdos = int(processed_kgrid_wfn_eigenvector[k_select][3])
        k_select_eigenvector = int(processed_kgrid_wfn_eigenvector[k_select][4])
        current_weighted_sum_valence = reversed_valence_df.loc[k_select_kpdos+1].mul(contrib_v_dict[(s_select, k_select_eigenvector)], axis=0).sum()
        current_weighted_sum_conduction = conduction_df.loc[k_select_kpdos+1].mul(contrib_c_dict[(s_select, k_select_eigenvector)], axis=0).sum()
        if accumulated_df_valence.empty:
            accumulated_df_valence = pd.DataFrame([current_weighted_sum_valence])
        else:
            accumulated_df_valence += current_weighted_sum_valence
        if accumulated_df_conduction.empty:
            accumulated_df_conduction = pd.DataFrame([current_weighted_sum_conduction])
        else:
            accumulated_df_conduction += current_weighted_sum_conduction
    ns_accumulated_df_valence = pd.concat([ns_accumulated_df_valence, accumulated_df_valence], axis=0)
    ns_accumulated_df_conduction = pd.concat([ns_accumulated_df_conduction, accumulated_df_conduction], axis=0)

print(accumulated_df_conduction)
#result_df
ns_accumulated_df_conduction


# %%
ns_accumulated_df_valence

# %%
max(accumulated_df_conduction.iloc[:,4:].sum(axis=0))

# %%
# Save comments and ns_accumulated_df_valence to a file, with every row in df is a single line
with open('pdos_sum_valence.dat', 'w') as f:
    for line in comments:
        f.write(line)
    for i in range(len(ns_accumulated_df_valence)):
        f.write(str(ns_accumulated_df_valence.iloc[i,:].values.tolist()).strip('[]') + '\n')


with open('pdos_sum_conduction.dat', 'w') as f:
    for line in comments:
        f.write(line)
    for i in range(len(ns_accumulated_df_conduction)):
        f.write(str(ns_accumulated_df_conduction.iloc[i,:].values.tolist()).strip('[]') + '\n')

# %% [markdown]
# ## Step 4: Calculate the projectiong into different part of the heterostructure
# 
# This part is system specific, for now I just hard-coded it

# %% [markdown]
# ### 4.0 for APD1_Na

# %%
# columns from 37 to 72, 361 to 424 as the Br-per. contribution
# columns from 1 to 36, 329 to 360 as the Br-non. contribution
# columns from 425 to 488 as the O contribution
# columns from 609 to 680 as the Na contribution
# Total is 1 to 680

conduction_perovskite_Pb = ns_accumulated_df_conduction.iloc[:, 37+3:73+3].sum(axis=1) 
conduction_perovskite_Cl = ns_accumulated_df_conduction.iloc[:, 361+3:425+3].sum(axis=1)
conduction_non_perovskite_Pb = ns_accumulated_df_conduction.iloc[:, 1+3:37+3].sum(axis=1)
conduction_non_perovskite_Cl = ns_accumulated_df_conduction.iloc[:, 73+3:361+3].sum(axis=1)
conduction_O = ns_accumulated_df_conduction.iloc[:, 425+3:489+3].sum(axis=1)
conduction_Na = ns_accumulated_df_conduction.iloc[:, 609+3:649+3].sum(axis=1)
conduction_total = ns_accumulated_df_conduction.iloc[:, 1+3:681+3].sum(axis=1)
conduction_other = ns_accumulated_df_conduction.iloc[:, 489+3:609+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 649+3:681+3].sum(axis=1)
#conduction_others = conduction_total - conduction_perovskite - conduction_non_perovskite - conduction_O - conduction_Na - conduction_other

# %%
conduction_perovskite_Pb/ conduction_total

# %%
conduction_perovskite_Cl/ conduction_total

# %%
conduction_non_perovskite_Cl/ conduction_total

# %%
conduction_non_perovskite_Pb/ conduction_total

# %%
# valence bands for APD1_Na
valence_perovskite = ns_accumulated_df_valence.iloc[:, 37+3:73+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 361+3:425+3].sum(axis=1)
valence_non_perovskite = ns_accumulated_df_valence.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 73+3:361+3].sum(axis=1)
valence_O = ns_accumulated_df_valence.iloc[:, 425+3:489+3].sum(axis=1)
valence_Na = ns_accumulated_df_valence.iloc[:, 609+3:649+3].sum(axis=1)
valence_total = ns_accumulated_df_valence.iloc[:, 1+3:681+3].sum(axis=1)
valence_other = ns_accumulated_df_valence.iloc[:, 489+3:609+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 649+3:681+3].sum(axis=1)
#valence_others = valence_total - valence_perovskite - valence_non_perovskite - valence_O - valence_Na - valence_other

# %%
valence_non_perovskite/ valence_total

# %%
valence_perovskite/ valence_total

# %%
valence_O/ valence_total

# %% [markdown]
# ### 4.1 for APD2_Pb

# %%
# columns from 37 to 72, 105 to 168 as the Br-per. contribution
# columns from 1 to 36, 73 to 104 as the Br-non. contribution
# columns from 169 to 232 as the O contribution
# columns from 609 to 624 as the Li contribution
# Total is 1 to 656

conduction_perovskite = ns_accumulated_df_conduction.iloc[:, 37+3:73+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 105+3:169+3].sum(axis=1)
conduction_non_perovskite = ns_accumulated_df_conduction.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 73+3:105+3].sum(axis=1)
conduction_O = ns_accumulated_df_conduction.iloc[:, 169+3:233+3].sum(axis=1)
conduction_Li = ns_accumulated_df_conduction.iloc[:, 609+3:625+3].sum(axis=1)
conduction_total = ns_accumulated_df_conduction.iloc[:, 4:660].sum(axis=1)
conduction_other = ns_accumulated_df_conduction.iloc[:, 233+3:609+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 625+3:657+3].sum(axis=1)
conduction_others = conduction_total - conduction_perovskite - conduction_non_perovskite - conduction_O - conduction_Li - conduction_other

# %%
conduction_perovskite/ conduction_total

# %%
valence_perovskite = ns_accumulated_df_valence.iloc[:, 37+3:73+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 105+3:169+3].sum(axis=1)
valence_non_perovskite = ns_accumulated_df_valence.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 73+3:105+3].sum(axis=1)
valence_O = ns_accumulated_df_valence.iloc[:, 169+3:233+3].sum(axis=1)
valence_Li = ns_accumulated_df_valence.iloc[:, 609+3:625+3].sum(axis=1)
valence_total = ns_accumulated_df_valence.iloc[:, 5:660].sum(axis=1)
valence_other = ns_accumulated_df_valence.iloc[:, 233+3:609+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 625+3:657+3].sum(axis=1)
valence_others = valence_total - valence_perovskite - valence_non_perovskite - valence_O - valence_Li - valence_other

# %%
valence_perovskite/valence_total

# %%
valence_non_perovskite/valence_total

# %%
valence_Li/valence_total

# %%
valence_O/valence_total

# %%
valence_other/valence_total

# %% [markdown]
# ### 4.2 for Hypothetical COCN - Pb-Pb Intergrowth

# %%
# columns from 1 to 36, 173 to 236 as the Br-per. contribution
# columns from 37 to 172 as the Br-non. contribution
# columns from 237 to 268 as the O contribution
# columns from 269 to 428 as the other contribution
# Total is 1 to 428


conduction_perovskite = ns_accumulated_df_conduction.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 173+3:237+3].sum(axis=1)
conduction_non_perovskite = ns_accumulated_df_conduction.iloc[:, 37+3:173+3].sum(axis=1)
conduction_O = ns_accumulated_df_conduction.iloc[:, 237+3:269+3].sum(axis=1)
conduction_other = ns_accumulated_df_conduction.iloc[:, 269+3:429+3].sum(axis=1)
conduction_total = ns_accumulated_df_conduction.iloc[:, 4:432].sum(axis=1)

# %%
conduction_non_perovskite/ conduction_total

# %%
valence_perovskite = ns_accumulated_df_valence.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 173+3:237+3].sum(axis=1)
valence_non_perovskite = ns_accumulated_df_valence.iloc[:, 37+3:173+3].sum(axis=1)
valence_O = ns_accumulated_df_valence.iloc[:, 237+3:269+3].sum(axis=1)
valence_other = ns_accumulated_df_valence.iloc[:, 269+3:429+3].sum(axis=1)
valence_total = ns_accumulated_df_valence.iloc[:, 5:432].sum(axis=1)

# %%
valence_perovskite/valence_total

# %% [markdown]
# ### 4.3 for Hypothetical HOCN - Pb-Pb Intergrowth

# %%
# columns from 1 to 36, 173 to 236 as the Br-per. contribution
# columns from 37 to 172 as the Br-non. contribution
# columns from 237 to 268 as the O contribution
# columns from 269 to 380 as the Li contribution
# Total is 1 to 380


conduction_perovskite = ns_accumulated_df_conduction.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_conduction.iloc[:, 173+3:237+3].sum(axis=1)
conduction_non_perovskite = ns_accumulated_df_conduction.iloc[:, 37+3:173+3].sum(axis=1)
conduction_O = ns_accumulated_df_conduction.iloc[:, 237+3:269+3].sum(axis=1)
conduction_other = ns_accumulated_df_conduction.iloc[:, 269+3:381+3].sum(axis=1)
conduction_total = ns_accumulated_df_conduction.iloc[:, 4:381].sum(axis=1)


valence_perovskite = ns_accumulated_df_valence.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 173+3:237+3].sum(axis=1)
valence_non_perovskite = ns_accumulated_df_valence.iloc[:, 37+3:173+3].sum(axis=1)
valence_O = ns_accumulated_df_valence.iloc[:, 237+3:269+3].sum(axis=1)
valence_other = ns_accumulated_df_valence.iloc[:, 269+3:381+3].sum(axis=1)
valence_total = ns_accumulated_df_valence.iloc[:, 5:381].sum(axis=1)

# %%
conduction_non_perovskite/ conduction_total

# %%
valence_perovskite/valence_total

# %%



