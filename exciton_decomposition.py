import pandas as pd
import numpy as np

# Things to modify by hand
iN_S_select =1

kgrid_file = 'kgrid.log_cocn.bak'
kgrid_file = 'kgrid.log_CYS.bak'
kgrid_file = 'kgrid.log_APD2.bak'
#kgrid_file = 'kgrid.log_APD1.bak'

bandstructure_file = 'bandstructure.dat_cocn.bak'
bandstructure_file = 'bandstructure.dat_CYS.bak'
bandstructure_file = 'bandstructure.dat_APD2.bak'
#bandstructure_file = 'bandstructure.dat_APD1.bak'

fermi = 280 # ! for HOCN
fermi = 304   # ! for COCN
fermi = 544   # ! for CYS
fermi = 408 # ! VBM - start + 1 , hard-coded for APD2_Li now, read from projbands file
#fermi = 432   # ! for APD1_Na


eigenvector_file = 'eigenvectors.h5_hocn.bak'
eigenvector_file = 'eigenvectors.h5_cocn.bak'
eigenvector_file = 'eigenvectors.h5.CYS.bak'
eigenvector_file = 'eigenvectors.h5.APD2.bak'
#eigenvector_file = 'eigenvectors.h5.APD1.bak'
#eigenvector_file = 'eigenvectors_APD1_relax.h5'

scf_file = 'relax.out_cocn.bak'
scf_file = 'scf.out_CYS.bak'
scf_file = 'scf.out_APD2_Li.bak'
#scf_file = 'scf.out_APD1.bak'

projbands_file = 'cl.projbands_hocn.bak'
projbands_file = 'cl.projbands_cocn.bak'
projbands_file = 'cl.projbands.CYS.bak'
projbands_file = 'cl.projbands_APD2_Li.bak'
#projbands_file = 'cl.projbands.apd1.bak'
#projbands_file = 'cl.projbands_APD1_relax.bak'

e_min = 0
e_high = 5



## 0.1 

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

# 0.2 

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
two_pi_over_alat = 2 * np.pi / alat
reciprocal_lattice_vectors = np.array(b_vectors) * two_pi_over_alat


# Convert cartesian coordinates to fractional coordinates
def cartesian_to_fractional(cartesian_coordinates, reciprocal_lattice_vectors):
    return np.dot(cartesian_coordinates, np.linalg.inv(reciprocal_lattice_vectors))

def fractional_to_cartesian(fractional_coordinates, reciprocal_lattice_vectors):
    return np.dot(fractional_coordinates, reciprocal_lattice_vectors)

fractional_coordinates = cartesian_to_fractional(data_matrix, reciprocal_lattice_vectors)
# Around to 3 decimals
fractional_coordinates = np.around(fractional_coordinates, decimals=3)

# Convert fractional coordinates to the BZ within (0,0,0) to (1,1,1) by translating the coordinates
def translate_to_BZ(fractional_coordinates):
    return fractional_coordinates - np.floor(fractional_coordinates)

translated_coordinates = translate_to_BZ(fractional_coordinates)

# Sort the translated coordinates first by kx, then by ky, then by kz, and keeping the original indices
def sort_by_k(translated_coordinates):
    indices = np.lexsort((translated_coordinates[:,2], translated_coordinates[:,1], translated_coordinates[:,0]))
    return translated_coordinates[indices], indices

sorted_coordinates, indices = sort_by_k(translated_coordinates)
# append the indices to the processed_data
processed_kgrid_wfn_eigenvector = np.hstack((processed_data, indices.reshape(-1,1)))


# Step 1:

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
data = [line.split() for line in data]
df = pd.DataFrame(data).astype(float)

## Read only partial

import numpy as np
import h5py as h5

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
            temp_contrib_cv = np.sum(abs(evc[:,:,:,0]+evc[:,:,:,1]*1j)**2, axis=0) # Summarize_eigenvectors and Ploteigenvectors - For a certain k and c, calculating the sum of Acvk across all vs, and vice versa, this is the correct way to do it, because for a certain k, valence band pdos idoes not change when we fix a v, and vice versa, But you ignore the correlation between c and v, thus you miss the ECF, but you can only plot ECF_noshift
            temp_contrib_v = np.sum(temp_contrib_cv, axis=0)
            temp_contrib_c = np.sum(temp_contrib_cv, axis=1)

            contrib_c_dict[(iN_S_select,k_select)] = temp_contrib_c
            contrib_v_dict[(iN_S_select,k_select)] = temp_contrib_v
        
        return 1, nk, nc, nv  # 由于只读取一个iN_S，nS始终为1

for k in range(0,nk):
    nS, nk, nc, nv = plot_for_k(eigenvector_file,k,iN_S_select,0,5)


conduction_df = df.groupby(0).apply(lambda x: x.iloc[fermi:fermi+nc,:])
valence_df = df.groupby(0).apply(lambda x: x.iloc[fermi-nv:fermi,:])
reversed_valence_df = valence_df.iloc[::-1,:]


# Normalize the reversed_valence_df and conduction_df by the sum of the pdos of each k (the 3rd column)

reversed_valence_df = reversed_valence_df.apply(lambda x: x/x[3], axis=1)
conduction_df = conduction_df.apply(lambda x: x/x[3], axis=1)

ns_accumulated_df_valence = pd.DataFrame()
ns_accumulated_df_conduction = pd.DataFrame()
#for s_select in range(nS):
for s_select in range(iN_S_select,iN_S_select+1):
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


## Hard coded

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


valence_perovskite = ns_accumulated_df_valence.iloc[:, 37+3:73+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 105+3:169+3].sum(axis=1)
valence_non_perovskite = ns_accumulated_df_valence.iloc[:, 1+3:37+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 73+3:105+3].sum(axis=1)
valence_O = ns_accumulated_df_valence.iloc[:, 169+3:233+3].sum(axis=1)
valence_Li = ns_accumulated_df_valence.iloc[:, 609+3:625+3].sum(axis=1)
valence_total = ns_accumulated_df_valence.iloc[:, 5:660].sum(axis=1)
valence_other = ns_accumulated_df_valence.iloc[:, 233+3:609+3].sum(axis=1) + ns_accumulated_df_valence.iloc[:, 625+3:657+3].sum(axis=1)
valence_others = valence_total - valence_perovskite - valence_non_perovskite - valence_O - valence_Li - valence_other


print(conduction_perovskite/ conduction_total)
print(conduction_non_perovskite/ conduction_total)

print(valence_perovskite/valence_total)
print(valence_non_perovskite/valence_total)
print(valence_O/valence_total)
print(valence_other/valence_total)