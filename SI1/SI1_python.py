#******************************************************************************************************
# Python code for generating isomers with branches not larger than isopropyl groups
#******************************************************************************************************

#******************************************************************************************************
'''
MIT License

Copyright (c) 2024 Shrinjay Sharma, Ping Yang, Yachan Liu, Kevin Rossi, Peng Bai, Marcello S. Rigutto, Erik Zuidema, Umang Agarwal, Richard Baur, Sofia Calero, David Dubbeldam, and Thijs J.H. Vlugt

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
#******************************************************************************************************
# import libraries
from itertools import permutations
import pandas as pd
import itertools
import re
import subprocess

def generate_isomers(chain_length, num_m_grp, num_e_grp, num_p_grp, num_ip_grp):

    backbone = (chain_length - num_m_grp - 2*num_e_grp - 3*num_p_grp - 3*num_ip_grp)-3
    
    # Generate positions using Python list
    #positions = ['C'] * backbone + ['(C)'] * num_m_grp + ['(CC)'] * num_e_grp + ['(CCC)'] * num_p_grp + ['(C(C)C)'] * num_ip_grp

    # Prepare the inputs for the C++ program
    input_data = f"{chain_length} {num_m_grp} {num_e_grp} {num_p_grp} {num_ip_grp}"

    # Call C++ program to get permutations
    #result = subprocess.run(['./permutation_generator'], input=' '.join(positions), text=True, capture_output=True)
    # Call the C++ program and pass the inputs
    result = subprocess.run(['./permutation_generator'], input=input_data, text=True, capture_output=True)

    
    #print(result)
    # Split the result into individual permutations (one permutation per line)
    permutations_output = result.stdout.strip().split('\n')
    unique_permutations = [line.split() for line in permutations_output]

    # Add two 'C's at the beginning and one 'C' at the end
    isomers_list = [['C', 'C'] + perm + ['C'] for perm in unique_permutations]
    #print(isomers_list)
    
    #*******************function to reject isomers with more than 2 consecutive '(C)'***************************
    def has_more_than_two_consecutive_C(sublist):
        count = 0
        for element in sublist:
            if element == '(C)' or element == '(CC)' or element == '(C(C)C)' or element == '(CCC)':
                count += 1
                if count > 2:
                    return True
            else:
                    count = 0
        return False
    
    # Filter the isomers based on validity
    filtered_isomer_list = [sublist for sublist in isomers_list if not has_more_than_two_consecutive_C(sublist)]
    #filtered_isomer_list = valid_isomers    
    

    
    # Filter the isomers based on validity
    #valid_isomers = [isomer for isomer in filtered_isomer_list if is_valid_isomer(isomer)]
    
    #*******************IUPAC nomenclature***************************
    #*******************compare and store arrays with lowest element position***************************
    
    def compare_lists(A, B, C, D):
        # First, compare lists A and B
        for a, b in zip(A, B):
            if a < b:
                return A,C
            elif a > b:
                return B,D
            
        # If A and B are identical, compare C and D based on priority
        priority = {'e': 4, 'm': 3, 'ip': 2, 'p': 1}
    
        for c, d in zip(C, D):
            if priority[c] > priority[d]:
                return A,C
            elif priority[c] < priority[d]:
                return B,D
    
        # If C and D are also identical, return A,C
        return A,C

    #***************function to store only valid isomers based on IUPAC convention************************
    def validate_and_store(AA, BB, NN):

        pairs = list(zip(AA, BB))
            
        # modification***************************************************
        has_2_m = (2, 'm') in pairs
        has_NNm1_m = (NN-1, 'm') in pairs
        requires_ip1 = (3, 'ip')
        requires_ip2 = (NN-2, 'ip')

            # Iterate over the elements in both lists A and B
        for a, b in pairs:
        
                # Check the conditions based on the value of B[i]
                if b == 'm' and a == 1:
                    return False, AA, BB  # Condition fails for 'm'
                elif b == 'e' and a in [1, 2, NN-1]:
                    return False, AA, BB  # Condition fails for 'e'
                elif b == 'ip' and a in [1, 2, 3, NN-1, NN-2]:
                    return False, AA, BB  # Condition fails for 'ip'
                elif b == 'p' and a in [1, 2, 3, NN-1, NN-2]:
                    return False, AA, BB  # Condition fails for 'p'            
                  
        # If all conditions are satisfied, return a success flag
        return True, AA, BB
    
    def validate_and_store_2(AA, BB, NN):

        pairs = list(zip(AA, BB))
            
        # modification***************************************************

        # Define required structural pairs to check
        required_1 = [(1, 'm'), (1, 'e'), (2, 'e'), (NN - 1, 'e'),
                      (1, 'p'), (2, 'p'), (3, 'p'), (NN - 1, 'p'), (NN - 2, 'p'),
                      (1, 'ip'), (2, 'ip'), (NN - 1, 'ip'), (NN - 2, 'ip')]

        required_2 = [(1, 'm'), (1, 'e'), (2, 'e'), (NN - 1, 'e'),
                      (1, 'p'), (2, 'p'), (3, 'p'), (NN - 1, 'p'), (NN - 2, 'p'),
                      (1, 'ip'), (2, 'ip'), (NN - 1, 'ip')]
    
        # Iterate over the elements in both lists A and B
        
        for a, b in pairs:
            # Case 1: Check for specific ip group at position 3
            if (2, 'm') in pairs and b == 'ip' and a == 3:
                if all(req not in pairs for req in required_1):
                    return True, AA, BB

            # Case 2: ip at NN-2 with (2, 'm') and (NN-1, 'm') present
            if (2, 'm') in pairs and (NN - 1, 'm') in pairs and b == 'ip' and a == NN - 2:
                if all(req not in pairs for req in required_2):
                    return True, AA, BB
        return False, AA, BB
    
    iupac_names, smiles_list = [], []
    
    for filtered_isomer in filtered_isomer_list:
        branch1, branch2 = [], []
        identity1, identity2 = [], []
        
        dummy1 = 0
        # loop from start to end
        for element in filtered_isomer:
            if element == 'C':
                dummy1 +=1
            elif element == '(C)':
                branch1.append(dummy1)
                identity1.append("m")
            
            elif element == '(CC)':
                branch1.append(dummy1)
                identity1.append("e")
                
            elif element == '(CCC)':
                branch1.append(dummy1)
                identity1.append("p")
                
            elif element == '(C(C)C)':
                branch1.append(dummy1)
                identity1.append("ip")
                
        dummy2 = 1
        # loop from end to start    
        for element in filtered_isomer[::-1]:
            if element == 'C':
                dummy2 +=1
            elif element == '(C)':
                branch2.append(dummy2)
                identity2.append("m")
                
            elif element == '(CC)':
                branch2.append(dummy2)
                identity2.append("e")
                
            elif element == '(CCC)':
                branch2.append(dummy2)
                identity2.append("p")
                
            elif element == '(C(C)C)':
                branch2.append(dummy2)
                identity2.append("ip")
                
                
        branch0, identity0 = compare_lists(branch1, branch2, identity1, identity2)
        
        
        is_valid, branch, identity = validate_and_store(branch0, identity0, backbone+3)
        if not is_valid:
            is_valid, branch, identity = validate_and_store_2(branch0, identity0, backbone+3)

        
        #iupac_names = ','.join(map(str,branch))+'-m-C'+str(main_chain)

        def name_isomer(numbers, groups, base_name):
            # Define the priority order
            priority = {'e': 1, 'm': 2, 'ip': 3, 'p': 4}
    
            # Combine numbers and groups into tuples and sort them
            paired_list = sorted(zip(numbers, groups), key=lambda x: (priority[x[1]], x[0]))
    
            # Create a dictionary to group numbers by their substituents
            grouped_parts = {}
            for num, group in paired_list:
                if group in grouped_parts:
                    grouped_parts[group].append(num)
                else:
                    grouped_parts[group] = [num]
    
            # Create the name by combining the numbers and group, sorted by priority
            name_parts = [f"{','.join(map(str, nums))}-{group}" for group, nums in grouped_parts.items()]
    
            # Join the name parts with hyphens and append the base name
            isomer_name = "-".join(name_parts) + f"-C{base_name}"
    
            return isomer_name
        
        base_name = chain_length - num_m_grp - 2*num_e_grp - 3*num_p_grp - 3*num_ip_grp

        if is_valid:
            iupac_names.append(name_isomer(branch, identity, base_name))
            smiles_list.append(''.join(filtered_isomer))

    # removing duplicates
    def remove_duplicates(A, B):
        seen = {}
        for i, item in enumerate(A):
            if item not in seen:
                seen[item] = B[i]  # Keep the corresponding element from B

        # Extract the lists from the dictionary
        new_A = list(seen.keys())
        new_B = list(seen.values())

        return new_A, new_B
    iupac_names, smiles_list = remove_duplicates(iupac_names, smiles_list)
    
    return iupac_names, smiles_list


# User input
chain_length = 20  # Total chain length
input_file = 'branch_index.txt' # input file with desired branch indices for the isomers
output_file = 'iupac_smiles_iC'+str(chain_length)+'.txt'

# Isomer generation
df_input = pd.read_csv(input_file, sep='\t') # tab separated
# Create empty lists to store the generated IUPAC names and SMILES strings
iupac_names = []
smiles_list = []

# Iterate over the rows of the input DataFrame
for index, row in df_input.iterrows():
    # Extract the input values from each row
    num_m_grp = row['m']
    num_e_grp = row['e']
    num_p_grp = row['p']
    num_ip_grp = row['ip']

    # Convert isomer lists to SMILES-like strings and print them
    iupac_name, smiles = generate_isomers(chain_length, num_m_grp, num_e_grp, num_p_grp, num_ip_grp)

    # Append the results to the lists
    iupac_names.append(iupac_name)
    smiles_list.append(smiles)

iupac_names=list(itertools.chain.from_iterable(iupac_names))
                 
smiles_list=list(itertools.chain.from_iterable(smiles_list))


# Open a file to write
with open(output_file, "w") as file:
    for iupac, smile in zip(iupac_names, smiles_list):
        file.write(f"{iupac}\t{smile}\n")







