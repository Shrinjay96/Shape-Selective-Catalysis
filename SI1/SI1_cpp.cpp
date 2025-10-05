//******************************************************************************************************
// C++ code for generating permutations for an array of C atoms and the branches like methyl, ethyl, propyl, and isopropyl groups. This code is used in the isomer generation python code.
//******************************************************************************************************

//******************************************************************************************************
/*
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
*/
//******************************************************************************************************

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <sstream>

// Function to generate unique permutations
std::vector<std::vector<std::string>> generate_permutations(const std::vector<std::string>& elements) {
    std::set<std::vector<std::string>> unique_permutations;

    std::vector<std::string> positions = elements;
    std::sort(positions.begin(), positions.end()); // Sort to ensure unique permutations

    do {
        unique_permutations.insert(positions); // Store only unique permutations
    } while (std::next_permutation(positions.begin(), positions.end()));

    // Convert the set into a vector of vectors for output
    std::vector<std::vector<std::string>> perm_list(unique_permutations.begin(), unique_permutations.end());
    return perm_list;
}

int main() {
    // Reading input values from the Python script
    int chain_length, num_m_grp, num_e_grp, num_p_grp, num_ip_grp;
    std::cin >> chain_length >> num_m_grp >> num_e_grp >> num_p_grp >> num_ip_grp;

    // Calculate backbone size
    int backbone = chain_length - num_m_grp - 2 * num_e_grp - 3 * num_p_grp - 3 * num_ip_grp - 3;

    // Generate positions
    std::vector<std::string> positions(backbone, "C");
    positions.insert(positions.end(), num_m_grp, "(C)");
    positions.insert(positions.end(), num_e_grp, "(CC)");
    positions.insert(positions.end(), num_p_grp, "(CCC)");
    positions.insert(positions.end(), num_ip_grp, "(C(C)C)");

    // Generate permutations
    std::vector<std::vector<std::string>> perm_list = generate_permutations(positions);

    // Output the permutations in a format Python can read (one permutation per line)
    for (const auto& perm : perm_list) {
        for (const auto& elem : perm) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
