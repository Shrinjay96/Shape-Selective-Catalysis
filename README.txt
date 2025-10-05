******************************************************************README******************************************************************

SI1:
1. SI1_python.py: 
	Python code for generating isomers with branches not larger than isopropyl groups
2. SI1_cpp.cpp : 
	C++ code for generating permutations for an array of C atoms and the branches like methyl, ethyl, propyl, and isopropyl groups. This code is used in the isomer generation python code (SI1_python.py)
3. permutation_generator: generated on compiling SI1_cpp.cpp (Already compiled in the SI1 folder)
	g++ -static -o permutation_generator SI1_cpp.cpp
   This is a static compilation on ubuntu 24.04.LTS
4. SI1_isolist.xlsx: 
	contains all isomers ranging from C1 to C20

SI2:
1. SI2_Descriptor.py: 
	Machine Learning (ML) script to predict Henry coefficients for alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523K 
	Descriptor based ML models are used in this script.                                                                                       
	(Random Forest, XGBoost, Cat Boost, and TabPFN)
2. SI2_DMPNN.py:
	Machine Learning (ML) script to predict Henry coefficients for alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523K                 
	Directed Message Passing Neural Network                                                                                        
	Chemprop package
	
3. Input data files to train the ML models:
	linear alkanes (C1-C30) + methyl-branched alkanes (C4-C20)
	SI2_m-Cn_523K_MTW/MTT/MRE/AFI.txt: for Random Forest, XG Boost, Cat Boost, and TabPFN
	SI2_m-Cn_523K_MTW/MTT/MRE/AFI.csv : for D-MPNN (Chemprop) Input feature: SMILES strings

	linear alkanes (C1-C30) + methyl-, ethyl-, propyl-, and isopropyl-branched alkanes (C4-C20)
	SI2_m-e-p-ip-Cn_523K_MTW/MTT/MRE/AFI.txt: for Random Forest, XG Boost, Cat Boost, and TabPFN
	SI2_m-e-p-ip-Cn_523K_MTW/MTT/MRE/AFI.csv : for D-MPNN (Chemprop) Input feature: SMILES strings

4. plot file for D-MPNN
	SI2_plot_file_DMPNN.py

5. run_DMPNN.sh (./run_DMPNN.sh)
	runs both SI2_DMPNN and SI2_plot_file_DMPNN.py

6. SI2_HC.xlsx
	excel file containing 
	a. Henry coefficients computed using molecular simulations for all isomers in MTW-, MTT-, MRE-, and AFI-type zeolites.
	b. Reaction equilibrium distribution for C16 isomers based on Henry coefficients predicted using our ML framework in MTW-, MTT-, MRE-, and AFI-		   type zeolites 

SI3:
1. SI3.pdf: 
	contains:
	 a. parity plots for prediction of negative logarithm of Henry coefficients for alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523 K
	 b. parity plots for prediction of negative logarithm of Henry coefficients for mono-methyl and di-methyl alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523 K using TabPFN model
         c. plots showing the influence of training data selection strategies (random sampling versus active learning) on the performance of the D-MPNN model for predicting Henry coefficients of linear alkanes (C1–C30) and methyl-branched alkanes (C4–C20) in MTW- and MTT-type zeolites.
	 d. average Henry coefficients for C16 isomers in MTW-, MTT-, MRE-, and AFI-type zeolites at 523 K 
	 e. average selectivities for C16 isomers relative to linear C16 in MTW-, MTT-, MRE-, and AFI-type zeolites at 523 K 