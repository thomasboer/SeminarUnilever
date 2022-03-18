# SeminarUnilever
Our research investigates what method serves best to model seasonality for multiple time series, which data consist of weekly Google Trends interest scores for Dutch dishes and ingredients over the course of 2018-2021.  More specifically, three different modeling techniques are compared: Fourier series, SARIMAX with Fourier adjusted residuals, Fourier series with SARIMA adjusted residuals. Each of these models is made two-fold, one using Lasso regression and one using AIC forward selection as variable selection techniques. All models are compared to a baseline SARIMAX.

The code to perform the research contains four parts:
1.	All necessary libraries (line 9)
2.	All definitions of self-created functions (line 41)
3.	Code to run the seasonality analysis (line 1241)
        3.1 Pure Fourier with AIC (line 1382)
        3.2 Pure Fourier with Lasso (line 1402)
        3.3 Compute SARIMA (line 1427)
        3.4 Compute FARIMA with Lasso and AIC (line 1433)
        3.5 Compute SARIMAF with Lasso and AIC forward selection (line 1474)
        3.6 Compute in-sample adjusted AIC (line 1487)
        3.7 Streak and peak predictions and evaluation score of all models (line 1499)
        3.8 Create output figures  (line 1609)
4.	Code to convert all output from the analysis to CSV files (line 1688)

How to use the code: 
Step 1. Run all libraries and functions
Step 2. Define the number of keywords you want to run. This can be done in the for loop from line 1303.  
Step 3. Alter the plots, in section 3.8, that need to be generated and stored in a pdf file.
Step 4. Run the entire loop until line 1683. If this is finished run line 1685.
Step 5. Running the code in section 4 allows to save all relevant information in CSV files.
