# SeminarUnilever
Our research investigates what method serves best to model seasonality for multiple time series, which data consist of weekly Google Trends interest scores for over Dutch dishes and ingredients over the course of 2018-2021. The code to perform the research contains four parts:
1.	All necessary libraries 
2.	All definitions of self-created functions 
3.	Code to run the seasonality analysis
        3.1 Pure Fourier with AIC
        3.2 Pure Fourier with Lasso
        3.3 Compute SARIMA
        3.4 Compute FARIMA with Lasso and AIC
        3.5 Compute SARIMAF with Lasso and AIC forward selection
        3.6 Compute in-sample adjusted AIC
        3.7 Streak and peak predictions and evaluation score of all models
        3.8 Create output figures   
4.	Code to convert all output from the analysis to CSV files
