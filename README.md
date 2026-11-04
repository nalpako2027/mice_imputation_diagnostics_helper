![License](https://img.shields.io/github/license/nalpako2027/mice_imputation_diagnostics_helper?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/nalpako2027/mice_imputation_diagnostics_helper?style=for-the-badge)
![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/orhan-kaplan-phd-5a4a84212/)
![Version](https://img.shields.io/badge/Version-1.0.0-orange) 

# 📊🔍 A Diagnostics Package for Multiple Imputation by Chained Equations (using PMM)
# Under Construction!!!

## 1. 📝 Summary
A from-scratch, `scikit-learn`-only implementation of Multiple Imputation by Chained Equations (MICE) with Predictive Mean Matching (PMM) based on *K*-nearest-neighbour (*K*-NN) donor selection, built for complex-survey data such as TIMSS and PISA. It runs the imputations and returns Rubin's (1987) pooling diagnostics (RVI, FMI, RE) and Von Hippel's (2020) recommended number of imputations, so that you can verify that your imputed results are stable and reproducible.


## 2. 🎯 Objectives

### Why: the case for principled imputation

In corporate data teams and academic research alike, listwise deletion (dropping every row that has at least one missing value in the variables used) remains the most common default for incomplete data (Guenther et al., 2023; Stavseth et al., 2019). Guenther et al. (2023) found that, since 2017, 33.1% of data analysts in business marketing research used listwise deletion and only 0.8% used imputation. Listwise deletion is quick and simple, but it puts internal, external and statistical conclusion validity at risk:  

- **Lost statistical power and signal.** Dropping rows removes valuable data and can shrink the sample substantially. If a customer skips one optional profile question, listwise deletion removes their entire purchase and behavioral history from the model; in a survey, a student who skips one item loses their whole record.
- **Selection bias.** Data are seldom missing completely at random. If particular groups (e.g., lower-income customers, historically underrepresented groups, specific regions) skip questionnaire items more often, listwise deletion **without a valid theoretical and methodological rationale** systematically removes those groups, and the resulting models reflect only respondents who complete every item. This reduces generalizability of the results by failing to accurately represent reality.
- **Wasted data acquisition costs.** Surveys, marketing pipelines and international assessment waves (e.g., TIMSS) require substantial investment, and discarding partially complete records throws away part of that investment in the absence of robust mathematical infrastructure necessary for the secure imputation of missing data elements.


### What: technical goals


Multiply imputing survey constructs for a complex-sample dataset such as TIMSS has three requirements that general-purpose Python imputation libraries do not jointly satisfy. This package is built to meet all three:

1. **Auditable PMM matching.** R's `mice` defaults to an asymmetric matching rule (`matchtype = 1`): donor-pool predictions use the non-stochastic OLS estimate (`beta_hat`), while the recipient's prediction uses a stochastic Bayesian posterior draw (`beta_star`). This is a documented parameter in `mice` and mirrors Stata's `mi impute pmm` default. Python alternatives either do not expose this choice or require reading undocumented source code to confirm what they do, which is a poor foundation for a method used in a study.
2. **Pooling diagnostics beyond point estimates.** Deciding *how many* imputations to run, and confirming that the imputation model behaves well, requires Rubin's Rules quantities (fraction of missing information, relative increase in variance, relative efficiency) and Von Hippel's (2020) two-stage recommended-*M* calculation. Most Python MICE wrappers pool parameter estimates but stop short of these diagnostics.
3. **Complex-survey compatibility.** TIMSS-style data come with sampling weights and jackknife repeated replication (JRR) variance structures. General-purpose imputers have no concept of a design weight and, because they are built as fixed `scikit-learn`-style pipelines, offer no natural place to keep weights, replicate-weight columns and other design metadata untouched and attached to each completed dataset.

Existing options were evaluated against these requirements. For example, `autoimpute` (Kearney & Barkat, 2019) has no weight argument in its `MiceImputer` or `PMMImputer` classes, does not expose the FMI/RVI/recommended-*M* diagnostics used here, and does not expose iteration-level convergence state (only the final completed dataset per chain).


## 3. 📂 Dataset

A sample application is run on the longitudinal **TIMSS 2023-L** wide-format panel dataset (Sweden's data). **[UPDATE: fill in the items below]**

- **Source:** **[UPDATE: IEA, TIMSS 2023 (longitudinal), with the full citation and a link to the data access page]**
- **Size:** **[UPDATE: number of students and number of variables]**
- **Subsetting before imputation:** **[UPDATE: any filtering, e.g. country, cohort, cases with missing weights]**
- **Variables imputed and auxiliary variables:** **[UPDATE: see Table 1 below]**

> **📋 Table 1 (to add):** Variables used in the sample application.
>
> | Variable | Description | Role (imputed / auxiliary) | Scale | % missing |
> |---|---|---|---|---|
> | **[UPDATE]** | | | | |



## 4. 🔧 Methodology

The package treats missing data under a Missing at Random (MAR) assumption and follows van Buuren and Groothuis-Oudshoorn's (2011) three steps of imputation, analysis and pooling. The full derivations, formulas and worked examples are in the notebook: **`mice_imputation_diagnostics_helper.ipynb`**. In short, `mice_and_diagnostics()` does the following:

1. **Stochastic initialization.** Each missing cell is filled with a random draw from the observed values of that variable, which keeps the initial values inside the range and data type of the original scale.
2. **Fully conditional specification (FCS).** For a configurable number of burn-in iterations, the function cycles through every incomplete variable. All other columns, including fully observed auxiliary variables, act as predictors (customizable per variable through `predictor_map`).
3. **Bayesian linear regression and PMM donor matching.** For each variable, the model is fitted on the rows where the variable was originally observed, using a non-informative prior (Rubin, 1987, p. 167; van Buuren, 2018, Sec. 3.2): a χ² draw for the residual variance, then a multivariate normal draw for the coefficients. Following `mice`'s Type 1 matching, donor-pool predictions use `beta_hat` and recipient predictions use the stochastic `beta_star`. Each recipient is matched to its *K* nearest donors on the predicted-mean scale (default *K* = 5), one donor is selected at random, and the donor's **real, observed value** is imputed.
4. **Independent parallel chains.** Each of the `m_datasets` imputations is generated from its own independent RNG stream (`numpy`'s `SeedSequence.spawn`), as Rubin's Rules require.
5. **Pooling diagnostics.** Rubin's Rules give the pooled estimate, within-imputation variance, between-imputation variance and total variance, from which the package reports RVI, FMI and RE, plus Von Hippel's (2020) recommended *M* for each imputed variable.
6. **Convergence and plausibility checks.** Trace plots of the mean (and standard deviation) of each imputed variable across burn-in cycles and chains, plus observed-versus-imputed KDE overlays.
7. **Survey-weighted margins.** Population-weighted means before and after imputation, using a user-specified design-weight column.

> **🖼️ Figure 1:** The structural workflow of the MICE imputation engine (`mice_and_diagnostics()`).
>
> <img width="1080" height="608" alt="Slide1" src="https://github.com/user-attachments/assets/6dfd2123-6531-48b6-b1d4-20334e1f04fe" />

### 🎯 Why PMM?

PMM preserves the natural boundaries and structure of the data. Unlike regular linear-regression imputation, which assumes multivariate normality and can produce unrealistic values (such as negative numbers, or decimals for categorical items), PMM copies every imputed value from a real, observed respondent (donor). This has three advantages for large-scale assessments:

- **Preserves scale boundaries:** categorical items (such as 1–4 Likert scales) stay within authentic, integer values.
- **Maintains non-linear relationships:** skewed variables can be handled without complex transformations.
- **Captures real variance:** randomised *K*-NN donor selection injects realistic, empirical noise, so downstream standard errors and variance components are not artificially deflated.

### 🔢 Choosing the number of imputations

Von Hippel (2020) recommends a two-stage procedure: run a pilot imputation (e.g., *M* = 10), read the recommended *M* for each imputed variable from the diagnostics table, and then run the final imputation with at least the largest recommended *M*.

### ⚖️ Design-weight caveat

Survey weights are used **only** for the before/after weighted margins reported at the end. They are not incorporated into the imputation regressions, which matches the unweighted default of R's `mice` and Stata's `mi impute pmm`. The function is written so that a case-weighted regression or a weight-based auxiliary predictor could be added directly to the regression step if an analysis calls for it (see Reiter, Raghunathan, & Kinney, 2006, on the risks of omitting design features from the imputation model; and Bouhlila & Sellaouti, 2013, who weighted each observation in their own TIMSS MICE application).


### Caution with MICE  
Performing MICE might introduce some technical problems (such as circular dependence, unimputed cells) for certain conditions and illogical or impossible combinations of variable values (such as pregnant father) into the data. Please refer to van Buuren and Groothuis-Oudshoorn (2011, p. 6) for some potential scenarios. As a precaution, perform data quality checks with logical assertions based on theoretical framework of your research. 

## 🔍 Desired Metrics for Successful Imputation  
1. For conservative purposes, run the highest number of imputation needed.
2. FMI and RVI generally should be below 0.15, with couple exception that do not exceed 0.50 (given that sufficient number of imputations are performed).
3. Relative efficiency (for each variable) > 0.90
4. Nicely converging iterations in traceplots
5. Imputed values not deviating from the distribution of observed values (specifically, check if any out-of-bounds exist after imputation).
6. Similar descriptive statistics (compare descriptive statistics of imputed variables before and after imputation)

> ⚠️ **Note:** Make sure you have completely cleaned and checked your raw data matrix, and evaluated the missing mechanism before running the diagnostics helper package! The missing data pattern should at least satisfy MAR condition.


### 📦 Installation

To install the required dependencies for this package, clone the repository, navigate to the project directory, and execute:

```bash
pip install -r requirements.txt
```  
See `requirements.txt` for the dependency list (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).

# A Sample Implementation

## Usage

```python
from mice_diagnostics import mice_and_diagnostics

imputed_list, trace_df, diagnostic_table = mice_and_diagnostics(
    df=jordan_df,
    imputed_cols=['ses_2023', 'digseff_2023', 'mathshortage_2023',
                  'instrclarity_2023', 'schemphsuccess_2023',
                  'confidentmath_2023', 'stud_expect_2023'],
    auxiliary_cols=['math1_2023', 'school_math_mean23'],
    m_datasets=40,
    burnin=10,
    knn=5,
    seed=1234,
    weight_var='TOTWGT',
    label="Jordan"
)
```
> [!WARNING]
> `imputed_list` is a list of `m_datasets` completed DataFrames, each indexed the same way as the input `df` (e.g. by student ID), so non-imputed columns (e.g., IDs, plausible values, replicate weights) can be reattached afterward via a simple index-aligned join.  

## Diagnostics workflow

This repo's diagnostics are meant to be read together, not individually — each catches a different failure mode:

| Check | Catches |
|---|---|
|Recommended M table |Required minimum number of imputations per varaiable|
| MCMC trace plots | Non-convergence within a chain (unstable burn-in) |
| Observed-vs-imputed KDE | Marginal distribution mismatch |
| Before/after weighted margins | Whether imputation shifted population-level estimates |



### 📚 References   

*   Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
*   Guenther, P., Guenther, M., Ringle, C. M., Zaefarian, G., & Cartwright, S. (2023). Improving PLS-SEM use for business marketing research. *Industrial Marketing Management, 111*, 127–142. https://doi.org/10.1016/j.indmarman.2023.03.010
*   Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*. John Wiley & Sons. 
*   Stavseth, M. R., Clausen, T., & Røislien, J. (2019). How handling missing data may impact conclusions: A comparison of six different imputation methods for categorical questionnaire data. *SAGE Open Medicine, 7*, 2050312118822912. https://doi.org/10.1177/2050312118822912
*   van Buuren, S. (2018). *Flexible imputation of missing data* (2nd ed.). CRC Press.
*   van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software, 45*, 3. https://doi.org/10.18637/jss.v045.i03
*   Von Hippel, P. T. (2020). How many imputations do you need? A two-stage calculation using a fractional missing information index. *Sociological Methods & Research*, *49*(3), 699–718. 

## License

![License](https://img.shields.io/github/license/nalpako2027/mice_imputation_diagnostics_helper?style=for-the-badge)


# Attribution

The helper package is free to use (MIT Licence). However, if you use this package in research, publications, presentations, or other work, please cite or acknowledge:

Kaplan, O. (2026). *A Diagnostics Package for Multiple Imputation by Chained Equations (using PMM)*.   
https://github.com/nalpako2027/mice_imputation_diagnostics_helper.git
