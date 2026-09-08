![License](https://img.shields.io/github/license/nalpako2027/mice_imputation_diagnostics_helper?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/nalpako2027/mice_imputation_diagnostics_helper?style=for-the-badge)
![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/orhan-kaplan-phd-5a4a84212/)

# 📊🔍 A Diagnostics Package for Multiple Imputation by Chained Equations (using PMM)
  
A from-scratch, `sklearn`-only implementation of Multiple Imputation by Chained Equations (MICE) with Predictive Mean Matching (PMM) based on the K-Nearest Neighbours (*K*-NN) donor selection mechanism, built for complex-survey data (e.g. TIMSS, PISA) where off-the-shelf imputation packages fall short. This engine bridges that fragmentation gap, runnning MICE imputations and computing a comprehensive Rubin's (1987) Diagnostics Matrix containing the Relative Variance Increase (RVI), Fraction of Missing Information (FMI), and Relative Efficiency (RE) alongside Von Hippel’s (2020) Two-Stage Quadratic Sufficiency Metric to calculate minimum number of imputation for reproducible standard errors.



## 📈 💼 The Scientific and Business Implications and Hidden Risks of Default Analytics  
In both corporate data departments and academic institutions, listwise deletion (completely dropping any row with at least one missing cell for any variable used) remains the most common default method for handling incomplete datasets (Guenther et al., 2023; Stavseth et al., 2019). A recent peer-reviewed article found that, since 2017, 33.1% of data anlaysts in business marketing research performed listwise deletion and only 0.8% used imputation (Guenther et al., 2023). Whilst data teams frequently utilise this approach due to its expeditious and straightforward nature, the blind dropping of valuable data engenders considerable risk with regard to operational decision-making, strategic investments, and especially the validity of the results (internal validity, external validity and statistical conclusion validity). 

The implementation of a robust MICE framework such as this engine for dealing with missing values is imperative for businesses and research for three reasons:  
1. **Destruction of Statistical Power and Revenue Signal:** The elimination of rows in an aggressive manner can result in a substantial reduction of the sample size. For an enterprise, this necessitates the discarding of valuable consumer or product tracking data. In the event that a customer elects to bypass a question pertaining to optional profiles, the consequence of listwise deletion is the eradication of their complete purchase and behavioural history from the predictive model, a development which has the potential to compromise the statistical power of the model and relevant inferences.
2. **The Danger of Selection Bias:** Generally, data is seldom missing completely at random. In instances where lower-income customers (or historically underrepresented groups) or specific regions exhibit a higher propensity to skip questionnaire columns, listwise deletion **without a valid theoretical and methodological rationale** systematically deletes those entire demographics from the database. Consequently, predictive/causal models will optimise exclusively for the subset of users who complete every form, resulting in algorithms that are significantly biased and consequently fail to capitalise on substantial problems/opportunities by failing to accurately represent reality in the real world.
3. **Wasted Data Acquisition Costs:** The procurement of data through user surveys, marketing pipelines, or panel research (e.g. international TIMSS waves) necessitates a substantial financial investment. The practice of listwise deletion in data science entails the deliberate discarding of 10% to 30% of a paid-for dataset, primarily due to the absence of robust mathematical infrastructure necessary for the secure imputation of missing data elements.

### 🎯 Why Choose MICE with Predictive Mean Matching (PMM)?  
Predictive Mean Matching (PMM) is highly recommended for this pipeline because it preserves the natural boundaries and structural integrity of the data. Unlike regular linear regression, which requires multivariate normality and frequently imputes unrealistic values (such as negative numbers or decimals for categorical items), PMM ensures that every imputed value is copied directly from a real, observed respondent (donor).This approach provides three critical advantages for large-scale assessments:  
- Preserves Scale Boundaries: It restricts categorical items (like 1–4 Likert scales) to authentic, integer values.
- Maintains Non-Linear Relationships: It handles skewed variables without requiring complex data transformations.
- Captures Real Variance: It avoids over-smoothing by injecting realistic, empirical noise through a randomized *k*-nearest-neighbor donor selection, ensuring your downstream standard errors and variance components are not artificially deflated.


## The problem

Multiply imputing survey constructs for a complex-sample dataset like TIMSS has three requirements that general-purpose Python imputation libraries don't jointly satisfy:

1. **Auditable PMM matching.** R's `mice` package defaults to an asymmetric PMM matching rule (`matchtype = 1`): the donor pool's predicted values use the non-stochastic OLS point estimate (`beta_hat`), while the recipient's predicted value uses a stochastic Bayesian posterior draw (`beta_star`). This is a documented, user-facing parameter in `mice` and mirrors Stata's `mi impute pmm` default. Python alternatives either don't expose this choice at all, or require reading undocumented source code to confirm what they actually do — which is a poor foundation for a method used in a peer-reviewed manuscript.
2. **Pooling diagnostics beyond point estimates.** Deciding *how many* imputations to run, and confirming the imputation model is behaving well, requires Rubin's Rules quantities (fraction of missing information, relative increase in variance, relative efficiency) and Von Hippel's (2020) two-stage recommended-*M* calculation. Most Python MICE wrappers pool parameter estimates but stop short of these diagnostics.
3. **Complex-survey compatibility.** TIMSS-style data comes with sampling weights and jackknife repeated replication (JRR) variance structures. General-purpose imputers have no concept of a design weight at any level, and — because they're built as fixed `scikit-learn`-style pipelines — offer no natural place to keep weights, replicate weight columns, and other design metadata untouched and attached to each completed dataset.

Existing options were evaluated and rejected on these grounds specifically. `autoimpute` (Kearney & Barkat, 2019), for example, has no weight argument anywhere in its `MiceImputer` or `PMMImputer` classes, does not expose the FMI/RVI/recommended-*M* diagnostics used here, and does not expose iteration-level convergence state (only the final completed dataset per chain).

## The solution

`mice_and_diagnostics()` implements the full pipeline as a single, readable, dependency-light function:

- **Imputation model**: Bayesian linear regression under a noninformative prior, following Rubin (1987, p. 167) and van Buuren (2018, Sec. 3.2, *Flexible Imputation of Missing Data*, 2nd ed.) — a $\chi^2$ draw for $\dot\sigma^2$, then a multivariate normal draw for $\dot\beta$.
- **PMM donor matching**: Implements `mice`'s default Type 1 matching explicitly — donor-pool predictions use `beta_hat`, recipient predictions use the stochastic `beta_star` draw — matched via k-nearest-neighbors on the predicted-mean scale, with the donated value always a real, previously observed value (not a regression prediction).
- **Fully conditional specification (FCS)**: Cycles through all incomplete variables for a configurable number of burn-in iterations, with per-variable predictor lists optionally customized via `predictor_map`.
- **Independent parallel chains**: Each of the `m_datasets` imputations is generated from its own independent RNG stream (via `numpy`'s `SeedSequence.spawn`), as required for Rubin's Rules.
- **Pooling diagnostics**: Computes pooled parameter estimates, within-imputation varaiance, between-imputation variance, total variance to calculate relative increase in variance (RVI), fraction of missing information (FMI), relative efficiency (RE), and Von Hippel's (2020) recommended number of imputations.
- **Convergence and plausibility diagnostics**: Markov Chain Monte Carlo (MCMC)-style trace plots of each imputed variable's mean across burn-in cycles and chains, plus observed-vs-imputed KDE density overlays.
- **Survey-weighted margins**: Reports population-weighted means before and after imputation (Rubin's grand average across the pooled datasets), using a user-specified design weight column.
- **Design-weight caveat, stated explicitly**: Survey weights are used only for the before/after weighted margins reported at the end — they are not incorporated into the imputation regressions themselves, matching the unweighted default behavior of R's `mice` and Stata's `mi impute pmm`. The function is written so that a case-weighted regression or a weight-based auxiliary predictor could be added directly to the OLS step if a given analysis calls for it (see Reiter, Raghunathan, & Kinney, 2006, on the risks of omitting design features from the imputation model; and Bouhlila & Sellaouti, 2013, who weighted each observation in their own TIMSS MICE application).

A sample application is performed on the longitudinal TIMSS 2023-L wide format panel dataset (Sweden's data) for representation. An illustrative approach to a problem is also analyzed and discusssed in the notebook. 


## ⚙️🛠️ Algorithmic Workflow & Methodology  
The multiple imputation engine treats missing data under a Missing at Random (MAR) structural assumption, executing an iterative chain of conditional univariate models on a variable-by-variable vasis (van Buuren \& Groothuis-Oudshoorn, 2011). Adopting van Buuren and Groothuis-Oudshoorn's (2011) steps of imputation, analysis and pooling, the current engine processes the data using a three-stage pipeline:

The structural workflow of the **MICE imputation engine (`mice_and_diagnostics()`)**:  

<img width="1080" height="608" alt="Slide1" src="https://github.com/user-attachments/assets/6dfd2123-6531-48b6-b1d4-20334e1f04fe" />


### Stage 1: Stochastic Initialization 
To begin the chain, missing parameters cannot remain empty. The function executes an unconditional random draw directly from the observed margins of the target variables, represented by the data matrix of $Y = (Y_{1}, \dots , Y_{n})$ for $n = (1, \dots , n)$ number of variables with missing values, with every observed values having an equal chance of being selected (i.e., uniformly distributed possibilities). For the iteration 0 (the baseline initialization stage) this can be shown as:    


$${\large Y_{mis,i}^{(0)} \sim \text{Uniform}(Y_{obs})}$$


This approach preserves the baseline range and data type boundaries of continuous, ordinal and nominal scales before modeling begins.


### Stage 2: Chained Regression & Predictive Mean Matching  
For each iteration $t \in {1, \dots, \text{burn\-in}}$ (where burn-in is the number of initial cycles the algorithm runs until the simulation is stabilized through statistical convergence, removing the noise introduced by the initial random guesses) and each target variable $Y_{j}$, all other columns $X$ (including completely observed covariates or auxiliary variables) act as predictors $X_{-j}$.  

The imputations is dynamically performed through these steps:  

1.	OLS Parameter Estimation: An ordinary least squares regression is fitted strictly on the rows where the target variable was originally present:

$$
\hat{\beta}^{(t)} = (X_{obs}^{T} X_{obs})^{-1} X_{obs}^{T} Y_{obs}
$$
  
2.	Linear Metric Projections: Expected values are predicted for both the observed donor pool and missing rows at iteration *t*:


$$
\hat{Y}_{obs} = X_{obs} \hat{\beta}^{(t)} \quad \text{and} \quad \hat{Y}_{mis} = X_{mis} \hat{\beta}^{(t)}
$$

3.	$K$-NN Donor Pooling: For each missing row vector \(i\), a distance metric is established across the projected space to isolate the \(K=5\) closest observed donors:

$$
d\left(\hat{y}_{mis,i}, \hat{y}_{obs,j}\right) = \left|\hat{y}_{mis,i} - \hat{y}_{obs,j}\right|
$$

4.	Stochastic Selection: One donor is selected at random from the 5 closest candidates, and their actual, raw observed value is mapped into the missing cell. This step avoids the artificial smoothing caused by linear mean imputations, maintaining the natural variance of the dataset.


### Missing Information Diagnostics Using Rubin's Rule (1987)

Once $M$ independent completed datasets are generated using unique random seeds, their stability is evaluated across the structural margins using **Rubin's (1987) Rules**.

Let $\hat{Q}_m$ be the sample mean of variable $Y$ in imputation $m$, and $W_m$ be its corresponding within-imputation sampling variance ($W_m = \sigma^2_m / M$), where *N* is the total number of rows.

The **Pooled Parameter Estimate** ($\overline{Q}$) and **Within-Imputation Variance** ($\overline{W}$) are calculated as:

$$\overline{Q} = \frac{1}{M}\sum_{m=1}^{M}\hat{Q}_m \quad \text{and} \quad \overline{W} = \frac{1}{M}\sum_{m=1}^{M}W_m$$

#### Between-Imputation Variance ($B$) and Total Variance ($T$)

The structural variation across the independent imputation streams represents the uncertainty added by the missing data points:

$$B = \frac{1}{M-1}\sum_{m=1}^{M}(\hat{Q}_m - \overline{Q})^2$$  

*Note: For single-variable scalar parameters like a column mean, the vector transpose inner product in Rubin (1987) simplifies mathematically to a standard squared difference because transposing a single number does not alter its dot product value.*

The **Total Variance** ($T$) combines both sources of error, incorporating a finite-sample correction multiplier $(1 + 1/M)$:

$$T = \overline{W} + \left(1 + \frac{1}{M}\right)B$$

*   **Relative Variance Increase** ($RVI$) measures the proportional increase in variance due to non-response:

$$\text{RVI} = \frac{\left(1+\frac{1}{M}\right)B}{\overline{W}}$$

*   **Fraction of Missing Information** ($FMI$) measures the proportion of total uncertainty directly caused by the missing values, adjusted for finite sample degrees of freedom ($\nu_{m}$):

$$\text{FMI} = \frac{\text{RVI}+\frac{2}{\nu_{m}+3}}{1+\text{RVI}} \quad \text{where} \quad \nu_{m}=(M-1)\left(1+\frac{1}{\text{RVI}}\right)^{2}$$


#### Relative Efficiency ($RE$)

**Relative Efficiency** replicates Stata's key verification check, tracking the stability of your current sample pool ($M$) relative to an infinite number of imputations ($M = \infty$):

$$\text{RE} = \frac{1}{1+\frac{\text{FMI}}{M}}$$

An $\text{RE}$ score exceeding $0.95$ (95% efficiency) is the standard benchmark for stable, publication-grade standard errors.


#### How Many Imputations? Von Hippel’s Two-Stage Sufficiency Rule

To determine if your choice of $M$ is sufficient, the engine calculates a **Recommended $M$** using Von Hippel's (2020) quadratic rule. The calculation of the M should be evaluated on a pilot imputation with 10 imputations. The final MICE should be performed based on the M that yield reproducible standard errors. To remain conservative, a logit transformation establishes the 95% upper confidence limit of the $\text{FMI}$ ($\gamma$):

$$\text{logit}(\gamma_{\text{mis}}) = \text{logit}(\hat{\gamma}_{\text{mis}}) \pm z\sqrt{\frac{2}{M}}$$  

Here z ($1.96$) represents the standard normal critical value for a 95% confidence interval. Given a target coefficient of variation tolerance of $\alpha = 0.05$, the structural sample size threshold is:

$$M_{\text{recommended}} = \left\lceil 1+0.5\left(\frac{\gamma}{\alpha}\right)^{2}\right\rceil$$

Recommended M's are calculated for each variable with missing value. To be on the safe side, the number of imputations should be--at least--equal to the largest `Recommended M`
### 📉 Convergence Monitoring via MCMC Trace Plots  

Because Multiple Imputation by Chained Equations (MICE) is structurally an iterative Markov Chain Monte Carlo (MCMC) algorithm, monitoring convergence is mandatory to ensure statistical validity (van Buuren & Groothuis-Oudshoorn, 2011). The Diagnostics Package generates iterative trace plots tracking the mean and standard deviation of imputed values across the full burn-in sequence for every parallel chain M. Healthy convergence is visually confirmed when the paths of the independent streams interweave randomly—resembling overlapping "fuzzy caterpillars"—which demonstrates that the arbitrary random noise from Stage 1 Initialization Y^{(0)} has been completely washed away.

### 📊 Distributional Plausibility Diagnostics  
To verify that the Predictive Mean Matching (PMM) donor selection mechanism is operating correctly, the package computes distributional plausibility diagnostics using kernel density estimation (KDE) overlays (van Buuren, 2018). This visualization isolates the subset of values generated by the *K*-nearest neighbors matching pool $knn=5$ and directly compares their probability distribution profile against the raw, observed records. A methodologically sound imputation requires the imputed distribution to mirror the bounds, peaks, and shapes of the valid data, providing empirical proof that the package has preserved the original data constraints without introducing mathematical distortion or out-of-bounds metrics.


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
