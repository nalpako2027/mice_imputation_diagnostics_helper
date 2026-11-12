# Executive Summary

## Problem

Missing data is common in survey, administrative, and operational datasets, and how it is handled affects every analysis built on top of it. Two common approaches carry risks that are easy to miss because neither throws an error:

- **Listwise deletion** discards information and can bias results when missingness is related to the variables being analyzed, which is common in real data.
- **Black-box imputation** is risky when you cannot see what the tool does. General-purpose packages do not always make their statistical steps or diagnostics easy to inspect.

## Solution

This project is a transparent multiple imputation pipeline in Python: multiple imputation by chained equations (MICE) with predictive mean matching (PMM). The code is open source and short enough to read line by line and adapt, for example to add survey weights to the regression step. Each step follows a documented method and comes with its own diagnostics.

**What it does**

- Imputes missing values with PMM, which only uses values actually observed for that variable, so imputations stay within the observed range.
- Runs `M` independent chains and pools them with Rubin's rules, so the uncertainty caused by missing data is quantified instead of hidden behind a single best guess.
- Reports RVI, FMI and relative efficiency for each variable, and a recommended number of imputations (von Hippel, 2020).
- Provides visual diagnostics: convergence trace plots, observed-vs-imputed distributions, and before/after survey-weighted means.

**Assumptions and limits**

- Data are assumed Missing at Random (MAR), and cleaning is done before the function is called.
- Survey weights are used only for the before/after means. The imputation model is unweighted, and the pooled diagnostics assume simple random sampling (no clustering).
- See [Limitations](#limitations) for details.

**Intended use:** settings where a silent data-quality error is costly and the imputation process needs to be inspectable and defensible.
