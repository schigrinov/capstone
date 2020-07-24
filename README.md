# WQU Capstone project - Short-term currency trading strategy using ML
## Master of Science in Financial Engineering program

* Sergey Chigrinov - chigrinov.s.88@gmail.com
* Dhruv Agrawal -  dhruva1@stanfordalumni.org
* Man Sing Ho - mshoalbert@gmail.com

### Jun-Aug-2020

This repository contains the code for the Capstone project for WQU university https://wqu.org
Thanks a lot, Igor Tulchinsky, for this amazing opportunity!

This repository contains the code (can be used as a package) and notebooks implementing a trading strategy based on machine learning algorythms. Major part of the code was put into a package "WQUcapstoneCode". Notebooks contain visual representation of analysis steps and results. The idea and code are inspired by Marcos Lopez de Prado:"Advances in Financial Machine Learning" and https://www.kaggle.com .

Required libraries:
* most standard libraries - pandas v1.0.5, numpy v1.18.1, matplotlib v3.1.3, seaborn v0.10.0, sklearn v0.22.1
* fxcmpy - the main data sourse. You can skip it and utilize the csv files from the "input_data" folder instead
* scipy v1.4.1 - statistics
* tqdm v4.43.0 - progress bar - can be completely removed from the code if you wish
* lightgbm v2.3.0, xgboost v0.90 - ML algos - didn't make it into the final ensemble - they're too good, so prone to overfitting
* pyfolio v0.9.2+73.gcfdf82a - performance analytics

The high-level diagram is below:

![Des](https://github.com/schigrinov/capstone/blob/master/results/Diagram.PNG)
