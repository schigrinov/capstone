# WQU Capstone project - Short-term trading strategy using ML
## Master of Sciense in Financial Engeneering program

* Sergey Chigrinov - chigrinov.s.88@gmail.com
* Dhruv Agrawal -  dhruva1@stanfordalumni.org
* Man Sing Ho - mshoalbert@gmail.com

### Jun-Aug-2020

This repository contains the code for the Capstone project for WQU university https://wqu.org
Thanks a lot, Igor Tulchinsky, for this amazing opportunity!

This repository contains the code (can be used as a package) and notebooks implementing a trading strategy based on machine learning algorythms. Major part of the code was put into a package "WQUcapstoneCode". Notebooks contain visual representation of analysis steps and results. The idea and code are inspired by Marcos Lopez de Prado:"Advances in Financial Machine Learning" and https://www.kaggle.com .

Required libraries:
* most standard libraries - pandas, numpy, matplotlib, seaborn, sklearn ...
* fxcmpy - the main data sourse. You can utilize the csv files from the "input_data" folder instead
* scipy - statistics
* tqdm - progress bar - can be completely removed from the code if you wish
* lightgbm, xgboost - ML algos - didn't make it into the final ensemble - they're too good, so prone to overfitting

The high-level diagram is below:
![Des](/results/Diagram.png)
