=========================== ======================= ====================================
Subkey                      Default                 Options                             
=========================== ======================= ====================================
Variance                    1.0                     Float                               
GroupwiseSearch             True                    Boolean(s)                          
SelectFromModel             0.2                     Float                               
SelectFromModel_estimator   Lasso, LR, RF           Lasso, LR, RF                       
SelectFromModel_lasso_alpha 0.1, 1.4                Two Floats: loc and scale           
SelectFromModel_n_trees     10, 90                  Two Integers: loc and scale         
UsePCA                      0.2                     Float                               
PCAType                     95variance, 10, 50, 100 Integer(s), 95variance              
StatisticalTestUse          0.2                     Float                               
StatisticalTestMetric       MannWhitneyU            ttest, Welch, Wilcoxon, MannWhitneyU
StatisticalTestThreshold    -3, 2.5                 Two Integers: loc and scale         
ReliefUse                   0.2                     Float                               
ReliefNN                    2, 4                    Two Integers: loc and scale         
ReliefSampleSize            0.75, 0.2               Two Floats: loc and scale           
ReliefDistanceP             1, 3                    Two Integers: loc and scale         
ReliefNumFeatures           10, 40                  Two Integers: loc and scale         
=========================== ======================= ====================================