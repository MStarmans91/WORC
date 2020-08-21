======================== ======================= ====================================
Subkey                   Default                 Options                             
======================== ======================= ====================================
Variance                 1.0                     Float                               
GroupwiseSearch          True                    Boolean(s)                          
SelectFromModel          0.0                     Float                               
UsePCA                   0.25                    Float                               
PCAType                  95variance, 10, 50, 100 Inteteger(s), 95variance            
StatisticalTestUse       0.25                    Float                               
StatisticalTestMetric    MannWhitneyU            ttest, Welch, Wilcoxon, MannWhitneyU
StatisticalTestThreshold -3, 2.5                 Two Integers: loc and scale         
ReliefUse                0.25                    Float                               
ReliefNN                 2, 4                    Two Integers: loc and scale         
ReliefSampleSize         0.75, 0.25              Two Floats: loc and scale           
ReliefDistanceP          1, 3                    Two Integers: loc and scale         
ReliefNumFeatures        10, 50, 100             Two Integers: loc and scale         
======================== ======================= ====================================