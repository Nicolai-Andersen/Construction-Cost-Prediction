# Construction-Cost-Prediction
Kaggle[1] is a data science and machine learning competition platform under Google LLC. where organizations pose real-world, data-driven problems. We have selected the Construction Cost Prediction competition provided by Solafune [2]. The competition tasks its participants with creating a novel method for predicting the construction cost per square meter for a given geographical area using a combination of economic data and satellite features. Crucially, Solafune require a dynamic model that can generalize well across different countries and rapidly changing economic factors.

The economic data is sourced from surveys conducted in Japan and the Philippines; Statistical Survey of Construction Starts and Construction Statistics from Approved Building Permits respectively. The Satellite data is divided into Sentinel-2, which is a 12-band, multispectral imaging tool encapsulating a wide range of data. And VIIRS, which provides data for DNB radiance; a measure of nighttime light emissions. All data is provided by Solafune.

# Data exploration and analysis
This is done in the Visualization notebook

# Individual work
This is the part of the project where each member of the group did their own implementation of the methods in the papers selected
## C-Mixup
By Nicolai Ramsvik Andersen

C-Mixup is a variant of the Mixup data augmentation technique designed to improve the generalization of regression models by creating synthetic training examples. Traditional Mixup generates synthetic samples by randomly interpolating pairs of training examples and their labels. This can produce arbitrarily incorrect labels in regression settings, where the label space is continuous mixing two samples with very different labels yields a synthetic label that may not correspond to any plausible real-world value.

C-Mixup addresses this by preferentially pairing samples with similar labels. A pairwise label-distance matrix is computed once before training using a Gaussian kernel, giving each sample a probability distribution over mixing partners. Samples with closer label values are assigned higher probability of being selected as a pair. For each training step, a batch of real samples is drawn, each sample's mixing partner is sampled from this distribution, and the interpolation ratio λ is drawn from a Beta(α, α) distribution. The two samples including their tabular features, satellite image embeddings, and label are then linearly interpolated to produce a synthetic training example. The model is trained exclusively on these synthetic examples, and the process repeats each epoch.

This approach has three key advantages over vanilla Mixup for regression: it improves in-distribution generalization by avoiding nonsensical interpolated labels, it improves out-of-distribution robustness by mixing across domains without requiring domain annotations, and it is computationally efficient since the kernel operates in label space rather than feature space.

## IBUG

## RegBN
Contribution by Lyder Samnøy

Main Implementation can be found in *RegBN_Implementation.ipynb*. *RegBN_Environment.py* refactores logic from *NNpipeline.ipynb* and adds features for RegBN in a stabile .py environment. *RegBN.py* is taken from RegBN's GitHub repository (not authored by us), only minimal changes are made for better cpu support.

RegBN is a tool for normalizing multimodal data created by Morteza Ghahremani and Christian Wachinger from the *Technical University of Munich (TUM), Germany*. The tool aims to remove features from a given dataset that are linearly dependent, thereby reducing confounding effects in the data. Unlike standard Batch Normalization, RegBN does not “learn” via packpropagation. Instead, it computes a Projection Matrix from the training data, meaning it has essentially no learnable parameters. It can therefore be added between the tabular/image encoders and the fusion step. Because of this, RegBN only needs to be computed once per batch, treating normalization as an algebraic problem rather than a learnable one.

Because RegBN requires the use of pytorch with CUDA-support to be installed, the notebook *RegBN_Implementation.ipynb* houses the separate, complete implementation for ease of use.

The methodology for implementing RegBN is found in the paper.

Our results from integrating RegBN into the model pipeline show that almost no features in our datasets are actually linearly dependent, leading to only a small change in model preformance. As the change was a slight increase in loss and not a decrease, RegBN is not used in the main implementation in this paper. First and foremost; this shows us that our datasets are already quite robust, with few dependent or confounding features. 
**Why did RegBN decrease model preformance?**
RegBN removes confounding effects in data. Naturally, this is most beneficial when such effects are detrimental to the prediction. For example, a model that classifies animals mught mislabel a wolf as a dog if that dog happens to be in snowy environment etc. However for our construction cost prediction, confounding effects likely have a positive effect on model performance, as they affect the regression head directly. Dependencies between economic and satellite data are likely a real signal more often than a spurrious one. 

Below are some validation results from *RegBN_Implementation.ipynb*. For illustration, these results are only from the full model:


### Full RegBN Validation:


| Metric | Value |
| :--- | :--- |
| **n_validation** | 205 |
| **tab_change_%** | 75.77 |
| **img_change_%** | 92.52 |
| **tab_img_corr_before** | 0.01 |
| **tab_img_corr_after** | 0.84 |
| **tab_img_corr_reduction_%** | -12399.10 |
| **dependence_direction** | increased |
| **pred_shift_mean** | 2601.01 |
| **pred_shift_median** | 1160.79 |
| **pred_before_mean** | 3685.30 |
| **pred_after_mean** | 1087.94 |
| **target_mean** | 1099.71 |
| **pred_before_target_gap** | 2585.59 |
| **pred_after_target_gap** | 11.77 |
| **target_gap_change** | -2573.82 |
| **prediction_alignment** | improved |

Note how *tab_change* and *img_change* both show large changes in the data despite a difference in loss of only 1.3% on average. This is caused by the transformations to the latent embeddings computed by the Projection Matrix; Although many features show change, they are likely minute changes of position that RegBN then further regularizes. The rest of the datapoints were an attempt to inspect changes in the data. However, the major problem here is that these metrics are not a comparison against our original model. This is all done inside the RegBN-trained model, with and without the RegBN step. This means that the downstream head is expecting RegBN-adjusted embeddings, breaking any semblance of callibration. Thus, the "before" values are wildly inaccurate, they can only show us that RegBN improves performance in the model it already used to train, which is expected. These validation results by themselves would not indicate a 1.3% performance difference.

In conclusion. RegBN is proven to be a valuable tool for normalizing and regularizing multimodal data, but only for domains where confounding effects are inherently disruptive. This appears not to be the case for our project domain. As mentioned above; dependent features likely have a positive effect on model performance as these correlations projected out by RegBN seem to have had a direct and positive effect on the final regression result, as noted by the marginally greater performance of the original model.

