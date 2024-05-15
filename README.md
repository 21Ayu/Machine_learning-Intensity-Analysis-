# Machine_learning-_Capstone-Project
The objective of this project is to develop an intelligent system using NLP to predict the intensity in the text reviews. By analyzing various parameters and process data, the system will predict the intensity where its happiness, angriness or sadness.

## Importing Necessary Libraies


![Screenshot (1679)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b6c08223-4f74-4e70-8a66-5b5da07639ed)
## Data Collection 


![Screenshot (1680)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/0e20fa43-93ae-4809-a641-1e756ae00184)

## Data Overview


![Screenshot (1681)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b6177991-c0e2-4962-b99d-c5f91fff6e26)

## EDA (Exploratory Data Analysis )


![Screenshot (1683)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/f05f41fe-2c70-495d-b2e7-b60f09c428ff)



![Screenshot (1682)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/cf708816-3f8f-4509-b07b-9e6117f2c8d4)




## Feature Enginerring


  ### Label ENCODING (Converting Textual Intensity Column into Numerical for Machine Learning Classification)
  
![Screenshot (1684)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/9e6511df-3235-4f2f-b265-8ba596705211)

### Tokenisation

![Screenshot (1685)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/41e87f37-aeba-4089-9f26-7eaabd6856b4)


### Lemmatization

![Screenshot (1687)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/640c000d-31c9-469b-aa92-c4bba51b1820)

 #### CLEANED Text
 
 ![Screenshot (1686)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/1765a63a-02e4-476b-9fb4-bed29ae96888)


### Vectorization



  #### Bag of Words
  
![Screenshot (1688)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/18d2ac8a-3f8d-4cd0-b201-9db1cc86b0ed)

  #### TFDIF
  
![Screenshot (1689)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/d732a7a3-6e93-44bb-b6fe-345919de8f2a)

  #### Embedding
  
![Screenshot (1690)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/5cfd4174-79c1-4427-baf2-663fcc6094a4)

## Split Train Data

![Screenshot (1691)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/cc0fd080-8787-43c0-8288-5bddab4b2c0b)

## Model Selection
##### Here we will use  Four Models for Analysis :
   #####  1.Randon Forest 
   ##### 2.XGBoost 
   ##### 3.Decision Tree
   ##### 4.Support Vector Machine



## 1.Random Forest 

![Screenshot (1692)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/caf495a6-6511-49b3-b35a-58cc9be1eac4)

  ### Accuracy 
  
![Screenshot (1693)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/3194f96b-dc06-4227-a2c8-82bfd7bcb027)
#### ROC curve

![Screenshot (1694)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b56623f6-4c57-4c72-abbc-1a217b59ae6c)
#### Confusion Matrix

![Screenshot (1695)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/0fc78c6a-eeef-41e6-a956-e520d16d9377)

  ### HyperParameter Tuning
  
  ![Screenshot (1696)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/ca79384a-6755-4c2b-ba55-ac3c093b6cb4)
  ![Screenshot (1697)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/9f134770-9429-43d0-93cd-fa6e30c46e43)

  ### Evaluation
  
  ![Screenshot (1698)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/047c72e3-a196-4904-967e-588062448ae6)



## XGBoost

![Screenshot (1699)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/cbd1cdd3-22c0-41b9-a67d-849f50f74c0a)

 
  ### Accuracy 
  ![Screenshot (1700)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/9e76241b-3f28-4e63-af96-fc72a6ccb9e8)
  
  ![Screenshot (1701)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/76e76697-a1a3-469a-945b-7b3e004546ab)



  #### ROC curve

![Screenshot (1702)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/eabcca04-8a35-4564-9cb1-a05d89c83faf)

  #### Confusion Matrix
![Screenshot (1703)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/1fdfeeb1-9d6a-4a85-b1e8-6334e89ae99b)


  ### HyperParameter Tuning
  
![Screenshot (1704)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b8485b43-e07a-4cc1-9e53-a2e00bb4ad6e)


  ### Evaluation
  #### The Accuracy of HyperTuned Model(74%) is Better than non Tuned Model(72%),thus we will take hypertuned model for our model.



  ##  Decision Tree

![Screenshot (1707)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/2bbe5a5c-29af-46b5-8411-e13ed9522bc7)


 
  ### Classification Report 
  ![Screenshot (1712)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/c8aeec05-d66e-4323-a40b-14dd19611911)

 ### Accuracy 

![Screenshot (1708)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/90157619-e070-4dc3-b6b4-675d25adefbe)

  #### ROC curve
![Screenshot (1713)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b5aad9c8-5e15-4a50-98ea-64234d5c2836)

  #### Confusion Matrix
  
  ![Screenshot (1716)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/15a1f930-7b51-421e-9e04-36a606f7a450)



  ### HyperParameter Tuning
  
![Screenshot (1709)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/db7d11d3-4a36-491c-aef2-8401be112c3f)

![Screenshot (1710)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/096acd27-ffa5-4f43-ba37-b5e7f3e1ae22)

  ### Evaluation
  
  #### The Accuracy of HyperTuned Model(74%) is Better than non Tuned Model(72%),thus we will take hypertuned model for our model.



  ## Decision Tree

![Screenshot (1699)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/cbd1cdd3-22c0-41b9-a67d-849f50f74c0a)

 
  ### Accuracy 
  ![Screenshot (1700)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/9e76241b-3f28-4e63-af96-fc72a6ccb9e8)
  
  ![Screenshot (1701)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/76e76697-a1a3-469a-945b-7b3e004546ab)



  #### ROC curve

![Screenshot (1702)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/eabcca04-8a35-4564-9cb1-a05d89c83faf)

  #### Confusion Matrix
![Screenshot (1703)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/1fdfeeb1-9d6a-4a85-b1e8-6334e89ae99b)


  ### HyperParameter Tuning
  
![Screenshot (1704)](https://github.com/21Ayu/Machine_learning-_Capstone-Project/assets/123882702/b8485b43-e07a-4cc1-9e53-a2e00bb4ad6e)


  ### Evaluation
  #### The Accuracy of HyperTuned Model(74%) is Better than non Tuned Model(72%),thus we will take hypertuned model for our model.



  # CONCLUSION

