# Heart Disease Prediction

The purpose of this project is to predict the provability of having any heart disease base on [heart disease prediction dataset ](https://www.kaggle.com/code/ahmedsta/heart-disease-prediction/data) using logistic regression and neural networks implemented by hand and using framework.

## Data understanding

The following sections will explain characteristics of the features, data behavior and preparation of the data set so it can be used as an input for the models.

### Data set features

This section will cover the definition and input for each feature.

#### Definition of the features

Choosing informative, discriminating and independent features is a crucial element of effective algorithms, so the first step is to understand the given features in the data set.

Most of the features given by the repository were defined, the rest was inferred by analyzing the definition of the word used and the input in the data set.

##### Describe on the data set.

The explanation of  the feature was given in the source of the data set

* HeartDisease: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
* BMI: Body Mass Index
* Smoking: Have you smoke at least 100 cigarettes in your entire life [ 5 packs]
* AlcoholDrinking: Heavy drinkers ( adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)
* Stroke: Ever told, you had a stroke
* PhysicalHealth: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30
* MentalHealth: Thinking about your mental health, for how many days during the past 30 days was your mental health not good?
* DiffWalking: Do you have serious difficulty walking or climbing stairs?
* Sex: Are you male or female?
* AgeCategory: Fourteen-level age category

##### Self-search.

The explanation of the feature was giving by searching the meaning of the feature and inputs found in the data set.

* Race: Your ancestor origin ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
* Diabetic: Are you diabetic?
* PhysicalActivity: According to [W.H.O](https://www.who.int/health-topics/physical-activity#tab=tab_1) Refers to all movement. Popular ways to be active include walking, cycling, wheeling, sports, active recreation and play, and can be done at any level of skill and for enjoyment by everybody.
* GenHealth:  General health status ['Very good', 'Fair', 'Good', 'Poor', 'Excellent']
* SleepTime: Sleep time per day
* Asthma: Do you have Asthma?
* Kidney Disease: Do you have Kidney Disease?
* SkinCancer: Do you have Skin Cancer?

#### Data input.

The following list shows what kind of information receives in each feature: ints, floats, or the specific strings. The data was extracting from the data set using the function unique of pandas.

* HeartDisease:['No' 'Yes']
* BMI:float64
* Smoking:['No' 'Yes']
* AlcoholDrinking:['No' 'Yes']
* Stroke:['No' 'Yes']
* PhysicalHealth:float64
* MentalHealth:float64
* DiffWalking:['No' 'Yes']
* Sex:['Female' 'Male']
* AgeCategory:['1-17','18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54','55-59','60-64', '65-69', '70-74', '75-79','80 or older']
* Race:['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
* Diabetic:['Yes' 'No' 'No, borderline diabetes' 'Yes (during pregnancy)']
* PhysicalActivity:['No' 'Yes']
* GenHealth:['Poor', 'Fair','Good','Very good','Excellent']
* SleepTime: float64
* Asthma:['No' 'Yes']
* KidneyDisease:['No' 'Yes']
* SkinCancer:['No' 'Yes']

##### Data example

|   | HeartDisease | BMI   | Smoking | AlcoholDrinking | Stroke | PhysicalHealth | MentalHealth | DiffWalking | Sex    | AgeCategory | Race  | Diabetic | PhysicalActivity | GenHealth | SleepTime | Asthma | KidneyDisease | SkinCancer |
| - | ------------ | ----- | ------- | --------------- | ------ | -------------- | ------------ | ----------- | ------ | ----------- | ----- | -------- | ---------------- | --------- | --------- | ------ | ------------- | ---------- |
| 0 | No           | 16.60 | Yes     | No              | No     | 3.0            | 30.0         | No          | Female | 55-59       | White | Yes      | Yes              | Very good | 5.0       | Yes    | No            | Yes        |
| 1 | No           | 20.34 | No      | No              | Yes    | 0.0            | 0.0          | No          | Female | 80 or older | White | No       | Yes              | Very good | 7.0       | No     | No            | No         |
| 2 | No           | 26.58 | Yes     | No              | No     | 20.0           | 30.0         | No          | Male   | 65-69       | White | Yes      | Yes              | Fair      | 8.0       | Yes    | No            | No         |
| 3 | No           | 24.21 | No      | No              | No     | 0.0            | 0.0          | No          | Female | 75-79       | White | No       | No               | Good      | 6.0       | No     | No            | Yes        |
| 4 | No           | 23.71 | No      | No              | No     | 28.0           | 0.0          | Yes         | Female | 40-44       | White | No       | Yes              | Very good | 8.0       | No     | No            | No         |

### Data behavior

The following section shows the relation of the features and heart diseases by plotting the total cases where the feature is present and the patient have heart disease over the total cases of the disease

<img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Csum%20_%7B%28heart.disease.case%3D%3D%20True%29%5Cin%20feature%20%3D%3D%20True%7D1%20%7D%7B%5Csum%20_%7Bheart.disease.case%3D%3D%20True%7D1%20%7D">



<img src="./Class/Images/dataSexVs.png">



<img src="./Class/Images/dataVs.png">

<img src="./Class/Images/dataDiabeticVs.png">

<img src="./Class/Images/dataAgeVs.png">

<img src="./Class/Images/dataRaceVs.png">

<img src="./Class/Images/dataGenHealthVs.png">

### Data preparation

#### Find missing values

Null values weren't found in the data set, as can be seen in the following table which was extracted using pandas.

RangeIndex: 319795 entries, 0 to 319794

| #  | Column           | Non-Null Count  | Dtypecol |
| -- | ---------------- | --------------- | -------- |
| 0  | HeartDisease     | 319795 non-null | object   |
| 1  | BMI              | 319795 non-null | float64  |
| 2  | Smoking          | 319795 non-null | object   |
| 3  | AlcoholDrinking  | 319795 non-null | object   |
| 4  | Stroke           | 319795 non-null | object   |
| 5  | PhysicalHealth   | 319795 non-null | float64  |
| 6  | MentalHealth     | 319795 non-null | float64  |
| 7  | DiffWalking      | 319795 non-null | object   |
| 8  | Sex              | 319795 non-null | object   |
| 9  | AgeCategory      | 319795 non-null | object   |
| 10 | Race             | 319795 non-null | object   |
| 11 | Diabetic         | 319795 non-null | object   |
| 12 | PhysicalActivity | 319795 non-null | object   |
| 13 | GenHealth        | 319795 non-null | object   |
| 14 | SleepTime        | 319795 non-null | float64  |
| 15 | Asthma           | 319795 non-null | object   |
| 16 | KidneyDisease    | 319795 non-null | object   |
| 17 | SkinCancer       | 319795 non-null | object   |

## Machine learning model.

## Deployment of the machine learning model.
