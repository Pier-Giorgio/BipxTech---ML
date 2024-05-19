# BIPxTech-TeamSystem: Analysis of Iva data and implement of Machine learning model
### **INTRODUCTION**
In this project, we aim to develop a predictive model capable of determining the VAT exemption code (IvaM) for each line item in an invoice, leveraging the information from other fields within the same line item. The accurate prediction of VAT exemption codes is crucial for financial accuracy, regulatory compliance, and efficient invoice processing in various business contexts.<br>

Ensuring financial accuracy through correctly predicting VAT exemption codes is crucial for maintaining the integrity of financial records. This accuracy helps prevent costly errors and discrepancies that could arise from incorrect VAT calculations. Moreover, automating the prediction of VAT exemption codes enhances operational efficiency by streamlining the invoicing process. This automation reduces the time and effort required for manual entry and validation, leading to faster processing times, lower administrative costs, and improved overall productivity.<br>

Invoices typically contain a wealth of information spread across multiple fields, such as product descriptions, articles, and various tax-related details. The challenge is to utilize this multi-dimensional data effectively to predict the correct VAT exemption code. Our dataset includes these fields, with the "IvaM" column serving as the target variable for our predictions.<br>

The project involves several key stages, starting from the initial data exploration and pre-processing to the development, evaluation, and implementation of the predictive model. Each stage is critical to ensure the robustness and accuracy of the final model. Below, we outline the main objectives and tasks undertaken:

•**Data exploration and analysis**: the initial phase involves a comprehensive analysis of the dataset to understand its structure, identify key features, and address any anomalies such as missing or unmapped values. This step is essential for setting a solid foundation for subsequent modeling efforts.<br> 
•**Data pre-processing**: this step involves cleaning the dataset, handling missing values, encoding categorical variables, and normalizing numerical features; effective pre-processing is crucial forenhancing the model's performance.<br>
•**Model development**: the core of the project is the development of the predictive model. Machine learning algorithms such as random forests and artificial neural networks are explored so as to find the 
model best suited to our goal. The choice of the final model is based on a thorough evaluation of their performance on the validation dataset.<br>
•**Model evaluation**: the selected model is rigorously evaluated using appropriate metrics such as accuracy and test loss. Comparisons between identical models but made up of different structures are also
used to understand and guarantee the best possible reliability and stability of performance for the chosen model.<br>
•**User interface development**: to facilitate user interaction with the model, a simple and intuitive user interface is developed. This interface allows users to input invoice line item details and receive predicted VAT exemption codes in real-time.<br>












<br>

### **ENCODING AND NORMALIZATION**
In this section we will describe the process of encoding and normalizing the data to prepare it for training the predictive model. These steps are critical to ensuring that the model can effectively learn from available features.<br>

We have chosen to use Ordinal Encoding to convert the categorical variables to numeric in our dataset. This technique assigns a unique integer value to each category, preserving the natural order of the categories (when it exists). The choice of Ordinal Encoding is primarily motivated by its simplicity and efficiency, as it is easier to implement and occupies less memory than one-hot encoding, which is ideal for working with large datasets. In addition, Ordinal Encoding is particularly compatible with certain machine learning algorithms that can benefit from the use of ordinal variables where the order of the categories could have an important meaning.<br>

To do this, we selected all non-numeric columns in the dataset, then to ensure compatibility with the encoder, all non-numeric columns were converted into strings and finally the categorical columns were transformed into ordinal values using the ordinal encoder.<br>

Concerning the normalisation process (normaliser Min-Max), we chose to scale the numerical variables in a range between 0 and 1. This process was crucial to ensure that all features have equal importance during the training of the model, preventing certain features from dominating due to their different scales. We chose to normalise rather than standardise because the distributions of our data did not follow a normal distribution. This step is particularly useful for models that are sensitive to feature scales, such as models based on neural networks, and can improve the speed and stability of convergence.


<br>

### **MODEL IMPLEMENTATION - ANN with 3 layers (256-128-64), 100 epochs, batch size 64**
**Choice of Algorithm: Artificial Neural Network (ANN)**<br>
The aim of our work is to develop a predictive model capable of predicting the VAT exemption code for each invoice line. To do this, we opted for a model based on an artificial neural network (ANN).
The decision to use an artificial neural network was guided by the complex and multidimensional nature of the dataset. Neural networks are particularly effective in recognising and modelling complex, non-linear relationships between input variables, making them ideal for this purpose and are particularly suitable for tabular data and can effectively handle multiclass classification. Furthermore, ANNs' ability to work with a large number of inputs and their flexibility in learning complex patterns make them well suited to make the most of the 26 available variables, which may contain significant hidden patterns for VAT code prediction.

**Use of All Variables**<br>
We initially decided to create an initial model using an 'inclusive' approach, using all 26 variables remaining after the data cleaning process. This decision was based on the project's recommendation to explore possible correlations or patterns between various attributes. Using all available variables makes it possible not to overlook potential important predictors, especially at an early stage of modelling, where the focus is on a broad understanding of the dynamics of the dataset.

**Description of the  model (ANN with 3 layers (256-128-64), 100 epochs, batch size 64)** <br>
The initial phase consisted of loading and preparing the dataset, selecting the independent variables (X) by deleting the column 'VatM', which represents the target for classification. The target (y) was transformed numerically through the use of LabelEncoder and then converted into a categorical format, suitable for neural network output.<br>

For the training phase, we divided the dataset into training, validation and test sets in the ratio of 60%, 20% and 20%, respectively. This distribution allowed an adequate evaluation of the model's performance, facilitating the optimisation of hyper-parameters and the prevention of overfitting during the validation phase.<br>

The neural network architecture we chose includes: an input layer equipped with 256 neurons and two hidden layers with 128 and 64 neurons, both activated via the ReLU function to introduce non-linearity essential for modelling complex data dynamics. The output layer contains neurons corresponding to the number of unique classes of 'IvaM' and employs a softmax activation function, which facilitates multinomial classification by providing a probability distribution over the possible classes.<br>

The model was fitted with the categorical_crossentropy loss function and optimised using adam, chosen for its efficiency in automatically adjusting the learning rate. Training was performed for 100 epochs with a batch size of 64, using the validation data to monitor the evolution of the model and prevent overfitting.<br>

The final results of the test showed an **accuracy** of 97.40% and a **loss** of 0.1643, indicating a high level of effectiveness of the model in predicting the VAT exemption code from the available information. This high level of accuracy indicates the correct calibration and generalisation of the model to new data. Next, we wanted to check how the model performed as the training epochs progressed, also to check for a possible risk of overfitting. To do this, we printed out two graphs, one inherent to loss and the other for accuracy.<br>

The loss graph shows a rapid drop in loss in both training and validation in the early epochs, indicating that the model has successfully learnt the fundamental relationships in the data. After this initial phase, the loss continues to decrease gradually, stabilising towards the end of the 100 epochs. Interestingly, despite a slight divergence between the training and validation loss after about 20 epochs, the difference remains relatively small, suggesting that the model does not suffer significantly from overfitting.

![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/afdc4999-5807-4b19-8dd3-a440e29d7321) <br>

The accuracy graph reflects a similar trend to that of loss. The accuracy on both sets increases rapidly at the beginning, reaching a plateau. The training accuracy curve shows a slight advantage over the validation accuracy curve, which is common in most machine learning models, but again, the closeness of the two curves suggests that the model generalises well.<br>

![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/dc3898d8-98da-4066-8a64-cfbe1d7f49b8) <br>
The results are as follows:<br>
**Final Train Loss**: 0.0368<br>
**Final Train Accuracy**: 98.6798%<br>
**Final Validation Loss**: 0.1502<br>
**Final Validation Accuracy**: 97.2761%<br>
The model shows excellent performance, with a balance between learning and generalisation, as demonstrated by the loss and accuracy metrics. These results are very promising and indicate that the model is well trained, with a good ability to generalise to unseen data.<br>

**Why an ANN with 3 layers (256-128-64) and 100 epochs (batch size 64)** <br>
The choice of our model architecture was the result of a rigorous testing process, during which we tested different configurations by varying the number of layers, number of epochs and batch size. This approach allowed us to identify the most effective structure based on key metrics such as accuracy and loss in testing, as well as performance during the training and validation phases.<br>

To determine the optimal configuration, we ran several models with varying architectures. Each model was evaluated in terms of test accuracy, test loss, and training and validation metrics. This systematic process ensured that our final selection was based on solid data representative of model performance in real-world scenarios.<br>

The model that showed the best performance was the one with three hidden layers of 256, 128, and 64 neurons, respectively, trained for 100 epochs with a batch size of 64. This configuration was shown to offer the best balance between effective learning and generalisation capability, being superior in key metrics. In particular, it achieved high test accuracy while maintaining low loss, indicating good resistance to overfitting as confirmed by validation data.<br>

To select the optimal structure of our model, in addition to considering test accuracy and loss, we paid special attention to the phenomenon of overfitting. We carefully assessed the difference between the training and validation metrics for both loss and accuracy. A small discrepancy between these metrics is indicative of a model that generalises well, while larger differences may signal potential overfitting.<br>

• **Loss discrepancy**: 0.1134 (0.1502 validation - 0.0368 train) <br>
• **Accuracy discrepancy**: 1.4037% (98.6798% train - 97.2761% validation) <br>

Our chosen model, characterised by three hidden levels and trained for 100 epochs with a batch size of 64, stood out not only for its high test accuracy, but also for the smallest discrepancies between the training and validation metrics. This indicates that the model is well balanced, showing good generalisation capability without suffering significantly from overfitting.
The decision to select this model as the 'best' was not only based on accuracy, but also on how effectively the model balances learning with generalisation capability.

<br>

### **MODEL OPTIMIZATION PROCESS** <br>







