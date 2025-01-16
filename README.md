MLFlow Example Running on Your Machine 

Steps:

1. Install MLFlow on your computer. 
2. Take any example of Classification problem. In my case, i used the famous Iris flower dataset for classification. Write code to do hyperparameter tuning for the chosen problem. Log each experiment run, including parameters, metrics, and artifacts, in MLFlow.
2. Use the best-performing model from MLFlow and serve the model.
3. Use curl or custom python code to make predictions

Using Logistic regression for classification for Iris dataset

mlflow interface:

$ mlflow ui

Best model is logged by registering it and can be accessed using experiment name and version e.g. "models:/Iris_Classification_Model/3"

Serving the model

$ mlflow models serve --model-uri "models:/Iris_Classification_Model/3" --host 127.0.0.1 --port 1234 --env-manager local 

Predicting using curl 

$ curl -X POST -H "Content-Type: application/json" -d '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}'  
http://127.0.0.1:1234/invocations
