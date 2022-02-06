
# classification-census-bureau

In this project, a classification model is developed on publicly available Census Bureau data. More information about the model and the used  data can be found in the model card in the ml folder. 

Unit tests were created to monitor the model performance on various slices of the data. The model was deployed using the FastAPI package and API tests were created. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

The model and the data were updated by using git and DVC.

## Run code

To run the code and train a mlp classifier to predict salary based on census data run: 

```
python ml/train_model.py
```

If you want to run the RESTful API run: 

```
uvicorn main:app --reload
```

