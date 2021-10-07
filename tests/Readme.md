# TODO: write  a test case to check if model is successfully getting created or not?
def test_model_writing():

    1. create some data

    2. run_classification_experiment(data, expeted-model-file)

    assert os.path.isfile(expected-model-file)


# TODO: write a test case to check fitting on training -- litmus test.

def test_small_data_overfit_checking():

    1. create a small amount of data / (digits / subsampling)

    2. train_metrics = run_classification_experiment(train=train, valid=train)

    assert train_metrics['acc']  > some threshold

    assert train_metrics['f1'] > some other threshold
    
 Solution:
 
 1.Small Data was fetched from dataset, only 100 images and targets.
 
 
 2.model was  created bu utils.modelcraetion function and returned.
 
 
 3.In test_utils(in tests folders) model was checked by  assert os.path.isfile(model_path+"/model.joblib")  for checking if created successfully.
 
 
 4.For second part, small amount of data was taken and was trained for 1000 iterations.
 
 
 5.accuracy and f1 score was printed with help of utils.run_classification_experiemnt.
 
 
 6.TRain accuaracy was higher than test because of over fitting.
 
 Screnshhot for pytest:
 ![Screenshot from 2021-10-07 19-47-48](https://user-images.githubusercontent.com/85408006/136407952-31176871-cd66-40ad-b980-87808614ed2f.png)

 
