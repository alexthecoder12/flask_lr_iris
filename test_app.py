'''
Auto-discoverable by pytest when named test_*.py.
Command is `pytest -v`.
Functions besides test_connect and test_index pull dynamic data.
'''

from flask import Flask
import pytest

import main

@pytest.fixture
def client(request):
    test_client = main.app.test_client()

    def teardown():
        pass # databases and resourses have to be freed at the end. But so far we don't have anything

    request.addfinalizer(teardown)
    return test_client

def test_index(client):
    response = client.get('/')
    assert response.status == '200 OK'

def test_predict(client):
	main.load_model() ## load model for predicting ##
	response = client.post('/results', json={'sepal_length' : 1, 
										'sepal_width' : 1, 
										'petal_length' : 1,
										'petal_width' : 1})

	assert response.json == '1'