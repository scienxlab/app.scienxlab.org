from application import application


with application.test_client() as c:
    response = c.get('/')
    assert response.status_code == 200
    assert b'few random little' in response.data

    response = c.get('/polarity')
    assert response.status_code == 200
    assert b'applet generates polarity' in response.data

    response = c.get('/bruges')
    assert response.status_code == 200
    assert b'really useful geophysical' in response.data
