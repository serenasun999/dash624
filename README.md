# Dash App Template for DATA 624

1. Fork the repository to your account
2. Clone the repository to your machine
3. Create a new environment and install the requirements 
4. Customize the dummy text a bit
5. Make sure it runs on your local machine

``` sh
python dashboard.py
```

You should be able to view the website at "http://0.0.0.0:8050/" or "http://localhost:8050/"

6. Create an account at "https://pythonanywhere.com"
7. Log in and open a Bash console
8. Clone your updated app into the home directory
9. Make a virtual environment (this time don't use conda if you used it before). 

``` sh
mkvirtualenv dash624env --python=/usr/bin/python3.10
```
10. Install the requirements into your new virtual environment
11. Navigate to the pythonanywhere "web" link (in the  top right menu in the bash window)
12. Click "Add a new web app"
13. Click on Flask 
14. Click on Python 3.10
15. Enter the path to dashboard.py in pythonanywhere
16. Under the "Virtualenv:" heading write what you named your environment (still in the web tab of pythonanywhere)
17. Uncer the "Code:" heading click on the "WSGI configuration file:" link to edit the file, and change 

``` python
from dashboard import app as application
```

to

``` python
from dashboard import app
application = app.server
```
18. Go back into the "Web" tab, hit reload, and then try the link. 


#### Notes
web on pythonanywhere will not aumatically updated based on any github changes, need to pull changes for integration
