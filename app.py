# Importer tout le contenu de run.py
from run import app

# Cette ligne est nécessaire pour que gunicorn trouve l'application
if __name__ == "__main__":
    app.run(host='0.0.0.0')
