# Setup
- geh auf [JupyterHub](https://jupyterhub.informatik.haw-hamburg.de)
## git setup
- erstelle einen ssh key auf jupyter:
    - ssh-keygen -t ed25519 -C "your_email@example.com"
    - eval "$(ssh-agent -s)"
    - ssh-add ~/.ssh/id_ed25519
    - füge die ausgabe von cat ~/.ssh/id_ed25519.pub als ssh in den github settings hinzu
## kaggle
- erstelle auf kaggle einen lagacy api key dabei wird eine kaggle.json generiert die in des root verzeichnis kopiert werden muss