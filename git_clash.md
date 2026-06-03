git config --global credential.https://github.com.useHttpPath true

git credential-manager reject

# Then immediately push (it should now prompt you)
git push -u origin main