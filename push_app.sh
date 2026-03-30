git init
git add app.py requirements.txt tokenizer.json image_captioning_model.h5 .gitignore
git commit -m "Initialize project for Streamlit Cloud"
git branch -M main
git remote remove origin || true
git remote add origin https://github.com/AyushNegi-11/NNA_Project.git
git push -u origin main
