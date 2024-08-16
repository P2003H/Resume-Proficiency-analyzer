import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import re
import json

def pdf_to_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_personal_details(resume_text):
    name_pattern = re.compile(r'Name:\s*(.*?)\s', re.IGNORECASE)
    email_pattern = re.compile(r'Email:\s*(.*?)\s', re.IGNORECASE)
    phone_pattern = re.compile(r'Phone:\s*(.*?)\s', re.IGNORECASE)

    name_match = re.search(name_pattern, resume_text)
    email_match = re.search(email_pattern, resume_text)
    phone_match = re.search(phone_pattern, resume_text)

    name = name_match.group(1) if name_match else None
    email = email_match.group(1) if email_match else None
    phone = phone_match.group(1) if phone_match else None

    return name, email, phone

def predict_proficiency_level(resume_text):
    data = [
        ("Beginner", "basic skills"),
        ("Beginner", "entry level"),
        ("Beginner", "novice"),
        ("Intermediate", "some experience"),
        ("Intermediate", "moderate skills"),
        ("Intermediate", "proficient"),
        ("Advanced", "extensive experience"),
        ("Advanced", "highly skilled"),
        ("Advanced", "expert level"),
        ("Advanced", "master")
    ]
    X, y = zip(*data)  

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(y)  

    clf = DecisionTreeClassifier()
    clf.fit(X_vectorized, X)

    resume_vectorized = vectorizer.transform([resume_text])

    proficiency_level = clf.predict(resume_vectorized)[0]

    return proficiency_level

def load_keywords():
    with open('keywords.json') as f:
        return json.load(f)

def predict_domain_with_decision_tree(resume_text, keywords):
    keyword_to_domain = {}
    for domain, domain_keywords in keywords.items():
        for keyword in domain_keywords:
            keyword_to_domain[keyword] = domain

    X_train = []
    y_train = []
    for word in resume_text.split():
        if word.lower() in keyword_to_domain:
            X_train.append(word.lower())
            y_train.append(keyword_to_domain[word.lower()])

    
    print("Input Data:")
    print(X_train)

    
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X_train)

    
    print("Vocabulary:")
    print(vectorizer.vocabulary_)

    
    domain_label_map = {label: idx for idx, label in enumerate(set(y_train))}
    y_train_encoded = [domain_label_map[label] for label in y_train]

    
    clf = DecisionTreeClassifier()
    clf.fit(X_vectorized, y_train_encoded)

    
    predicted_domain_encoded = clf.predict(vectorizer.transform(X_train))

    
    predicted_domain = np.bincount(predicted_domain_encoded).argmax()

  
    predicted_domain_label = [label for label, idx in domain_label_map.items() if idx == predicted_domain][0]

    return predicted_domain_label

def main():
    st.title('Resume Proficiency Analyzer')

    pdf = st.file_uploader(label='Upload Your Resume', type='pdf')

    if pdf is not None:
        resume_text = pdf_to_text(pdf)

        name_from_resume, email_from_resume, phone_from_resume = extract_personal_details(resume_text)

        name = name_from_resume if name_from_resume else st.sidebar.text_input("Name")
        email = email_from_resume if email_from_resume else st.sidebar.text_input("Email")
        phone = phone_from_resume if phone_from_resume else st.sidebar.text_input("Phone")

        proficiency_level = predict_proficiency_level(resume_text)

        keywords = load_keywords()

        domain = predict_domain_with_decision_tree(resume_text, keywords)

        st.subheader('Uploaded Resume')
        st.write(resume_text)

        st.subheader("Personal Details")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")

        st.subheader('Proficiency Level')
        st.write(f'Predicted Proficiency Level: **{proficiency_level}**')

        st.subheader('Predicted Domain (Decision Tree)')
        st.write(f'Based on your resume, you are predicted for: **{domain}**')

if __name__ == "__main__":
    main()
