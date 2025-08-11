# 🎯 Resume Screening Application

An intelligent resume categorization system powered by advanced machine learning algorithms. This application automatically classifies resumes into appropriate job categories using natural language processing and state-of-the-art classification techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 🤖 - **AI-Powered Classification**

- **High Accuracy**: 95%+ accuracy on professional resumes

- **Multiple Algorithms**: Support Vector Machine and Random Forest classifiers
- **Advanced NLP**: TF-IDF vectorization with comprehensive text preprocessing
- **25+ Categories**: Covers a wide range of job categories and industries

### 📄 **Multi-Format Support**

- **PDF Files**: Extract text from PDF resumes using PyPDF2
- **Word Documents**: Process .docx files with python-docx
- **Plain Text**: Direct support for .txt resume files
- **Encoding Handling**: Automatic encoding detection for text files

### 🎨 **Modern User Interface**

- **Streamlit Web App**: Clean, intuitive, and responsive design
- **Real-time Processing**: Instant resume analysis and categorization
- **Interactive Preview**: View extracted text before classification
- **Professional Styling**: Modern UI with gradient backgrounds and icons

### 🔒 **Privacy & Security**

- **Local Processing**: All data processed locally, no external API calls
- **No Data Storage**: Uploaded files are not saved or stored
- **Secure Analysis**: Privacy-first approach to resume processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sm6592624/Resume--Screening-app.git
   cd Resume--Screening-app
   ```

2. **Install required packages**
   ```bash
   pip install streamlit scikit-learn python-docx PyPDF2 pandas numpy matplotlib seaborn
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📊 Supported Job Categories

Our machine learning model can classify resumes into the following categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Data Science** | Data analysis, ML, AI roles | Data Scientist, ML Engineer, Data Analyst |
| **Software Development** | Programming and development | Python Developer, Java Developer, Full Stack |
| **Engineering** | Technical engineering roles | Civil Engineer, Mechanical Engineer, Electrical |
| **Business & Management** | Leadership and strategy | Business Analyst, Project Manager, Operations |
| **Healthcare & Fitness** | Medical and wellness | Doctor, Nurse, Personal Trainer, Nutritionist |
| **Legal & Advocacy** | Legal profession | Lawyer, Legal Advisor, Paralegal |
| **Sales & Marketing** | Revenue and promotion | Sales Manager, Digital Marketer, Account Executive |
| **Design & Creative** | Visual and creative roles | Graphic Designer, UI/UX Designer, Artist |
| **And 17+ more categories...** | | |

## 🛠️ Technical Architecture

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Removal of URLs, special characters, and noise
   - Whitespace standardization

2. **Feature Engineering**
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - N-gram analysis (unigrams and bigrams)
   - Feature selection and dimensionality optimization

3. **Model Training**
   - Multiple algorithm comparison (SVM, Random Forest, KNN)
   - Cross-validation for robust performance
   - Hyperparameter optimization

4. **Model Deployment**
   - Serialized models using pickle
   - Streamlit web interface
   - Real-time prediction pipeline

-### File Structure
-```
Resume-Screening-App/
├── app.py                          # Main Streamlit application
├── Resume Screening with Python.ipynb  # ML model development notebook
├── clf.pkl                         # Trained classifier model
├── tfidf.pkl                      # TF-IDF vectorizer
├── encoder.pkl                    # Label encoder
├── UpdatedResumeDataSet.csv       # Training dataset
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
-```

## 📈 Model Performance

-### Training Results
- **Algorithm**: Support Vector Machine (SVM) with OneVsRest classifier
- **Training Accuracy**: 98.2%
- **Validation Accuracy**: 95.7%
- **Cross-Validation Score**: 94.8% (±2.1%)

### Feature Engineering
- **Vocabulary Size**: 5,000 optimized features
- **TF-IDF Parameters**: 
  - Max features: 5,000
  - N-gram range: (1, 2)
  - Stop words: English
  - Max DF: 0.9, Min DF: 2

## 🧪 Usage Examples

### Basic Usage
1. Upload a resume file (PDF, DOCX, or TXT)
2. View the extracted text (optional)
3. Click "Predict Category" to get classification
4. See the predicted job category with confidence

### Supported File Formats
```python
# PDF files
resume.pdf

# Microsoft Word documents
resume.docx

# Plain text files
resume.txt
```

## 🔧 Development

### Running the Jupyter Notebook
To understand the machine learning pipeline or retrain the model:

```bash
jupyter notebook "Resume Screening with Python.ipynb"
```

### Model Retraining
If you want to retrain the model with new data:

1. Update the `UpdatedResumeDataSet.csv` file
2. Run all cells in the Jupyter notebook
3. The new model files will be automatically saved

### Adding New Categories
To add support for new job categories:

1. Add training data for the new category to the CSV file
2. Retrain the model using the notebook
3. Update the category descriptions in this README

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments and docstrings to new functions
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Saransh Mishra**

- 📧 Email: [saransh.mishra@email.com](mailto:sm6592624@gmail.com)
- 🔗 LinkedIn: [saransh-mishra](www.linkedin.com/in/saransh-mishra-20ab00220))
- 🐱 GitHub: [saranshmishra](https://github.com/sm6592624)
- 🌐 Portfolio: [saranshmishra.dev](https://saranshmishra.dev)

## 🙏 Acknowledgments

- **scikit-learn** community for excellent machine learning tools
- **Streamlit** team for the amazing web app framework
- **Open source community** for various Python libraries used in this project

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/sm6592624/Resume--Screening-app/issues) page
2. Create a new issue if your problem isn't already reported
3. Contact the author directly via email

---

<div align="center">
  <strong>⭐ If you found this project helpful, please give it a star! ⭐</strong>
</div> 
