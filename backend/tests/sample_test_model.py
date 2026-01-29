"""
Test Script for Advanced Career Path Classifier

This script loads the trained model and allows you to:
1. Test with sample resumes
2. Input your own resume text
3. Evaluate predictions with confidence scores
"""

import os
import sys
import numpy as np
import pandas as pd

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_dir)

# Direct import to avoid __init__.py issues
from app.models.naive_bayes.train_model_advanced import AdvancedCareerPathClassifier

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: PDF support not available. Install PyPDF2 or pdfplumber: pip install PyPDF2")


def load_model():
    """Load the trained model."""
    print("="*80)
    print("LOADING TRAINED MODEL")
    print("="*80 + "\n")
    
    try:
        # Get backend directory (tests -> backend)
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(backend_dir, 'data', 'trained_models')
        
        classifier = AdvancedCareerPathClassifier.load_model(
            model_dir, 
            'advanced_career_path_cnb'
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Number of career paths: {len(classifier.classes_)}")
        print(f"✓ Vocabulary size: {len(classifier.vectorizer.vocabulary_)}")
        
        return classifier
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you have trained the model first:")
        print("  python train_model_advanced.py")
        sys.exit(1)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    if not PDF_AVAILABLE:
        print("✗ PDF support not available. Please install PyPDF2:")
        print("  pip install PyPDF2")
        return None
    
    try:
        # Try PyPDF2 first
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text.strip()
        except:
            # Fallback to pdfplumber
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
    
    except FileNotFoundError:
        print(f"✗ File not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"✗ Error reading PDF: {e}")
        return None


def predict_single(classifier, resume_text, show_top_n=5):
    """
    Predict career path for a single resume.
    
    Args:
        classifier: Trained classifier
        resume_text: Resume text string
        show_top_n: Number of top predictions to show
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    # Get prediction (display name)
    prediction = classifier.predict(resume_text, return_display_name=True)
    prob_dict = classifier.predict_proba(resume_text, return_display_names=True)

    # Sort top N predictions by probability
    top_n = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:show_top_n]

    print(f"\n✓ Top Predicted Career Path: {prediction}")
    print(f"\nTop {show_top_n} Career Paths with Confidence Scores:")
    print("-" * 60)
    for i, (career, prob) in enumerate(top_n, 1):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{i}. {career:<40} {prob:.4f} {bar}")
        print(f"{i}. {career:<40} {prob*100:5.2f}% {bar}")
    print("-" * 60)


def test_sample_resumes(classifier):
    """Test with pre-defined sample resumes."""
    print("\n" + "="*80)
    print("TESTING WITH SAMPLE RESUMES")
    print("="*80)
    
    samples = {
        "Machine Learning Engineer": """
            Senior Machine Learning Engineer with 5+ years of experience in developing
            and deploying ML models. Expertise in Python, TensorFlow, PyTorch, and 
            scikit-learn. Built deep learning models for computer vision and NLP tasks.
            Experience with neural networks, CNNs, RNNs, transformers, and BERT.
            Skilled in MLOps, model deployment, AWS, Docker, and Kubernetes.
            Strong background in data preprocessing, feature engineering, and model
            optimization. Published research in top ML conferences.
        """,
        
        "Full Stack Developer": """
            Full Stack Web Developer with expertise in modern web technologies.
            Frontend: React, Angular, Vue.js, HTML5, CSS3, JavaScript, TypeScript.
            Backend: Node.js, Express, Python, Django, Flask, Ruby on Rails.
            Databases: MongoDB, PostgreSQL, MySQL, Redis. Experience with RESTful
            APIs, GraphQL, microservices architecture. Cloud platforms: AWS, Azure.
            Version control: Git, CI/CD pipelines. Agile methodologies and Scrum.
        """,
        
        "Data Scientist": """
            Data Scientist with strong analytical and statistical skills. Proficient
            in Python, R, SQL, and big data tools like Spark and Hadoop. Experience
            with data analysis, data visualization (Tableau, Power BI, matplotlib),
            and statistical modeling. Expertise in A/B testing, experimental design,
            predictive modeling, and machine learning algorithms. Strong communication
            skills for presenting insights to stakeholders and business leaders.
        """,
        
        "DevOps Engineer": """
            DevOps Engineer with extensive experience in CI/CD pipelines and cloud
            infrastructure. Expert in Docker, Kubernetes, Jenkins, GitLab CI, and
            GitHub Actions. Proficient in AWS, Azure, and GCP cloud platforms.
            Infrastructure as Code using Terraform and Ansible. Monitoring with
            Prometheus, Grafana, and ELK stack. Scripting in Bash, Python, and
            PowerShell. Strong focus on automation, reliability, and scalability.
        """,
        
        "Cybersecurity Analyst": """
            Cybersecurity professional specializing in network security and threat
            detection. Experience with security tools: Wireshark, Metasploit, Nessus,
            Burp Suite. Knowledge of penetration testing, vulnerability assessment,
            and security audits. Familiar with SIEM systems, firewalls, IDS/IPS, and
            VPN technologies. Understanding of compliance standards (GDPR, HIPAA, PCI-DSS).
            Certifications: CISSP, CEH, CompTIA Security+, and OSCP.
        """,
        
        "Mobile App Developer": """
            Mobile application developer with experience in iOS and Android development.
            iOS: Swift, Objective-C, Xcode, SwiftUI. Android: Kotlin, Java, Android
            Studio, Jetpack Compose. Cross-platform: React Native, Flutter. Experience
            with mobile UI/UX design, API integration, push notifications, and app
            store deployment. Strong understanding of mobile app architecture, testing,
            and performance optimization.
        """
    }
    
    for expected_role, resume in samples.items():
        print("\n" + "-"*80)
        print(f"Expected Role: {expected_role}")
        print("-"*80)
        print(f"Resume snippet: {resume.strip()[:100]}...")
        
        predict_single(classifier, resume, show_top_n=3)
    
    print("\n" + "="*80)
    print("✓ Sample resume testing completed!")
    print("="*80)


def interactive_mode(classifier):
    """Interactive mode for testing custom resume texts."""
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    print("\nEnter resume text to predict career path.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'sample' to see a sample format.")
    print("-"*80)
    
    sample_format = """
Sample Resume Format:

Software Engineer with 3 years of experience in web development.
Proficient in JavaScript, React, Node.js, and MongoDB. Built scalable
web applications with RESTful APIs. Experience with Agile methodologies
and version control using Git.
"""
    
    while True:
        print("\n" + ">"*80)
        user_input = input("Enter resume text (or command): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Exiting interactive mode. Goodbye!")
            break
        
        if user_input.lower() == 'sample':
            print(sample_format)
            continue
        
        if not user_input:
            print("⚠️  Please enter some text.")
            continue
        
        # Allow multi-line input if text is short
        if len(user_input) < 50:
            print("(Short text detected. Enter more lines, or press Enter on empty line to predict)")
            lines = [user_input]
            while True:
                line = input("  ")
                if not line:
                    break
                lines.append(line)
            user_input = " ".join(lines)
        
        try:
            predict_single(classifier, user_input, show_top_n=5)
        except Exception as e:
            print(f"✗ Error making prediction: {e}")


def test_with_own_resume(classifier):
    """Test with user's own resume (text or PDF)."""
    print("\n" + "="*80)
    print("TEST WITH YOUR OWN RESUME")
    print("="*80)
    
    while True:
        if not PDF_AVAILABLE:
            print("\n✗ PDF support not installed. Install with: pip install PyPDF2")
            break
        pdf_path = input("\nEnter path to your PDF resume (or type 'back' to return): ").strip()
        if pdf_path.lower() in ['back', 'exit', 'quit']:
            break
        if pdf_path:
            pdf_path = pdf_path.strip('"').strip("'")
            print(f"\nExtracting text from PDF: {pdf_path}")
            resume_text = extract_text_from_pdf(pdf_path)
            if resume_text:
                print(f"✓ Extracted {len(resume_text)} characters")
                print(f"\nPreview: {resume_text[:200]}...\n")
                predict_single(classifier, resume_text, show_top_n=5)
            else:
                print("✗ Failed to extract text from PDF")
        again = input("\nTest another resume? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            break
    """Test resumes from a text file (one resume per line or separated by blank lines)."""
    print("\n" + "="*80)
    print("BATCH TESTING FROM FILE")
    print("="*80)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (resumes separated by blank lines)
        resumes = [r.strip() for r in content.split('\n\n') if r.strip()]
        
        print(f"\nFound {len(resumes)} resumes in file: {filepath}\n")
        
        for i, resume in enumerate(resumes, 1):
            print("\n" + "-"*80)
            print(f"Resume {i}/{len(resumes)}")
            print("-"*80)
            print(f"Text: {resume[:100]}...")
            
            predict_single(classifier, resume, show_top_n=3)
        
        print("\n" + "="*80)
        print(f"✓ Batch testing completed! Processed {len(resumes)} resumes.")
        print("="*80)
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
    except Exception as e:
        print(f"✗ Error reading file: {e}")

def main():
    """Main function."""
    print("\n" + "="*80)
    print("CAREER PATH PREDICTION - MODEL TESTING")
    print("="*80 + "\n")
    
    # Load model
    classifier = load_model()
    
    # Interactive menu
    while True:
        print("\n" + "="*80)
        print("CHOOSE TEST MODE")
        print("="*80)
        print("\n1. Test with sample resumes")
        print("2. Test with your own resume (PDF or text)")
        print("3. Exit")
        print("\n" + "-"*80)
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            test_sample_resumes(classifier)
        
        elif choice == '2':
            test_with_own_resume(classifier)
        
        elif choice == '3':
            print("\n" + "="*80)
            print("✓ Testing completed! Goodbye!")
            print("="*80)
            break
        
        else:
            print("\n⚠️  Invalid option. Please select 1-3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
