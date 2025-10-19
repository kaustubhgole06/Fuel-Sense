# Setup instructions for FuelSense project

# Create necessary directories if they don't exist
import os
import sys

def setup_project():
    """Set up the FuelSense project structure"""
    
    print("🔧 Setting up FuelSense project...")
    
    # Create directories
    directories = ['data', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")
    
    # Check if required packages are installed
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'streamlit', 'plotly', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Please install them using: pip install -r requirements.txt")
    else:
        print("\n🎉 All required packages are installed!")
        print("🚀 Ready to run the project!")
        print("\nNext steps:")
        print("1. Run the analysis: jupyter notebook fuel_sense_analysis.ipynb")
        print("2. Launch dashboard: streamlit run app.py")

if __name__ == "__main__":
    setup_project()