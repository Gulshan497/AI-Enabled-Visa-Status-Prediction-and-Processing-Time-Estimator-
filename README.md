AI Enabled Visa Status Prediction and Processing Time Estimator

## 🚀 Live Demo
**[Visit VisaAI](https://ai-visa-prediction-production.up.railway.app)** - Deployed on Railway

This project leverages Machine Learning and data-driven analytics to provide intelligent insights into visa application outcomes. By analyzing historical visa datasets across multiple countries, 
the system predicts the likelihood of visa approval, estimates processing durations, and identifies key factors influencing decision timelines.

## 📋 Table of Contents
- [Features](#-key-capabilities)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [Technology Stack](#-technology-stack)

## ✨ Key Capabilities

- **Visa Approval Prediction**
  Utilizes trained ML models to estimate the probability of visa acceptance based on applicant profiles, visa category, and past trends.

- **Processing Time Estimation**
  Calculates expected processing days using statistical modelling, allowing applicants to plan their timelines more effectively.

- **Data Validation & Cleaning**
  Automatically handles missing values, date inconsistencies, and data-format issues to ensure reliable predictions.

- **Visual Analytics**
  Generates interactive charts and trend insights for better understanding of visa behaviour across regions and years.

- **Country & Visa-Type Insights**
  Offers comparative analysis of processing speeds and decision patterns for different countries and visa categories.

- **Separated Frontend & Backend**
  Independent deployment allows scaling and updates without affecting each component.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VisaAI Platform                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   Frontend (Vercel)  │      │   Backend (Railway)  │    │
│  │                      │      │                      │    │
│  │  - Landing Page      │◄────►│  - Flask API Server  │    │
│  │  - Config Page       │ HTTPS│  - ML Model          │    │
│  │  - Form Interface    │      │  - Predictions       │    │
│  │  - Analytics Charts  │      │  - Data Processing   │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AI Visa Prediction/
├── api.py                          # Backend Flask API (NEW)
├── Procfile.backend                # Railway backend config (NEW)
├── requirements.txt                # Python dependencies
├── visa_dataset.csv                # Training data
├── visa_processing_model.pkl       # Trained ML model
├── preprocessing_info.pkl          # Feature preprocessing info
│
├── frontend/                        # Separated Frontend (NEW)
│   ├── index.html                  # Premium landing page
│   ├── config.html                 # Backend config UI
│   ├── vercel.json                 # Vercel deployment config
│   └── static/
│       ├── css/
│       │   └── styles.css          # Premium styling
│       └── js/
│           ├── main.js             # UI interactions
│           └── api.js              # API communication (NEW)
│
├── Milestone/                       # Project milestones
│   ├── Milestone1.ipynb            # Data exploration
│   ├── MileStone1ProcessingDays.py # Processing time analysis
│   ├── MileStone2EDAandFE.py       # EDA & Feature Engineering
│   ├── Milestone3.py               # ML Model Development
│   └── Milestone4.py               # Full-stack app (original)
│
├── test_prediction.py              # Prediction testing script
├── predict_processing_days.py      # Data processing
├── dataset_tracking.json           # Dataset metadata
│
├── DEPLOYMENT_GUIDE.md             # Detailed deployment steps
├── DEPLOYMENT_CHECKLIST.md         # Quick checklist
└── README.md                       # This file
```

## 🎯 Quick Start

### Prerequisites
- Python 3.12+
- Git
- pip

### Local Development

1. **Clone & Setup**
```bash
cd "C:\Users\gulsh\.cursor\AI Visa Prediction"
pip install -r requirements.txt
```

2. **Run Backend Locally**
```bash
python api.py
# Starts at http://127.0.0.1:5000
```

3. **Open Frontend**
```bash
# Open frontend/index.html in your browser
# Configure backend: http://127.0.0.1:5000
```

4. **Test Prediction**
```bash
python test_prediction.py
```

## 🚀 Deployment

### Current Status
- ✅ Backend ready for Railway deployment
- ✅ Frontend ready for Vercel deployment
- ✅ Documentation complete
- ⏳ Cloud deployment (follow guide below)

### Deploy Backend to Railway
```bash
railway login
railway up
# Get your backend URL from Railway dashboard
```

### Deploy Frontend to Vercel
```bash
cd frontend
vercel --prod
# Or push to GitHub and connect Vercel
```

### Configure Frontend with Backend
1. Open your Vercel frontend URL
2. Click ⚙️ Backend Config
3. Paste your Railway backend URL
4. Click "Test Connection" then "Save"

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions**

## 🛠️ Technology Stack

**Backend**
- Flask 3.1.2 - Web framework
- scikit-learn 1.5.1 - Machine learning
- pandas 2.3.3 - Data processing
- joblib 1.5.3 - Model serialization
- gunicorn 22.0.0 - WSGI server
- flask-cors 4.0.0 - Cross-origin requests

**Frontend**
- HTML5 - Markup
- CSS3 - Styling with animations
- JavaScript - Interactions
- Chart.js - Data visualization
- Google Fonts - Typography

**Deployment**
- Railway - Backend hosting
- Vercel - Frontend hosting
- GitHub - Version control

**ML Model**
- Algorithm: Linear Regression
- Training Data: 800 visa applications
- Features: Country, Visa Type, Application Season, Processing Office
- Output: Estimated processing days

## 📊 Model Performance

- Trained on 800+ visa application records
- Features: Country, Visa Type, Application Month, Processing Office
- Predictions: Processing time in days
- Confidence: Based on historical data accuracy

## 🔐 Security & Privacy

- No personal data stored in predictions
- Frontend-Backend communication via HTTPS on cloud
- CORS enabled for safe cross-origin requests
- Model files kept secure with git

## 📈 Future Enhancements

- [ ] Real-time visa status tracking
- [ ] Multiple ML model comparison
- [ ] User account system
- [ ] Prediction history storage
- [ ] API rate limiting
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] Email notifications

## 📝 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Step-by-step deployment
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Quick reference
- **test_prediction.py** - Model testing examples
- **Milestone/\*.ipynb** - Data science notebooks

## 👨‍💻 Author

**Gulshan Kumar**
- GitHub: [Repository Link]
- LinkedIn: [gulshan-kumar19](https://www.linkedin.com/in/gulshan-kumar19)
- Email: gulshan19112005@gmail.com

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This tool provides estimates based on historical data patterns. Actual visa processing times may vary based on individual circumstances, policy changes, and other factors not covered in the training data. Use these predictions as a reference only.

---

**Last Updated**: January 2026
**Status**: ✅ Production Ready
**Backend**: Live on Railway
**Frontend**: Ready for Vercel

