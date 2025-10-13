# Oil RAG DRA - Bilingual RAG with Dynamic Retrieval Augmentation

A bilingual document alignment and retrieval-augmented generation system for processing Equinor's annual reports (English/Norwegian, 2010-2024).

## 🚀 Features

- **Bilingual Document Alignment**: High-quality alignment between English and Norwegian documents
- **Dynamic Retrieval Augmentation (DRA)**: Adaptive retrieval system that improves over time
- **Multi-language Support**: English and Norwegian text processing
- **Comprehensive Pipeline**: From PDF parsing to aligned embeddings
- **Statistical Analysis**: Detailed alignment quality metrics

## 📊 Project Status

- ✅ **Data Processing**: Complete bilingual alignment pipeline
- ✅ **Document Parsing**: PDF extraction and chunking
- 🔄 **Core Models**: DRA controller and retrieval system (in progress)
- ⏳ **Web Interface**: Streamlit/API applications (planned)
- ⏳ **Training Pipeline**: Model fine-tuning (planned)

## 🏗️ Project Structure

```
oil_rag_dra/
├── oil_rag/               # Core package
│   ├── core/              # DRA controller and main logic
│   ├── data/processors/   # Data processing pipeline
│   ├── models/            # Model definitions
│   ├── retrieval/         # Retrieval system
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utilities and helpers
├── config/                # Configuration files
├── scripts/               # Processing scripts
│   ├── data/              # Data processing scripts
│   └── deployment/        # Deployment scripts
├── apps/                  # Applications
│   ├── web/               # Web interfaces
│   └── api/               # API services
├── data/                  # Data storage
│   ├── raw/               # Original PDFs
│   ├── processed/         # Aligned and processed data
│   └── metadata/          # Metadata and configurations
├── experiments/           # Experiments and analysis
│   └── notebooks/         # Jupyter notebooks
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 📈 Data Quality Metrics

Based on 2024 data:
- **Total Alignments**: 981 pairs
- **Coverage**: 72% (EN) / 76% (NO)  
- **Average Similarity**: 0.83
- **High Confidence**: 70% of alignments
- **Bidirectional Matches**: 68%

## 🛠️ Installation

```bash
# Clone the repository
git clone <repo-url>
cd oil_rag_dra

# Install dependencies
pip install -r requirements.txt

# Run data processing
python scripts/data/align_reports.py
```

## 📚 Usage

### Data Processing
```bash
# Align bilingual reports
python scripts/data/align_reports.py

# Validate alignments
python scripts/data/validate_alignment.py
```

### Configuration
Edit configuration files in `config/`:
- `data.yaml`: Data processing settings
- `model.yaml`: Model and training configuration

## 🔬 Experiments

Jupyter notebooks in `experiments/notebooks/`:
- Data exploration and analysis
- Model comparison and evaluation
- DRA system visualization

## 📊 Results

The system achieves high-quality bilingual alignment:
- Strong semantic similarity (0.8+ average)
- Good coverage across document sections
- Reliable bidirectional matching

---

**Note**: This project is actively under development. Core DRA functionality and web interfaces are planned for future releases.