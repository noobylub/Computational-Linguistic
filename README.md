# Computational Linguistic Project

A collection of Jupyter notebooks for computational linguistics and natural language processing experiments, including attention mechanisms, transformer models, and reinforcement learning applications.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Computational-Linguistic
   ```

2. **Create and activate virtual environment**
   
   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 📁 Project Structure

- **SmallAttentionBasedLM_(with_TODOs).ipynb** - Implementation of a small decoder-only Transformer with attention mechanisms
- **AttentionAsKernelRegression_(with_TODOs).ipynb** - Attention mechanisms viewed through kernel regression lens
- **Vector,_matrix,_and_data_frame_operations;_sklearn.ipynb** - Linear algebra and data manipulation exercises
- **ReinforcementLearning_DR1_challenge.ipynb** - RL applications in computational linguistics
- **LELA60331_Week_*.ipynb** - Weekly seminar notebooks for LELA60331 course
- **Additional notebooks** - Various computational linguistics experiments and tutorials

## 🧠 Key Topics Covered

- **Attention Mechanisms**: Self-attention, multi-head attention, positional embeddings
- **Transformer Architecture**: Encoder-decoder models, language modeling
- **Linear Algebra**: Matrix operations, vector spaces, embeddings
- **Machine Learning**: Scikit-learn implementations, data preprocessing
- **Reinforcement Learning**: RL applications in NLP tasks
- **Byte-level Tokenization**: UTF-8 based tokenization strategies

## 🛠️ Dependencies

Core libraries:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformer models
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `jupyter` - Interactive notebook environment

See `requirements.txt` for complete list with version specifications.

## 📖 Usage

1. Activate the virtual environment (see setup instructions)
2. Launch Jupyter Notebook: `jupyter notebook`
3. Navigate to the desired notebook
4. Run cells sequentially or experiment with modifications

## 🔧 Development Notes

- The project uses a byte-based tokenization approach for multilingual support
- Virtual environment is excluded from Git via `.gitignore`
- Notebooks contain TODO items for further development and experimentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is for educational purposes. Please refer to the specific licenses of the libraries used.

## 🆘 Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies installed
   ```bash
   source venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Jupyter not found**: Install Jupyter in the virtual environment
   ```bash
   pip install jupyter
   ```

3. **Kernel issues**: Register the virtual environment with Jupyter
   ```bash
   python -m ipykernel install --user --name=venv --display-name="Python (venv)"
   ```

For additional issues, check that you're using the correct Python interpreter from the virtual environment.
